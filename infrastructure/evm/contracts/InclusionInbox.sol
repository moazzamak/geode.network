// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @notice GEODE on-chain force-inclusion inbox (M336, registered
/// 28 Aug 2026 before the build — the EVM mirror of M312's R-A14a
/// semantics). Any party posts an entry with a content digest; the
/// librarian must incorporate it within the registered window or
/// the chain is invalid.
///
/// M312 semantics, mirrored exactly:
///   - post: any party posts directly to the settlement contract
///     (here: this inbox) — no librarian intermediation;
///   - incorporate: librarian-only, within the window;
///   - validity: the chain is INVALID while any entry sits
///     unincorporated past its deadline — queryable, never silent;
///   - a late incorporation is a recorded violation.
///
/// M365 (G24, 29 Aug 2026) — the inbox was a free bloat weapon. The
/// deposit was fully refunded on the honest path, so spam cost only
/// gas (~160k per post) while every entry obliged the librarian
/// within one window: an attacker bought chain-invalidity, or
/// unbounded ledger bloat, for gas alone. Three repairs, registered
/// in analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md §G24:
///
///   1. The payment splits into a REFUNDABLE bond and a
///      NON-REFUNDABLE posting fee. The fee accrues to the
///      operations line, because the librarian's incorporation cost
///      is the actual externality a poster imposes.
///   2. The fee escalates SUPERLINEARLY past a per-address
///      per-epoch free allowance: the k-th post beyond the
///      allowance costs base*(k+2)^2, so N extra posts cost
///      O(N^3). Spamming the queue is priced, not merely gassed.
///   3. The per-epoch incorporation obligation is CAPPED. Entries
///      beyond the cap roll forward in posting order without
///      invalidating the chain. A censored entry is still
///      guaranteed inclusion — just not instantly — so the
///      censorship guarantee survives while the bloat weapon does
///      not.
///
/// Incorporation is FIFO. That is what makes validity an O(1) head
/// check instead of a scan over an attacker-grown array, and it is
/// what makes "rolls forward in posting order" mean something: the
/// librarian cannot serve a friend ahead of a rival.
///
/// Deadlines are monotonic by construction (each is at least the
/// previous one), so the queue head always carries the earliest
/// deadline.
///
/// Residual, stated not hidden: a spammer who posts first can push
/// an honest poster's deadline out by filling the backlog ahead of
/// it. The superlinear fee is what bounds that — buying N slots of
/// delay costs O(N^3) — but it is a cost barrier, not a proof.
///
/// M382 (G53, 29 Aug 2026) — the librarian address used to be
/// `immutable` here. This is the contract that contains a
/// misbehaving librarian, and it was the one contract that could
/// never learn the librarian had been replaced: a convicted operator
/// kept the inbox role permanently, or the inbox was redeployed and
/// the open queue — including whatever entry someone was being
/// censored over — was abandoned. The address is now READ from a
/// single source, so a replacement propagates without redeployment
/// and without touching the queue.
interface ILibrarianSource {
    function librarian() external view returns (address);
}

contract InclusionInbox {
    error NotLibrarian();
    error WrongPayment(uint256 required, uint256 given);
    error EntryClosed(bytes32 entryId);
    error NotPoster(bytes32 entryId);
    error WindowNotElapsed(bytes32 entryId, uint256 deadlineBlock);
    error EmptyDigest();
    error NotHeadOfQueue(bytes32 entryId, bytes32 head);
    error QueueEmpty();
    error NothingToClaim();
    error NotOperationsLine();

    event Posted(bytes32 indexed entryId, address indexed poster,
                 bytes32 digest, uint256 bond, uint256 fee,
                 uint256 postedBlock, uint256 deadlineBlock);
    event Incorporated(bytes32 indexed entryId, address indexed poster,
                       address indexed incorporatedBy,
                       uint256 incorporatedBlock, bool late);
    event BondReturned(bytes32 indexed entryId, address indexed poster,
                       uint256 amount);
    event FeeAwarded(bytes32 indexed entryId, address indexed earner,
                     uint256 amount);
    event OperationsClaimed(address indexed to, uint256 amount);

    struct Entry {
        address poster;
        bytes32 digest;
        uint256 bond;
        uint256 fee;               // held until incorporation pays it
        uint256 postedBlock;
        uint256 deadlineBlock;
        uint256 incorporatedBlock; // 0 = unincorporated
    }

    address public immutable operationsLine;
    ILibrarianSource public immutable librarianSource;
    uint256 public immutable inclusionWindowBlocks;
    uint256 public immutable bondAmount;
    uint256 public immutable epochBlocks;
    uint256 public immutable basePostingFee;
    uint256 public immutable freePostsPerEpoch;
    uint256 public immutable maxIncorporationsPerEpoch;

    mapping(bytes32 => Entry) public entries;
    mapping(address => mapping(uint256 => uint256)) public postsInEpoch;

    bytes32[] private _queue;
    uint256 private _head;
    uint256 public operationsAccrued;
    uint256 private _lastDeadline;

    /// M383 pull payments: the fee for an entry is credited here to
    /// whoever earned it, and claimed separately. A push would let a
    /// reverting recipient block the queue.
    mapping(address => uint256) public claimable;
    uint256 public librarianIncorporations;
    uint256 public foreignIncorporations;

    constructor(address librarianSource_, uint256 windowBlocks_,
                uint256 bondAmount_, address operationsLine_,
                uint256 epochBlocks_, uint256 basePostingFee_,
                uint256 freePostsPerEpoch_,
                uint256 maxIncorporationsPerEpoch_) {
        require(epochBlocks_ > 0, "epochBlocks must be positive");
        require(maxIncorporationsPerEpoch_ > 0,
                "incorporation cap must be positive");
        require(operationsLine_ != address(0),
                "operations line must be set");
        require(librarianSource_ != address(0),
                "librarian source must be set");
        librarianSource = ILibrarianSource(librarianSource_);
        inclusionWindowBlocks = windowBlocks_;
        bondAmount = bondAmount_;
        operationsLine = operationsLine_;
        epochBlocks = epochBlocks_;
        basePostingFee = basePostingFee_;
        freePostsPerEpoch = freePostsPerEpoch_;
        maxIncorporationsPerEpoch = maxIncorporationsPerEpoch_;
    }

    function currentEpoch() public view returns (uint256) {
        return block.number / epochBlocks;
    }

    /// @notice the librarian of record, read live so a replacement
    /// takes effect here the moment it takes effect anywhere.
    function librarian() public view returns (address) {
        return librarianSource.librarian();
    }

    /// @notice what the next post from `poster` costs this epoch.
    /// Callers must send exactly bondAmount + this.
    function postingFee(address poster) public view
        returns (uint256) {
        uint256 made = postsInEpoch[poster][currentEpoch()];
        if (made < freePostsPerEpoch) return basePostingFee;
        uint256 over = made - freePostsPerEpoch + 2;
        return basePostingFee * over * over;
    }

    function openCount() public view returns (uint256) {
        return _queue.length - _head;
    }

    /// @notice the oldest unincorporated entry, or zero if none.
    function headEntry() public view returns (bytes32) {
        if (_head >= _queue.length) return bytes32(0);
        return _queue[_head];
    }

    /// @notice any party posts an entry with a digest, a bond, and
    /// a non-refundable posting fee.
    function post(bytes32 entryId, bytes32 digest) external payable {
        if (digest == bytes32(0)) revert EmptyDigest();
        if (entries[entryId].postedBlock != 0)
            revert EntryClosed(entryId);

        uint256 fee = postingFee(msg.sender);
        uint256 required = bondAmount + fee;
        if (msg.value != required)
            revert WrongPayment(required, msg.value);

        // the obligation is capped per epoch, so a backlog beyond
        // the cap rolls forward instead of invalidating the chain
        uint256 backlog = openCount();
        uint256 extraEpochs = backlog / maxIncorporationsPerEpoch;
        uint256 deadline = block.number + inclusionWindowBlocks
            + extraEpochs * epochBlocks;
        if (deadline < _lastDeadline) deadline = _lastDeadline;
        _lastDeadline = deadline;

        entries[entryId] = Entry({
            poster: msg.sender,
            digest: digest,
            bond: bondAmount,
            fee: fee,
            postedBlock: block.number,
            deadlineBlock: deadline,
            incorporatedBlock: 0
        });
        _queue.push(entryId);
        postsInEpoch[msg.sender][currentEpoch()] += 1;

        emit Posted(entryId, msg.sender, digest, bondAmount, fee,
                    block.number, deadline);
    }

    /// @notice Incorporate the head of the queue. PERMISSIONLESS by
    /// M383: every line below is forced by on-chain state, so gating
    /// the call bought no safety and cost the whole censorship
    /// surface. A poster who is being ignored incorporates its own
    /// entry; nobody has to be asked.
    ///
    /// FIFO still binds, so a caller cannot reorder around a rival.
    /// The bond returns to the poster; the fee does not. The caller
    /// is recorded, which is a sharper liveness signal than a missed
    /// deadline was: an entry a stranger had to incorporate is direct
    /// evidence the librarian was not doing its job.
    function incorporate(bytes32 entryId) external {
        if (_head >= _queue.length) revert QueueEmpty();
        bytes32 head = _queue[_head];
        if (entryId != head) revert NotHeadOfQueue(entryId, head);

        Entry storage e = entries[entryId];
        if (e.incorporatedBlock != 0) revert EntryClosed(entryId);
        e.incorporatedBlock = block.number;
        bool late = block.number > e.deadlineBlock;
        _head += 1;

        // The fee follows the work. Paying it to the operations line
        // regardless of who did the job left a stalled librarian
        // earning exactly as much as a prompt one.
        address earner;
        if (msg.sender == librarian()) {
            librarianIncorporations += 1;
            earner = operationsLine;
        } else {
            foreignIncorporations += 1;
            // Within the window the librarian still holds the job, so
            // a foreign caller may clear the queue but earns nothing.
            // Otherwise a poster could post and instantly refund its
            // own fee, and the superlinear spam price would collapse.
            earner = late ? msg.sender : operationsLine;
        }
        uint256 fee = e.fee;
        if (fee > 0) {
            e.fee = 0;
            claimable[earner] += fee;
            if (earner == operationsLine) operationsAccrued += fee;
        }

        emit Incorporated(entryId, e.poster, msg.sender, block.number,
                          late);
        emit FeeAwarded(entryId, earner, fee);
        // The bond is CREDITED to the poster, never pushed: a poster
        // whose contract reverts on receive must not be able to jam
        // the queue head (the M383 pull lesson, applied to the bond).
        uint256 amount = e.bond;
        if (amount > 0) {
            e.bond = 0;
            claimable[e.poster] += amount;
            emit BondReturned(entryId, e.poster, amount);
        }
    }

    /// @notice if the librarian misses the deadline, the poster
    /// withdraws the bond. The bond is CREDITED to the poster's
    /// claimable balance, never pushed (a reverting receiver cannot
    /// jam the queue). The entry stays open and the chain stays
    /// invalid meanwhile — withdrawing does not discharge the
    /// obligation. The posting fee is never returned.
    function withdrawBond(bytes32 entryId) external {
        Entry storage e = entries[entryId];
        if (e.postedBlock == 0) revert NotPoster(entryId);
        if (e.incorporatedBlock != 0) revert EntryClosed(entryId);
        if (msg.sender != e.poster) revert NotPoster(entryId);
        if (block.number <= e.deadlineBlock)
            revert WindowNotElapsed(entryId, e.deadlineBlock);
        uint256 amount = e.bond;
        e.bond = 0;
        claimable[e.poster] += amount;
        emit BondReturned(entryId, e.poster, amount);
    }

    /// @notice the operations line pulls the fees it earned.
    function claimOperations() external {
        if (msg.sender != operationsLine) revert NotOperationsLine();
        uint256 amount = claimable[operationsLine];
        claimable[operationsLine] = 0;
        operationsAccrued = 0;
        (bool ok, ) = payable(operationsLine).call{value: amount}("");
        require(ok, "operations claim failed");
        emit OperationsClaimed(operationsLine, amount);
    }

    /// @notice pull whatever bounties the caller has earned by
    /// clearing entries the librarian left past their deadline.
    function claim() external {
        uint256 amount = claimable[msg.sender];
        if (amount == 0) revert NothingToClaim();
        claimable[msg.sender] = 0;
        if (msg.sender == operationsLine) operationsAccrued = 0;
        (bool ok, ) = payable(msg.sender).call{value: amount}("");
        require(ok, "claim failed");
        emit OperationsClaimed(msg.sender, amount);
    }

    function isLate(bytes32 entryId) public view returns (bool) {
        Entry storage e = entries[entryId];
        if (e.postedBlock == 0 || e.incorporatedBlock != 0)
            return false;
        return block.number > e.deadlineBlock;
    }

    /// @notice M312 chain validity: false while the oldest
    /// unincorporated entry sits past its deadline. Deadlines are
    /// monotonic in posting order, so the head carries the earliest
    /// one and this is O(1) — an attacker cannot make the check
    /// itself expensive.
    function chainValid() public view returns (bool) {
        if (_head >= _queue.length) return true;
        return block.number <= entries[_queue[_head]].deadlineBlock;
    }
}
