// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @notice GEODE on-chain force-inclusion inbox (M336, registered
/// 28 Aug 2026 before the build — the EVM mirror of M312's R-A14a
/// semantics). Any party posts an entry with a content digest and a
/// deposit; the librarian must incorporate it within the registered
/// window or the chain is invalid. The deposit returns on
/// incorporation, or to the poster if the librarian fails.
///
/// M312 semantics, mirrored exactly:
///   - post: any party posts directly to the settlement contract
///     (here: this inbox) — no librarian intermediation;
///   - incorporate: librarian-only, within the window;
///   - validity: the chain is INVALID while any entry sits
///     unincorporated past its window — queryable, never silent;
///   - a late incorporation is a recorded violation (the chain was
///     invalid from the posting deadline until then).
///
/// The window is registered in BLOCKS: the on-chain stand-in for the
/// M312 epoch. inclusionWindowBlocks is charter-fixed at deploy (no
/// setter). chainValid() scans the open-entry array, which is
/// bounded by posters' deposits (each post locks the minimum
/// deposit), so the scan is finite and spend-bounded — the
/// registered honest limit.
contract InclusionInbox {
    error NotLibrarian();
    error DepositTooSmall(uint256 required, uint256 given);
    error EntryClosed(bytes32 entryId);
    error NotPoster(bytes32 entryId);
    error WindowNotElapsed(bytes32 entryId, uint256 deadlineBlock);
    error EmptyDigest();
    error NotOpen(bytes32 entryId);

    event Posted(bytes32 indexed entryId, address indexed poster,
                 bytes32 digest, uint256 deposit,
                 uint256 postedBlock);
    event Incorporated(bytes32 indexed entryId, address indexed poster,
                       uint256 incorporatedBlock, bool late);
    event DepositReturned(bytes32 indexed entryId, address indexed poster,
                          uint256 amount);

    struct Entry {
        address poster;
        bytes32 digest;
        uint256 deposit;
        uint256 postedBlock;
        uint256 incorporatedBlock; // 0 = unincorporated
    }

    address public immutable librarian;
    uint256 public immutable inclusionWindowBlocks;
    uint256 public immutable minDeposit;

    mapping(bytes32 => Entry) public entries;
    bytes32[] private _openIds;      // unincorporated entries

    constructor(address librarian_, uint256 windowBlocks_,
                uint256 minDeposit_) {
        librarian = librarian_;
        inclusionWindowBlocks = windowBlocks_;
        minDeposit = minDeposit_;
    }

    function _deadline(uint256 postedBlock) internal view
        returns (uint256) {
        return postedBlock + inclusionWindowBlocks;
    }

    function _isLate(uint256 postedBlock) internal view
        returns (bool) {
        return block.number > _deadline(postedBlock);
    }

    /// @notice any party posts an entry with a digest and a deposit
    function post(bytes32 entryId, bytes32 digest) external payable {
        if (digest == bytes32(0)) revert EmptyDigest();
        if (entries[entryId].postedBlock != 0)
            revert EntryClosed(entryId);
        if (msg.value < minDeposit)
            revert DepositTooSmall(minDeposit, msg.value);
        entries[entryId] = Entry({
            poster: msg.sender,
            digest: digest,
            deposit: msg.value,
            postedBlock: block.number,
            incorporatedBlock: 0
        });
        _openIds.push(entryId);
        emit Posted(entryId, msg.sender, digest, msg.value,
                    block.number);
    }

    /// @notice the librarian incorporates a posted entry (only the
    /// librarian). The deposit returns to the poster. A late
    /// incorporation is recorded as a violation.
    function incorporate(bytes32 entryId) external {
        if (msg.sender != librarian) revert NotLibrarian();
        Entry storage e = entries[entryId];
        if (e.postedBlock == 0) revert NotPoster(entryId);
        if (e.incorporatedBlock != 0) revert EntryClosed(entryId);
        e.incorporatedBlock = block.number;
        bool late = _isLate(e.postedBlock);
        _closeOpen(entryId);
        emit Incorporated(entryId, e.poster, block.number, late);
        uint256 amount = e.deposit;
        e.deposit = 0;
        (bool ok, ) = payable(e.poster).call{value: amount}("");
        require(ok, "deposit return failed");
        emit DepositReturned(entryId, e.poster, amount);
    }

    /// @notice if the librarian fails to incorporate within the
    /// window, the poster withdraws the deposit (the chain stays
    /// invalid meanwhile — the entry remains open and visible).
    function withdrawDeposit(bytes32 entryId) external {
        Entry storage e = entries[entryId];
        if (e.postedBlock == 0) revert NotPoster(entryId);
        if (e.incorporatedBlock != 0) revert EntryClosed(entryId);
        if (msg.sender != e.poster) revert NotPoster(entryId);
        if (!_isLate(e.postedBlock))
            revert WindowNotElapsed(entryId,
                                    _deadline(e.postedBlock));
        uint256 amount = e.deposit;
        e.deposit = 0;
        (bool ok, ) = payable(e.poster).call{value: amount}("");
        require(ok, "deposit return failed");
        emit DepositReturned(entryId, e.poster, amount);
    }

    function _closeOpen(bytes32 entryId) internal {
        uint256 n = _openIds.length;
        for (uint256 i = 0; i < n; i++) {
            if (_openIds[i] == entryId) {
                _openIds[i] = _openIds[n - 1];
                _openIds.pop();
                return;
            }
        }
        revert NotOpen(entryId);
    }

    function openCount() public view returns (uint256) {
        return _openIds.length;
    }

    function isLate(bytes32 entryId) public view returns (bool) {
        Entry storage e = entries[entryId];
        if (e.postedBlock == 0 || e.incorporatedBlock != 0)
            return false;
        return _isLate(e.postedBlock);
    }

    /// @notice M312 chain validity: false while any entry sits
    /// unincorporated past its window. Scans the open list, which
    /// is spend-bounded by the posting deposit.
    function chainValid() public view returns (bool) {
        for (uint256 i = 0; i < _openIds.length; i++) {
            Entry storage e = entries[_openIds[i]];
            if (e.incorporatedBlock == 0 &&
                _isLate(e.postedBlock)) {
                return false;
            }
        }
        return true;
    }
}
