// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/Ownable2StepUpgradeable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuardTransient.sol";
import "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";

/// @notice GEODE settlement ledger — whitepaper-aligned (24 Aug 2026).
///
/// Asset: native ETH. Users deposit ETH; 2.5% routes to the
/// development fund; the rest is attributed by the librarian as
/// epoch-vested credits to registration payout addresses.
///
/// Registered rules implemented here:
/// - One registration form for arms and primitives: operator key +
///   payout address (may differ) + price per unit of work + sealed
///   claim. A primitive's royalty is simply its payout address.
/// - Self-payment exclusion keys on the PAYOUT address (C1).
/// - Vesting: linear over N = 4 epochs; claims are pull-only and
///   account-bound; nothing is pushed.
/// - Slash = burn, graded L0-L3, replay-gated: the librarian files
///   the verdict's evidence hash; the off-chain replay decides guilt.
/// - Settlement batches skip-and-emit: a malformed credit never
///   reverts the batch.
/// - Registration fee and dev-fund changes are timelocked; price
///   changes carry a one-epoch notice period.
///
/// Session-level rules (the unit-of-work table, max unit price, max
/// spend, challenge sessions, shadow probes) are enforced by the
/// gateway and the off-chain registry/ledger; this contract settles
/// the metered outcome.
contract CreditLedger is
    Initializable,
    UUPSUpgradeable,
    Ownable2StepUpgradeable,
    ReentrancyGuardTransient,
    PausableUpgradeable
{
    error ZeroAmount();
    error ZeroAddress();
    error NothingToClaim();
    error BatchShapeMismatch();
    error BatchTooLarge(uint256 length, uint256 max);
    error NotLibrarian();
    error RootAlreadyPosted();
    error NoAttributionRoot();
    error BadProof();
    error EpochNotClosed();
    error NotOperator();
    error AlreadyRegistered();
    error NotRegistered();
    error NoPendingChange();
    error ChangeTooSoon(uint256 changeAt, uint256 now);
    error FeeMismatch(uint256 sent, uint256 required);
    error SendFailed();
    error InvalidLevel(uint8 level);
    error WrongTarget();
    error NotGovernance();
    error NotOwnerOrGovernance();
    error WrongBond();
    error NoSuchFiling();
    error AlreadyResolved();
    error AlreadyChallenged();
    error WindowOpen();
    error WindowClosed();
    error ChallengePending();
    error NotChallenged();
    error InvalidKind(uint8 kind);

    uint256 public constant DEV_FUND_BPS = 25;      // 2.5% of 1000
    uint256 public constant EPOCH = 7 days;
    uint256 public constant VESTING_EPOCHS = 4;     // N=4: tranche 1 after epoch 1
    uint256 public constant MAX_BATCH = 64;
    uint256 public constant CHANGE_DELAY = 2 days;      // admin timelocks
    uint256 public constant PRICE_CHANGE_DELAY = 7 days; // 1-epoch notice
    uint256 public constant SLASH_BOND = 1 ether;   // M386 filing/challenge stake
    uint256 public constant SLASH_WINDOW = 7 days;  // M386 challenge window = 1 epoch

    address public devFund;
    address public librarian;

    struct Registration {
        address operatorKey;
        address payoutAddress;
        uint256 pricePerUnit; // wei per unit of work
        bytes32 sealedClaim;  // artifact hash + fingerprint + contract proof
        bool admitted;
        bool delisted;        // M294 quorum takedown: permanent
        uint256 frozenUntil;  // M323 ministerial freeze: escrow window
        bytes32 freezeEvidence; // commitment-only reference (M323-G3)
    }

    struct CreditEntry {
        bytes32 artifactId;
        address who;   // credited payout address
        uint256 amount;
    }

    mapping(bytes32 => Registration) public regs;
    uint256 public registrationFee;

    // vesting state
    mapping(address => uint256) public creditsOf;   // outstanding (vested + unvested)
    mapping(address => uint256) public claimedOf;
    mapping(address => mapping(uint256 => uint256)) public epochCredits;
    mapping(address => uint256) public lastCollapsed;
    mapping(address => uint256) public matureOf;    // fully vested, collapsed

    uint256 public epochStart;
    uint256 public epochId;

    // solvency accounting (never read raw address(this).balance)
    uint256 public ethHeld;       // gross ETH deposited
    uint256 public devFundShare;  // accrued 2.5% + registration fees
    uint256 public burnedTotal;   // slashed: no claim path ever
    uint256 public attributable;  // undrawn attribution pool

    address public pendingDevFund;
    uint256 public devFundChangeAt;
    uint256 public pendingRegistrationFee;
    uint256 public registrationFeeChangeAt;
    mapping(bytes32 => uint256) public pendingPrice;
    mapping(bytes32 => uint256) public priceChangeAt;

    /// @notice M382 (G53): the executor of the librarian-replacement
    /// vote, in the pattern GovernanceFloors.governance establishes.
    /// It is a SECOND authority over the librarian address, not a
    /// second owner: it holds this one power and nothing else.
    ///
    /// Why it has to exist. setLibrarian was onlyOwner, and the
    /// registered endgame is that the developer renounces ownership.
    /// The suite's own "a renounced owner closes every admin path"
    /// gate proves setLibrarian reverts thereafter — so the on-chain
    /// librarian could never change again, and the two-thirds
    /// earned-weight replacement vote decided something it could not
    /// execute. Splitting the two powers lets the developer's key
    /// close while the replacement path stays open.
    address public governance;

    /// @notice M385 (G54): the pull path for attribution.
    ///
    /// recordCredits is a PUSH: the librarian must name every payee
    /// and every amount, and a payee that is never named is never
    /// paid. That makes one address a liveness dependency for every
    /// participant's income, which is the failure the pull model
    /// exists to remove.
    ///
    /// Here the librarian publishes ONE 32-byte commitment per epoch
    /// instead of N payments, and each payee draws its own credit
    /// against it by proving membership. The librarian cannot pay the
    /// wrong person more than the tree says, cannot decide the order,
    /// cannot withhold an individual payee while paying others, and
    /// cannot be needed a second time once the root is up. A payee
    /// whose leaf is missing can show that against the published
    /// root, which a silent omission from a push batch never allowed.
    mapping(uint256 => bytes32) public attributionRoot;
    /// @notice Amount already drawn per (epoch, leaf). Leaves carry a
    /// CUMULATIVE amount, so a repeated claim is a no-op rather than a
    /// double payment, and a later corrective root can only ever raise
    /// what a payee has already drawn.
    mapping(bytes32 => uint256) public attributionDrawn;

    /// @notice M386 (G54): a pending slash accusation. Anyone files
    /// with SLASH_BOND; a challenge inside SLASH_WINDOW escalates the
    /// filing to the replay quorum. `bondRefundee` names the party
    /// that won the dispute and may pull SLASH_BOND; the loser's bond
    /// is burned (slashBondsBurned) and never paid to anyone.
    struct SlashFiling {
        address who;           // accused payout address
        bytes32 artifactId;
        uint256 amount;
        uint8 level;
        bytes32 evidenceHash;
        address filer;
        uint256 filedAt;
        address challenger;    // address(0) if never challenged
        uint256 resolvedAt;    // 0 while live
        address bondRefundee;  // winner, set at resolution
        bool guilty;           // quorum verdict, challenged resolutions
    }

    SlashFiling[] public slashFilings;   // id = index
    /// @notice Bond ETH held against live filings and challenges.
    /// Accounting note: bonds are NOT part of ethHeld (they are not
    /// deposits); address(this).balance = ethHeld + slashBondHeld +
    /// slashBondsBurned + registryBondHeld + registryBondsBurned.
    uint256 public slashBondHeld;
    /// @notice Bond ETH forfeited by losing a dispute. Unreachable by
    /// any claim path — nobody gains from a false accusation.
    uint256 public slashBondsBurned;

    /// @notice M387 (G54): a proposed registry-status change. The
    /// same bond/window/quorum game as the slash filing, applied to
    /// admission, delisting, and the ministerial freeze — the three
    /// remaining decisions the librarian merely FILES. Kinds:
    /// 0 = admit (admitValue), 1 = delist, 2 = freeze (freezeEpochs).
    struct RegistryFiling {
        uint8 kind;            // 0 admit, 1 delist, 2 freeze
        bytes32 artifactId;
        bool admitValue;       // kind 0
        uint256 freezeEpochs;  // kind 2
        bytes32 evidenceHash;  // quorum record / order reference
        address filer;
        uint256 filedAt;
        address challenger;    // address(0) if never challenged
        uint256 resolvedAt;    // 0 while live
        address bondRefundee;  // winner, set at resolution
        bool guilty;           // quorum verdict, challenged resolutions
    }

    RegistryFiling[] public registryFilings;   // id = index
    uint256 public registryBondHeld;
    uint256 public registryBondsBurned;

    event AttributionRootPosted(uint256 indexed epochId, bytes32 root);
    event AttributionClaimed(uint256 indexed epochId,
                             bytes32 indexed artifactId,
                             address indexed who, uint256 amount);
    event SlashFiled(uint256 indexed filingId, address indexed who,
                     bytes32 indexed artifactId, uint256 amount,
                     uint8 level, bytes32 evidenceHash, address filer);
    event SlashChallenged(uint256 indexed filingId,
                          address indexed challenger);
    event SlashExecuted(uint256 indexed filingId);
    event SlashResolved(uint256 indexed filingId, bool guilty,
                        bytes32 quorumRecordHash);
    event SlashSkipped(uint256 indexed filingId, string reason);
    event SlashBondClaimed(uint256 indexed filingId,
                           address indexed refundee, uint256 amount);
    event RegistryChangeFiled(uint256 indexed filingId, uint8 kind,
                              bytes32 indexed artifactId, bool admitValue,
                              uint256 freezeEpochs, bytes32 evidenceHash,
                              address filer);
    event RegistryChangeChallenged(uint256 indexed filingId,
                                   address indexed challenger);
    event RegistryChangeExecuted(uint256 indexed filingId);
    event RegistryChangeResolved(uint256 indexed filingId, bool guilty,
                                 bytes32 quorumRecordHash);
    event RegistryBondClaimed(uint256 indexed filingId,
                              address indexed refundee, uint256 amount);

    event Deposited(address indexed from, uint256 amount, uint256 devCut);
    event Registered(bytes32 indexed artifactId, address indexed operatorKey,
                     address indexed payoutAddress, uint256 pricePerUnit,
                     bytes32 sealedClaim);
    event Admitted(bytes32 indexed artifactId, bool admitted);
    event Delisted(bytes32 indexed artifactId, bytes32 quorumRecordHash);
    event Frozen(bytes32 indexed artifactId, bytes32 evidenceHash,
                 uint256 frozenUntil);
    event Unfrozen(bytes32 indexed artifactId);
    event Credited(bytes32 indexed artifactId, address indexed who,
                   uint256 amount);
    event CreditSkipped(bytes32 indexed artifactId, address indexed who,
                        uint256 amount, string reason);
    event Claimed(address indexed who, uint256 amount);
    event DevFundClaimed(uint256 amount);
    event Burned(address indexed who, bytes32 indexed artifactId,
                 uint256 amount, uint8 level, bytes32 evidenceHash);
    event LibrarianChanged(address indexed newLibrarian);
    event GovernanceChanged(address indexed newGovernance);
    event DevFundChangeScheduled(address indexed newFund, uint256 changeAt);
    event DevFundChanged(address indexed newFund);
    event RegistrationFeeChangeScheduled(uint256 newFee, uint256 changeAt);
    event RegistrationFeeChanged(uint256 newFee);
    event PriceChangeScheduled(bytes32 indexed artifactId, uint256 newPrice,
                               uint256 changeAt);
    event PriceChanged(bytes32 indexed artifactId, uint256 newPrice);
    event EpochRolled(uint256 epochStart, uint256 epochId);

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize(address devFund_, uint256 registrationFee_)
        public initializer {
        if (devFund_ == address(0)) revert ZeroAddress();
        __Ownable2Step_init();
        __Ownable_init(msg.sender);
        __Pausable_init();
        devFund = devFund_;
        registrationFee = registrationFee_;
        epochStart = block.timestamp;
    }

    /// @notice Deposit native ETH for sessions: 2.5% to the dev-fund
    /// share, the rest to the attribution pool.
    function deposit() external payable whenNotPaused {
        if (msg.value == 0) revert ZeroAmount();
        uint256 devCut = (msg.value * DEV_FUND_BPS) / 1000;
        ethHeld += msg.value;
        devFundShare += devCut;
        attributable += msg.value - devCut;
        emit Deposited(msg.sender, msg.value, devCut);
    }

    /// @notice One registration form for arms and primitives: the
    /// operator key registers, the payout address receives, the price
    /// is per unit of work, the claim is sealed. The fee goes to the
    /// development fund; admission is recorded by the librarian.
    function register(bytes32 artifactId, address payoutAddress,
                      uint256 pricePerUnit, bytes32 sealedClaim)
        external payable whenNotPaused {
        if (payoutAddress == address(0)) revert ZeroAddress();
        if (regs[artifactId].operatorKey != address(0)) {
            revert AlreadyRegistered();
        }
        if (msg.value != registrationFee) {
            revert FeeMismatch(msg.value, registrationFee);
        }
        ethHeld += msg.value;
        devFundShare += msg.value;
        regs[artifactId] = Registration({
            operatorKey: msg.sender,
            payoutAddress: payoutAddress,
            pricePerUnit: pricePerUnit,
            sealedClaim: sealedClaim,
            admitted: false,
            delisted: false,
            frozenUntil: 0,
            freezeEvidence: bytes32(0)
        });
        emit Registered(artifactId, msg.sender, payoutAddress,
                        pricePerUnit, sealedClaim);
    }

    /// @notice The librarian files the admission verdict; the
    /// deterministic rule decides, the key files.
    function setAdmitted(bytes32 artifactId, bool admitted_)
        external onlyLibrarian {
        if (regs[artifactId].operatorKey == address(0)) {
            revert NotRegistered();
        }
        regs[artifactId].admitted = admitted_;
        emit Admitted(artifactId, admitted_);
    }

    /// @notice M294 quorum takedown: the librarian FILES a ratified
    /// validator verdict (the deterministic vote count decides; the
    /// key files). Permanent — no un-delist path. Distinct from the
    /// slash ladder: a takedown burns nothing and moves no credits.
    function setDelisted(bytes32 artifactId, bytes32 quorumRecordHash)
        external onlyLibrarian {
        if (regs[artifactId].operatorKey == address(0)) {
            revert NotRegistered();
        }
        regs[artifactId].delisted = true;
        emit Delisted(artifactId, quorumRecordHash);
    }

    /// @notice M323 ministerial freeze: the librarian FILES an
    /// authenticated, in-nexus order (the off-chain ministerial flow
    /// verifies format, signature, and nexus — the key files, never
    /// decides). The freeze suspends earning and listing for the
    /// order's window. The evidence reference is commitment-only:
    /// no content ever enters this contract (M323-G3).
    ///
    /// There is NO validator function that can lift or bypass the
    /// freeze: release comes only from liftFreeze (confirmation
    /// failure or expiry filed by the librarian) or the expiry of
    /// frozenUntil. Fail-closed by construction (M323-G2).
    function freezeArtifact(bytes32 artifactId, bytes32 evidenceHash,
                            uint256 epochs) external onlyLibrarian {
        if (regs[artifactId].operatorKey == address(0)) {
            revert NotRegistered();
        }
        if (epochs == 0) revert ZeroAmount();
        regs[artifactId].frozenUntil =
            block.timestamp + epochs * EPOCH;
        regs[artifactId].freezeEvidence = evidenceHash;
        emit Frozen(artifactId, evidenceHash, regs[artifactId].frozenUntil);
    }

    /// @notice Librarian-filed release (confirmation failure or the
    /// registered expiry rule). Never validator action.
    function liftFreeze(bytes32 artifactId) external onlyLibrarian {
        if (regs[artifactId].operatorKey == address(0)) {
            revert NotRegistered();
        }
        regs[artifactId].frozenUntil = 0;
        emit Unfrozen(artifactId);
    }

    function isFrozen(bytes32 artifactId) public view returns (bool) {
        return regs[artifactId].frozenUntil > block.timestamp;
    }

    /// @notice Operator-keyed price change, timelocked one epoch so
    /// the next epoch's price table (a ledger entry) is known ahead.
    function schedulePriceChange(bytes32 artifactId, uint256 newPrice)
        external {
        if (msg.sender != regs[artifactId].operatorKey) {
            revert NotOperator();
        }
        if (priceChangeAt[artifactId] != 0) revert NoPendingChange();
        pendingPrice[artifactId] = newPrice;
        priceChangeAt[artifactId] = block.timestamp + PRICE_CHANGE_DELAY;
        emit PriceChangeScheduled(artifactId, newPrice,
                                  priceChangeAt[artifactId]);
    }

    function applyPriceChange(bytes32 artifactId) external {
        if (priceChangeAt[artifactId] == 0) revert NoPendingChange();
        if (block.timestamp < priceChangeAt[artifactId]) {
            revert ChangeTooSoon(priceChangeAt[artifactId], block.timestamp);
        }
        regs[artifactId].pricePerUnit = pendingPrice[artifactId];
        pendingPrice[artifactId] = 0;
        priceChangeAt[artifactId] = 0;
        emit PriceChanged(artifactId, regs[artifactId].pricePerUnit);
    }

    /// @notice The librarian records attribution batches. A malformed
    /// credit is SKIPPED with a logged event, never a revert:
    /// unregistered artifact, self-payment (payer is the payout
    /// address), or an amount above the undrawn pool. Delisted
    /// artifacts (M294 quorum takedown) never earn.
    function recordCredits(address[] calldata payers,
                           CreditEntry[] calldata entries)
        external whenNotPaused {
        if (msg.sender != librarian) revert NotLibrarian();
        if (payers.length != entries.length) revert BatchShapeMismatch();
        if (payers.length == 0) revert ZeroAmount();
        if (payers.length > MAX_BATCH) {
            revert BatchTooLarge(payers.length, MAX_BATCH);
        }
        _rollEpochIfDue();
        for (uint256 i = 0; i < entries.length; ++i) {
            CreditEntry memory e = entries[i];
            if (e.amount == 0) continue;
            Registration storage r = regs[e.artifactId];
            if (r.operatorKey == address(0)) {
                emit CreditSkipped(e.artifactId, e.who, e.amount,
                                   "unregistered artifact");
                continue;
            }
            if (!r.admitted) {
                emit CreditSkipped(e.artifactId, e.who, e.amount,
                                   "not admitted");
                continue;
            }
            if (isFrozen(e.artifactId)) {
                emit CreditSkipped(e.artifactId, e.who, e.amount,
                                   "frozen artifact (M323)");
                continue;
            }
            if (r.delisted) {
                emit CreditSkipped(e.artifactId, e.who, e.amount,
                                   "delisted");
                continue;
            }
            if (payers[i] == r.payoutAddress) {
                emit CreditSkipped(e.artifactId, e.who, e.amount,
                                   "self-payment");
                continue;
            }
            if (e.amount > attributable) {
                emit CreditSkipped(e.artifactId, e.who, e.amount,
                                   "insufficient pool");
                continue;
            }
            attributable -= e.amount;
            creditsOf[e.who] += e.amount;
            epochCredits[e.who][epochId] += e.amount;
            emit Credited(e.artifactId, e.who, e.amount);
        }
    }

    /// @notice M385 (G54): publish the attribution commitment for a
    /// CLOSED epoch. One call, one 32-byte word, no payee names — the
    /// librarian's whole remaining part in getting people paid.
    ///
    /// A root is write-once. Allowing a rewrite would let the
    /// librarian pay a payee, replace the tree, and strand the rest,
    /// which is the withholding this milestone removes. Correcting an
    /// under-payment is a later epoch's root, never an edit to a
    /// published one.
    function postAttributionRoot(uint256 forEpoch, bytes32 root)
        external whenNotPaused {
        if (msg.sender != librarian) revert NotLibrarian();
        if (root == bytes32(0)) revert NoAttributionRoot();
        _rollEpochIfDue();
        // Only a closed epoch can be summarised: its work is done and
        // the tree is final. Committing to an open epoch would be a
        // promise about work that has not happened yet.
        if (forEpoch >= epochId) revert EpochNotClosed();
        if (attributionRoot[forEpoch] != bytes32(0)) {
            revert RootAlreadyPosted();
        }
        attributionRoot[forEpoch] = root;
        emit AttributionRootPosted(forEpoch, root);
    }

    /// @notice M385 (G54): draw a credit against a published root.
    /// PERMISSIONLESS — anyone may push the proof, but the credit
    /// always lands on the `who` named in the leaf, so a third party
    /// can only ever pay someone on time, never redirect them.
    ///
    /// `cumulative` is the payee's running total for this artifact in
    /// this epoch, not a delta, so the call is idempotent: replaying
    /// it pays nothing and reverts nothing that matters.
    function claimAttribution(uint256 forEpoch, bytes32 artifactId,
                              address who, uint256 cumulative,
                              bytes32[] calldata proof)
        external whenNotPaused {
        bytes32 root = attributionRoot[forEpoch];
        if (root == bytes32(0)) revert NoAttributionRoot();
        // Double-hashed leaf: the standard guard against a proof node
        // being passed off as a leaf.
        bytes32 leaf = keccak256(abi.encode(
            keccak256(abi.encode(forEpoch, artifactId, who, cumulative))));
        if (!MerkleProof.verifyCalldata(proof, root, leaf)) {
            revert BadProof();
        }

        bytes32 slot = keccak256(abi.encode(forEpoch, artifactId, who));
        uint256 drawn = attributionDrawn[slot];
        if (cumulative <= drawn) return;   // already paid; idempotent
        uint256 amount = cumulative - drawn;

        // The same eligibility rules the push path applies, in the
        // same order. A root cannot buy an artifact past them.
        Registration storage r = regs[artifactId];
        if (r.operatorKey == address(0)) {
            emit CreditSkipped(artifactId, who, amount,
                               "unregistered artifact");
            return;
        }
        if (!r.admitted) {
            emit CreditSkipped(artifactId, who, amount, "not admitted");
            return;
        }
        if (isFrozen(artifactId)) {
            emit CreditSkipped(artifactId, who, amount,
                               "frozen artifact (M323)");
            return;
        }
        if (r.delisted) {
            emit CreditSkipped(artifactId, who, amount, "delisted");
            return;
        }
        if (amount > attributable) {
            emit CreditSkipped(artifactId, who, amount,
                               "insufficient pool");
            return;
        }

        attributionDrawn[slot] = cumulative;
        attributable -= amount;
        creditsOf[who] += amount;
        // Vesting keys on the epoch the credit is DRAWN in, matching
        // the push path. A late claim vests late; it is not backdated.
        epochCredits[who][epochId] += amount;
        emit Credited(artifactId, who, amount);
        emit AttributionClaimed(forEpoch, artifactId, who, amount);
    }

    /// @notice Pull-only claim of the caller's vested credits;
    /// account-bound — no transfer or assignment path exists.
    function claim() external nonReentrant whenNotPaused {
        address who = msg.sender;
        uint256 vested = _vestedOf(who);
        if (vested <= claimedOf[who]) revert NothingToClaim();
        uint256 amount = vested - claimedOf[who];
        claimedOf[who] = vested;
        (bool ok, ) = payable(who).call{value: amount}("");
        if (!ok) revert SendFailed();
        emit Claimed(who, amount);
    }

    /// @notice The development fund pulls its accrued share
    /// (2.5% of fees + registration fees).
    function claimDevFund() external nonReentrant {
        if (devFundShare == 0) revert NothingToClaim();
        uint256 amount = devFundShare;
        devFundShare = 0;
        (bool ok, ) = payable(devFund).call{value: amount}("");
        if (!ok) revert SendFailed();
        emit DevFundClaimed(amount);
    }

    /// @notice Replay-gated burn (slash ladder). The librarian files
    /// the verdict's evidence hash; the off-chain replay of sealed
    /// data decides guilt. Level 1 burns the unvested promise only;
    /// levels 2-3 additionally delist the artifact. Nobody gains a
    /// slashed amount: it moves to burnedTotal, unreachable by any
    /// claim path.
    function slash(address who, bytes32 artifactId, uint256 amount,
                   uint8 level, bytes32 evidenceHash) external
        whenNotPaused {
        if (msg.sender != librarian) revert NotLibrarian();
        Registration storage r = regs[artifactId];
        if (level == 0 || level > 3) revert InvalidLevel(level);
        if (level >= 2) {
            if (r.operatorKey == address(0)) revert NotRegistered();
            if (r.payoutAddress != who) revert WrongTarget();
        }
        if (amount == 0) revert ZeroAmount();
        uint256 unclaimed = creditsOf[who] - claimedOf[who];
        if (amount > unclaimed) revert NothingToClaim();
        if (level == 1) {
            uint256 unvested = creditsOf[who] - _vestedOf(who);
            if (amount > unvested) revert NothingToClaim();
        }
        creditsOf[who] -= amount;
        _burnFromBuckets(who, amount, level);
        burnedTotal += amount;
        if (level >= 2) {
            r.admitted = false;
            emit Admitted(artifactId, false);
        }
        emit Burned(who, artifactId, amount, level, evidenceHash);
    }

    // --- M386 (G54): propose-and-challenge slash -------------------
    //
    // The direct `slash` above is the librarian's trusted fast path:
    // it files a verdict it already ran the replay for. This block
    // removes the librarian from the COMMON path. Anyone files with a
    // bond, the filing sits in a challenge window, and if nobody
    // refutes it the slash executes with no privileged party anywhere
    // in the call. A refutation escalates to the replay quorum, whose
    // verdict decides guilt by the same replay as ever; the loser of
    // the dispute loses its bond.

    /// @notice File a slash accusation, permissionless, with a bond.
    /// The filing is validated for shape immediately, so a filing
    /// that can never execute is rejected at filing time. It does NOT
    /// burn anything: the amount is only touched on execution, after
    /// the challenge window, or on a quorum verdict.
    function fileSlash(address who, bytes32 artifactId, uint256 amount,
                       uint8 level, bytes32 evidenceHash)
        external payable whenNotPaused returns (uint256 filingId) {
        if (msg.value != SLASH_BOND) revert WrongBond();
        _validateSlashFiling(who, artifactId, amount, level);
        filingId = slashFilings.length;
        slashFilings.push(SlashFiling({
            who: who,
            artifactId: artifactId,
            amount: amount,
            level: level,
            evidenceHash: evidenceHash,
            filer: msg.sender,
            filedAt: block.timestamp,
            challenger: address(0),
            resolvedAt: 0,
            bondRefundee: address(0),
            guilty: false
        }));
        slashBondHeld += SLASH_BOND;
        emit SlashFiled(filingId, who, artifactId, amount, level,
                        evidenceHash, msg.sender);
    }

    /// @notice Refute a filing by replay, within the window, posting
    /// the same bond. The challenger asserts the replay of the filed
    /// evidence does not establish this guilt. A challenged filing can
    /// no longer auto-execute; the replay quorum decides it instead.
    function challengeSlash(uint256 filingId)
        external payable whenNotPaused {
        if (msg.value != SLASH_BOND) revert WrongBond();
        if (filingId >= slashFilings.length) revert NoSuchFiling();
        SlashFiling storage s = slashFilings[filingId];
        if (s.resolvedAt != 0) revert AlreadyResolved();
        if (s.challenger != address(0)) revert AlreadyChallenged();
        if (block.timestamp >= s.filedAt + SLASH_WINDOW) {
            revert WindowClosed();
        }
        s.challenger = msg.sender;
        slashBondHeld += SLASH_BOND;
        emit SlashChallenged(filingId, msg.sender);
    }

    /// @notice Execute an UNCHALLENGED filing after its window has
    /// elapsed. Permissionless — the librarian is nowhere in this
    /// call. If the accused drained its balance during the window the
    /// burn is skipped (emitted) and the filing still resolves so the
    /// filer's bond is returnable; the filing can never burn more than
    /// is actually present at execution time.
    function executeSlash(uint256 filingId) external whenNotPaused {
        if (filingId >= slashFilings.length) revert NoSuchFiling();
        SlashFiling storage s = slashFilings[filingId];
        if (s.resolvedAt != 0) revert AlreadyResolved();
        if (s.challenger != address(0)) revert ChallengePending();
        if (block.timestamp < s.filedAt + SLASH_WINDOW) {
            revert WindowOpen();
        }
        s.resolvedAt = block.timestamp;
        if (!_applySlash(s)) {
            emit SlashSkipped(filingId, "insufficient balance");
        }
        s.bondRefundee = s.filer;
        emit SlashExecuted(filingId);
    }

    /// @notice Resolve a CHALLENGED filing by the replay quorum's
    /// verdict. The librarian is the quorum's filer (the M294
    /// pattern: the deterministic vote count decides, the key files).
    /// Guilty: the slash executes and the challenger's bond is burned
    /// — it staked on a replay the quorum contradicted. Innocent: the
    /// filing is void and the FILER's bond is burned — a false
    /// accusation loses its bond. The winner in each case pulls its
    /// own bond back; no bond is ever paid to the network.
    function resolveSlash(uint256 filingId, bool guilty,
                          bytes32 quorumRecordHash)
        external whenNotPaused {
        if (msg.sender != librarian) revert NotLibrarian();
        if (filingId >= slashFilings.length) revert NoSuchFiling();
        SlashFiling storage s = slashFilings[filingId];
        if (s.resolvedAt != 0) revert AlreadyResolved();
        if (s.challenger == address(0)) revert NotChallenged();
        s.resolvedAt = block.timestamp;
        s.guilty = guilty;
        if (guilty) {
            if (!_applySlash(s)) {
                emit SlashSkipped(filingId, "insufficient balance");
            }
            s.bondRefundee = s.filer;
            slashBondHeld -= SLASH_BOND;
            slashBondsBurned += SLASH_BOND;   // challenger's stake
        } else {
            s.bondRefundee = s.challenger;
            slashBondHeld -= SLASH_BOND;
            slashBondsBurned += SLASH_BOND;   // filer's stake
        }
        emit SlashResolved(filingId, guilty, quorumRecordHash);
    }

    /// @notice Pull the winner's bond after a resolved filing. One
    /// claim, exactly SLASH_BOND, to the party the resolution named.
    function claimSlashBond(uint256 filingId)
        external nonReentrant whenNotPaused {
        if (filingId >= slashFilings.length) revert NoSuchFiling();
        SlashFiling storage s = slashFilings[filingId];
        address refundee = s.bondRefundee;
        if (refundee == address(0)) revert NothingToClaim();
        s.bondRefundee = address(0);
        slashBondHeld -= SLASH_BOND;
        (bool ok, ) = payable(refundee).call{value: SLASH_BOND}("");
        if (!ok) revert SendFailed();
        emit SlashBondClaimed(filingId, refundee, SLASH_BOND);
    }

    /// @dev Shape checks shared by filing-time (revert) and execution
    /// (skip). Identical to the librarian's direct slash path.
    /// Not view: _vestedOf collapses matured buckets.
    function _validateSlashFiling(address who, bytes32 artifactId,
                                  uint256 amount, uint8 level)
        internal {
        Registration storage r = regs[artifactId];
        if (level == 0 || level > 3) revert InvalidLevel(level);
        if (level >= 2) {
            if (r.operatorKey == address(0)) revert NotRegistered();
            if (r.payoutAddress != who) revert WrongTarget();
        }
        if (amount == 0) revert ZeroAmount();
        uint256 unclaimed = creditsOf[who] - claimedOf[who];
        if (amount > unclaimed) revert NothingToClaim();
        if (level == 1) {
            uint256 unvested = creditsOf[who] - _vestedOf(who);
            if (amount > unvested) revert NothingToClaim();
        }
    }

    /// @dev Applies a filing's burn against CURRENT balances; returns
    /// false (and burns nothing) if the balance no longer covers the
    /// amount. Level 2-3 also delist the artifact.
    function _applySlash(SlashFiling storage s) internal returns (bool) {
        Registration storage r = regs[s.artifactId];
        if (s.level == 0 || s.level > 3) return false;
        if (s.level >= 2) {
            if (r.operatorKey == address(0)) return false;
            if (r.payoutAddress != s.who) return false;
        }
        if (s.amount == 0) return false;
        uint256 unclaimed = creditsOf[s.who] - claimedOf[s.who];
        if (s.amount > unclaimed) return false;
        if (s.level == 1) {
            uint256 unvested = creditsOf[s.who] - _vestedOf(s.who);
            if (s.amount > unvested) return false;
        }
        creditsOf[s.who] -= s.amount;
        _burnFromBuckets(s.who, s.amount, s.level);
        burnedTotal += s.amount;
        if (s.level >= 2) {
            r.admitted = false;
            emit Admitted(s.artifactId, false);
        }
        emit Burned(s.who, s.artifactId, s.amount, s.level,
                    s.evidenceHash);
        return true;
    }

    // --- M387 (G54): registry changes are propose-and-challenge ----
    //
    // setAdmitted, setDelisted and freezeArtifact are all filings of
    // an OFF-CHAIN decision (the published evaluation rule, a
    // two-thirds quorum, a confirmed ministerial order) that this
    // contract cannot re-derive. So the privilege is not deleted, it
    // is opened: anyone files the decision with the same bond and
    // window as a slash filing, and an unchallenged filing executes
    // with no privileged party in the call. A refutation escalates to
    // the replay quorum; the loser of the dispute loses its bond.
    // The librarian keeps its direct fast paths (it is the quorum's
    // filer); they are convenience, no longer the only way.
    //
    // liftFreeze (early release) is deliberately NOT in this set. Its
    // challenge incentives invert: the party that benefits from a
    // false filing is the frozen artifact itself, so there is no
    // natural challenger to police an unfreeze. Early release stays
    // with the librarian, who filed the freeze it is now releasing.

    /// @notice File a registry-status change, permissionless, with
    /// the same bond and window as a slash filing. Kind 0 admits (or
    /// de-admits) an artifact, kind 1 delists it permanently, kind 2
    /// freezes it for `freezeEpochs`. Nothing changes until the
    /// window elapses unchallenged, or a quorum verdict confirms it.
    function fileRegistryChange(uint8 kind, bytes32 artifactId,
                                bool admitValue, uint256 freezeEpochs,
                                bytes32 evidenceHash)
        external payable whenNotPaused returns (uint256 filingId) {
        if (msg.value != SLASH_BOND) revert WrongBond();
        _validateRegistryFiling(kind, artifactId, freezeEpochs);
        filingId = registryFilings.length;
        registryFilings.push(RegistryFiling({
            kind: kind,
            artifactId: artifactId,
            admitValue: admitValue,
            freezeEpochs: freezeEpochs,
            evidenceHash: evidenceHash,
            filer: msg.sender,
            filedAt: block.timestamp,
            challenger: address(0),
            resolvedAt: 0,
            bondRefundee: address(0),
            guilty: false
        }));
        registryBondHeld += SLASH_BOND;
        emit RegistryChangeFiled(filingId, kind, artifactId, admitValue,
                                 freezeEpochs, evidenceHash, msg.sender);
    }

    /// @notice Refute a registry filing inside its window, posting
    /// the same bond. The challenger asserts the off-chain decision
    /// was not actually made (the rule fails, the quorum was short,
    /// the order was not confirmed). Escalates to the replay quorum.
    function challengeRegistryChange(uint256 filingId)
        external payable whenNotPaused {
        if (msg.value != SLASH_BOND) revert WrongBond();
        if (filingId >= registryFilings.length) revert NoSuchFiling();
        RegistryFiling storage s = registryFilings[filingId];
        if (s.resolvedAt != 0) revert AlreadyResolved();
        if (s.challenger != address(0)) revert AlreadyChallenged();
        if (block.timestamp >= s.filedAt + SLASH_WINDOW) {
            revert WindowClosed();
        }
        s.challenger = msg.sender;
        registryBondHeld += SLASH_BOND;
        emit RegistryChangeChallenged(filingId, msg.sender);
    }

    /// @notice Execute an UNCHALLENGED registry filing after its
    /// window. Permissionless. The affected party (the operator who
    /// stands to lose income or listing) is the natural challenger of
    /// a false filing, exactly as the accused polices a false slash.
    function executeRegistryChange(uint256 filingId)
        external whenNotPaused {
        if (filingId >= registryFilings.length) revert NoSuchFiling();
        RegistryFiling storage s = registryFilings[filingId];
        if (s.resolvedAt != 0) revert AlreadyResolved();
        if (s.challenger != address(0)) revert ChallengePending();
        if (block.timestamp < s.filedAt + SLASH_WINDOW) {
            revert WindowOpen();
        }
        s.resolvedAt = block.timestamp;
        _applyRegistryChange(s);
        s.bondRefundee = s.filer;
        emit RegistryChangeExecuted(filingId);
    }

    /// @notice Resolve a CHALLENGED registry filing by the replay
    /// quorum's verdict (librarian-filed, the M294 pattern). Guilty:
    /// the change applies and the challenger's bond is burned.
    /// Innocent: the filing is void and the FILER's bond is burned —
    /// a false registry change loses its bond.
    function resolveRegistryChange(uint256 filingId, bool guilty,
                                   bytes32 quorumRecordHash)
        external whenNotPaused {
        if (msg.sender != librarian) revert NotLibrarian();
        if (filingId >= registryFilings.length) revert NoSuchFiling();
        RegistryFiling storage s = registryFilings[filingId];
        if (s.resolvedAt != 0) revert AlreadyResolved();
        if (s.challenger == address(0)) revert NotChallenged();
        s.resolvedAt = block.timestamp;
        s.guilty = guilty;
        if (guilty) {
            _applyRegistryChange(s);
            s.bondRefundee = s.filer;
            registryBondHeld -= SLASH_BOND;
            registryBondsBurned += SLASH_BOND;   // challenger's stake
        } else {
            s.bondRefundee = s.challenger;
            registryBondHeld -= SLASH_BOND;
            registryBondsBurned += SLASH_BOND;   // filer's stake
        }
        emit RegistryChangeResolved(filingId, guilty, quorumRecordHash);
    }

    /// @notice Pull the winner's bond after a resolved registry
    /// filing. One claim, exactly SLASH_BOND, to the party the
    /// resolution named.
    function claimRegistryBond(uint256 filingId)
        external nonReentrant whenNotPaused {
        if (filingId >= registryFilings.length) revert NoSuchFiling();
        RegistryFiling storage s = registryFilings[filingId];
        address refundee = s.bondRefundee;
        if (refundee == address(0)) revert NothingToClaim();
        s.bondRefundee = address(0);
        registryBondHeld -= SLASH_BOND;
        (bool ok, ) = payable(refundee).call{value: SLASH_BOND}("");
        if (!ok) revert SendFailed();
        emit RegistryBondClaimed(filingId, refundee, SLASH_BOND);
    }

    /// @dev Shape checks for a registry filing, applied at filing
    /// time so a filing that can never apply is rejected early.
    function _validateRegistryFiling(uint8 kind, bytes32 artifactId,
                                     uint256 freezeEpochs)
        internal view {
        if (kind > 2) revert InvalidKind(kind);
        if (regs[artifactId].operatorKey == address(0)) {
            revert NotRegistered();
        }
        if (kind == 2 && freezeEpochs == 0) revert ZeroAmount();
    }

    /// @dev Applies a registry filing to current state. Returns false
    /// only if the artifact has ceased to exist, which cannot happen
    /// (there is no unregister path); kept for symmetry with the
    /// slash executor. A freeze never shortens an existing longer
    /// freeze: it may only extend the escrow window.
    function _applyRegistryChange(RegistryFiling storage s)
        internal returns (bool) {
        Registration storage r = regs[s.artifactId];
        if (r.operatorKey == address(0)) return false;
        if (s.kind == 0) {
            r.admitted = s.admitValue;
            emit Admitted(s.artifactId, s.admitValue);
        } else if (s.kind == 1) {
            r.delisted = true;
            emit Delisted(s.artifactId, s.evidenceHash);
        } else if (s.kind == 2) {
            uint256 until = block.timestamp + s.freezeEpochs * EPOCH;
            if (until > r.frozenUntil) {
                r.frozenUntil = until;
                r.freezeEvidence = s.evidenceHash;
                emit Frozen(s.artifactId, s.evidenceHash, until);
            }
        } else {
            return false;
        }
        return true;
    }

    /// @notice Librarian role management: the key that appends
    /// ledger entries and executes deterministic registry updates —
    /// an operator key at bootstrap, a governance contract with no
    /// human key at maturity.
    ///
    /// Callable by the owner (bootstrap) or by governance (the
    /// replacement vote's executor). The second caller is what keeps
    /// the role replaceable after the developer renounces ownership.
    function setLibrarian(address newLibrarian)
        external onlyOwnerOrGovernance {
        librarian = newLibrarian;
        emit LibrarianChanged(newLibrarian);
    }

    function renounceLibrarian() external onlyOwnerOrGovernance {
        librarian = address(0);
        emit LibrarianChanged(address(0));
    }

    /// @notice Owner-only, and only while the owner exists: naming
    /// the governance executor is a bootstrap act.
    function setGovernance(address newGovernance) external onlyOwner {
        governance = newGovernance;
        emit GovernanceChanged(newGovernance);
    }

    /// @notice Governance hands itself on, so the role survives its
    /// own succession once no owner remains to re-appoint it.
    function transferGovernance(address newGovernance) external {
        if (msg.sender != governance || governance == address(0))
            revert NotGovernance();
        governance = newGovernance;
        emit GovernanceChanged(newGovernance);
    }

    function scheduleDevFundChange(address newFund) external onlyOwner {
        if (newFund == address(0)) revert ZeroAddress();
        if (pendingDevFund != address(0)) revert NoPendingChange();
        pendingDevFund = newFund;
        devFundChangeAt = block.timestamp + CHANGE_DELAY;
        emit DevFundChangeScheduled(newFund, devFundChangeAt);
    }

    function applyDevFundChange() external {
        if (pendingDevFund == address(0)) revert NoPendingChange();
        if (block.timestamp < devFundChangeAt) {
            revert ChangeTooSoon(devFundChangeAt, block.timestamp);
        }
        devFund = pendingDevFund;
        pendingDevFund = address(0);
        devFundChangeAt = 0;
        emit DevFundChanged(devFund);
    }

    function scheduleRegistrationFee(uint256 newFee) external onlyOwner {
        if (registrationFeeChangeAt != 0) revert NoPendingChange();
        pendingRegistrationFee = newFee;
        registrationFeeChangeAt = block.timestamp + CHANGE_DELAY;
        emit RegistrationFeeChangeScheduled(newFee,
                                            registrationFeeChangeAt);
    }

    function applyRegistrationFeeChange() external {
        if (registrationFeeChangeAt == 0) revert NoPendingChange();
        if (block.timestamp < registrationFeeChangeAt) {
            revert ChangeTooSoon(registrationFeeChangeAt,
                                 block.timestamp);
        }
        registrationFee = pendingRegistrationFee;
        pendingRegistrationFee = 0;
        registrationFeeChangeAt = 0;
        emit RegistrationFeeChanged(registrationFee);
    }

    function pause() external onlyOwner {
        _pause();
    }

    function unpause() external onlyOwner {
        _unpause();
    }

    /// @notice Vested amount for `who`, computed against wall time
    /// without mutating state: buckets older than N epochs count as
    /// fully vested, the current window vests linearly (tranche 1
    /// after the first epoch, full at the fourth).
    function vestedOf(address who) public view returns (uint256 vested) {
        uint256 currentId = epochId
            + (block.timestamp - epochStart) / EPOCH;
        uint256 first = lastCollapsed[who];
        vested = matureOf[who];
        for (uint256 e = first; e <= currentId; ++e) {
            uint256 bucket = epochCredits[who][e];
            if (bucket == 0) continue;
            uint256 elapsed = currentId - e;
            uint256 frac = elapsed >= VESTING_EPOCHS
                ? 1e18 : (elapsed * 1e18) / VESTING_EPOCHS;
            vested += (bucket * frac) / 1e18;
        }
    }

    function _vestedOf(address who) internal returns (uint256 vested) {
        _rollEpochIfDue();
        uint256 first = lastCollapsed[who];
        if (epochId >= VESTING_EPOCHS
            && first <= epochId - VESTING_EPOCHS) {
            uint256 matured = 0;
            uint256 collapseUntil = epochId - VESTING_EPOCHS;
            for (uint256 e = first; e <= collapseUntil; ++e) {
                matured += epochCredits[who][e];
                delete epochCredits[who][e];
            }
            matureOf[who] += matured;
            lastCollapsed[who] = collapseUntil + 1;
            first = collapseUntil + 1;
        }
        vested = matureOf[who];
        for (uint256 e = first; e <= epochId; ++e) {
            uint256 bucket = epochCredits[who][e];
            if (bucket == 0) continue;
            uint256 elapsed = epochId - e;
            // Invariant: buckets older than N epochs were collapsed
            // above, so elapsed < VESTING_EPOCHS always holds here.
            uint256 frac = (elapsed * 1e18) / VESTING_EPOCHS;
            vested += (bucket * frac) / 1e18;
        }
    }

    /// @dev Removes `amount` from the vesting buckets, newest first.
    /// Level 1 may only consume each bucket's unvested part; levels
    /// 2-3 may consume whole buckets and then the mature balance.
    function _burnFromBuckets(address who, uint256 amount, uint8 level)
        internal {
        uint256 first = lastCollapsed[who];
        uint256 remaining = amount;
        uint256 i = epochId + 1;
        while (remaining > 0 && i > first) {
            --i;
            uint256 b = epochCredits[who][i];
            if (b == 0) continue;
            uint256 elapsed = epochId - i;
            uint256 frac = elapsed >= VESTING_EPOCHS
                ? 1e18 : (elapsed * 1e18) / VESTING_EPOCHS;
            uint256 unvestedHere = b - (b * frac) / 1e18;
            uint256 limit = level == 1 ? unvestedHere : b;
            uint256 remove = remaining < limit ? remaining : limit;
            epochCredits[who][i] = b - remove;
            remaining -= remove;
        }
        if (remaining > 0) {
            matureOf[who] -= remaining;
        }
    }

    function _rollEpochIfDue() internal {
        if (block.timestamp < epochStart + EPOCH) return;
        uint256 elapsed = (block.timestamp - epochStart) / EPOCH;
        epochStart += elapsed * EPOCH;
        epochId += elapsed;
        emit EpochRolled(epochStart, epochId);
    }

    modifier onlyLibrarian() {
        if (msg.sender != librarian) revert NotLibrarian();
        _;
    }

    modifier onlyOwnerOrGovernance() {
        if (msg.sender != owner() && msg.sender != governance)
            revert NotOwnerOrGovernance();
        _;
    }

    /// @dev UUPS authorisation (owner-only upgrades).
    function _authorizeUpgrade(address) internal override onlyOwner {}
}
