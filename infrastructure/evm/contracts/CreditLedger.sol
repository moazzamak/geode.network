// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/Ownable2StepUpgradeable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuardTransient.sol";
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
    error NotOperator();
    error AlreadyRegistered();
    error NotRegistered();
    error NoPendingChange();
    error ChangeTooSoon(uint256 changeAt, uint256 now);
    error FeeMismatch(uint256 sent, uint256 required);
    error SendFailed();
    error InvalidLevel(uint8 level);
    error WrongTarget();

    uint256 public constant DEV_FUND_BPS = 25;      // 2.5% of 1000
    uint256 public constant EPOCH = 7 days;
    uint256 public constant VESTING_EPOCHS = 4;     // N=4: tranche 1 after epoch 1
    uint256 public constant MAX_BATCH = 64;
    uint256 public constant CHANGE_DELAY = 2 days;      // admin timelocks
    uint256 public constant PRICE_CHANGE_DELAY = 7 days; // 1-epoch notice

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

    /// @notice Librarian role management (owner-only): the key that
    /// appends ledger entries and executes deterministic registry
    /// updates — an operator key at bootstrap, a governance contract
    /// with no human key at maturity.
    function setLibrarian(address newLibrarian) external onlyOwner {
        librarian = newLibrarian;
        emit LibrarianChanged(newLibrarian);
    }

    function renounceLibrarian() external onlyOwner {
        librarian = address(0);
        emit LibrarianChanged(address(0));
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

    /// @dev UUPS authorisation (owner-only upgrades).
    function _authorizeUpgrade(address) internal override onlyOwner {}
}
