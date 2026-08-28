// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @notice GEODE governance floors and charter-fixed voting constants
/// (the registered EVM mirror, 27 Aug 2026 — M314 floors, M315
/// takedown floor, M327 voting cap, M328 vote machinery).
///
/// Two classes, deliberately different:
///
/// 1. FLOORS — the security parameters carry hard floors that sit
///    outside ordinary governance. A governance action may RAISE
///    any floor through a two-step timelock; it may never lower
///    one. The raise is a pending state with a delay, so no
///    parameter moves without public notice.
///
/// 2. CHARTER-FIXED — the voting-weight cap and the quorum/diversity
///    constants have NO setter of any kind. They cannot be raised,
///    lowered, or removed by any key. A captured quorum's first
///    plausible vote — raising its own cap — has no execution
///    surface in this contract.
///
/// The zakat end-state rule and the development-fund routing
/// prohibition live in the fund contract; this contract holds the
/// governance parameters only and is deliberately free of any
/// payable path (no funds can ever be held here).
contract GovernanceFloors {
    error NotGovernance();
    error RaiseBelowCurrent(uint256 current, uint256 proposed);
    error TimelockNotElapsed(uint256 readyAt, uint256 now);
    error NoPendingRaise();

    address public immutable governance; // the timelocked governance
                                         // executor (multisig or
                                         // governance contract)

    uint256 public constant MIN_DELAY = 7 days;

    // --- floors: raisable only ------------------------------------
    uint256 public probeRateBps = 50;             // 0.05 floor
    uint256 public vestingEpochsFloor = 4;        // N >= 4 epochs
    uint256 public admissionSampleFloor = 3;      // k >= 3
    uint256 public referenceSampleFloor = 2;      // k_e >= 2
    uint256 public auditFractionBps = 100;        // 1 in 10
    uint256 public takedownMinResponders = 3;     // M315 floor, raisable only

    // --- charter-fixed: no setters exist --------------------------
    uint256 public constant VOTING_CAP_BPS = 2000;      // 20% of total weight
    uint256 public constant DIVERSITY_BASIS_BPS = 2000; // d = ceil(0.2*n)
    uint256 public constant DIVERSITY_MIN = 3;          // d >= 3
    uint256 public constant QUORUM_BPS = 6667;          // 2/3, rounded up
    uint256 public constant UNOPENED_FAIL_CLOSED_BPS = 3334; // > 1/3

    struct PendingRaise {
        uint8  field;   // which floor
        uint256 value;  // proposed value
        uint256 readyAt;
    }

    PendingRaise public pending;

    constructor(address governance_) {
        require(governance_ != address(0), "zero governance");
        governance = governance_;
    }

    modifier onlyGovernance() {
        if (msg.sender != governance) revert NotGovernance();
        _;
    }

    function _floor(uint8 field) internal view returns (uint256) {
        if (field == 0) return probeRateBps;
        if (field == 1) return vestingEpochsFloor;
        if (field == 2) return admissionSampleFloor;
        if (field == 3) return referenceSampleFloor;
        if (field == 4) return auditFractionBps;
        if (field == 5) return takedownMinResponders;
        revert();
    }

    /// @notice Propose raising one floor. Reverts if the proposal is
    /// at or below the current value — lowering is inexpressible.
    function proposeRaise(uint8 field, uint256 value)
        external onlyGovernance
    {
        uint256 current = _floor(field);
        if (value <= current) revert RaiseBelowCurrent(current, value);
        pending = PendingRaise(field, value, block.timestamp + MIN_DELAY);
    }

    function cancelRaise() external onlyGovernance {
        if (pending.readyAt == 0) revert NoPendingRaise();
        delete pending;
    }

    /// @notice Apply a timelocked raise after the delay has elapsed.
    function executeRaise() external onlyGovernance {
        if (pending.readyAt == 0) revert NoPendingRaise();
        if (block.timestamp < pending.readyAt)
            revert TimelockNotElapsed(pending.readyAt, block.timestamp);
        _setFloor(pending.field, pending.value);
        delete pending;
    }

    function _setFloor(uint8 field, uint256 value) internal {
        if (field == 0) probeRateBps = value;
        else if (field == 1) vestingEpochsFloor = value;
        else if (field == 2) admissionSampleFloor = value;
        else if (field == 3) referenceSampleFloor = value;
        else if (field == 4) auditFractionBps = value;
        else if (field == 5) takedownMinResponders = value;
        else revert();
    }

    /// @notice The diversity floor: d = max(3, ceil(0.2 * n).
    function diversityFloor(uint256 responders) public pure returns (uint256) {
        uint256 ceilTerm = (responders * DIVERSITY_BASIS_BPS + 9999)
            / 10000;
        return ceilTerm > DIVERSITY_MIN ? ceilTerm : DIVERSITY_MIN;
    }
}
