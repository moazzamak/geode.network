// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @notice GEODE keyless governance executor (M388 — the M382
/// remainder, registered 29 Aug 2026 in
/// analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md §10.5 before the
/// build).
///
/// The paper says the librarian-replacement vote's executor is "a
/// governance contract with no human key at maturity". M382 created
/// the `governance` address slot on CreditLedger and proved the
/// power lands there, but no such contract existed: `governance`
/// was an EOA, and the retirement claim was only *possible*, not
/// *true*. This contract is that referent.
///
/// Properties:
///  - NO OWNER. No owner(), no admin path, no upgrade path, no
///    pause. The only state is the single pending action and the
///    bond accounting; every transition is permissionless-and-
///    recorded or time-gated. No human key exists after the ledger's
///    owner is renounced.
///  - ONE POWER. As the `governance` on CreditLedger it may replace
///    the librarian (`setLibrarian`) and hand its own role on
///    (`transferGovernance`). Nothing else — the ledger's own
///    modifiers confine it.
///  - TIMELOCKED. Every action waits REPLACEMENT_DELAY (7 days,
///    matching GovernanceFloors.MIN_DELAY). Decision registered: the
///    instant path is the worse failure — a captured proposer could
///    swap the librarian with no notice, and a slow-but-visible
///    replacement lets the network see and respond during the
///    window.
///  - MECHANICAL TRIGGER. A proposal must carry a recorded
///    divergence reason hash; an action with no recorded reason is
///    inexpressible (the paper: "a recorded divergence reason is a
///    fact a validator can replay, and no amount of weight
///    substitutes for it"). Whether a filed reason is real is
///    decided by the replay quorum off-chain — the shared
///    quorum-authentication residual of the G54 block, registered in
///    the review, not improvised here. A fabricated reason is a
///    recorded, replay-visible deviation against its filer, and the
///    delay is the notice window.
///  - PULL BONDS. A proposer's bond is credited to a claimable
///    balance and pulled, never pushed: a recipient that reverts on
///    receive cannot block the path (the InclusionInbox M383
///    lesson).
interface ILedgerGoverned {
    function setLibrarian(address newLibrarian) external;
    function transferGovernance(address newGovernance) external;
    function librarian() external view returns (address);
}

contract LibrarianGovernance {
    error ZeroAddress();
    error NoPending();
    error ZeroReasonHash();
    error WrongBond(uint256 sent, uint256 required);
    error TimelockNotElapsed(uint256 readyAt, uint256 now);
    error NotProposer();
    error NothingToClaim();
    error SendFailed();

    event ReplacementProposed(address indexed newLibrarian,
                              bytes32 reasonHash, uint256 readyAt);
    event SuccessionProposed(address indexed nextGovernance,
                             bytes32 reasonHash, uint256 readyAt);
    event ReplacementExecuted(address indexed newLibrarian,
                              bytes32 reasonHash);
    event SuccessionExecuted(address indexed nextGovernance,
                             bytes32 reasonHash);
    event PendingCancelled(address indexed proposer);
    event BondCredited(address indexed to, uint256 amount);
    event BondClaimed(address indexed to, uint256 amount);

    /// @notice the ledger whose librarian this contract may replace.
    ILedgerGoverned public immutable ledger;
    uint256 public constant REPLACEMENT_DELAY = 7 days;
    uint256 public constant PROPOSAL_BOND = 1 ether;

    enum Kind { NONE, REPLACE_LIBRARIAN, GOVERNANCE_SUCCESSION }

    struct Pending {
        Kind kind;
        address target;      // the new librarian, or the next governance
        bytes32 reasonHash;  // the recorded, replay-checkable trigger
        address proposer;
        uint256 readyAt;
    }

    Pending public pending;
    /// @notice Bond value held against the live pending proposal.
    uint256 public pendingBond;
    /// @notice Pullable bond refunds (payments are pull, not push).
    mapping(address => uint256) public bondClaimable;
    uint256 public claimableTotal;

    constructor(address ledger_) {
        require(ledger_ != address(0), "zero ledger");
        ledger = ILedgerGoverned(ledger_);
    }

    /// @notice File a librarian replacement. Permissionless with a
    /// PROPOSAL_BOND; the recorded divergence reason is required and
    /// the action waits REPLACEMENT_DELAY. A new proposal supersedes
    /// any live pending (crediting the prior proposer's bond to
    /// their claimable balance), so a griefer holding the slot with
    /// a bad filing cannot block a real one forever.
    function proposeReplacement(address newLibrarian, bytes32 reasonHash)
        external payable {
        if (newLibrarian == address(0)) revert ZeroAddress();
        if (reasonHash == bytes32(0)) revert ZeroReasonHash();
        if (msg.value != PROPOSAL_BOND)
            revert WrongBond(msg.value, PROPOSAL_BOND);
        _replacePending(Kind.REPLACE_LIBRARIAN, newLibrarian, reasonHash);
        emit ReplacementProposed(newLibrarian, reasonHash, pending.readyAt);
    }

    /// @notice File a governance succession: this role hands itself
    /// on to `nextGovernance`. Same bond, recorded reason, and
    /// timelock as a replacement, so the role survives its own
    /// succession without a human key.
    function proposeSuccession(address nextGovernance, bytes32 reasonHash)
        external payable {
        if (nextGovernance == address(0)) revert ZeroAddress();
        if (reasonHash == bytes32(0)) revert ZeroReasonHash();
        if (msg.value != PROPOSAL_BOND)
            revert WrongBond(msg.value, PROPOSAL_BOND);
        _replacePending(Kind.GOVERNANCE_SUCCESSION, nextGovernance,
                        reasonHash);
        emit SuccessionProposed(nextGovernance, reasonHash,
                                pending.readyAt);
    }

    /// @notice Execute a timelocked pending action. Permissionless —
    /// anyone may press the button once the delay has elapsed. The
    /// proposer's bond becomes claimable (pull, never push).
    function execute() external {
        Pending memory p = pending;
        if (p.kind == Kind.NONE) revert NoPending();
        if (block.timestamp < p.readyAt)
            revert TimelockNotElapsed(p.readyAt, block.timestamp);
        delete pending;
        _creditBond(p.proposer, PROPOSAL_BOND);
        if (p.kind == Kind.REPLACE_LIBRARIAN) {
            ledger.setLibrarian(p.target);
            emit ReplacementExecuted(p.target, p.reasonHash);
        } else {
            ledger.transferGovernance(p.target);
            emit SuccessionExecuted(p.target, p.reasonHash);
        }
    }

    /// @notice The proposer withdraws a pending action before the
    /// delay elapses; the bond becomes claimable.
    function cancel() external {
        Pending memory p = pending;
        if (p.kind == Kind.NONE) revert NoPending();
        if (msg.sender != p.proposer) revert NotProposer();
        delete pending;
        _creditBond(p.proposer, PROPOSAL_BOND);
        emit PendingCancelled(p.proposer);
    }

    /// @notice Pull a credited bond refund. Pull, never push — a
    /// recipient that reverts on receive cannot block the path.
    function claimBond() external {
        uint256 amount = bondClaimable[msg.sender];
        if (amount == 0) revert NothingToClaim();
        bondClaimable[msg.sender] = 0;
        claimableTotal -= amount;
        (bool ok, ) = msg.sender.call{value: amount}("");
        if (!ok) revert SendFailed();
        emit BondClaimed(msg.sender, amount);
    }

    function _replacePending(Kind kind, address target,
                             bytes32 reasonHash) internal {
        Pending memory p = pending;
        if (p.kind != Kind.NONE) {
            delete pending;
            _creditBond(p.proposer, PROPOSAL_BOND);
        }
        pending = Pending(kind, target, reasonHash, msg.sender,
                          block.timestamp + REPLACEMENT_DELAY);
        pendingBond = PROPOSAL_BOND;
    }

    function _creditBond(address to, uint256 amount) internal {
        bondClaimable[to] += amount;
        claimableTotal += amount;
        pendingBond = 0;
        emit BondCredited(to, amount);
    }
}
