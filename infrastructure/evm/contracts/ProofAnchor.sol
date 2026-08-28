// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @notice M214 — the per-query on-chain proof-hash anchor: any
/// party anchors keccak256(proof) for every settled query's M193b
/// log-sized proof, giving the settlement batches a Byzantine-safe
/// dispute trail at anchor cost (full verification runs off-chain;
/// full-width ON-CHAIN verification requires a pairing-based SNARK —
/// the production zk stack, M211).
contract ProofAnchor {
    event Anchored(address indexed anchorer, bytes32 indexed proofHash,
                   uint256 blockNumber);

    mapping(bytes32 => uint256) public anchoredAt;

    error NothingToAnchor();

    /// @notice Records the first block a proof hash was anchored.
    /// Append-only: a re-anchor of the same hash is a no-op.
    function anchor(bytes calldata proof) external returns (bytes32 h) {
        if (proof.length == 0) revert NothingToAnchor();
        h = keccak256(proof);
        if (anchoredAt[h] == 0) {
            anchoredAt[h] = block.number;
            emit Anchored(msg.sender, h, block.number);
        }
    }
}
