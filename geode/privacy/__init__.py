"""Privacy and integrity cryptography: secret sharing (MPC) and the
zero-knowledge arguments for the ridge-head relation.

Layering rule: imports only ``geode.hashing`` and standard libraries.
"""
from geode.privacy.secret_sharing import (
    recombine_additive,
    replicated_gram_shares,
    shamir_reconstruct,
    shamir_split,
    signed_from_field,
    split_additive,
    to_field,
)
from geode.privacy.zk_bulletproofs import (
    commit_vec,
    proof_size_bytes,
    prove,
    verify,
)
from geode.privacy.zk_dispute import (
    build_dispute_payload,
    verify_dispute_payload,
)
from geode.privacy.zk_onchain import (
    challenge_serialization,
    generator_label,
    proof_length_bytes,
    rounds_of,
    serialize,
    serialize_hex,
    words_hex,
)

__all__ = [
    "build_dispute_payload",
    "challenge_serialization",
    "commit_vec",
    "generator_label",
    "proof_length_bytes",
    "proof_size_bytes",
    "prove",
    "recombine_additive",
    "replicated_gram_shares",
    "rounds_of",
    "serialize",
    "serialize_hex",
    "shamir_reconstruct",
    "shamir_split",
    "signed_from_field",
    "split_additive",
    "to_field",
    "verify",
    "words_hex",
    "verify_dispute_payload",
]
