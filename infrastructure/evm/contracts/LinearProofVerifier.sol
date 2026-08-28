// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @notice M213 — the EVM verification hook for the M193b log-sized
/// (Bulletproofs-style) argument for the ridge-head relation
/// y = <w, x> + b. A DIRECT port of `geode/zk_bulletproofs.py`
/// `verify`: same group constants, same seed-derived hash-to-point
/// generators (sha256 precompile), same Fiat-Shamir serialization
/// (minimal lowercase hex joined by ';'), same fold equations.
///
/// Registered proof layout (32 bytes per element, r = log2(n)):
///   [C][L_0..L_{r-1}][R_0..R_{r-1}][a_final][b_final][r_final]
/// The claim is a separate argument (the public statement) — the
/// proof bytes are exactly the sealed M193b size (1,024 B at r = 14).
///
/// Registered honesty: the weight vector w is PUBLIC here, so the
/// modexp count is O(n) (the H-multiexp plus the G/H generator folds
/// the Python verifier performs). The committed-weights variant
/// (O(log n) verifier) is the registered follow-up cell.
contract LinearProofVerifier {
    // The M193b group: seed-derived 256-bit safe prime p = 2q + 1.
    uint256 private constant P =
        0xe7e29e4580e8d316e042f3b524116998e03e693f404bac69c94eac22d7e7842b;
    uint256 private constant Q =
        0x73f14f22c074698b702179da9208b4cc701f349fa025d634e4a756116bf3c215;

    error BadProofLength(uint256 given, uint256 expected);
    error NotPowerOfTwo(uint256 n);

    event Verification(uint256 indexed n, bool result);

    // ----------------------------------------------------------- precompiles
    function _sha256(bytes memory input) private view
        returns (bytes32 out) {
        (bool ok, bytes memory data) =
            address(0x02).staticcall(input);
        require(ok, "sha256 failed");
        out = abi.decode(data, (bytes32));
    }

    // NOTE (M213, registered finding): the 0x05 modexp precompile is
    // NOT used — the local Hardhat EIP-198 implementation fails for
    // base values >= ~2^200 (measured; small bases pass with the
    // identical input layout). The native mulmod square-and-multiply
    // is EXACT for 256-bit operands (mulmod computes the full 512-bit
    // product modulo), deterministic, and sidesteps the anomaly.
    function _modExp(uint256 base, uint256 exp, uint256 mod) private pure
        returns (uint256 result) {
        if (mod == 0) return 0;
        result = 1;
        base %= mod;
        while (exp != 0) {
            if ((exp & 1) == 1) result = mulmod(result, base, mod);
            base = mulmod(base, base, mod);
            exp >>= 1;
        }
    }

    // ----------------------------------------------------- field primitives
    function _modInv(uint256 a) private pure returns (uint256) {
        // a^(q-2) mod q (Fermat, q prime).
        return _powModQ(a, Q - 2);
    }

    function _powModQ(uint256 base, uint256 exp) private pure
        returns (uint256 result) {
        result = 1;
        base %= Q;
        while (exp != 0) {
            if ((exp & 1) == 1) result = mulmod(result, base, Q);
            base = mulmod(base, base, Q);
            exp >>= 1;
        }
    }

    // -------------------------------------------------- string serialization
    function _dec(uint256 v) private pure returns (bytes memory) {
        if (v == 0) return bytes("0");
        uint256 len = 0;
        uint256 t = v;
        while (t > 0) { t /= 10; len++; }
        bytes memory b = new bytes(len);
        while (len > 0) {
            len--;
            b[len] = bytes1(uint8(48 + (v % 10)));
            v /= 10;
        }
        return b;
    }

    function _hexMin(uint256 v) private pure returns (bytes memory) {
        if (v == 0) return bytes("0");
        uint256 len = 0;
        uint256 t = v;
        while (t > 0) { t /= 16; len++; }
        bytes memory b = new bytes(len);
        while (len > 0) {
            len--;
            uint256 d = v & 0xf;
            b[len] = bytes1(uint8(d < 10 ? 48 + d : 87 + d));
            v >>= 4;
        }
        return b;
    }

    // --------------------------------------------------- the generators
    function _generator(uint256 index) private view returns (uint256) {
        bytes memory label = abi.encodePacked("geode-bp-", _dec(index));
        uint256 h = uint256(_sha256(label)) % P;
        h = mulmod(h, h, P);
        if (h == 1) h = 4;
        return h;
    }

    // ---------------------------------------------------- the challenge
    // Python verify calls _challenge(_ser(l), _ser(r), _ser(c)) — three
    // SEPARATE single-value _ser calls, so the hashed bytes are the
    // three minimal lowercase hex strings concatenated with NO
    // separators (';' only appears when _ser receives several values
    // in one call, which never happens in the verify path).
    function _challenge(uint256 l, uint256 r, uint256 c) private view
        returns (uint256) {
        bytes memory s = abi.encodePacked(_hexMin(l), _hexMin(r),
                                          _hexMin(c));
        return uint256(_sha256(s)) % Q;
    }

    // ---------------------------------------------------------- multiexps
    function _prod(uint256[] memory bases, uint256[] memory exps)
        private view returns (uint256) {
        uint256 acc = 1;
        for (uint256 i = 0; i < bases.length; i++) {
            acc = mulmod(acc, _modExp(bases[i], exps[i] % Q, P), P);
        }
        return acc;
    }

    // ------------------------------------------------------------ verify
    struct ProofData {
        uint256 cCommit;
        uint256[] ls;
        uint256[] rs;
        uint256 aFinal;
        uint256 bFinal;
        uint256 rFinal;
    }

    function _roundsOf(uint256 n) private pure returns (uint256 r) {
        for (uint256 t = n; t > 1; t >>= 1) r++;
    }

    function _parse(bytes calldata proof, uint256 n) private pure
        returns (ProofData memory pd) {
        uint256 rounds = _roundsOf(n);
        pd.ls = new uint256[](rounds);
        pd.rs = new uint256[](rounds);
        pd.cCommit = _u256(proof, 0);
        for (uint256 j = 0; j < rounds; j++) {
            pd.ls[j] = _u256(proof, 32 + j * 32);
            pd.rs[j] = _u256(proof, 32 + rounds * 32 + j * 32);
        }
        uint256 off = 32 + 2 * rounds * 32;
        pd.aFinal = _u256(proof, off);
        pd.bFinal = _u256(proof, off + 32);
        pd.rFinal = _u256(proof, off + 64);
    }

    function verify(bytes calldata proof, uint256 claim,
                    uint256[] calldata w) public view returns (bool) {
        uint256 n = w.length;
        if (n == 0 || (n & (n - 1)) != 0) revert NotPowerOfTwo(n);
        uint256 expected = (1 + 2 * _roundsOf(n) + 3) * 32;
        if (proof.length != expected) {
            revert BadProofLength(proof.length, expected);
        }
        ProofData memory pd = _parse(proof, n);
        return _verify(pd, claim, w, n);
    }

    /// @notice Transaction-path entry point (M213): executes the same
    /// verification in a transaction and emits the verdict, so the
    /// gas cost is measurable from the receipt at widths beyond the
    /// eth_call gas cap, and production verification is exercised as
    /// it would run on a network.
    function verifyTx(bytes calldata proof, uint256 claim,
                      uint256[] calldata w) external returns (bool) {
        bool result = verify(proof, claim, w);
        emit Verification(w.length, result);
        return result;
    }

    struct RoundState {
        uint256[] b;
        uint256[] gens;
        uint256[] hgens;
        uint256 pAcc;
        uint256 qGen;
    }

    function _materialize(uint256[] calldata w, uint256 n) private view
        returns (uint256[] memory b, uint256[] memory gens,
                 uint256[] memory hgens, uint256 qGen) {
        b = new uint256[](n);
        gens = new uint256[](n);
        hgens = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            b[i] = w[i] % Q;
            gens[i] = _generator(i);
            hgens[i] = _generator(n + i);
        }
        qGen = _generator(2 * n);
    }

    function _foldRound(RoundState memory st, uint256 u, uint256 uInv,
                        uint256 u2, uint256 u2Inv, uint256 lj,
                        uint256 rj) private view {
        uint256 half = st.b.length / 2;
        uint256[] memory b2 = new uint256[](half);
        uint256[] memory g2 = new uint256[](half);
        uint256[] memory h2 = new uint256[](half);
        for (uint256 i = 0; i < half; i++) {
            b2[i] = addmod(mulmod(st.b[i], uInv, Q),
                           mulmod(st.b[half + i], u, Q), Q);
            g2[i] = mulmod(_modExp(st.gens[i], uInv, P),
                           _modExp(st.gens[half + i], u, P), P);
            h2[i] = mulmod(_modExp(st.hgens[i], u, P),
                           _modExp(st.hgens[half + i], uInv, P), P);
        }
        st.b = b2;
        st.gens = g2;
        st.hgens = h2;
        st.pAcc = mulmod(st.pAcc,
                         mulmod(_modExp(lj, u2, P),
                                _modExp(rj, u2Inv, P), P), P);
    }

    function _foldAll(ProofData memory pd, uint256 claim,
                      uint256[] calldata w, uint256 n) private view
        returns (RoundState memory st) {
        (st.b, st.gens, st.hgens, st.qGen) = _materialize(w, n);
        st.pAcc = mulmod(
            pd.cCommit,
            mulmod(_prod(st.hgens, st.b),
                   _modExp(st.qGen, claim % Q, P), P), P);
        uint256 rounds = pd.ls.length;
        for (uint256 j = 0; j < rounds; j++) {
            uint256 u = _challenge(pd.ls[j], pd.rs[j], pd.cCommit);
            uint256 uInv = _modInv(u);
            uint256 u2 = mulmod(u, u, Q);
            uint256 u2Inv = mulmod(uInv, uInv, Q);
            _foldRound(st, u, uInv, u2, u2Inv, pd.ls[j], pd.rs[j]);
        }
    }

    function _finalRhs(RoundState memory st, ProofData memory pd)
        private view returns (uint256 rhs) {
        uint256 aFinal = pd.aFinal % Q;
        uint256 bFinal = pd.bFinal % Q;
        uint256 rFinal = pd.rFinal % Q;
        uint256 t1 = _modExp(_BPG(), rFinal, P);
        uint256 t2 = mulmod(_modExp(st.gens[0], aFinal, P),
                            _modExp(st.hgens[0], bFinal, P), P);
        uint256 t3 = mulmod(t2,
                            _modExp(st.qGen, mulmod(aFinal, bFinal, Q),
                                    P), P);
        rhs = mulmod(t1, t3, P);
    }

    function _verify(ProofData memory pd, uint256 claim,
                     uint256[] calldata w, uint256 n) private view
        returns (bool) {
        RoundState memory st = _foldAll(pd, claim, w, n);
        if (pd.bFinal % Q != st.b[0]) return false;
        return st.pAcc == _finalRhs(st, pd);
    }

    function _BPG() private view returns (uint256) {
        // BP_G = _generator(-1) — Python uses index -1, i.e. the label
        // "geode-bp--1". Mirrored literally for bit-exactness.
        bytes memory label = "geode-bp--1";
        uint256 h = uint256(_sha256(label)) % P;
        h = mulmod(h, h, P);
        if (h == 1) h = 4;
        return h;
    }

    function _u256(bytes calldata data, uint256 pos) private pure
        returns (uint256 out) {
        assembly { out := calldataload(add(data.offset, pos)) }
    }
}
