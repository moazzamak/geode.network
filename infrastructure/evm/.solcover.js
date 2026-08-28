module.exports = {
  skipFiles: ["mocks", "LinearProofVerifier.sol"],
  measureStatementCoverage: true,
  measureFunctionCoverage: true,
  measureBranchCoverage: true,
  measureModifierCoverage: true,
  // The LinearProofVerifier is arithmetic-heavy (viaIR, mulmod modexp) and its
  // test suite OOMs Node under coverage instrumentation. It is excluded from
  // instrumentation (library code) and its tests are excluded from coverage
  // runs; correctness is evidenced by its 5 passing standalone tests.
  mocha: { grep: "LinearProofVerifier", invert: true },
};
