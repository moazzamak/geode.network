require("@nomicfoundation/hardhat-toolbox");
require("@openzeppelin/hardhat-upgrades");
require("solidity-coverage");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: { enabled: true, runs: 10 },
      evmVersion: "cancun",
      viaIR: true, // M213: required by the LinearProofVerifier's
                   // arithmetic-heavy port (optimizer inlining otherwise
                   // exceeds the 16-slot stack limit)
    },
  },
  networks: {
    hardhat: {
      chainId: 31337,
      hardfork: "cancun", // M213: Hardhat's default (fusaka) caps each
                          // transaction at 2^24 gas; the O(n) verifier
                          // gas sweep needs the cancun rules, matching
                          // the compile target
      blockGasLimit: 200_000_000, // the direct-port verifier is O(n)
                                  // modexps; the sweep at n=512 needs
                                  // headroom above the default 30M
    },
    // Whitepaper chain decision (24 Aug): anchors on Ethereum Sepolia
    // during development (mainnet thereafter); settlement on Arbitrum
    // One with an Arbitrum Sepolia rehearsal. Keys come from the
    // environment; nothing is committed.
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "",
      chainId: 11155111,
      accounts: process.env.SEPOLIA_PRIVATE_KEY
        ? [process.env.SEPOLIA_PRIVATE_KEY] : [],
    },
    arbitrumSepolia: {
      url: process.env.ARBITRUM_SEPOLIA_RPC_URL || "",
      chainId: 421614,
      accounts: process.env.ARBITRUM_SEPOLIA_PRIVATE_KEY
        ? [process.env.ARBITRUM_SEPOLIA_PRIVATE_KEY] : [],
    },
    arbitrumOne: {
      url: process.env.ARBITRUM_ONE_RPC_URL || "",
      chainId: 42161,
      accounts: process.env.ARBITRUM_ONE_PRIVATE_KEY
        ? [process.env.ARBITRUM_ONE_PRIVATE_KEY] : [],
    },
  },
  mocha: { timeout: 120000 },
};





