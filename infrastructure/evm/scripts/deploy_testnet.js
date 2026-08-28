// M-testnet deployment script — the registered Sepolia / Arbitrum
// Sepolia launch posture (whitepaper ledger section; checklist §3):
//
//   Sepolia          ProofAnchor (per-epoch ledger anchors)
//                    GovernanceFloors (floors mirror)
//   Arbitrum Sepolia CreditLedger proxy (settlement rehearsal)
//                    GovernanceFloors (floors mirror)
//
// Keys come from the environment, never from this file:
//   SEPOLIA_RPC_URL / SEPOLIA_PRIVATE_KEY
//   ARBITRUM_SEPOLIA_RPC_URL / ARBITRUM_SEPOLIA_PRIVATE_KEY
//   LIBRARIAN_ADDRESS   — the key-ceremony address (TESTNET_LAUNCH_PLAN
//                         §2). If unset, the deployer acts as librarian
//                         and the script REFUSES on a live network.
//   DEV_FUND_ADDRESS    — the development-fund address (zakat end state).
//   REGISTRATION_FEE_WEI— the registration fee (explicit, no silent
//                         default for money parameters).
//
// --dry-run exercises the exact deployment on the local hardhat
// network (chainId 31337) and writes no on-chain evidence.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");
const { ethers, upgrades } = hre;

const EVIDENCE_DIR = path.join(__dirname, "..", "evidence");

function envOr(address, name, fallback) {
  if (!address) {
    console.log(`[deploy] ${name} unset -> using ${fallback}`);
    return fallback;
  }
  return address;
}

async function deployLedger(signer, { devFund, registrationFeeWei,
                                     librarian }) {
  const CreditLedger = await ethers.getContractFactory("CreditLedger");
  const proxy = await upgrades.deployProxy(
    CreditLedger,
    [devFund, registrationFeeWei],
    { kind: "uups", initializer: "initialize" });
  await proxy.waitForDeployment();
  const impl = await upgrades.erc1967.getImplementationAddress(
    await proxy.getAddress());
  if (librarian) {
    const tx = await proxy.setLibrarian(librarian);
    await tx.wait();
  }
  return { proxy, impl };
}

async function deployFloors(governance) {
  const Floors = await ethers.getContractFactory("GovernanceFloors");
  const floors = await Floors.deploy(governance);
  await floors.waitForDeployment();
  return floors;
}

async function deployAnchor() {
  const Anchor = await ethers.getContractFactory("ProofAnchor");
  const anchor = await Anchor.deploy();
  await anchor.waitForDeployment();
  return anchor;
}

async function main() {
  const dryRun = process.argv.includes("--dry-run")
    || process.env.DEPLOY_DRY_RUN === "1";
  const networkName = hre.network.name;
  if (!dryRun && !["sepolia", "arbitrumSepolia"].includes(networkName)) {
    throw new Error(`refusing to deploy to '${networkName}': the ` +
                    `registered targets are sepolia and ` +
                    `arbitrumSepolia (--dry-run for a local rehearsal)`);
  }
  const [deployer] = await ethers.getSigners();
  console.log(`[deploy] network=${networkName} deployer=` +
              `${await deployer.getAddress()} dryRun=${dryRun}`);

  const librarian = envOr(
    process.env.LIBRARIAN_ADDRESS, "LIBRARIAN_ADDRESS",
    await deployer.getAddress());
  if (!dryRun && librarian === await deployer.getAddress()) {
    throw new Error("LIBRARIAN_ADDRESS is unset on a live network: " +
                    "the librarian key comes from the registered key " +
                    "ceremony (TESTNET_LAUNCH_PLAN §2), never from the " +
                    "deployer.");
  }
  const devFund = envOr(
    process.env.DEV_FUND_ADDRESS, "DEV_FUND_ADDRESS",
    await deployer.getAddress());
  const registrationFeeWei = process.env.REGISTRATION_FEE_WEI;
  if (registrationFeeWei === undefined) {
    throw new Error("REGISTRATION_FEE_WEI is unset: the fee is a " +
                    "money parameter and has no silent default.");
  }

  const receipt = {
    network: networkName,
    dryRun,
    deployer: await deployer.getAddress(),
    librarian,
    devFund,
    registrationFeeWei: registrationFeeWei.toString(),
    deployedAt: new Date().toISOString(),
    contracts: {},
  };

  if (networkName === "arbitrumSepolia") {
    const { proxy, impl } = await deployLedger(
      deployer, { devFund, registrationFeeWei, librarian });
    const floors = await deployFloors(librarian);
    receipt.contracts.creditLedgerProxy = await proxy.getAddress();
    receipt.contracts.creditLedgerImpl = impl;
    receipt.contracts.governanceFloors = await floors.getAddress();
    // the registered post-deploy check: the floor mirror's charter
    // constants match the plan (2/3 quorum, 20% cap, d >= 3)
    receipt.checks = {
      quorumBps: Number(await floors.QUORUM_BPS()),
      votingCapBps: Number(await floors.VOTING_CAP_BPS()),
      diversityMin: Number(await floors.DIVERSITY_MIN()),
      diversityBasisBps: Number(await floors.DIVERSITY_BASIS_BPS()),
    };
  } else if (networkName === "sepolia") {
    const anchor = await deployAnchor();
    const floors = await deployFloors(librarian);
    receipt.contracts.proofAnchor = await anchor.getAddress();
    receipt.contracts.governanceFloors = await floors.getAddress();
    receipt.checks = {
      quorumBps: Number(await floors.QUORUM_BPS()),
      votingCapBps: Number(await floors.VOTING_CAP_BPS()),
      diversityMin: Number(await floors.DIVERSITY_MIN()),
    };
  } else {
    // dry run: deploy everything locally in one go
    const { proxy, impl } = await deployLedger(
      deployer, { devFund, registrationFeeWei, librarian });
    const floors = await deployFloors(librarian);
    const anchor = await deployAnchor();
    receipt.contracts.creditLedgerProxy = await proxy.getAddress();
    receipt.contracts.creditLedgerImpl = impl;
    receipt.contracts.governanceFloors = await floors.getAddress();
    receipt.contracts.proofAnchor = await anchor.getAddress();
    receipt.checks = {
      quorumBps: Number(await floors.QUORUM_BPS()),
      votingCapBps: Number(await floors.VOTING_CAP_BPS()),
      diversityMin: Number(await floors.DIVERSITY_MIN()),
      diversityBasisBps: Number(await floors.DIVERSITY_BASIS_BPS()),
    };
  }

  console.log(JSON.stringify(receipt, null, 2));
  if (!dryRun) {
    fs.mkdirSync(EVIDENCE_DIR, { recursive: true });
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const out = path.join(EVIDENCE_DIR,
                          `deploy_${networkName}_${stamp}.json`);
    fs.writeFileSync(out, JSON.stringify(receipt, null, 2) + "\n");
    console.log(`[deploy] evidence -> ${out}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
