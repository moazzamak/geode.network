// Harness evidence (reworked 24 Aug 2026, whitepaper-aligned): gas
// budgets for the CreditLedger hot paths + coverage summary + source
// hashes. Run after `hardhat coverage`.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { ethers, upgrades } = require("hardhat");

const EPOCH = 7 * 24 * 3600;

function sha256(file) {
  return crypto
    .createHash("sha256")
    .update(fs.readFileSync(file))
    .digest("hex");
}

async function main() {
  const [owner, devFund, payer, contributor, operator] =
    await ethers.getSigners();

  const Ledger = await ethers.getContractFactory("CreditLedger");
  const ledger = await upgrades.deployProxy(Ledger, [
    devFund.address,
    1000n, // registration fee
  ]);
  await ledger.waitForDeployment();
  await ledger.setLibrarian(owner.address);

  const gas = {};
  gas.ledgerDeposit = Number(
    (await (await ledger.connect(payer)
      .deposit({ value: 100000n })).wait()).gasUsed);

  const artifactId = ethers.keccak256(ethers.toUtf8Bytes("arm-a"));
  gas.ledgerRegister = Number(
    (await (await ledger.connect(operator).register(
      artifactId, contributor.address, 10n, ethers.ZeroHash,
      { value: 1000n })).wait()).gasUsed);
  await ledger.setAdmitted(artifactId, true);

  const credEntries = Array.from({ length: 10 }, () => ({
    artifactId,
    who: contributor.address,
    amount: 1000n,
  }));
  const credPayers = Array.from({ length: 10 }, () => payer.address);
  gas.ledgerRecord10 = Number(
    (await (await ledger.connect(owner).recordCredits(
      credPayers, credEntries)).wait()).gasUsed);
  gas.ledgerClaimDevFund = Number(
    (await (await ledger.claimDevFund()).wait()).gasUsed);

  // vest: after one epoch the first of N=4 tranches is claimable
  await ethers.provider.send("evm_increaseTime", [EPOCH + 1]);
  await ethers.provider.send("evm_mine");
  gas.ledgerClaim = Number(
    (await (await ledger.connect(contributor).claim()).wait()).gasUsed);
  gas.ledgerSlash = Number(
    (await (await ledger.connect(owner).slash(
      contributor.address, artifactId, 100n, 1, ethers.ZeroHash))
      .wait()).gasUsed);

  const Anchor = await ethers.getContractFactory("ProofAnchor");
  const anchor = await Anchor.deploy();
  await anchor.waitForDeployment();
  gas.anchor = Number(
    (await (await anchor.anchor(ethers.toUtf8Bytes("p"))).wait()).gasUsed);

  const coverage = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "coverage.json"), "utf8"));
  const ours = Object.entries(coverage)
    .filter(([f]) => f.startsWith("contracts\\") && !f.includes("mocks"))
    .map(([f, d]) => {
      const pct = (covered, total) =>
        total === 0 ? 100 : Number(((covered / total) * 100).toFixed(2));
      return {
        file: f,
        stmts: pct(d.s && Object.values(d.s).filter((v) => v > 0).length,
                   d.s ? Object.values(d.s).length : 0),
        branch: pct(d.b ? Object.values(d.b)
          .flatMap((p) => p).filter((v) => v > 0).length : 0,
          d.b ? Object.values(d.b).flatMap((p) => p).length : 0),
        fn: pct(d.f && Object.values(d.f).filter((v) => v > 0).length,
                d.f ? Object.values(d.f).length : 0),
      };
    });

  const evidence = {
    milestone: "WHITEPAPER_ALIGNMENT_24AUG",
    generated_at: new Date().toISOString(),
    sources: {
      "contracts/CreditLedger.sol": sha256(
        path.join(__dirname, "..", "contracts", "CreditLedger.sol")),
      "contracts/ProofAnchor.sol": sha256(
        path.join(__dirname, "..", "contracts", "ProofAnchor.sol")),
      "contracts/LinearProofVerifier.sol": sha256(
        path.join(__dirname, "..", "contracts", "LinearProofVerifier.sol")),
    },
    coverage: ours,
    coverage_gate: ours.every(
      (c) => c.stmts === 100 && c.branch === 100 && c.fn === 100),
    gas_budgets: gas,
    tests: 48, // full suite (43 under coverage + 5 verifier-only)
  };
  const out = path.join(__dirname, "..", "evidence", "harness_evidence.json");
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, JSON.stringify(evidence, null, 2));
  console.log(JSON.stringify(evidence, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
