// M212 g4 (reworked 24 Aug 2026, whitepaper-aligned) — the
// cross-language post gate: post a Python-built settlement batch
// JSON via recordCredits to the deployed CreditLedger on the local
// EVM and verify the credited amounts match the Python-side expected
// values exactly (no revert; malformed entries are skipped with
// logged events, never a batch revert).
//
// The batch JSON path arrives via the POST_BATCH_PATH environment
// variable (Hardhat does not forward script arguments):
//   POST_BATCH_PATH=<abs path> npx hardhat run scripts/post_batch.js
//
// Report schema:
// {
//   "registration_fee": 1000,
//   "registrations": [{"artifactId": "0x..", "operator": "0x..",
//                      "payoutAddress": "0x..", "pricePerUnit": 10,
//                      "sealedClaim": "0x.."}],
//   "deposits": [{"payer": "0x..", "amount": 1000}],
//   "batches": [{"payers": ["0x.."],
//                "entries": [{"artifactId": "0x..", "who": "0x..",
//                             "amount": 10}]}],
//   "expected": {"credits": {"0x..": 100},
//                "skipped": [{"payer": "0x..", "artifactId": "0x..",
//                             "who": "0x..", "amount": 5}]}
// }
const fs = require("fs");
const { ethers, upgrades } = require("hardhat");

function fail(msg) {
  console.log(`POST_FAIL ${msg}`);
  process.exit(1);
}

// Derived addresses (sha256 prefixed) are not default signers: fund
// them with ETH and impersonate them so they can register/pay.
async function signerFor(addr) {
  await ethers.provider.send("hardhat_impersonateAccount", [addr]);
  const bal = await ethers.provider.getBalance(addr);
  if (bal < ethers.parseEther("1")) {
    const [owner] = await ethers.getSigners();
    await owner.sendTransaction({ to: addr, value: ethers.parseEther("1") });
  }
  return ethers.getSigner(addr);
}

async function main() {
  const path = process.env.POST_BATCH_PATH;
  if (!path || !path.endsWith(".json")) fail("no POST_BATCH_PATH json");
  const report = JSON.parse(fs.readFileSync(path, "utf8"));
  const registrations = report.registrations || [];
  const deposits = report.deposits || [];
  const batches = report.batches || [];
  const expectedCredits = (report.expected && report.expected.credits) || {};
  const skipped = (report.expected && report.expected.skipped) || [];
  const fee = BigInt(report.registration_fee || 0);

  const [owner, devFund, librarian] = await ethers.getSigners();
  const Ledger = await ethers.getContractFactory("CreditLedger");
  const ledger = await upgrades.deployProxy(Ledger, [
    devFund.address,
    fee,
  ]);
  await ledger.waitForDeployment();
  await ledger.setLibrarian(librarian.address);

  // the whitepaper's on-chain proof-hash anchor: each batch's digest
  // of the proofs of the computations it pays for is recorded on the
  // anchor contract with the batch.
  const Anchor = await ethers.getContractFactory("ProofAnchor");
  const anchor = await Anchor.deploy();
  await anchor.waitForDeployment();

  // ---- registrations (one form: operator key + payout address +
  // price per unit + sealed claim; the fee funds the dev fund) ------
  // The post gate represents post-admission settlement, so every
  // registration in the report is admitted before credits flow.
  for (const r of registrations) {
    const operator = await signerFor(r.operator);
    await ledger.connect(operator).register(
      r.artifactId, r.payoutAddress, BigInt(r.pricePerUnit),
      r.sealedClaim, { value: fee });
    await ledger.connect(librarian).setAdmitted(r.artifactId, true);
  }

  // ---- deposits (native ETH; the contract splits 2.5% to the fund)
  for (const dep of deposits) {
    const payer = await signerFor(dep.payer);
    await ledger.connect(payer).deposit({ value: BigInt(dep.amount) });
  }

  const poolBefore = await ledger.attributable();
  if (poolBefore !== BigInt(report.pool_expected)) {
    fail(`attributable ${poolBefore} != pool_expected ${report.pool_expected}`);
  }

  // ---- post every built batch (the gate: no revert) ----------------
  let anchored = 0;
  for (const batch of batches) {
    const entries = batch.entries.map((e) => ({
      artifactId: e.artifactId,
      who: e.who,
      amount: BigInt(e.amount),
    }));
    await ledger.connect(librarian).recordCredits(batch.payers, entries);
    if (batch.proof_hash) {
      await anchor.anchor(ethers.getBytes(batch.proof_hash));
      anchored += 1;
    }
  }

  // ---- the credited amounts must match the Python-side expected
  let creditedSum = 0n;
  for (const [who, amount] of Object.entries(expectedCredits)) {
    const onChain = await ledger.creditsOf(who);
    const expected = BigInt(amount);
    if (onChain !== expected) {
      fail(`creditsOf ${who}: ${onChain} != expected ${expected}`);
    }
    creditedSum += onChain;
  }

  const poolAfter = await ledger.attributable();
  const poolExpected = poolBefore - creditedSum;
  if (poolAfter !== poolExpected) {
    fail(`attributable after ${poolAfter} != expected ${poolExpected}`);
  }
  const devShare = await ledger.devFundShare();

  // ---- raw control: a batch containing the builder-excluded entries
  // must be SKIPPED by the contract with logged events, not credited.
  if (skipped.length > 0) {
    const poolBeforeControl = await ledger.attributable();
    const controlPayers = skipped.map((s) => s.payer);
    const controlEntries = skipped.map((s) => ({
      artifactId: s.artifactId, who: s.who, amount: BigInt(s.amount),
    }));
    const tx = await ledger.connect(librarian).recordCredits(
      controlPayers, controlEntries);
    const rc = await tx.wait();
    const skipEvents = rc.logs
      .filter((l) => {
        try { return ledger.interface.parseLog(l).name ===
              "CreditSkipped"; } catch { return false; }
      });
    if (skipEvents.length !== skipped.length) {
      fail(`expected ${skipped.length} CreditSkipped events, ` +
           `got ${skipEvents.length}`);
    }
    for (const s of skipped) {
      const onChain = await ledger.creditsOf(s.who);
      const before = (expectedCredits[s.who] === undefined)
        ? 0n : BigInt(expectedCredits[s.who]);
      if (onChain !== before) {
        fail(`skipped entry ${s.payer} changed credits of ${s.who} ` +
             `(${before} -> ${onChain})`);
      }
    }
    const poolAfterControl = await ledger.attributable();
    if (poolAfterControl !== poolBeforeControl) {
      fail(`control batch moved the pool ` +
           `(${poolBeforeControl} -> ${poolAfterControl})`);
    }
  }

  console.log(`POST_OK credited=${creditedSum} ` +
              `pool_remaining=${poolAfter} ` +
              `skipped=${skipped.length} dev_share=${devShare} ` +
              `anchored=${anchored}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
