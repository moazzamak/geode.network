// M213 — the cross-language verification gate: deploy
// LinearProofVerifier and run Python-serialized M193b proofs through
// it. The input JSON arrives via the POST_VERIFY_PATH environment
// variable (Hardhat does not forward script arguments):
//
//   POST_VERIFY_PATH=<abs path> npx hardhat run scripts/verify_onchain.js
//
// JSON shape:
//   { "cases": [
//       { "n": 64, "proof": "0x...", "claim": "0x...",
//         "w": ["0x...", ...], "expect": true } ] }
// Output: one VERIFY_CASE line per case, then VERIFY_OK.
const fs = require("fs");
const { ethers } = require("hardhat");

function fail(msg) {
  console.log(`VERIFY_FAIL ${msg}`);
  process.exit(1);
}

async function main() {
  const path = process.env.POST_VERIFY_PATH;
  if (!path || !path.endsWith(".json")) fail("no POST_VERIFY_PATH json");
  const spec = JSON.parse(fs.readFileSync(path, "utf8"));
  const cases = spec.cases || [];
  if (cases.length === 0) fail("no cases");

  const Verifier = await ethers.getContractFactory("LinearProofVerifier");
  const verifier = await Verifier.deploy();
  await verifier.waitForDeployment();

  const results = [];
  // The eth_call gas cap (2^24) is below the O(n) verifier cost at
  // n>=128, so the verdict rides in a TRANSACTION via verifyTx and
  // the gas is read from the receipt.
  const GAS = 190000000n;
  for (let i = 0; i < cases.length; i++) {
    const c = cases[i];
    const wArr = c.w.map((x) => BigInt(x));
    const tx = await verifier.verifyTx(
      c.proof, BigInt(c.claim), wArr, { gasLimit: GAS });
    const rc = await tx.wait();
    let ok = null;
    for (const log of rc.logs) {
      try {
        const parsed = verifier.interface.parseLog(log);
        if (parsed && parsed.name === "Verification") {
          ok = parsed.args.result;
        }
      } catch { /* not ours */ }
    }
    if (ok === null) fail(`case ${i}: no Verification event`);
    const gas = Number(rc.gasUsed);
    console.log(`VERIFY_CASE ${i} ok=${ok} gas=${gas} ` +
                `expect=${Boolean(c.expect)}`);
    results.push({ ok: Boolean(ok), gas, expect: Boolean(c.expect) });
  }
  const bad = results.filter((r) => r.ok !== r.expect);
  if (bad.length > 0) fail(`${bad.length} case(s) mismatched`);
  console.log("VERIFY_OK");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
