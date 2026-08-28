// M214 — the proof-hash anchor gate: anchor the real-SIZE M193b
// proof on-chain, verify retrievability, append-only semantics, and
// measure the gas. Input JSON via POST_ANCHOR_PATH.
const fs = require("fs");
const { ethers } = require("hardhat");

function fail(msg) {
  console.log(`ANCHOR_FAIL ${msg}`);
  process.exit(1);
}

async function main() {
  const path = process.env.POST_ANCHOR_PATH;
  if (!path || !path.endsWith(".json")) fail("no POST_ANCHOR_PATH json");
  const spec = JSON.parse(fs.readFileSync(path, "utf8"));

  const A = await ethers.getContractFactory("ProofAnchor");
  const a = await A.deploy();
  await a.waitForDeployment();

  const sender = (await ethers.getSigners())[0];
  const tx = await a.connect(sender).anchor(spec.proof, { gasLimit: 1000000n });
  const rc = await tx.wait();
  const h1 = ethers.keccak256(spec.proof);
  const block1 = await a.anchoredAt(h1);
  if (block1 === 0n) fail("anchor not recorded");

  // append-only: re-anchoring the same hash is a no-op
  const tx2 = await a.connect(sender).anchor(spec.proof, { gasLimit: 1000000n });
  const rc2 = await tx2.wait();
  const block2 = await a.anchoredAt(h1);
  if (block2 !== block1) fail("re-anchor moved the block number");

  // a different proof gets a different hash
  const h2 = ethers.keccak256(spec.proof_alt);
  if (h1 === h2) fail("tampered proof hashed identically");
  await a.connect(sender).anchor(spec.proof_alt, { gasLimit: 1000000n });
  if ((await a.anchoredAt(h2)) === 0n) fail("alt anchor not recorded");

  console.log(`ANCHOR_OK gas=${Number(rc.gasUsed)} ` +
              `reanchor_gas=${Number(rc2.gasUsed)} ` +
              `block1=${Number(block1)} distinct=true`);
}

main().catch((e) => { console.error(e); process.exit(1); });
