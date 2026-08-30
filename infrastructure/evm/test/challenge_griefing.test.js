// CreditLedger — the anti-griefing challenge escalation (the
// repeated-settlement-DoS repair, 30 Aug 2026). A funded attacker
// used to be able to delay EVERY epoch's settlement by one
// attestation window at the flat bond price: file a root, challenge
// it, do nothing, let it resolve as "unsubstantiated" after the
// window, repeat. Seven days per attempt, flat cost.
//
// The repair (registered before dispatch, never after a reading):
//   (a) ATTEST_WINDOW is 3 days (was 7), cutting the per-attempt
//       delay;
//   (b) a GLOBAL decaying challenge counter ("heat") drives an
//       escalating fee: the Nth live challenge within a 21-day
//       half-life pays bond * 2^(heat-1), capped at bond * 2^9, on
//       top of the refundable base bond;
//   (c) the fee funds the operations pool (settlement bounties),
//       never a party;
//   (d) heat is global, so rotating addresses cannot reset it, and
//       it decays only on 21 days of silence (a weekly campaign
//       never decays);
//   (e) dead challenges (resolved, already-challenged, or
//       window-closed filings) revert at the state guard BEFORE the
//       fee logic, so they cost nothing and cannot heat the counter.
//
// Gates: first challenge in a quiet network is free; the fee doubles
// per live challenge; fees reach the operations pool; heat is global
// across addresses; heat decays after 21 days of silence; a dead
// challenge reverts cheaply without heating the counter.
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");

const EPOCH = 7 * 24 * 3600;
const SLASH_WINDOW = 7 * 24 * 3600;
const HEAT_HALF_LIFE = 21 * 24 * 3600;
const BOND = ethers.parseEther("1");
const FEE = 1000n;

async function advanceTime(seconds) {
  await ethers.provider.send("evm_increaseTime", [seconds]);
  await ethers.provider.send("evm_mine");
}

async function griefFixture() {
  const [owner, devFund, filer, c1, c2, c3] = await ethers.getSigners();
  const Ledger = await ethers.getContractFactory("CreditLedger");
  const ledger = await upgrades.deployProxy(Ledger, [devFund.address, FEE]);
  await ledger.waitForDeployment();
  await advanceTime(EPOCH);   // close epoch 0 so it can be filed
  return { owner, devFund, filer, c1, c2, c3, ledger };
}

let rootSeq = 0;
async function fileRoot(f, who) {
  const root = ethers.keccak256(
    ethers.toUtf8Bytes(`grief-root-${rootSeq++}`));
  const tx = await f.ledger.connect(who).fileAttributionRoot(
    0n, root, { value: BOND });
  const rc = await tx.wait();
  // the filing id comes from the emitted RootFiled event, so the id
  // always matches the (snapshot-reverted) on-chain state
  for (const log of rc.logs) {
    const parsed = f.ledger.interface.parseLog(log);
    if (parsed && parsed.name === "RootFiled") return parsed.args[0];
  }
  throw new Error("no RootFiled event");
}

describe("challenge griefing (repeated-settlement-DoS repair, 30 Aug 2026)",
         function () {
  it("the first challenge in a quiet network is free", async function () {
    const f = await loadFixture(griefFixture);
    const id = await fileRoot(f, f.filer);
    await f.ledger.connect(f.c1).challengeAttributionRoot(id,
      { value: BOND });
    expect(await f.ledger.challengeHeat()).to.equal(1n);
    expect(await f.ledger.operationsPool()).to.equal(0n);
  });

  it("the second live challenge pays bond * 2, and the fee funds " +
     "the operations pool", async function () {
    const f = await loadFixture(griefFixture);
    const id1 = await fileRoot(f, f.filer);
    await f.ledger.connect(f.c1).challengeAttributionRoot(id1,
      { value: BOND });
    const id2 = await fileRoot(f, f.filer);
    // a second challenge at the flat bond price is refused
    await expect(f.ledger.connect(f.c1).challengeAttributionRoot(
      id2, { value: BOND }))
      .to.be.revertedWithCustomError(f.ledger, "WrongBond");
    // the escalated price clears it
    await f.ledger.connect(f.c1).challengeAttributionRoot(id2,
      { value: BOND + BOND });
    expect(await f.ledger.challengeHeat()).to.equal(2n);
    // the extra ETH is the pool's, not the challenger's or the filer's
    expect(await f.ledger.operationsPool()).to.equal(BOND);
  });

  it("a sustained campaign compounds: the fee doubles per live " +
     "challenge", async function () {
    const f = await loadFixture(griefFixture);
    // challenge k costs bond * 2^(k-1) for k >= 2
    const expected = [
      BOND * 0n,            // 1st: free
      BOND * 1n,            // 2nd
      BOND * 2n,            // 3rd
      BOND * 4n,            // 4th
      BOND * 8n,            // 5th
    ];
    let pool = 0n;
    for (let k = 0; k < expected.length; k++) {
      const id = await fileRoot(f, f.filer);
      if (expected[k] === 0n) {
        await f.ledger.connect(f.c1).challengeAttributionRoot(id,
          { value: BOND });
      } else {
        await f.ledger.connect(f.c1).challengeAttributionRoot(id,
          { value: BOND + expected[k] });
      }
      pool += expected[k];
      expect(await f.ledger.challengeHeat()).to.equal(BigInt(k + 1));
      expect(await f.ledger.operationsPool()).to.equal(pool);
    }
    // the 6th challenge needs 16x the bond
    const id = await fileRoot(f, f.filer);
    await expect(f.ledger.connect(f.c1).challengeAttributionRoot(
      id, { value: BOND + BOND * 8n }))
      .to.be.revertedWithCustomError(f.ledger, "WrongBond");
    await f.ledger.connect(f.c1).challengeAttributionRoot(
      id, { value: BOND + BOND * 16n });
    expect(await f.ledger.operationsPool())
      .to.equal(pool + BOND * 16n);
  });

  it("heat is global: a fresh address pays the escalated fee too",
     async function () {
    const f = await loadFixture(griefFixture);
    const id1 = await fileRoot(f, f.filer);
    await f.ledger.connect(f.c1).challengeAttributionRoot(id1,
      { value: BOND });
    // c2 is a brand-new challenger but the heat did not reset
    const id2 = await fileRoot(f, f.filer);
    await expect(f.ledger.connect(f.c2).challengeAttributionRoot(
      id2, { value: BOND }))
      .to.be.revertedWithCustomError(f.ledger, "WrongBond");
    await f.ledger.connect(f.c2).challengeAttributionRoot(
      id2, { value: BOND + BOND });
    expect(await f.ledger.challengeHeat()).to.equal(2n);
  });

  it("heat decays only on 21 days of silence", async function () {
    const f = await loadFixture(griefFixture);
    const id1 = await fileRoot(f, f.filer);
    await f.ledger.connect(f.c1).challengeAttributionRoot(id1,
      { value: BOND });
    const id1b = await fileRoot(f, f.filer);
    await f.ledger.connect(f.c2).challengeAttributionRoot(id1b,
      { value: BOND + BOND });
    expect(await f.ledger.challengeHeat()).to.equal(2n);
    // 20 days is NOT silence: the heat survives
    await advanceTime(20 * 24 * 3600);
    const id2 = await fileRoot(f, f.filer);
    await expect(f.ledger.connect(f.c1).challengeAttributionRoot(
      id2, { value: BOND }))
      .to.be.revertedWithCustomError(f.ledger, "WrongBond");
    // 21 days since the last challenge IS silence: heat halves to 1
    await advanceTime(HEAT_HALF_LIFE - 20 * 24 * 3600 + 1);
    const id3 = await fileRoot(f, f.filer);
    await f.ledger.connect(f.c1).challengeAttributionRoot(id3,
      { value: BOND + BOND });   // heat 2 >> 1 -> heat 1 -> fee bond
    expect(await f.ledger.challengeHeat()).to.equal(2n);
    // a full 42 days of silence clears the counter entirely
    await advanceTime(2 * HEAT_HALF_LIFE);
    const id4 = await fileRoot(f, f.filer);
    await f.ledger.connect(f.c1).challengeAttributionRoot(id4,
      { value: BOND });   // free again
    expect(await f.ledger.challengeHeat()).to.equal(1n);
  });

  it("a weekly campaign never decays: 1 ETH per window of delay " +
     "compounds", async function () {
    const f = await loadFixture(griefFixture);
    // 7-day cadence (the attacker's per-attempt delay), repeated:
    // heat grows without bound because 7 < 21 days of silence
    let pool = 0n;
    let heat = 0n;
    for (let k = 0; k < 4; k++) {
      const id = await fileRoot(f, f.filer);
      const fee = heat === 0n ? 0n : BOND * (1n << (heat - 1n));
      await f.ledger.connect(f.c1).challengeAttributionRoot(
        id, { value: BOND + fee });
      pool += fee;
      heat += 1n;
      expect(await f.ledger.challengeHeat()).to.equal(heat);
      expect(await f.ledger.operationsPool()).to.equal(pool);
      await advanceTime(7 * 24 * 3600);
    }
    // after 4 weekly attacks the next one already costs 8x the bond
    expect(heat).to.equal(4n);
    const id = await fileRoot(f, f.filer);
    await expect(f.ledger.connect(f.c1).challengeAttributionRoot(
      id, { value: BOND + BOND * 4n }))
      .to.be.revertedWithCustomError(f.ledger, "WrongBond");
    await f.ledger.connect(f.c1).challengeAttributionRoot(
      id, { value: BOND + BOND * 8n });
  });

  it("a dead challenge reverts cheaply without heating the counter",
     async function () {
    const f = await loadFixture(griefFixture);
    // an UNchallenged filing that executes lands the root; a
    // subsequent challenge is refused at the state guard, before any
    // fee logic, so it neither pays nor heats
    const id = await fileRoot(f, f.filer);
    await advanceTime(SLASH_WINDOW + 1);
    await f.ledger.connect(f.c1).executeAttributionRoot(id);
    expect(await f.ledger.challengeHeat()).to.equal(0n);
    await expect(f.ledger.connect(f.c1).challengeAttributionRoot(
      id, { value: BOND }))
      .to.be.revertedWithCustomError(f.ledger, "AlreadyResolved");
    expect(await f.ledger.challengeHeat()).to.equal(0n);
    expect(await f.ledger.operationsPool()).to.equal(0n);
  });

  it("a double challenge on the same filing is refused cheaply too",
     async function () {
    const f = await loadFixture(griefFixture);
    const id = await fileRoot(f, f.filer);
    await f.ledger.connect(f.c1).challengeAttributionRoot(id,
      { value: BOND });
    expect(await f.ledger.challengeHeat()).to.equal(1n);
    await expect(f.ledger.connect(f.c2).challengeAttributionRoot(
      id, { value: BOND }))
      .to.be.revertedWithCustomError(f.ledger, "AlreadyChallenged");
    expect(await f.ledger.challengeHeat()).to.equal(1n);
  });

  it("the attestation window is 3 days (the per-attempt delay cap)",
     async function () {
    const f = await loadFixture(griefFixture);
    expect(await f.ledger.ATTEST_WINDOW()).to.equal(3 * 24 * 3600);
  });
});
