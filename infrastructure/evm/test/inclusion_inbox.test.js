// M336 — InclusionInbox tests: on-chain force-inclusion, the
// registered M312 mirror. Gates: post -> incorporate -> valid;
// post -> withhold -> invalid; post -> incorporate-late -> invalid
// (recorded violation); bond returns on incorporation and on
// poster withdrawal after the librarian fails.
//
// M365 (G24, 29 Aug 2026) — the inbox was a free bloat weapon: the
// deposit was fully refunded, so spam cost only gas while every
// entry obliged the librarian within one window. The registered
// gates for the repair:
//   (a) honest posting costs a non-refundable fee plus a returned
//       bond, and the fee reaches the operations line;
//   (b) a spam campaign costs superlinearly;
//   (c) a spam campaign never reaches chainValid == false while the
//       librarian meets the capped per-epoch obligation;
//   (d) a censored entry from a fresh address is still incorporated
//       within a registered, finite bound.
const { expect } = require("chai");
const { ethers, network } = require("hardhat");

const WINDOW = 10;                 // blocks
const BOND = 100n;                 // refundable
const EPOCH_BLOCKS = 5000;         // long enough that one test is one epoch
const BASE_FEE = 10n;              // non-refundable, per post
const FREE_POSTS = 4;              // per address per epoch at base fee
const MAX_INC_PER_EPOCH = 8;       // capped librarian obligation

async function deployInbox(windowBlocks = WINDOW) {
  const [poster, librarian, other, operations, fresh] =
    await ethers.getSigners();
  const S = await ethers.getContractFactory("LibrarianSourceMock");
  const source = await S.deploy(librarian.address);
  await source.waitForDeployment();
  const I = await ethers.getContractFactory("InclusionInbox");
  const inbox = await I.deploy(
    await source.getAddress(), windowBlocks, BOND, operations.address,
    EPOCH_BLOCKS, BASE_FEE, FREE_POSTS, MAX_INC_PER_EPOCH);
  await inbox.waitForDeployment();
  return { inbox, source, poster, librarian, other, operations, fresh };
}

async function mineBlocks(n) {
  for (let i = 0; i < n; i++) {
    await network.provider.send("evm_mine");
  }
}

const id = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

// post paying exactly what the contract asks for right now
async function postExact(inbox, signer, entryId, digest) {
  const fee = await inbox.postingFee(signer.address);
  return inbox.connect(signer).post(entryId, digest, {
    value: BOND + fee,
  });
}

// the analytic fee schedule, mirrored in JS so the tests check the
// contract against an independently written formula rather than
// against itself
function expectedFee(postsAlreadyMade) {
  if (postsAlreadyMade < FREE_POSTS) return BASE_FEE;
  const over = BigInt(postsAlreadyMade - FREE_POSTS + 2);
  return BASE_FEE * over * over;
}

describe("InclusionInbox (M336)", function () {
  it("post -> incorporate -> valid, bond returned", async function () {
    const { inbox, poster, librarian } = await deployInbox();
    const entryId = id("entry-1");
    await postExact(inbox, poster, entryId, id("payload-1"));
    expect(await inbox.chainValid()).to.equal(true);
    await inbox.connect(librarian).incorporate(entryId);
    expect(await inbox.chainValid()).to.equal(true);
    expect(await inbox.openCount()).to.equal(0n);
    const e = await inbox.entries(entryId);
    expect(e.incorporatedBlock).to.be.greaterThan(0n);
    expect(e.bond).to.equal(0n);
  });

  it("post -> withhold -> chain invalid", async function () {
    const { inbox, poster } = await deployInbox();
    const entryId = id("entry-2");
    await postExact(inbox, poster, entryId, id("payload-2"));
    await mineBlocks(WINDOW + 1);
    expect(await inbox.chainValid()).to.equal(false);
    expect(await inbox.isLate(entryId)).to.equal(true);
  });

  it("post -> incorporate-late -> violation recorded, chain valid again",
     async function () {
    const { inbox, poster, librarian } = await deployInbox();
    const entryId = id("entry-3");
    await postExact(inbox, poster, entryId, id("payload-3"));
    await mineBlocks(WINDOW + 1);
    expect(await inbox.chainValid()).to.equal(false); // was invalid
    const { anyValue } = require(
      "@nomicfoundation/hardhat-chai-matchers/withArgs");
    await expect(inbox.connect(librarian).incorporate(entryId))
      .to.emit(inbox, "Incorporated")
      .withArgs(entryId, poster.address, librarian.address, anyValue,
                true); // late = true
    expect(await inbox.chainValid()).to.equal(true);
  });

  it("poster withdraws the bond after the librarian fails",
     async function () {
    const { inbox, poster } = await deployInbox();
    const entryId = id("entry-4");
    await postExact(inbox, poster, entryId, id("payload-4"));
    // inside the window: no withdrawal yet
    await expect(inbox.connect(poster).withdrawBond(entryId))
      .to.be.revertedWithCustomError(inbox, "WindowNotElapsed");
    await mineBlocks(WINDOW + 1);
    await inbox.connect(poster).withdrawBond(entryId);
    expect((await inbox.entries(entryId)).bond).to.equal(0n);
    // the entry stays open: the violation remains visible
    expect(await inbox.chainValid()).to.equal(false);
  });

  it("anyone may incorporate: the queue does not need permission",
     async function () {
    // M383: every line of incorporate() is forced by on-chain state,
    // so gating the call bought no safety and cost the censorship
    // surface. A stranger clearing the head is not an attack.
    const { inbox, poster, other } = await deployInbox();
    const entryId = id("entry-5");
    await postExact(inbox, poster, entryId, id("payload-5"));
    await inbox.connect(other).incorporate(entryId);
    expect((await inbox.entries(entryId)).incorporatedBlock)
      .to.be.greaterThan(0n);
    expect(await inbox.foreignIncorporations()).to.equal(1n);
    expect(await inbox.librarianIncorporations()).to.equal(0n);
  });

  it("rejects empty digest and reused ids", async function () {
    const { inbox, poster } = await deployInbox();
    const entryId = id("entry-6");
    const fee = await inbox.postingFee(poster.address);
    await expect(inbox.connect(poster).post(
      entryId, ethers.ZeroHash, { value: BOND + fee }))
      .to.be.revertedWithCustomError(inbox, "EmptyDigest");
    await postExact(inbox, poster, entryId, id("p6"));
    const fee2 = await inbox.postingFee(poster.address);
    await expect(inbox.connect(poster).post(
      entryId, id("p6b"), { value: BOND + fee2 }))
      .to.be.revertedWithCustomError(inbox, "EntryClosed");
  });

  it("gas budget: post, incorporate, chainValid", async function () {
    const { inbox, poster, librarian } = await deployInbox();
    const entryId = id("entry-7");
    const txPost = await postExact(inbox, poster, entryId, id("p7"));
    const receiptPost = await txPost.wait();
    const txInc = await inbox.connect(librarian).incorporate(entryId);
    const receiptInc = await txInc.wait();
    await mineBlocks(WINDOW + 1);
    // Seal the gas budget as a generous ceiling, not a measurement.
    // M365 raised the post ceiling from 200k to 300k: pricing spam
    // costs storage. A post now writes the deadline, the per-address
    // per-epoch counter, and the accrued operations balance, which
    // the pre-repair version did not. That is the price of the
    // repair and it is recorded rather than absorbed.
    expect(receiptPost.gasUsed).to.be.lessThan(300_000n);
    expect(receiptInc.gasUsed).to.be.lessThan(200_000n);
    console.log(
      `M336 gas: post ${receiptPost.gasUsed}, ` +
      `incorporate ${receiptInc.gasUsed}`);
  });
});

describe("InclusionInbox spam pricing (M365, G24)", function () {
  it("reproduces the defect's shape: the bond alone is fully refunded",
     async function () {
    // The finding was that posting was free on the honest path. The
    // bond still is — that is deliberate. What must NOT be free is
    // the fee, and this test pins which half is which.
    const { inbox, poster, librarian } = await deployInbox();
    const entryId = id("m365-shape");
    await postExact(inbox, poster, entryId, id("m365-shape-p"));
    expect((await inbox.entries(entryId)).bond).to.equal(BOND);
    await inbox.connect(librarian).incorporate(entryId);
    expect((await inbox.entries(entryId)).bond).to.equal(0n);
    // and the fee did not come back with it
    expect(await inbox.operationsAccrued).to.not.equal(undefined);
    expect(await inbox.operationsAccrued()).to.equal(BASE_FEE);
  });

  it("honest posting costs the fee and returns the bond", async function () {
    const { inbox, poster, librarian } = await deployInbox();
    const entryId = id("m365-honest");
    const before = await ethers.provider.getBalance(poster.address);
    const rc = await (await postExact(
      inbox, poster, entryId, id("m365-honest-p"))).wait();
    await inbox.connect(librarian).incorporate(entryId);
    // M383 pull lesson: the bond is credited, not pushed — the
    // poster pulls it back, so the net cost is still only the fee
    const rc2 = await (await inbox.connect(poster).claim()).wait();
    const gas = rc.gasUsed * rc.gasPrice + rc2.gasUsed * rc2.gasPrice;
    const after = await ethers.provider.getBalance(poster.address);
    // net of gas, the poster is out exactly the posting fee
    expect(before - after - gas).to.equal(BASE_FEE);
  });

  it("the fee reaches the operations line and only it can claim",
     async function () {
    const { inbox, poster, librarian, other, operations } =
      await deployInbox();
    await postExact(inbox, poster, id("m365-ops-1"), id("p1"));
    await postExact(inbox, poster, id("m365-ops-2"), id("p2"));
    // M383: the fee is held against the entry until someone does the
    // work, so it accrues on incorporation, not on posting
    expect(await inbox.operationsAccrued()).to.equal(0n);
    await inbox.connect(librarian).incorporate(id("m365-ops-1"));
    await inbox.connect(librarian).incorporate(id("m365-ops-2"));
    expect(await inbox.operationsAccrued()).to.equal(BASE_FEE * 2n);
    await expect(inbox.connect(other).claimOperations())
      .to.be.revertedWithCustomError(inbox, "NotOperationsLine");
    const before = await ethers.provider.getBalance(operations.address);
    const rc = await (
      await inbox.connect(operations).claimOperations()).wait();
    const after = await ethers.provider.getBalance(operations.address);
    expect(after - before + rc.gasUsed * rc.gasPrice)
      .to.equal(BASE_FEE * 2n);
    expect(await inbox.operationsAccrued()).to.equal(0n);
  });

  it("rejects a payment that is not exactly bond + fee",
     async function () {
    const { inbox, poster } = await deployInbox();
    await expect(inbox.connect(poster).post(
      id("m365-underpay"), id("p"), { value: BOND }))
      .to.be.revertedWithCustomError(inbox, "WrongPayment");
    await expect(inbox.connect(poster).post(
      id("m365-overpay"), id("p"), { value: BOND + BASE_FEE + 1n }))
      .to.be.revertedWithCustomError(inbox, "WrongPayment");
  });

  it("the fee escalates superlinearly past the free allowance",
     async function () {
    const { inbox, fresh } = await deployInbox();
    const fees = [];
    for (let k = 0; k < FREE_POSTS + 6; k++) {
      const quoted = await inbox.postingFee(fresh.address);
      expect(quoted).to.equal(expectedFee(k));
      fees.push(quoted);
      await postExact(inbox, fresh, id(`m365-esc-${k}`),
                      id(`m365-esc-p-${k}`));
    }
    // flat inside the allowance
    for (let k = 1; k < FREE_POSTS; k++) {
      expect(fees[k]).to.equal(fees[0]);
    }
    // strictly increasing outside it, and the SECOND difference is
    // positive too: that is what "superlinear" means
    for (let k = FREE_POSTS + 1; k < fees.length; k++) {
      expect(fees[k]).to.be.greaterThan(fees[k - 1]);
    }
    for (let k = FREE_POSTS + 2; k < fees.length; k++) {
      const d1 = fees[k] - fees[k - 1];
      const d2 = fees[k - 1] - fees[k - 2];
      expect(d1).to.be.greaterThan(d2);
    }
    // the campaign cost the plan cares about: 1,000 entries priced
    // by the same schedule, against the linear cost it replaces
    let campaign = 0n;
    for (let k = 0; k < 1000; k++) campaign += expectedFee(k);
    const linear = BASE_FEE * 1000n;
    expect(campaign / linear).to.be.greaterThan(100_000n);
    console.log(
      `M365: 1,000-entry campaign costs ${campaign} vs ${linear} ` +
      `flat — a factor of ${campaign / linear}`);
  });

  it("a spam campaign never reaches chainValid == false while the " +
     "librarian meets the capped obligation", async function () {
    // The window has to exceed the time the campaign itself takes to
    // land, or the test measures the harness rather than the cap:
    // every post is a block, so a 24-post campaign burns 24 blocks
    // before the librarian gets a turn.
    const bigWindow = 200;
    const { inbox, poster, librarian } = await deployInbox(bigWindow);
    const n = 3 * MAX_INC_PER_EPOCH; // three epochs of obligation
    const ids = [];
    for (let k = 0; k < n; k++) {
      const e = id(`m365-spam-${k}`);
      ids.push(e);
      await postExact(inbox, poster, e, id(`m365-spam-p-${k}`));
    }
    expect(await inbox.openCount()).to.equal(BigInt(n));
    expect(await inbox.chainValid()).to.equal(true);

    // the cap is what buys the time: the entry just past the first
    // epoch's obligation carries a deadline a full epoch later
    const atCap = await inbox.entries(ids[MAX_INC_PER_EPOCH - 1]);
    const pastCap = await inbox.entries(ids[MAX_INC_PER_EPOCH]);
    expect(pastCap.deadlineBlock - atCap.deadlineBlock)
      .to.be.greaterThanOrEqual(BigInt(EPOCH_BLOCKS));

    // the librarian works at exactly the capped rate, FIFO, and
    // rests between epochs
    for (let k = 0; k < n; k++) {
      expect(await inbox.chainValid()).to.equal(true);
      await inbox.connect(librarian).incorporate(ids[k]);
      if ((k + 1) % MAX_INC_PER_EPOCH === 0) {
        await mineBlocks(bigWindow); // rest until the next obligation
      }
    }
    expect(await inbox.chainValid()).to.equal(true);
    expect(await inbox.openCount()).to.equal(0n);
  });

  it("incorporation is FIFO: the librarian cannot skip a rival",
     async function () {
    const { inbox, poster, fresh, librarian } = await deployInbox();
    const rival = id("m365-fifo-rival");
    const friend = id("m365-fifo-friend");
    await postExact(inbox, fresh, rival, id("m365-fifo-p1"));
    await postExact(inbox, poster, friend, id("m365-fifo-p2"));
    expect(await inbox.headEntry()).to.equal(rival);
    await expect(inbox.connect(librarian).incorporate(friend))
      .to.be.revertedWithCustomError(inbox, "NotHeadOfQueue");
    await inbox.connect(librarian).incorporate(rival);
    await inbox.connect(librarian).incorporate(friend);
    await expect(inbox.connect(librarian).incorporate(friend))
      .to.be.revertedWithCustomError(inbox, "QueueEmpty");
  });

  it("a censored entry from a fresh address is incorporated within " +
     "the registered bound", async function () {
    const { inbox, poster, fresh, librarian } = await deployInbox();
    // an attacker fills the queue ahead of the honest poster
    const backlog = 2 * MAX_INC_PER_EPOCH;
    const ids = [];
    for (let k = 0; k < backlog; k++) {
      const e = id(`m365-cen-${k}`);
      ids.push(e);
      await postExact(inbox, poster, e, id(`m365-cen-p-${k}`));
    }
    const censored = id("m365-censored");
    const rc = await (await postExact(
      inbox, fresh, censored, id("m365-censored-p"))).wait();
    const entry = await inbox.entries(censored);

    // the bound is finite, computable at posting time, and equals
    // window + ceil-free backlog rollover
    const extraEpochs = BigInt(Math.floor(backlog / MAX_INC_PER_EPOCH));
    const bound = BigInt(rc.blockNumber) + BigInt(WINDOW)
      + extraEpochs * BigInt(EPOCH_BLOCKS);
    expect(entry.deadlineBlock).to.equal(bound);
    expect(entry.deadlineBlock).to.be.greaterThan(0n);

    // and it is actually reached: the librarian clears the backlog
    // and the censored entry lands before its deadline
    for (const e of ids) {
      await inbox.connect(librarian).incorporate(e);
    }
    await inbox.connect(librarian).incorporate(censored);
    const done = await inbox.entries(censored);
    expect(done.incorporatedBlock).to.be.greaterThan(0n);
    expect(done.incorporatedBlock)
      .to.be.lessThanOrEqual(entry.deadlineBlock);
    expect(await inbox.chainValid()).to.equal(true);
  });

  it("chainValid stays O(1) as the queue grows", async function () {
    const { inbox, poster } = await deployInbox();
    await postExact(inbox, poster, id("m365-o1-0"), id("m365-o1-p-0"));
    const gasSmall = await inbox.chainValid.estimateGas();
    for (let k = 1; k < 30; k++) {
      await postExact(inbox, poster, id(`m365-o1-${k}`),
                      id(`m365-o1-p-${k}`));
    }
    const gasLarge = await inbox.chainValid.estimateGas();
    expect(await inbox.openCount()).to.equal(30n);
    // the pre-repair implementation scanned the open array, so this
    // would have grown with the queue
    expect(gasLarge).to.equal(gasSmall);
    console.log(
      `M365: chainValid gas ${gasSmall} at 1 entry, ` +
      `${gasLarge} at 30`);
  });
});

describe("InclusionInbox librarian source (M382, G53)", function () {
  it("reads the librarian live rather than freezing it at deploy",
     async function () {
    const { inbox, source, librarian, other } = await deployInbox();
    expect(await inbox.librarian()).to.equal(librarian.address);
    await source.setLibrarian(other.address);
    expect(await inbox.librarian()).to.equal(other.address);
  });

  it("a replaced librarian loses the role and the deputy gains it, " +
     "with the open queue intact", async function () {
    const { inbox, source, poster, librarian, other } =
      await deployInbox();
    // an entry is posted under the old librarian and left open —
    // this is the censored-entry case, mid-replacement
    const entryId = id("m382-mid-replacement");
    await postExact(inbox, poster, entryId, id("m382-p"));
    expect(await inbox.openCount()).to.equal(1n);

    // the replacement vote fires; the deputy takes the address
    await source.setLibrarian(other.address);

    // the convicted librarian no longer holds the role here. Before
    // M382 this address was immutable, so it kept it permanently.
    // It can still push the queue forward — since M383 anyone can —
    // but it does so as a stranger and earns nothing for it.
    expect(await inbox.librarian()).to.equal(other.address);

    // and the deputy inherits the queue rather than a fresh contract
    expect(await inbox.openCount()).to.equal(1n);
    await inbox.connect(other).incorporate(entryId);
    expect(await inbox.librarianIncorporations()).to.equal(1n);
    expect(await inbox.openCount()).to.equal(0n);
    expect(await inbox.chainValid()).to.equal(true);
  });

  it("the old librarian's incorporation counts as a stranger's",
     async function () {
    const { inbox, source, poster, librarian, other } =
      await deployInbox();
    const entryId = id("m382-old-librarian");
    await postExact(inbox, poster, entryId, id("m382-p2"));
    await source.setLibrarian(other.address);
    await inbox.connect(librarian).incorporate(entryId);
    expect(await inbox.foreignIncorporations()).to.equal(1n);
    expect(await inbox.librarianIncorporations()).to.equal(0n);
  });

  it("rejects a zero librarian source at deploy", async function () {
    const [, , , operations] = await ethers.getSigners();
    const I = await ethers.getContractFactory("InclusionInbox");
    await expect(I.deploy(
      ethers.ZeroAddress, WINDOW, BOND, operations.address,
      EPOCH_BLOCKS, BASE_FEE, FREE_POSTS, MAX_INC_PER_EPOCH))
      .to.be.revertedWith("librarian source must be set");
  });
});

describe("InclusionInbox incentives (M383)", function () {
  it("censorship is structurally impossible: the poster clears its " +
     "own entry without asking", async function () {
    const { inbox, poster } = await deployInbox();
    const entryId = id("m383-self");
    await postExact(inbox, poster, entryId, id("m383-self-p"));
    // the librarian never acts and is never asked
    await inbox.connect(poster).incorporate(entryId);
    expect(await inbox.openCount()).to.equal(0n);
    expect(await inbox.chainValid()).to.equal(true);
  });

  it("a prompt librarian is paid; a stalled one is not",
     async function () {
    const { inbox, poster, librarian, operations } = await deployInbox();
    await postExact(inbox, poster, id("m383-paid"), id("m383-paid-p"));
    await inbox.connect(librarian).incorporate(id("m383-paid"));
    expect(await inbox.claimable(operations.address))
      .to.equal(BASE_FEE);

    // second entry: the librarian sits on it past the deadline and a
    // stranger clears it. The fee follows the work.
    const { inbox: i2, poster: p2, other, operations: ops2 } =
      await deployInbox();
    await postExact(i2, p2, id("m383-stalled"), id("m383-stalled-p"));
    await mineBlocks(WINDOW + 1);
    await i2.connect(other).incorporate(id("m383-stalled"));
    expect(await i2.claimable(ops2.address)).to.equal(0n);
    expect(await i2.claimable(other.address)).to.equal(BASE_FEE);
  });

  it("the bounty is claimable by whoever earned it", async function () {
    const { inbox, poster, other } = await deployInbox();
    await postExact(inbox, poster, id("m383-bounty"), id("m383-b-p"));
    await mineBlocks(WINDOW + 1);
    await inbox.connect(other).incorporate(id("m383-bounty"));
    const before = await ethers.provider.getBalance(other.address);
    const rc = await (await inbox.connect(other).claim()).wait();
    const after = await ethers.provider.getBalance(other.address);
    expect(after - before + rc.gasUsed * rc.gasPrice)
      .to.equal(BASE_FEE);
    await expect(inbox.connect(other).claim())
      .to.be.revertedWithCustomError(inbox, "NothingToClaim");
  });

  it("stalling is self-penalising without any vote or slash",
     async function () {
    // the point of the incentive: a librarian that ignores the queue
    // loses its income to whoever does the work instead. No quorum,
    // no governance, no burn.
    const { inbox, poster, other, operations } = await deployInbox(200);
    let toOps = 0n, toStranger = 0n;
    for (let k = 0; k < 4; k++) {
      const e = id(`m383-stall-${k}`);
      await postExact(inbox, poster, e, id(`m383-stall-p-${k}`));
    }
    await mineBlocks(220);
    for (let k = 0; k < 4; k++) {
      await inbox.connect(other).incorporate(id(`m383-stall-${k}`));
    }
    toOps = await inbox.claimable(operations.address);
    toStranger = await inbox.claimable(other.address);
    expect(toOps).to.equal(0n);
    expect(toStranger).to.be.greaterThan(0n);
  });

  it("self-incorporating inside the window does not refund the fee, " +
     "so the spam price survives", async function () {
    // the exploit this guards: if a poster could post and instantly
    // reclaim its own fee, the superlinear spam schedule of M365
    // would collapse to gas.
    const { inbox, poster, operations } = await deployInbox();
    const entryId = id("m383-no-refund");
    await postExact(inbox, poster, entryId, id("m383-nr-p"));
    await inbox.connect(poster).incorporate(entryId);
    // the bond is credited back to the poster (pull), but the fee
    // is never refunded — the spam price is the fee, not the bond
    expect(await inbox.claimable(poster.address)).to.equal(BOND);
    expect(await inbox.claimable(operations.address))
      .to.equal(BASE_FEE);
  });

  it("a censored poster pays only gas, never the fee twice",
     async function () {
    const { inbox, poster } = await deployInbox();
    const entryId = id("m383-censored");
    await postExact(inbox, poster, entryId, id("m383-c-p"));
    await mineBlocks(WINDOW + 1);
    // past the deadline the poster clears its own entry and takes
    // the bounty, which is the fee it paid in the first place, and
    // pulls the bond back (M383 pull lesson)
    await inbox.connect(poster).incorporate(entryId);
    expect(await inbox.claimable(poster.address))
      .to.equal(BOND + BASE_FEE);
  });

  it("a reverting poster cannot jam the queue: the bond is credited, " +
     "not pushed", async function () {
    // the M383 pull lesson applied to the bond. Under a PUSH bond
    // return, a poster contract that reverts on receive would make
    // incorporate revert forever and the queue head would be invalid
    // for everyone. With the pull, incorporation succeeds, the bond
    // sits in claimable, and only the poster's own claim could fail.
    const { inbox, librarian, other } = await deployInbox();
    const RP = await ethers.getContractFactory("RejectingPoster");
    const rp = await RP.deploy();
    await rp.waitForDeployment();
    const rpAddr = await rp.getAddress();
    const inboxAddr = await inbox.getAddress();
    const entryId = id("m383-rejecting-poster");
    const fee = await inbox.postingFee(rpAddr);
    await rp.post(inboxAddr, entryId, id("m383-rp-p"),
      { value: BOND + fee });
    expect((await inbox.entries(entryId)).bond).to.equal(BOND);
    // the librarian clears it; the queue head advances
    await inbox.connect(librarian).incorporate(entryId);
    expect((await inbox.entries(entryId)).incorporatedBlock)
      .to.not.equal(0n);
    expect((await inbox.entries(entryId)).bond).to.equal(0n);
    expect(await inbox.chainValid()).to.equal(true);
    // the reverting poster's bond is pullable, never pushed
    expect(await inbox.claimable(rpAddr)).to.equal(BOND);
    // and a foreign caller can still clear entries behind it
    const entry2 = id("m383-rejecting-poster-2");
    await rp.post(inboxAddr, entry2, id("m383-rp-p2"),
      { value: BOND + fee });
    await inbox.connect(other).incorporate(entry2);
    expect(await inbox.chainValid()).to.equal(true);
  });
});
