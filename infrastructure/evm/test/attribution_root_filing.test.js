// CreditLedger — R3-F1: any party may post the epoch attribution root.
// Covers the bonded propose-and-challenge root filing: any party files
// a closed epoch's root under a bond; an unchallenged filing executes
// after the window (write-once); a challenge escalates to the replay
// quorum; the loser's bond is burned; a stalled librarian can no
// longer freeze an epoch's payments.
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");

const FEE = 1000n;
const EPOCH = 7 * 24 * 3600;
const SLASH_WINDOW = 7 * 24 * 3600;
const BOND = ethers.parseEther("1");

async function advanceTime(seconds) {
  await ethers.provider.send("evm_increaseTime", [seconds]);
  await ethers.provider.send("evm_mine");
}

const ROOT = ethers.keccak256(ethers.toUtf8Bytes("epoch-0-root"));
const ROOT_WRONG = ethers.keccak256(ethers.toUtf8Bytes("epoch-0-wrong"));

async function deployFixture() {
  const [owner, devFund, payer, contributor, operator, librarian,
        rando, stranger] = await ethers.getSigners();
  const Ledger = await ethers.getContractFactory("CreditLedger");
  const ledger = await upgrades.deployProxy(Ledger, [devFund.address, FEE]);
  await ledger.waitForDeployment();
  await ledger.setLibrarian(librarian.address);
  const artifactId = ethers.keccak256(ethers.toUtf8Bytes("arm-a"));
  return { owner, devFund, payer, contributor, operator, librarian,
           rando, stranger, ledger, artifactId };
}

async function closedEpochFixture() {
  const f = await loadFixture(deployFixture);
  // register, admit, and credit one session so the epoch has work
  await f.ledger.connect(f.operator).register(
    f.artifactId, f.contributor.address, 10n, ethers.ZeroHash,
    { value: FEE });
  await f.ledger.connect(f.librarian).setAdmitted(f.artifactId, true);
  await f.ledger.connect(f.payer).deposit({ value: 100000n });
  await f.ledger.connect(f.librarian).recordCredits(
    [f.payer.address],
    [{ artifactId: f.artifactId, who: f.contributor.address,
       amount: 2000n }]);
  return f;
}

describe("any party may post the attribution root (R3-F1)", function () {
  describe("the permissionless common path", function () {
    it("any party files and executes a closed epoch's root, " +
       "librarian absent", async function () {
      const f = await loadFixture(closedEpochFixture);
      // close epoch 0 and file its root from a stranger
      await advanceTime(EPOCH);
      await expect(f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND }))
        .to.emit(f.ledger, "RootFiled")
        .withArgs(0n, 0n, ROOT, f.stranger.address);
      // cannot execute inside the window
      await expect(f.ledger.connect(f.stranger).executeAttributionRoot(0n))
        .to.be.revertedWithCustomError(f.ledger, "WindowOpen");
      // after the window, anyone presses the button
      await advanceTime(SLASH_WINDOW);
      await expect(f.ledger.connect(f.rando).executeAttributionRoot(0n))
        .to.emit(f.ledger, "AttributionRootPosted").withArgs(0n, ROOT);
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT);
      // the filer pulls its bond
      await f.ledger.connect(f.stranger).claimRootBond(0n);
    });

    it("shape is rejected at filing time", async function () {
      const f = await loadFixture(closedEpochFixture);
      // open epoch (not yet rolled) rejected
      await expect(f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND }))
        .to.be.revertedWithCustomError(f.ledger, "EpochNotClosed");
      await advanceTime(EPOCH);
      // zero root rejected
      await expect(f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ethers.ZeroHash, { value: BOND }))
        .to.be.revertedWithCustomError(f.ledger, "NoAttributionRoot");
      // wrong bond rejected
      await expect(f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: 1n }))
        .to.be.revertedWithCustomError(f.ledger, "WrongBond");
    });

    it("a stalled librarian cannot freeze the epoch's payments",
       async function () {
      const f = await loadFixture(closedEpochFixture);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await f.ledger.connect(f.rando).executeAttributionRoot(0n);
      // the payee draws its credit against the stranger-posted root
      // (bad proof here only proves the root is the one on file)
      await expect(f.ledger.connect(f.contributor).claimAttribution(
        0n, f.artifactId, f.contributor.address, 2000n, [ethers.ZeroHash]))
        .to.be.revertedWithCustomError(f.ledger, "BadProof");
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT);
    });

    it("the root is write-once: a second filing cannot displace it",
       async function () {
      const f = await loadFixture(closedEpochFixture);
      await advanceTime(EPOCH);
      // a filing for an epoch that already has a root is rejected
      await f.ledger.connect(f.librarian).postAttributionRoot(0n, ROOT);
      await expect(f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT_WRONG, { value: BOND }))
        .to.be.revertedWithCustomError(f.ledger, "RootAlreadyPosted");
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT);
    });

    it("a filing for an epoch the fast path later covers skips " +
       "gracefully and returns the bond", async function () {
      const f = await loadFixture(closedEpochFixture);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND });
      // the librarian posts the same root during the window
      await f.ledger.connect(f.librarian).postAttributionRoot(0n, ROOT);
      await advanceTime(SLASH_WINDOW);
      await expect(f.ledger.connect(f.rando).executeAttributionRoot(0n))
        .to.emit(f.ledger, "RootSkipped")
        .withArgs(0n, "root already posted");
      await f.ledger.connect(f.stranger).claimRootBond(0n);
    });
  });

  describe("a false root is refutable (challenge + quorum)", function () {
    it("a challenged filing cannot auto-execute", async function () {
      const f = await loadFixture(closedEpochFixture);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT_WRONG, { value: BOND });
      // the contributor it would mis-pay refutes it
      await f.ledger.connect(f.contributor).challengeAttributionRoot(
        0n, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await expect(f.ledger.connect(f.rando).executeAttributionRoot(0n))
        .to.be.revertedWithCustomError(f.ledger, "ChallengePending");
    });

    it("quorum innocent: the root lands and the challenger's bond burns",
       async function () {
      const f = await loadFixture(closedEpochFixture);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND });
      await f.ledger.connect(f.contributor).challengeAttributionRoot(
        0n, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      const quorum = ethers.keccak256(ethers.toUtf8Bytes("quorum-ok"));
      await f.ledger.connect(f.librarian).resolveAttributionRoot(
        0n, false, quorum);
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT);
      // the filer is refunded; the challenger's stake is burned
      await f.ledger.connect(f.stranger).claimRootBond(0n);
      await expect(f.ledger.connect(f.contributor).claimRootBond(0n))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("quorum guilty: the false root is void and the filer's bond burns",
       async function () {
      const f = await loadFixture(closedEpochFixture);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT_WRONG, { value: BOND });
      await f.ledger.connect(f.contributor).challengeAttributionRoot(
        0n, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      const quorum = ethers.keccak256(ethers.toUtf8Bytes("quorum-void"));
      await f.ledger.connect(f.librarian).resolveAttributionRoot(
        0n, true, quorum);
      // nothing lands for the epoch
      expect(await f.ledger.attributionRoot(0n)).to.equal(ethers.ZeroHash);
      // the challenger is refunded; the filer's stake is burned
      await f.ledger.connect(f.contributor).claimRootBond(0n);
      await expect(f.ledger.connect(f.stranger).claimRootBond(0n))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("resolution is quorum-filed only (stranger -> NotLibrarian)",
       async function () {
      const f = await loadFixture(closedEpochFixture);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND });
      await f.ledger.connect(f.contributor).challengeAttributionRoot(
        0n, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await expect(f.ledger.connect(f.rando).resolveAttributionRoot(
        0n, false, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NotLibrarian");
    });
  });

  describe("the reward for the work (ops line -> root bounty)", function () {
    // the inbox names the ledger as its operations line, so the
    // non-refundable posting fees accrue to the ledger and are pulled
    // into the operations pool; the party whose root lands earns the
    // registered bounty. One prompt librarian incorporation pays one
    // base fee (10) into the line.
    async function inboxFixture() {
      const f = await loadFixture(closedEpochFixture);
      const Inbox = await ethers.getContractFactory("InclusionInbox");
      const inbox = await Inbox.deploy(
        await f.ledger.getAddress(),   // librarian source: the ledger
        10,                            // window blocks
        100n,                          // refundable inbox bond
        await f.ledger.getAddress(),   // operations line: the ledger
        5000,                          // epoch blocks (long)
        10n,                           // base posting fee
        4,                             // free posts per epoch
        8                              // capped librarian obligation
      );
      await inbox.waitForDeployment();
      return { ...f, inbox };
    }

    const BOUNTY = 10n;                  // == one base posting fee
    const INBOX_BOND = 100n;
    const entryId = (s) => ethers.keccak256(ethers.toUtf8Bytes(s));

    async function fundLine(f) {
      const eid = entryId("entry-0");
      const digest = ethers.keccak256(ethers.toUtf8Bytes("digest-0"));
      const fee = await f.inbox.postingFee(f.stranger.address);
      await f.inbox.connect(f.stranger).post(eid, digest, {
        value: INBOX_BOND + fee });
      await f.inbox.connect(f.librarian).incorporate(eid);
      await f.ledger.pullOperations(await f.inbox.getAddress());
      return fee;
    }

    async function registerBounty(f, amount) {
      await f.ledger.connect(f.owner).scheduleRootPostingBounty(amount);
      await advanceTime(2 * 24 * 3600);   // CHANGE_DELAY = 2 days
      await f.ledger.connect(f.owner).applyRootPostingBountyChange();
    }

    it("the filer whose root lands earns the registered bounty from " +
       "the operations-line pool", async function () {
      const f = await loadFixture(inboxFixture);
      const fee = await fundLine(f);
      await registerBounty(f, BOUNTY);
      expect(await f.ledger.rootPostingBounty()).to.equal(BOUNTY);
      expect(await f.ledger.operationsPool()).to.equal(fee);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await expect(f.ledger.connect(f.rando).executeAttributionRoot(0n))
        .to.emit(f.ledger, "RootBountyAwarded")
        .withArgs(0n, f.stranger.address, BOUNTY);
      expect(await f.ledger.operationsPool()).to.equal(fee - BOUNTY);
      expect(await f.ledger.rootBountyClaimable(f.stranger.address))
        .to.equal(BOUNTY);
      // the filer pulls the bounty; the pool never overdraws
      await f.ledger.connect(f.stranger).claimRootBounty();
      expect(await f.ledger.rootBountyClaimable(f.stranger.address))
        .to.equal(0n);
    });

    it("two concurrent filings with different roots: the first to " +
       "execute lands, the second skips, one bounty, both bonds " +
       "accounted", async function () {
      const f = await loadFixture(inboxFixture);
      await fundLine(f);
      await registerBounty(f, BOUNTY);
      await advanceTime(EPOCH);
      // two parties file the same closed epoch with different roots
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND });
      await f.ledger.connect(f.rando).fileAttributionRoot(
        0n, ROOT_WRONG, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      // the first to execute lands and earns the bounty
      await expect(f.ledger.connect(f.rando).executeAttributionRoot(0n))
        .to.emit(f.ledger, "AttributionRootPosted").withArgs(0n, ROOT)
        .to.emit(f.ledger, "RootBountyAwarded")
        .withArgs(0n, f.stranger.address, BOUNTY);
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT);
      // the second filing resolves as a skip: no root, no bounty
      await expect(f.ledger.connect(f.rando).executeAttributionRoot(1n))
        .to.emit(f.ledger, "RootSkipped")
        .withArgs(1n, "root already posted");
      expect(await f.ledger.rootBountyClaimable(f.stranger.address))
        .to.equal(BOUNTY);
      expect(await f.ledger.rootBountyClaimable(f.rando.address))
        .to.equal(0n);
      // the pool lost exactly one bounty, never more
      expect(await f.ledger.operationsPool()).to.equal(0n);
      // both bonds are accounted: each filer pulls its own back
      await f.ledger.connect(f.stranger).claimRootBond(0n);
      await f.ledger.connect(f.rando).claimRootBond(1n);
      expect(await f.ledger.rootBondHeld()).to.equal(0n);
    });

    it("a quorum-confirmed root also pays the filer " +
       "(challenge -> innocent)", async function () {
      const f = await loadFixture(inboxFixture);
      await fundLine(f);
      await registerBounty(f, BOUNTY);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND });
      await f.ledger.connect(f.contributor).challengeAttributionRoot(
        0n, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      const quorum = ethers.keccak256(ethers.toUtf8Bytes("quorum-ok"));
      await expect(f.ledger.connect(f.librarian).resolveAttributionRoot(
        0n, false, quorum))
        .to.emit(f.ledger, "RootBountyAwarded")
        .withArgs(0n, f.stranger.address, BOUNTY);
      expect(await f.ledger.operationsPool()).to.equal(0n);
    });

    it("a quorum-voided false root pays no bounty", async function () {
      const f = await loadFixture(inboxFixture);
      await fundLine(f);
      await registerBounty(f, BOUNTY);
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT_WRONG, { value: BOND });
      await f.ledger.connect(f.contributor).challengeAttributionRoot(
        0n, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      const quorum = ethers.keccak256(ethers.toUtf8Bytes("quorum-void"));
      await f.ledger.connect(f.librarian).resolveAttributionRoot(
        0n, true, quorum);
      expect(await f.ledger.operationsPool()).to.equal(BOUNTY);
      expect(await f.ledger.rootBountyClaimable(f.stranger.address))
        .to.equal(0n);
    });

    it("an underfunded pool skips the bounty publicly — nothing is minted",
       async function () {
      const f = await loadFixture(closedEpochFixture);
      await registerBounty(f, 500n);     // no inbox pull -> empty pool
      await advanceTime(EPOCH);
      await f.ledger.connect(f.stranger).fileAttributionRoot(
        0n, ROOT, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await expect(f.ledger.connect(f.rando).executeAttributionRoot(0n))
        .to.emit(f.ledger, "RootBountySkipped").withArgs(0n, 500n);
      expect(await f.ledger.operationsPool()).to.equal(0n);
      expect(await f.ledger.rootBountyClaimable(f.stranger.address))
        .to.equal(0n);
    });

    it("only the inbox that names this ledger as its line can be pulled",
       async function () {
      const f = await loadFixture(closedEpochFixture);
      await expect(f.ledger.pullOperations(ethers.ZeroAddress))
        .to.be.revertedWithCustomError(f.ledger, "ZeroAddress");
      const [, , , , , , , , otherOps] = await ethers.getSigners();
      const Inbox = await ethers.getContractFactory("InclusionInbox");
      const inbox = await Inbox.deploy(
        await f.ledger.getAddress(), 10, 100n, otherOps.address,
        5000, 10n, 4, 8);
      await inbox.waitForDeployment();
      await expect(f.ledger.pullOperations(await inbox.getAddress()))
        .to.be.revertedWithCustomError(inbox, "NotOperationsLine");
    });
  });
});
