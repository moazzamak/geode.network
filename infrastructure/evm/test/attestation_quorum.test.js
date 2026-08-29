// CreditLedger — the on-chain attestation quorum (closure of the R3
// shared residual). A challenged filing is decided by the network's
// own credit-holding identities, never by a single key: each eligible
// identity (vested credits > 0) attests whether the filing is wrong;
// weight is vested credits, capped per identity at 20% of the eligible
// total; a VERDICT needs two-thirds of capped participating weight, a
// participation floor of one-third of eligible weight, and at least
// three distinct identities. A challenge that reaches no verdict by
// the attestation window is unsubstantiated: the filing proceeds as
// if unchallenged and the challenger's bond burns. The same primitive
// decides slash, registry-change, and attribution-root filings.
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");

const FEE = 1000n;
const EPOCH = 7 * 24 * 3600;
const SLASH_WINDOW = 7 * 24 * 3600;
const ATTEST_WINDOW = 7 * 24 * 3600;
const BOND = ethers.parseEther("1");
const VESTING = 5 * EPOCH;          // fully vest the early credits + close epoch 0

async function advanceTime(seconds) {
  await ethers.provider.send("evm_increaseTime", [seconds]);
  await ethers.provider.send("evm_mine");
}

const ROOT = ethers.keccak256(ethers.toUtf8Bytes("epoch-0-root"));
const ROOT_WRONG = ethers.keccak256(ethers.toUtf8Bytes("epoch-0-wrong"));

async function deployFixture() {
  const [owner, devFund, payer, operator, librarian, rando,
        v1, v2, v3, v4, v5, v6, stranger, filer] = await ethers.getSigners();
  const Ledger = await ethers.getContractFactory("CreditLedger");
  const ledger = await upgrades.deployProxy(Ledger, [devFund.address, FEE]);
  await ledger.waitForDeployment();
  await ledger.setLibrarian(librarian.address);
  const artifactId = ethers.keccak256(ethers.toUtf8Bytes("arm-a"));
  return { owner, devFund, payer, operator, librarian, rando,
           v1, v2, v3, v4, v5, v6, stranger, filer, ledger, artifactId };
}

// Six equal voters, 100 vested credits each. Eligible total 600,
// per-identity cap 120 (never binds), participation floor 199
// (two voters cross it). A verdict therefore needs three distinct
// identities and two-thirds of the participating weight.
async function quorumFixture() {
  const f = await loadFixture(deployFixture);
  await f.ledger.connect(f.operator).register(
    f.artifactId, f.v1.address, 10n, ethers.ZeroHash, { value: FEE });
  await f.ledger.connect(f.librarian).setAdmitted(f.artifactId, true);
  await f.ledger.connect(f.payer).deposit({ value: 100000n });
  const voters = [f.v1, f.v2, f.v3, f.v4, f.v5, f.v6];
  const payers = voters.map(() => f.payer.address);
  const entries = voters.map((v) => ({
    artifactId: f.artifactId, who: v.address, amount: 100n }));
  await f.ledger.connect(f.librarian).recordCredits(payers, entries);
  await advanceTime(VESTING);
  f.voters = voters;
  return f;
}

// A whale (1000) plus four 100-credit voters. Eligible total 1400,
// cap 280, floor 466. Used to show the cap binds the whale.
async function whaleFixture() {
  const f = await loadFixture(deployFixture);
  await f.ledger.connect(f.operator).register(
    f.artifactId, f.v1.address, 10n, ethers.ZeroHash, { value: FEE });
  await f.ledger.connect(f.librarian).setAdmitted(f.artifactId, true);
  await f.ledger.connect(f.payer).deposit({ value: 100000n });
  await f.ledger.connect(f.librarian).recordCredits(
    [f.payer.address, f.payer.address, f.payer.address,
     f.payer.address, f.payer.address],
    [{ artifactId: f.artifactId, who: f.v1.address, amount: 1000n },
     { artifactId: f.artifactId, who: f.v2.address, amount: 100n },
     { artifactId: f.artifactId, who: f.v3.address, amount: 100n },
     { artifactId: f.artifactId, who: f.v4.address, amount: 100n },
     { artifactId: f.artifactId, who: f.v5.address, amount: 100n }]);
  await advanceTime(VESTING);
  f.voters = [f.v1, f.v2, f.v3, f.v4, f.v5];
  return f;
}

async function fileAndChallenge(f, root) {
  await f.ledger.connect(f.filer).fileAttributionRoot(0n, root,
    { value: BOND });
  await f.ledger.connect(f.v1).challengeAttributionRoot(0n,
    { value: BOND });
}

async function openAttestationWindow() {
  await advanceTime(SLASH_WINDOW);   // challenge window closes
}

async function finalize(f, who) {
  const rec = ethers.keccak256(ethers.toUtf8Bytes("quorum-record"));
  await f.ledger.connect(who).finalizeAttributionRoot(0n, rec);
}

describe("the on-chain attestation quorum", function () {
  describe("who may attest, and when", function () {
    it("attestation requires vested credits (NotEligible)", async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT_WRONG);
      await openAttestationWindow();
      await expect(f.ledger.connect(f.rando).attestAttributionRoot(0n, true))
        .to.be.revertedWithCustomError(f.ledger, "NotEligible");
    });

    it("attestation requires a challenged filing (NotChallenged)",
       async function () {
      const f = await loadFixture(quorumFixture);
      await f.ledger.connect(f.filer).fileAttributionRoot(0n, ROOT,
        { value: BOND });
      await openAttestationWindow();
      await expect(f.ledger.connect(f.v1).attestAttributionRoot(0n, true))
        .to.be.revertedWithCustomError(f.ledger, "NotChallenged");
    });

    it("attestation only inside the attestation window", async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT_WRONG);
      // too early: the challenge window is still open
      await expect(f.ledger.connect(f.v2).attestAttributionRoot(0n, true))
        .to.be.revertedWithCustomError(f.ledger, "WindowOpen");
      await openAttestationWindow();
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, true);
      await advanceTime(ATTEST_WINDOW);   // window closed
      await expect(f.ledger.connect(f.v3).attestAttributionRoot(0n, true))
        .to.be.revertedWithCustomError(f.ledger, "WindowClosed");
    });

    it("an identity attests once per filing (AlreadyAttested)",
       async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT_WRONG);
      await openAttestationWindow();
      await expect(f.ledger.connect(f.v1).attestAttributionRoot(0n, true))
        .to.emit(f.ledger, "Attested")
        .withArgs(2, 0, f.v1.address, true, 100n);
      await expect(f.ledger.connect(f.v1).attestAttributionRoot(0n, true))
        .to.be.revertedWithCustomError(f.ledger, "AlreadyAttested");
    });
  });

  describe("the verdict", function () {
    it("a void verdict needs two-thirds, the floor, and three distinct " +
       "identities; anyone finalizes", async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT_WRONG);
      await openAttestationWindow();
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v3).attestAttributionRoot(0n, true);
      const rec = ethers.keccak256(ethers.toUtf8Bytes("quorum-void"));
      await expect(f.ledger.connect(f.rando).finalizeAttributionRoot(0n, rec))
        .to.emit(f.ledger, "RootResolved").withArgs(0, true, rec);
      // the wrong root never lands; the filer's bond burns
      expect(await f.ledger.attributionRoot(0n)).to.equal(ethers.ZeroHash);
      expect(await f.ledger.rootBondsBurned()).to.equal(BOND);
      // the challenger pulls its own bond back; the filer has nothing
      await f.ledger.connect(f.v1).claimRootBond(0);
      expect(await f.ledger.rootBondHeld()).to.equal(0n);
      await expect(f.ledger.connect(f.filer).claimRootBond(0))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("an uphold verdict lands the root and burns the challenger's bond",
       async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT);
      await openAttestationWindow();
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, false);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, false);
      await f.ledger.connect(f.v3).attestAttributionRoot(0n, false);
      await finalize(f, f.stranger);
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT);
      expect(await f.ledger.rootBondsBurned()).to.equal(BOND);
      await f.ledger.connect(f.filer).claimRootBond(0);
      await expect(f.ledger.connect(f.v1).claimRootBond(0))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("a one-third minority cannot void; the filing proceeds by default",
       async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT_WRONG);
      await openAttestationWindow();
      // 200 void vs 300 uphold: neither side reaches two-thirds
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v3).attestAttributionRoot(0n, false);
      await f.ledger.connect(f.v4).attestAttributionRoot(0n, false);
      await f.ledger.connect(f.v5).attestAttributionRoot(0n, false);
      // no verdict while the window is open
      await expect(f.ledger.connect(f.rando).finalizeAttributionRoot(
        0n, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "WindowOpen");
      await advanceTime(ATTEST_WINDOW);
      await finalize(f, f.rando);
      // unsubstantiated challenge: the root lands, the challenger burns
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT_WRONG);
      expect(await f.ledger.rootBondsBurned()).to.equal(BOND);
    });

    it("two distinct identities cannot reach a verdict (distinct floor)",
       async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT_WRONG);
      await openAttestationWindow();
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, true);
      await expect(f.ledger.connect(f.rando).finalizeAttributionRoot(
        0n, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "WindowOpen");
      // window closes: the challenge is unsubstantiated, the filing proceeds
      await advanceTime(ATTEST_WINDOW);
      await finalize(f, f.rando);
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT_WRONG);
      expect(await f.ledger.rootBondsBurned()).to.equal(BOND);
    });

    it("the default: an unsubstantiated challenge proceeds and the " +
       "challenger's bond burns", async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT);
      await openAttestationWindow();
      // nobody attests
      await advanceTime(ATTEST_WINDOW);
      await finalize(f, f.rando);
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT);
      expect(await f.ledger.rootBondsBurned()).to.equal(BOND);
      await f.ledger.connect(f.filer).claimRootBond(0);
      expect(await f.ledger.rootBondHeld()).to.equal(0n);
    });

    it("the 20% cap: a whale cannot carry a void against the field",
       async function () {
      const f = await loadFixture(whaleFixture); // v1=1000, others 100, cap 280
      await f.ledger.connect(f.filer).fileAttributionRoot(0n, ROOT_WRONG,
        { value: BOND });
      await f.ledger.connect(f.v1).challengeAttributionRoot(0n,
        { value: BOND });
      await openAttestationWindow();
      // the whale attests void (capped to 280); two voters uphold
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, false);
      await f.ledger.connect(f.v3).attestAttributionRoot(0n, false);
      // uncapped, the whale alone would pass two-thirds of the
      // participants; capped, no verdict is reachable and the filing
      // proceeds at the window close
      await advanceTime(ATTEST_WINDOW);
      await finalize(f, f.rando);
      expect(await f.ledger.attributionRoot(0n)).to.equal(ROOT_WRONG);
      expect(await f.ledger.rootBondsBurned()).to.equal(BOND);
    });

    it("the librarian key cannot decide a challenge: it has no vote " +
       "without earned standing, and finalization is permissionless",
       async function () {
      const f = await loadFixture(quorumFixture);
      await fileAndChallenge(f, ROOT_WRONG);
      await openAttestationWindow();
      // the librarian has never earned credits: no vote
      await expect(f.ledger.connect(f.librarian).attestAttributionRoot(0n, true))
        .to.be.revertedWithCustomError(f.ledger, "NotEligible");
      // a real quorum decides; anyone finalizes
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v3).attestAttributionRoot(0n, true);
      await finalize(f, f.rando);
      expect(await f.ledger.attributionRoot(0n)).to.equal(ethers.ZeroHash);
    });
  });

  describe("the same quorum decides slash filings", function () {
    it("a false accusation is voided: nothing burns, the filer's bond burns",
       async function () {
      const f = await loadFixture(quorumFixture);
      await f.ledger.connect(f.filer).fileSlash(
        f.v1.address, f.artifactId, 50n, 2, ethers.ZeroHash,
        { value: BOND });
      await f.ledger.connect(f.v1).challengeSlash(0, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await f.ledger.connect(f.v1).attestSlash(0, true);
      await f.ledger.connect(f.v2).attestSlash(0, true);
      await f.ledger.connect(f.v3).attestSlash(0, true);
      const rec = ethers.keccak256(ethers.toUtf8Bytes("quorum-void"));
      await expect(f.ledger.connect(f.rando).finalizeSlash(0, rec))
        .to.emit(f.ledger, "SlashResolved").withArgs(0, false, rec);
      expect(await f.ledger.burnedTotal()).to.equal(0n);
      expect(await f.ledger.slashBondsBurned()).to.equal(BOND);
      await f.ledger.connect(f.v1).claimSlashBond(0);
      await expect(f.ledger.connect(f.filer).claimSlashBond(0))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("an upheld slash executes and the challenger's bond burns",
       async function () {
      const f = await loadFixture(quorumFixture);
      await f.ledger.connect(f.filer).fileSlash(
        f.v1.address, f.artifactId, 50n, 2, ethers.ZeroHash,
        { value: BOND });
      await f.ledger.connect(f.v1).challengeSlash(0, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await f.ledger.connect(f.v1).attestSlash(0, false);
      await f.ledger.connect(f.v2).attestSlash(0, false);
      await f.ledger.connect(f.v3).attestSlash(0, false);
      const rec = ethers.keccak256(ethers.toUtf8Bytes("quorum-uphold"));
      await expect(f.ledger.connect(f.rando).finalizeSlash(0, rec))
        .to.emit(f.ledger, "SlashResolved").withArgs(0, true, rec);
      expect(await f.ledger.burnedTotal()).to.equal(50n);
      expect(await f.ledger.slashBondsBurned()).to.equal(BOND);
      await f.ledger.connect(f.filer).claimSlashBond(0);
      expect(await f.ledger.slashBondHeld()).to.equal(0n);
    });
  });

  describe("the same quorum decides registry-change filings", function () {
    it("a false delist is voided by the quorum", async function () {
      const f = await loadFixture(quorumFixture);
      await f.ledger.connect(f.filer).fileRegistryChange(
        1, f.artifactId, false, 0, ethers.ZeroHash, { value: BOND });
      await f.ledger.connect(f.operator).challengeRegistryChange(
        0, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await f.ledger.connect(f.v1).attestRegistryChange(0, true);
      await f.ledger.connect(f.v2).attestRegistryChange(0, true);
      await f.ledger.connect(f.v3).attestRegistryChange(0, true);
      const rec = ethers.keccak256(ethers.toUtf8Bytes("quorum-void"));
      await expect(f.ledger.connect(f.rando).finalizeRegistryChange(0, rec))
        .to.emit(f.ledger, "RegistryChangeResolved").withArgs(0, false, rec);
      expect((await f.ledger.regs(f.artifactId)).delisted).to.equal(false);
      expect(await f.ledger.registryBondsBurned()).to.equal(BOND);
      await f.ledger.connect(f.operator).claimRegistryBond(0);
      await expect(f.ledger.connect(f.filer).claimRegistryBond(0))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("an upheld admission applies", async function () {
      const f = await loadFixture(quorumFixture);
      const fresh = ethers.keccak256(ethers.toUtf8Bytes("arm-b"));
      await f.ledger.connect(f.operator).register(
        fresh, f.v2.address, 10n, ethers.ZeroHash, { value: FEE });
      await f.ledger.connect(f.filer).fileRegistryChange(
        0, fresh, true, 0, ethers.ZeroHash, { value: BOND });
      await f.ledger.connect(f.operator).challengeRegistryChange(
        0, { value: BOND });
      await advanceTime(SLASH_WINDOW);
      await f.ledger.connect(f.v1).attestRegistryChange(0, false);
      await f.ledger.connect(f.v2).attestRegistryChange(0, false);
      await f.ledger.connect(f.v3).attestRegistryChange(0, false);
      await f.ledger.connect(f.rando).finalizeRegistryChange(
        0, ethers.ZeroHash);
      expect((await f.ledger.regs(fresh)).admitted).to.equal(true);
      await f.ledger.connect(f.filer).claimRegistryBond(0);
    });
  });

  describe("the attestation reward (ops line -> verdict work)", function () {
    async function inboxFixture() {
      const f = await loadFixture(quorumFixture);
      const Inbox = await ethers.getContractFactory("InclusionInbox");
      const inbox = await Inbox.deploy(
        await f.ledger.getAddress(),   // librarian source: the ledger
        10, 100n,                      // window blocks, inbox bond
        await f.ledger.getAddress(),   // operations line: the ledger
        5000, 10n, 4, 8);              // epoch blocks, fee, free, cap
      await inbox.waitForDeployment();
      return { ...f, inbox };
    }

    const POT = 6n;                    // one registered verdict pot (<= one posting fee)
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

    async function registerReward(f, amount) {
      await f.ledger.connect(f.owner).scheduleAttestationReward(amount);
      await advanceTime(2 * 24 * 3600);   // CHANGE_DELAY = 2 days
      await f.ledger.connect(f.owner).applyAttestationRewardChange();
    }

    it("the attestors on the endorsed side split the registered reward",
       async function () {
      const f = await loadFixture(inboxFixture);
      const fee = await fundLine(f);
      await registerReward(f, POT);
      await fileAndChallenge(f, ROOT_WRONG);
      await advanceTime(SLASH_WINDOW);
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v3).attestAttributionRoot(0n, true);
      await finalize(f, f.rando);
      // three equal winning voters: the 6-wei pot splits 2 each
      expect(await f.ledger.attestationClaimable(f.v1.address)).to.equal(2n);
      expect(await f.ledger.attestationClaimable(f.v2.address)).to.equal(2n);
      expect(await f.ledger.attestationClaimable(f.v3.address)).to.equal(2n);
      // the pool lost exactly the pot, never more
      expect(await f.ledger.operationsPool()).to.equal(fee - POT);
      // a winning attestor pulls its share
      await f.ledger.connect(f.v1).claimAttestationReward();
      expect(await f.ledger.attestationClaimable(f.v1.address)).to.equal(0n);
    });

    it("the default path pays no reward", async function () {
      const f = await loadFixture(inboxFixture);
      const fee = await fundLine(f);
      await registerReward(f, POT);
      await fileAndChallenge(f, ROOT);
      await advanceTime(SLASH_WINDOW);
      // only two attestors: no verdict, the window closes, the filing
      // proceeds by default
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, false);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, false);
      await advanceTime(ATTEST_WINDOW);
      await finalize(f, f.rando);
      expect(await f.ledger.operationsPool()).to.equal(fee);
      expect(await f.ledger.attestationClaimable(f.v1.address)).to.equal(0n);
      expect(await f.ledger.attestationClaimable(f.v2.address)).to.equal(0n);
    });

    it("an underfunded pool skips the reward publicly — nothing is minted",
       async function () {
      const f = await loadFixture(quorumFixture);   // no inbox -> empty pool
      await registerReward(f, POT);
      await fileAndChallenge(f, ROOT_WRONG);
      await advanceTime(SLASH_WINDOW);
      await f.ledger.connect(f.v1).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v2).attestAttributionRoot(0n, true);
      await f.ledger.connect(f.v3).attestAttributionRoot(0n, true);
      await expect(f.ledger.connect(f.rando).finalizeAttributionRoot(
        0n, ethers.ZeroHash))
        .to.emit(f.ledger, "AttestationRewardSkipped").withArgs(0, POT);
      expect(await f.ledger.operationsPool()).to.equal(0n);
      expect(await f.ledger.attestationClaimable(f.v1.address)).to.equal(0n);
    });
  });
});
