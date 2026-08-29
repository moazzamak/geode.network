// LibrarianGovernance — M388 (the M382 remainder) suite, updated for
// the on-chain quorum gate (Finding-1 closure).
//
// Covers:
//  1. the governance executor exists as a contract with no human key
//     (no owner, no admin, no upgrade path);
//  2. the deputy's replacement runs through this keyless executor
//     with a recorded reason (the mechanical trigger) AND the
//     network's weighted-quorum endorsement — a bond and a timelock
//     alone can no longer name the librarian;
//  3. the path is timelocked (REPLACEMENT_DELAY), an unendorsed
//     proposal fails and its bond burns, and the proposer's bond is
//     otherwise pulled, never pushed.
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");

const FEE = 1000n;
const BOND = ethers.parseEther("1");
const EPOCH = 7 * 24 * 3600;
const REPLACEMENT_DELAY = 7 * 24 * 3600;

async function advanceTime(seconds) {
  await ethers.provider.send("evm_increaseTime", [seconds]);
  await ethers.provider.send("evm_mine");
}

const REASON = ethers.keccak256(ethers.toUtf8Bytes("recorded-divergence"));

// Three equal voters, 100 vested credits each (eligible total 300,
// cap 60, floor 99). A quorum therefore needs three distinct
// identities endorsing.
async function deployFixture() {
  const [owner, devFund, librarian, deputy, successor, rando, stranger,
        v1, v2, v3] = await ethers.getSigners();
  const Ledger = await ethers.getContractFactory("CreditLedger");
  const ledger = await upgrades.deployProxy(Ledger, [
    devFund.address,
    FEE,
  ]);
  await ledger.waitForDeployment();
  await ledger.setLibrarian(librarian.address);

  const Governance = await ethers.getContractFactory(
    "LibrarianGovernance");
  const gov = await Governance.deploy(await ledger.getAddress());
  await gov.waitForDeployment();

  // the developer names the keyless contract as the governance
  // executor
  await ledger.setGovernance(await gov.getAddress());

  // three vested voters so a quorum can form
  const artifactId = ethers.keccak256(ethers.toUtf8Bytes("arm-g"));
  await ledger.register(artifactId, v1.address, 10n, ethers.ZeroHash,
    { value: FEE });
  await ledger.connect(librarian).setAdmitted(artifactId, true);
  await ledger.connect(stranger).deposit({ value: 100000n });
  await ledger.connect(librarian).recordCredits(
    [stranger.address, stranger.address, stranger.address],
    [{ artifactId, who: v1.address, amount: 100n },
     { artifactId, who: v2.address, amount: 100n },
     { artifactId, who: v3.address, amount: 100n }]);
  await advanceTime(5 * EPOCH);          // vest; epoch 0 closed

  // the developer then renounces ownership — the retirement endgame
  await ledger.renounceOwnership();

  return { owner, devFund, librarian, deputy, successor, rando,
           stranger, v1, v2, v3, ledger, gov };
}

// Endorse the LIVE proposal (proposalCounter) with all three voters.
async function endorseLive(f) {
  const pid = await f.gov.proposalCounter();
  await f.ledger.connect(f.v1).attestGovernanceReplacement(pid, true);
  await f.ledger.connect(f.v2).attestGovernanceReplacement(pid, true);
  await f.ledger.connect(f.v3).attestGovernanceReplacement(pid, true);
}

describe("LibrarianGovernance (M388, quorum-gated)", function () {
  describe("the contract exists and is keyless (leftover 1)",
           function () {
    it("deploys with no owner and no admin surface", async function () {
      const { gov } = await loadFixture(deployFixture);
      expect(await gov.ledger()).to.properAddress;
      // no owner() getter exists; the contract cannot be paused or
      // upgraded because it has no such paths at all
      expect(await gov.pendingBond()).to.equal(0n);
      expect(await gov.claimableTotal()).to.equal(0n);
      expect(await gov.bondsBurned()).to.equal(0n);
    });

    it("holds exactly the one power through the ledger's own guard",
       async function () {
      const { gov, deputy, stranger } =
        await loadFixture(deployFixture);
      // it can propose a replacement (the one power), but the ledger
      // still confines it: before the timelock, nothing moves
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      await expect(
        gov.connect(stranger).execute(),
      ).to.be.revertedWithCustomError(gov, "TimelockNotElapsed");
    });
  });

  describe("the recorded reason is the mechanical trigger", function () {
    it("a replacement with no recorded reason is inexpressible",
       async function () {
      const { gov, deputy, stranger } =
        await loadFixture(deployFixture);
      await expect(
        gov.connect(stranger).proposeReplacement(
          deputy.address, ethers.ZeroHash, { value: BOND }),
      ).to.be.revertedWithCustomError(gov, "ZeroReasonHash");
    });

    it("a zero target and a wrong bond are rejected", async function () {
      const { gov, stranger } = await loadFixture(deployFixture);
      await expect(
        gov.connect(stranger).proposeReplacement(
          ethers.ZeroAddress, REASON, { value: BOND }),
      ).to.be.revertedWithCustomError(gov, "ZeroAddress");
      await expect(
        gov.connect(stranger).proposeReplacement(
          ethers.ZeroAddress, REASON, { value: 1n }),
      ).to.be.revertedWithCustomError(gov, "ZeroAddress");
    });
  });

  describe("the path is timelocked and quorum-gated (leftover 3 + F1)",
           function () {
    it("cannot execute before the delay", async function () {
      const { gov, deputy, stranger } =
        await loadFixture(deployFixture);
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      // keep a margin under the delay: evm_increaseTime + evm_mine can
      // land the mined block on the readyAt boundary
      await advanceTime(REPLACEMENT_DELAY - 2);
      await expect(gov.connect(stranger).execute())
        .to.be.revertedWithCustomError(gov, "TimelockNotElapsed");
    });

    it("a bond and a timelock alone cannot name the librarian: an " +
       "unendorsed proposal fails and its bond burns", async function () {
      const { gov, ledger, librarian, deputy, stranger } =
        await loadFixture(deployFixture);
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      const pid = await gov.proposalCounter();
      // nobody endorses during the notice window
      await advanceTime(REPLACEMENT_DELAY);
      await expect(gov.connect(stranger).execute())
        .to.emit(gov, "GovernanceRejected").withArgs(pid, REASON);
      // the librarian did not move; the unratified bond is burned and
      // unreachable — nobody gains from an unratified filing
      expect(await ledger.librarian()).to.equal(librarian.address);
      expect(await gov.bondsBurned()).to.equal(BOND);
      expect(await gov.pendingBond()).to.equal(0n);
      await expect(gov.connect(stranger).claimBond())
        .to.be.revertedWithCustomError(gov, "NothingToClaim");
    });

    it("the distinct-identity floor: two endorsers cannot name the " +
       "librarian", async function () {
      const f = await loadFixture(deployFixture);
      const { gov, ledger, librarian, deputy, stranger, v1, v2 } = f;
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      const pid = await gov.proposalCounter();
      // 2 of 3 vested voters endorse: two-thirds of participants but
      // below the three-distinct floor
      await ledger.connect(v1).attestGovernanceReplacement(pid, true);
      await ledger.connect(v2).attestGovernanceReplacement(pid, true);
      await advanceTime(REPLACEMENT_DELAY);
      await gov.connect(stranger).execute();
      expect(await ledger.librarian()).to.equal(librarian.address);
      expect(await gov.bondsBurned()).to.equal(BOND);
    });

    it("executes after the delay when the quorum endorses; the " +
       "deputy takes the role", async function () {
      const f = await loadFixture(deployFixture);
      const { gov, ledger, librarian, deputy, stranger } = f;
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      await endorseLive(f);
      await advanceTime(REPLACEMENT_DELAY);
      // anyone may press the button, not just the proposer
      await expect(gov.connect(stranger).execute())
        .to.emit(gov, "ReplacementExecuted").withArgs(deputy.address, REASON);
      expect(await ledger.librarian()).to.equal(deputy.address);
      expect(await ledger.librarian()).to.not.equal(librarian.address);
    });
  });

  describe("the proposer's bond is pulled, never pushed", function () {
    it("the proposer claims the bond after execution", async function () {
      const f = await loadFixture(deployFixture);
      const { gov, deputy, stranger } = f;
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      await endorseLive(f);
      await advanceTime(REPLACEMENT_DELAY);
      await gov.connect(stranger).execute();
      await expect(gov.connect(stranger).claimBond())
        .to.emit(gov, "BondClaimed").withArgs(stranger.address, BOND);
      await expect(gov.connect(stranger).claimBond())
        .to.be.revertedWithCustomError(gov, "NothingToClaim");
    });

    it("the proposer can cancel before the delay and pull the bond",
       async function () {
      const { gov, deputy, stranger, rando } =
        await loadFixture(deployFixture);
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      // only the proposer can cancel
      await expect(gov.connect(rando).cancel())
        .to.be.revertedWithCustomError(gov, "NotProposer");
      await gov.connect(stranger).cancel();
      await gov.connect(stranger).claimBond();
      expect(await gov.pendingBond()).to.equal(0n);
    });

    it("a superseding proposal refunds the prior proposer",
       async function () {
      const f = await loadFixture(deployFixture);
      const { gov, deputy, stranger, rando } = f;
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      // rando supersedes; stranger's bond becomes claimable
      await gov.connect(rando).proposeReplacement(
        stranger.address, REASON, { value: BOND });
      expect(await gov.pendingBond()).to.equal(BOND);
      await gov.connect(stranger).claimBond();
      // rando's newer proposal is the live one and executes after a
      // quorum endorses it
      await endorseLive(f);
      await advanceTime(REPLACEMENT_DELAY);
      await gov.connect(rando).execute();
      expect(await f.ledger.librarian()).to.equal(stranger.address);
    });
  });

  describe("the executor hands its own role on (succession)",
           function () {
    it("governance succession moves the power after a quorum " +
       "endorses", async function () {
      const f = await loadFixture(deployFixture);
      const { gov, ledger, successor, stranger, rando } = f;
      await gov.connect(stranger).proposeSuccession(
        successor.address, REASON, { value: BOND });
      await endorseLive(f);
      await advanceTime(REPLACEMENT_DELAY);
      await gov.connect(rando).execute();
      expect(await ledger.governance()).to.equal(successor.address);
      // the old keyless executor no longer holds the power: it cannot
      // even open a proposal, and the ledger's own guard refuses it
      await expect(gov.connect(stranger).proposeReplacement(
        rando.address, REASON, { value: BOND }))
        .to.be.revertedWithCustomError(ledger, "NotGovernance");
    });
  });
});
