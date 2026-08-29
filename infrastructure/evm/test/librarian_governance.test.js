// LibrarianGovernance — M388 (the M382 remainder) suite.
// Covers the three registered closures:
//  1. the governance executor exists as a contract with no human key
//     (no owner, no admin, no upgrade path);
//  2. the deputy's replacement is executed through this keyless
//     executor with a recorded reason (the mechanical trigger);
//  3. the path is timelocked (REPLACEMENT_DELAY), and the proposer's
//     bond is pulled, never pushed.
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");

const FEE = 1000n;
const BOND = ethers.parseEther("1");
const REPLACEMENT_DELAY = 7 * 24 * 3600;

async function advanceTime(seconds) {
  await ethers.provider.send("evm_increaseTime", [seconds]);
  await ethers.provider.send("evm_mine");
}

const REASON = ethers.keccak256(ethers.toUtf8Bytes("recorded-divergence"));

async function deployFixture() {
  const [owner, devFund, librarian, deputy, successor, rando, stranger] =
    await ethers.getSigners();
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
  // executor and then renounces ownership — the retirement endgame
  await ledger.setGovernance(await gov.getAddress());
  await ledger.renounceOwnership();

  return { owner, devFund, librarian, deputy, successor, rando,
           stranger, ledger, gov };
}

describe("LibrarianGovernance (M388)", function () {
  describe("the contract exists and is keyless (leftover 1)",
           function () {
    it("deploys with no owner and no admin surface", async function () {
      const { gov } = await loadFixture(deployFixture);
      expect(await gov.ledger()).to.properAddress;
      // no owner() getter exists; the contract cannot be paused or
      // upgraded because it has no such paths at all
      expect(await gov.pendingBond()).to.equal(0n);
      expect(await gov.claimableTotal()).to.equal(0n);
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

  describe("the path is timelocked (leftover 3)", function () {
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

    it("executes after the delay and the deputy takes the role",
       async function () {
      const { gov, ledger, librarian, deputy, stranger } =
        await loadFixture(deployFixture);
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
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
      const { gov, deputy, stranger } =
        await loadFixture(deployFixture);
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
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
      const { gov, deputy, stranger, rando } =
        await loadFixture(deployFixture);
      await gov.connect(stranger).proposeReplacement(
        deputy.address, REASON, { value: BOND });
      // rando supersedes; stranger's bond becomes claimable
      await gov.connect(rando).proposeReplacement(
        stranger.address, REASON, { value: BOND });
      expect(await gov.pendingBond()).to.equal(BOND);
      await gov.connect(stranger).claimBond();
      // rando's newer proposal is the live one and executes
      await advanceTime(REPLACEMENT_DELAY);
      await gov.connect(rando).execute();
      expect(await gov.ledger()).to.properAddress;
    });
  });

  describe("the executor hands its own role on (succession)",
           function () {
    it("governance succession moves the power after the timelock",
       async function () {
      const { gov, ledger, successor, stranger, rando } =
        await loadFixture(deployFixture);
      await gov.connect(stranger).proposeSuccession(
        successor.address, REASON, { value: BOND });
      await advanceTime(REPLACEMENT_DELAY);
      await gov.connect(rando).execute();
      expect(await ledger.governance()).to.equal(successor.address);
      // the old keyless executor no longer holds the power: a
      // replacement it files can no longer move the librarian
      await gov.connect(stranger).proposeReplacement(
        rando.address, REASON, { value: BOND });
      await advanceTime(REPLACEMENT_DELAY);
      await expect(gov.connect(stranger).execute())
        .to.be.revertedWithCustomError(ledger, "NotOwnerOrGovernance");
    });
  });
});
