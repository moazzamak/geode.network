// CreditLedger — whitepaper-aligned suite (24 Aug 2026).
// Covers: native-ETH deposit split, the unified registration form
// (operator key + payout address + price + sealed claim), admission,
// timelocked price/registration-fee changes, skip-and-emit batches
// (self-payment keys on the PAYOUT address), N=4 linear vesting with
// pull claims, the graded burn ladder, librarian management, and the
// transient reentrancy guard.
const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");
const { loadFixture } = require("@nomicfoundation/hardhat-toolbox/network-helpers");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");

const EPOCH = 7 * 24 * 3600;
const CHANGE_DELAY = 2 * 24 * 3600;
const PRICE_DELAY = 7 * 24 * 3600;
const FEE = 1000n;

async function advanceTime(seconds) {
  await ethers.provider.send("evm_increaseTime", [seconds]);
  await ethers.provider.send("evm_mine");
}

async function deployFixture() {
  const [owner, devFund, payer, contributor, operator, librarian, rando] =
    await ethers.getSigners();
  const Ledger = await ethers.getContractFactory("CreditLedger");
  const ledger = await upgrades.deployProxy(Ledger, [
    devFund.address,
    FEE,
  ]);
  await ledger.waitForDeployment();
  await ledger.setLibrarian(librarian.address);
  const artifactId = ethers.keccak256(ethers.toUtf8Bytes("arm-a"));
  return { owner, devFund, payer, contributor, operator, librarian,
           rando, ledger, artifactId, Ledger };
}

async function registeredFixture() {
  const f = await loadFixture(deployFixture);
  await f.ledger.connect(f.operator).register(
    f.artifactId, f.contributor.address, 10n, ethers.ZeroHash,
    { value: FEE });
  await f.ledger.connect(f.librarian).setAdmitted(f.artifactId, true);
  await f.ledger.connect(f.payer).deposit({ value: 100000n });
  return f;
}

async function creditedFixture() {
  const f = await loadFixture(registeredFixture);
  await f.ledger.connect(f.librarian).recordCredits(
    [f.payer.address],
    [{ artifactId: f.artifactId, who: f.contributor.address,
       amount: 2000n }]);
  return f;
}

describe("CreditLedger", function () {
  describe("initialize", function () {
    it("sets dev fund, registration fee, owner, librarian",
       async function () {
      const { ledger, devFund, owner, librarian } =
        await loadFixture(deployFixture);
      expect(await ledger.devFund()).to.equal(devFund.address);
      expect(await ledger.registrationFee()).to.equal(FEE);
      expect(await ledger.owner()).to.equal(owner.address);
      expect(await ledger.librarian()).to.equal(librarian.address);
      expect(await ledger.epochId()).to.equal(0n);
    });

    it("cannot be initialised twice and rejects a zero dev fund",
       async function () {
      const { ledger, devFund } = await loadFixture(deployFixture);
      await expect(
        ledger.initialize(devFund.address, FEE),
      ).to.be.revertedWithCustomError(ledger, "InvalidInitialization");
      const Ledger = await ethers.getContractFactory("CreditLedger");
      await expect(
        upgrades.deployProxy(Ledger, [ethers.ZeroAddress, FEE]),
      ).to.be.reverted;
    });
  });

  describe("deposit / dev fund", function () {
    it("splits 2.5% to the dev share and 97.5% to attribution",
       async function () {
      const { ledger, payer } = await loadFixture(deployFixture);
      await expect(ledger.connect(payer).deposit({ value: 1000n }))
        .to.emit(ledger, "Deposited")
        .withArgs(payer.address, 1000n, 25n);
      expect(await ledger.attributable()).to.equal(975n);
      expect(await ledger.devFundShare()).to.equal(25n);
      expect(await ledger.ethHeld()).to.equal(1000n);
    });

    it("deposit(0) reverts", async function () {
      const { ledger, payer } = await loadFixture(deployFixture);
      await expect(ledger.connect(payer).deposit({ value: 0n }))
        .to.be.revertedWithCustomError(ledger, "ZeroAmount");
    });

    it("the dev fund pulls its share, then has nothing to claim",
       async function () {
      const { ledger, payer, devFund } = await loadFixture(deployFixture);
      await expect(ledger.claimDevFund())
        .to.be.revertedWithCustomError(ledger, "NothingToClaim");
      await ledger.connect(payer).deposit({ value: 1000n });
      const before = await ethers.provider.getBalance(devFund.address);
      await expect(ledger.connect(payer).claimDevFund())
        .to.emit(ledger, "DevFundClaimed")
        .withArgs(25n);
      expect(await ethers.provider.getBalance(devFund.address))
        .to.equal(before + 25n);
      await expect(ledger.claimDevFund())
        .to.be.revertedWithCustomError(ledger, "NothingToClaim");
    });

    it("claimDevFund reverts when the fund rejects ETH (SendFailed)",
       async function () {
      const [owner] = await ethers.getSigners();
      const Mocks = await ethers.getContractFactory("RejectingReceiver");
      const rejector = await Mocks.deploy();
      await rejector.waitForDeployment();
      const Ledger = await ethers.getContractFactory("CreditLedger");
      const ledger = await upgrades.deployProxy(Ledger, [
        await rejector.getAddress(), FEE,
      ]);
      await ledger.waitForDeployment();
      const payer = await ethers.getSigner(
        (await ethers.getSigners())[2].address);
      await ledger.connect(payer).deposit({ value: 1000n });
      await expect(ledger.claimDevFund())
        .to.be.revertedWithCustomError(ledger, "SendFailed");
      expect(await ledger.devFundShare()).to.equal(25n); // unclaimed
    });
  });

  describe("registration (one form for arms and primitives)",
           function () {
    it("stores operator key, payout address, price, sealed claim",
       async function () {
      const f = await loadFixture(deployFixture);
      const claim = ethers.keccak256(ethers.toUtf8Bytes("sealed"));
      await expect(f.ledger.connect(f.operator).register(
        f.artifactId, f.contributor.address, 10n, claim, { value: FEE }))
        .to.emit(f.ledger, "Registered")
        .withArgs(f.artifactId, f.operator.address,
                  f.contributor.address, 10n, claim);
      const r = await f.ledger.regs(f.artifactId);
      expect(r.operatorKey).to.equal(f.operator.address);
      expect(r.payoutAddress).to.equal(f.contributor.address);
      expect(r.pricePerUnit).to.equal(10n);
      expect(r.sealedClaim).to.equal(claim);
      expect(r.admitted).to.equal(false);
      expect(await f.ledger.devFundShare()).to.equal(FEE);
    });

    it("rejects a zero payout, a wrong fee, and a duplicate",
       async function () {
      const f = await loadFixture(deployFixture);
      await expect(f.ledger.connect(f.operator).register(
        f.artifactId, ethers.ZeroAddress, 10n, ethers.ZeroHash,
        { value: FEE }))
        .to.be.revertedWithCustomError(f.ledger, "ZeroAddress");
      await expect(f.ledger.connect(f.operator).register(
        f.artifactId, f.contributor.address, 10n, ethers.ZeroHash,
        { value: 999n }))
        .to.be.revertedWithCustomError(f.ledger, "FeeMismatch")
        .withArgs(999n, FEE);
      await f.ledger.connect(f.operator).register(
        f.artifactId, f.contributor.address, 10n, ethers.ZeroHash,
        { value: FEE });
      await expect(f.ledger.connect(f.operator).register(
        f.artifactId, f.contributor.address, 11n, ethers.ZeroHash,
        { value: FEE }))
        .to.be.revertedWithCustomError(f.ledger, "AlreadyRegistered");
    });

    it("the librarian files admission; others cannot",
       async function () {
      const f = await loadFixture(registeredFixture);
      await expect(f.ledger.connect(f.rando).setAdmitted(
        f.artifactId, true))
        .to.be.revertedWithCustomError(f.ledger, "NotLibrarian");
      const missing = ethers.keccak256(ethers.toUtf8Bytes("nope"));
      await expect(f.ledger.connect(f.librarian).setAdmitted(
        missing, true))
        .to.be.revertedWithCustomError(f.ledger, "NotRegistered");
      await expect(f.ledger.connect(f.librarian).setAdmitted(
        f.artifactId, true))
        .to.emit(f.ledger, "Admitted")
        .withArgs(f.artifactId, true);
      expect((await f.ledger.regs(f.artifactId)).admitted).to.equal(true);
    });

    it("price changes are operator-keyed and timelocked one epoch",
       async function () {
      const f = await loadFixture(registeredFixture);
      await expect(f.ledger.connect(f.rando).schedulePriceChange(
        f.artifactId, 20n))
        .to.be.revertedWithCustomError(f.ledger, "NotOperator");
      await expect(f.ledger.connect(f.operator).schedulePriceChange(
        f.artifactId, 20n))
        .to.emit(f.ledger, "PriceChangeScheduled")
        .withArgs(f.artifactId, 20n, anyValue);
      await expect(f.ledger.applyPriceChange(f.artifactId))
        .to.be.revertedWithCustomError(f.ledger, "ChangeTooSoon")
        .withArgs(anyValue, anyValue);
      await expect(f.ledger.connect(f.operator).schedulePriceChange(
        f.artifactId, 30n))
        .to.be.revertedWithCustomError(f.ledger, "NoPendingChange");
      await advanceTime(PRICE_DELAY + 1);
      await expect(f.ledger.applyPriceChange(f.artifactId))
        .to.emit(f.ledger, "PriceChanged")
        .withArgs(f.artifactId, 20n);
      expect((await f.ledger.regs(f.artifactId)).pricePerUnit)
        .to.equal(20n);
      await expect(f.ledger.applyPriceChange(f.artifactId))
        .to.be.revertedWithCustomError(f.ledger, "NoPendingChange");
    });

    it("the registration fee is registry-set with a timelock",
       async function () {
      const f = await loadFixture(registeredFixture);
      await expect(f.ledger.connect(f.rando).scheduleRegistrationFee(
        500n))
        .to.be.reverted; // Ownable
      await expect(f.ledger.applyRegistrationFeeChange())
        .to.be.revertedWithCustomError(f.ledger, "NoPendingChange");
      await f.ledger.scheduleRegistrationFee(500n);
      await expect(f.ledger.applyRegistrationFeeChange())
        .to.be.revertedWithCustomError(f.ledger, "ChangeTooSoon")
        .withArgs(anyValue, anyValue);
      await expect(f.ledger.scheduleRegistrationFee(600n))
        .to.be.revertedWithCustomError(f.ledger, "NoPendingChange");
      await advanceTime(CHANGE_DELAY + 1);
      await expect(f.ledger.applyRegistrationFeeChange())
        .to.emit(f.ledger, "RegistrationFeeChanged")
        .withArgs(500n);
      expect(await f.ledger.registrationFee()).to.equal(500n);
      const id2 = ethers.keccak256(ethers.toUtf8Bytes("arm-b"));
      await expect(f.ledger.connect(f.operator).register(
        id2, f.contributor.address, 10n, ethers.ZeroHash, { value: FEE }))
        .to.be.revertedWithCustomError(f.ledger, "FeeMismatch");
    });
  });

  describe("recordCredits (skip-and-emit batches)", function () {
    it("credits the payout address and draws the pool",
       async function () {
      const f = await loadFixture(registeredFixture);
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 2000n }]))
        .to.emit(f.ledger, "Credited")
        .withArgs(f.artifactId, f.contributor.address, 2000n);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(2000n);
      expect(await f.ledger.attributable()).to.equal(97500n - 2000n);
    });

    it("only the librarian records; shape and size are checked",
       async function () {
      const f = await loadFixture(registeredFixture);
      const entries = [{ artifactId: f.artifactId,
                         who: f.contributor.address, amount: 10n }];
      await expect(f.ledger.connect(f.rando).recordCredits(
        [f.payer.address], entries))
        .to.be.revertedWithCustomError(f.ledger, "NotLibrarian");
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [], entries))
        .to.be.revertedWithCustomError(f.ledger, "BatchShapeMismatch");
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [], []))
        .to.be.revertedWithCustomError(f.ledger, "ZeroAmount");
      const big = Array.from({ length: 65 }, () => ({
        artifactId: f.artifactId, who: f.contributor.address,
        amount: 1n,
      }));
      await expect(f.ledger.connect(f.librarian).recordCredits(
        big.map(() => f.payer.address), big))
        .to.be.revertedWithCustomError(f.ledger, "BatchTooLarge")
        .withArgs(65n, 64n);
    });

    it("zero-amount entries are no-ops", async function () {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 0n }]);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(0n);
      expect(await f.ledger.attributable()).to.equal(97500n);
    });

    it("skips credits for an unadmitted registration",
       async function () {
      const f = await loadFixture(deployFixture);
      await f.ledger.connect(f.operator).register(
        f.artifactId, f.contributor.address, 10n, ethers.ZeroHash,
        { value: FEE });
      await f.ledger.connect(f.payer).deposit({ value: 10000n });
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 100n }]))
        .to.emit(f.ledger, "CreditSkipped")
        .withArgs(f.artifactId, f.contributor.address, 100n,
                  "not admitted");
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(0n);
      // admission unblocks the same batch shape
      await f.ledger.connect(f.librarian).setAdmitted(
        f.artifactId, true);
      await f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 100n }]);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(100n);
    });

    it("skips self-payment (payer is the payout address), " +
       "unregistered artifacts, and amounts above the pool — " +
       "never reverting the batch", async function () {
      const f = await loadFixture(registeredFixture);
      // self-payment: the payout address pays itself
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [f.contributor.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 500n }]))
        .to.emit(f.ledger, "CreditSkipped")
        .withArgs(f.artifactId, f.contributor.address, 500n,
                  "self-payment");
      // unregistered artifact
      const missing = ethers.keccak256(ethers.toUtf8Bytes("ghost"));
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: missing, who: f.contributor.address,
           amount: 500n }]))
        .to.emit(f.ledger, "CreditSkipped")
        .withArgs(missing, f.contributor.address, 500n,
                  "unregistered artifact");
      // above the pool: skipped, batch survives, good entries land
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address, f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 100000n },
         { artifactId: f.artifactId, who: f.contributor.address,
           amount: 100n }]))
        .to.emit(f.ledger, "CreditSkipped")
        .withArgs(f.artifactId, f.contributor.address, 100000n,
                  "insufficient pool");
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(100n);
    });

    it("the librarian files a quorum takedown (M294); others cannot, "
       + "and unknown artifacts revert", async function () {
      const f = await loadFixture(registeredFixture);
      await expect(f.ledger.connect(f.rando).setDelisted(
        f.artifactId, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NotLibrarian");
      const missing = ethers.keccak256(ethers.toUtf8Bytes("nope"));
      await expect(f.ledger.connect(f.librarian).setDelisted(
        missing, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NotRegistered");
      const recordHash = ethers.keccak256(
        ethers.toUtf8Bytes("quorum-record"));
      await expect(f.ledger.connect(f.librarian).setDelisted(
        f.artifactId, recordHash))
        .to.emit(f.ledger, "Delisted")
        .withArgs(f.artifactId, recordHash);
      expect((await f.ledger.regs(f.artifactId)).delisted).to.equal(true);
    });

    it("skips credits for a delisted artifact (M294), permanently",
       async function () {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.librarian).setDelisted(
        f.artifactId, ethers.ZeroHash);
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 100n }]))
        .to.emit(f.ledger, "CreditSkipped")
        .withArgs(f.artifactId, f.contributor.address, 100n,
                  "delisted");
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(0n);
    });
  });

  describe("vesting and claims (N=4, pull-only)", function () {
    it("nothing is claimable before the first tranche", async function () {
      const f = await loadFixture(creditedFixture);
      expect(await f.ledger.vestedOf(f.contributor.address)).to.equal(0n);
      await expect(f.ledger.connect(f.contributor).claim())
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("vests linearly: 1/4 after epoch 1, 1/2 after epoch 2, " +
       "full after epoch 4", async function () {
      const f = await loadFixture(creditedFixture);
      await advanceTime(EPOCH + 1);
      expect(await f.ledger.vestedOf(f.contributor.address))
        .to.equal(500n);
      const before = await ethers.provider.getBalance(
        f.contributor.address);
      await expect(f.ledger.connect(f.contributor).claim())
        .to.emit(f.ledger, "Claimed")
        .withArgs(f.contributor.address, 500n);
      // balance grew by the claim minus the claim tx's gas
      expect(await ethers.provider.getBalance(f.contributor.address))
        .to.be.closeTo(before + 500n, 1000000000000000n);
      await advanceTime(EPOCH);
      // claim again: the next tranche
      await f.ledger.connect(f.contributor).claim();
      expect(await f.ledger.claimedOf(f.contributor.address))
        .to.equal(1000n);
      await advanceTime(2 * EPOCH);
      expect(await f.ledger.vestedOf(f.contributor.address))
        .to.equal(2000n);
      await f.ledger.connect(f.contributor).claim();
      expect(await f.ledger.claimedOf(f.contributor.address))
        .to.equal(2000n);
      await expect(f.ledger.connect(f.contributor).claim())
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("rolls multiple epochs at once", async function () {
      const f = await loadFixture(creditedFixture);
      await advanceTime(5 * EPOCH + 1);
      // the view reads wall time without rolling state
      expect(await f.ledger.vestedOf(f.contributor.address))
        .to.equal(2000n);
      await f.ledger.connect(f.contributor).claim();
      expect(await f.ledger.epochId()).to.equal(5n);
      expect(await f.ledger.claimedOf(f.contributor.address))
        .to.equal(2000n);
    });

    it("forced ETH cannot be claimed (solvency counter)",
       async function () {
      const f = await loadFixture(creditedFixture); // 2000 credited
      // a forced donation (no deposit() call) must not become
      // claimable: the ledger never reads raw address(this).balance
      await ethers.provider.send("hardhat_setBalance", [
        await f.ledger.getAddress(), "0xDE0B6B3A7640000"]);
      await advanceTime(5 * EPOCH);
      await f.ledger.connect(f.contributor).claim();
      expect(await f.ledger.claimedOf(f.contributor.address))
        .to.equal(2000n); // only the credited amount, not the gift
      await expect(f.ledger.connect(f.contributor).claim())
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("claim reverts when the beneficiary rejects ETH",
       async function () {
      const [owner] = await ethers.getSigners();
      const Mocks = await ethers.getContractFactory("RejectingReceiver");
      const rejector = await Mocks.deploy();
      await rejector.waitForDeployment();
      const Ledger = await ethers.getContractFactory("CreditLedger");
      const ledger = await upgrades.deployProxy(Ledger, [
        (await ethers.getSigners())[1].address, FEE,
      ]);
      await ledger.waitForDeployment();
      await ledger.setLibrarian(owner.address);
      const operator = (await ethers.getSigners())[4];
      const id = ethers.keccak256(ethers.toUtf8Bytes("arm-c"));
      await ledger.connect(operator).register(
        id, await rejector.getAddress(), 1n, ethers.ZeroHash,
        { value: FEE });
      await ledger.setAdmitted(id, true);
      const payer = (await ethers.getSigners())[2];
      await ledger.connect(payer).deposit({ value: 10000n });
      await ledger.recordCredits(
        [payer.address],
        [{ artifactId: id, who: await rejector.getAddress(),
           amount: 1000n }]);
      await advanceTime(5 * EPOCH);
      await ethers.provider.send("hardhat_impersonateAccount",
                                 [await rejector.getAddress()]);
      await ethers.provider.send("hardhat_setBalance", [
        await rejector.getAddress(), "0xDE0B6B3A7640000"]);
      const claimant = await ethers.getSigner(
        await rejector.getAddress());
      await expect(ledger.connect(claimant).claim())
        .to.be.revertedWithCustomError(ledger, "SendFailed");
      expect(await ledger.claimedOf(await rejector.getAddress()))
        .to.equal(0n);
    });
  });

  describe("slashing (burn, graded, replay-gated)", function () {
    it("burns unvested promise at level 1", async function () {
      const f = await loadFixture(creditedFixture);
      await expect(f.ledger.connect(f.rando).slash(
        f.contributor.address, f.artifactId, 100n, 1, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NotLibrarian");
      await expect(f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 100n, 0, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "InvalidLevel");
      await expect(f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 100n, 4, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "InvalidLevel");
      await expect(f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 0n, 1, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "ZeroAmount");
      await expect(f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 5000n, 1, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
      await expect(f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 100n, 1, ethers.ZeroHash))
        .to.emit(f.ledger, "Burned")
        .withArgs(f.contributor.address, f.artifactId, 100n, 1,
                  ethers.ZeroHash);
      expect(await f.ledger.burnedTotal()).to.equal(100n);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(1900n);
      // level 1 cannot burn what is already vested
      await advanceTime(5 * EPOCH);
      await expect(f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 1900n, 1, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("level 2+ delists the artifact", async function () {
      const f = await loadFixture(creditedFixture);
      await f.ledger.connect(f.librarian).setAdmitted(f.artifactId, true);
      await f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 500n, 2, ethers.ZeroHash);
      expect((await f.ledger.regs(f.artifactId)).admitted)
        .to.equal(false);
      expect(await f.ledger.burnedTotal()).to.equal(500n);
    });

    it("burns unvested first, then whole buckets, then the mature " +
       "balance", async function () {
      const f = await loadFixture(creditedFixture); // 2000 @ epoch 0
      await advanceTime(5 * EPOCH);
      await f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 1000n }]); // rolls to epoch 5; bucket 5
      // level 1 collapses the old bucket into the mature balance and
      // burns unvested parts only
      await f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 500n, 1, ethers.ZeroHash);
      expect(await f.ledger.matureOf(f.contributor.address))
        .to.equal(2000n);
      // level 2 drains the live bucket, then the mature balance
      await f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 2500n, 2, ethers.ZeroHash);
      expect(await f.ledger.burnedTotal()).to.equal(3000n);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(0n);
      expect(await f.ledger.matureOf(f.contributor.address))
        .to.equal(0n);
    });

    it("burns fully-vested but uncollapsed buckets at level 2",
       async function () {
      const f = await loadFixture(creditedFixture); // 2000 @ epoch 0
      await advanceTime(5 * EPOCH);
      await f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 1000n }]); // rolls to epoch 5; bucket 5
      await f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 1500n, 2, ethers.ZeroHash);
      expect(await f.ledger.burnedTotal()).to.equal(1500n);
      expect(await f.ledger.epochCredits(
        f.contributor.address, 5)).to.equal(0n);
      expect(await f.ledger.epochCredits(
        f.contributor.address, 0)).to.equal(1500n);
    });

    it("reverts level 2 when the artifact is unregistered",
       async function () {
      const f = await loadFixture(creditedFixture);
      const ghost = ethers.keccak256(ethers.toUtf8Bytes("ghost"));
      await expect(f.ledger.connect(f.librarian).slash(
        f.contributor.address, ghost, 100n, 2, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NotRegistered");
    });

    it("reverts level 2 when the victim is not the payout address",
       async function () {
      const f = await loadFixture(creditedFixture);
      await expect(f.ledger.connect(f.librarian).slash(
        f.rando.address, f.artifactId, 100n, 2, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "WrongTarget");
      expect((await f.ledger.regs(f.artifactId)).admitted)
        .to.equal(true); // untouched
    });
  });

  describe("slash is propose-and-challenge (M386, G54)", function () {
    const SLASH_BOND = ethers.parseEther("1");

    async function filedFixture() {
      const f = await loadFixture(creditedFixture); // 2000 @ epoch 0
      // a stranger files a level-1 accusation against the contributor
      await f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 500n, 1, ethers.ZeroHash,
        { value: SLASH_BOND });
      return f;
    }

    it("anyone files with a bond and nothing burns until resolution",
       async function () {
      const f = await loadFixture(creditedFixture);
      await expect(f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 500n, 1, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.emit(f.ledger, "SlashFiled")
        .withArgs(0, f.contributor.address, f.artifactId, 500n, 1,
                  ethers.ZeroHash, f.rando.address);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(2000n);
      expect(await f.ledger.burnedTotal()).to.equal(0n);
      expect(await f.ledger.slashBondHeld()).to.equal(SLASH_BOND);
      const filing = await f.ledger.slashFilings(0);
      expect(filing.filer).to.equal(f.rando.address);
      expect(filing.resolvedAt).to.equal(0n);
      expect(filing.challenger).to.equal(ethers.ZeroAddress);
    });

    it("a guilty artifact is slashed with the librarian absent",
       async function () {
      const f = await loadFixture(filedFixture);
      await advanceTime(8 * 24 * 3600);      // past SLASH_WINDOW (7d)
      await expect(f.ledger.connect(f.rando).executeSlash(0))
        .to.emit(f.ledger, "Burned")
        .withArgs(f.contributor.address, f.artifactId, 500n, 1,
                  ethers.ZeroHash);
      expect(await f.ledger.burnedTotal()).to.equal(500n);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(1500n);
      // the filer's bond is pullable; nothing was burned from it
      expect(await f.ledger.slashBondsBurned()).to.equal(0n);
      await f.ledger.connect(f.rando).claimSlashBond(0);
      expect(await f.ledger.slashBondHeld()).to.equal(0n);
    });

    it("an unchallenged filing cannot execute inside the window",
       async function () {
      const f = await loadFixture(filedFixture);
      await expect(f.ledger.connect(f.rando).executeSlash(0))
        .to.be.revertedWithCustomError(f.ledger, "WindowOpen");
    });

    it("a false accusation loses the bond to the quorum's verdict",
       async function () {
      const f = await loadFixture(filedFixture);
      // the accused refutes by replay, inside the window
      await f.ledger.connect(f.contributor).challengeSlash(
        0, { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      // a challenged filing cannot be executed unilaterally
      await expect(f.ledger.connect(f.rando).executeSlash(0))
        .to.be.revertedWithCustomError(f.ledger, "ChallengePending");
      // the quorum's verdict (filed by the librarian) is innocent
      const rec = ethers.keccak256(ethers.toUtf8Bytes("v1"));
      await expect(f.ledger.connect(f.librarian).resolveSlash(0, false, rec))
        .to.emit(f.ledger, "SlashResolved").withArgs(0, false, rec);
      // nothing burned; the false accuser's bond is forfeited
      expect(await f.ledger.burnedTotal()).to.equal(0n);
      expect(await f.ledger.slashBondsBurned()).to.equal(SLASH_BOND);
      expect(await f.ledger.slashBondHeld()).to.equal(SLASH_BOND);
      // the challenger pulls its own bond back
      await f.ledger.connect(f.contributor).claimSlashBond(0);
      expect(await f.ledger.slashBondHeld()).to.equal(0n);
      // the false accuser has nothing to pull
      await expect(f.ledger.connect(f.rando).claimSlashBond(0))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("a true filing survives its own challenge: the challenger's " +
       "bond is burned and the slash still lands", async function () {
      const f = await loadFixture(filedFixture);
      // the guilty party challenges its own true filing
      await f.ledger.connect(f.contributor).challengeSlash(
        0, { value: SLASH_BOND });
      const rec = ethers.keccak256(ethers.toUtf8Bytes("v2"));
      await expect(f.ledger.connect(f.librarian).resolveSlash(0, true, rec))
        .to.emit(f.ledger, "Burned")
        .withArgs(f.contributor.address, f.artifactId, 500n, 1,
                  ethers.ZeroHash);
      expect(await f.ledger.burnedTotal()).to.equal(500n);
      // the challenger (guilty party) lost its stake; the filer (a
      // stranger) keeps the right to pull its own bond
      expect(await f.ledger.slashBondsBurned()).to.equal(SLASH_BOND);
      expect(await f.ledger.slashBondHeld()).to.equal(SLASH_BOND);
      await f.ledger.connect(f.rando).claimSlashBond(0);
      expect(await f.ledger.slashBondHeld()).to.equal(0n);
    });

    it("only the librarian resolves a challenged filing",
       async function () {
      const f = await loadFixture(filedFixture);
      await f.ledger.connect(f.contributor).challengeSlash(
        0, { value: SLASH_BOND });
      await expect(f.ledger.connect(f.rando).resolveSlash(
        0, false, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NotLibrarian");
    });

    it("a wrong bond is refused at filing and at challenge time",
       async function () {
      const f = await loadFixture(creditedFixture);
      await expect(f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 500n, 1, ethers.ZeroHash,
        { value: 1n }))
        .to.be.revertedWithCustomError(f.ledger, "WrongBond");
      await f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 500n, 1, ethers.ZeroHash,
        { value: SLASH_BOND });
      await expect(f.ledger.connect(f.contributor).challengeSlash(
        0, { value: 2n }))
        .to.be.revertedWithCustomError(f.ledger, "WrongBond");
    });

    it("a filing that can never apply is rejected at filing time",
       async function () {
      const f = await loadFixture(creditedFixture);
      await expect(f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 500n, 0, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "InvalidLevel");
      await expect(f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 500n, 4, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "InvalidLevel");
      await expect(f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 0n, 1, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "ZeroAmount");
      await expect(f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 5000n, 1, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
      // level 2 must target the payout address of a registered artifact
      const ghost = ethers.keccak256(ethers.toUtf8Bytes("ghost"));
      await expect(f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, ghost, 100n, 2, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "NotRegistered");
      await expect(f.ledger.connect(f.rando).fileSlash(
        f.rando.address, f.artifactId, 100n, 2, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "WrongTarget");
    });

    it("level 2 delists through the permissionless path",
       async function () {
      const f = await loadFixture(creditedFixture);
      await f.ledger.connect(f.rando).fileSlash(
        f.contributor.address, f.artifactId, 500n, 2, ethers.ZeroHash,
        { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      await f.ledger.connect(f.rando).executeSlash(0);
      expect((await f.ledger.regs(f.artifactId)).admitted).to.equal(false);
      expect(await f.ledger.burnedTotal()).to.equal(500n);
    });

    it("a filing resolves once and a second settlement reverts",
       async function () {
      const f = await loadFixture(filedFixture);
      await advanceTime(8 * 24 * 3600);
      await f.ledger.connect(f.rando).executeSlash(0);
      await expect(f.ledger.connect(f.rando).executeSlash(0))
        .to.be.revertedWithCustomError(f.ledger, "AlreadyResolved");
      const g = await loadFixture(filedFixture);
      await g.ledger.connect(g.contributor).challengeSlash(
        0, { value: SLASH_BOND });
      await g.ledger.connect(g.librarian).resolveSlash(0, false,
                                                       ethers.ZeroHash);
      await expect(g.ledger.connect(g.rando).challengeSlash(
        0, { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(g.ledger, "AlreadyResolved");
    });

    it("a challenge is refused after the window closes",
       async function () {
      const f = await loadFixture(filedFixture);
      await advanceTime(8 * 24 * 3600);
      await expect(f.ledger.connect(f.contributor).challengeSlash(
        0, { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "WindowClosed");
    });

    it("an execution whose balance was drained in the window skips " +
       "and still resolves", async function () {
      const f = await loadFixture(filedFixture);
      // the accused claims everything it can during the window
      await advanceTime(5 * EPOCH);          // fully vested
      await f.ledger.connect(f.contributor).claim();
      await advanceTime(8 * 24 * 3600);
      await expect(f.ledger.connect(f.rando).executeSlash(0))
        .to.emit(f.ledger, "SlashSkipped")
        .withArgs(0, "insufficient balance");
      // the filing resolved; the filer's bond is returnable
      await f.ledger.connect(f.rando).claimSlashBond(0);
      expect(await f.ledger.slashBondHeld()).to.equal(0n);
    });
  });

  describe("registry changes are propose-and-challenge (M387, G54)",
           function () {
    const SLASH_BOND = ethers.parseEther("1");

    async function freshArtifactFixture() {
      // an artifact that is registered but NOT yet admitted
      const f = await loadFixture(registeredFixture);
      f.freshId = ethers.keccak256(ethers.toUtf8Bytes("arm-b"));
      await f.ledger.connect(f.operator).register(
        f.freshId, f.contributor.address, 10n, ethers.ZeroHash,
        { value: FEE });
      return f;
    }

    it("admission follows the published rule with no privileged " +
       "filer", async function () {
      const f = await loadFixture(freshArtifactFixture);
      expect((await f.ledger.regs(f.freshId)).admitted).to.equal(false);
      // a stranger files the admission verdict and executes it; the
      // librarian is nowhere in the call
      await f.ledger.connect(f.rando).fileRegistryChange(
        0, f.freshId, true, 0, ethers.ZeroHash,
        { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      await expect(f.ledger.connect(f.rando).executeRegistryChange(0))
        .to.emit(f.ledger, "Admitted").withArgs(f.freshId, true);
      expect((await f.ledger.regs(f.freshId)).admitted).to.equal(true);
      // the filer's bond is pullable
      await f.ledger.connect(f.rando).claimRegistryBond(0);
      expect(await f.ledger.registryBondHeld()).to.equal(0n);
    });

    it("admission is bidirectional: a stranger may also file a " +
       "de-admission", async function () {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.rando).fileRegistryChange(
        0, f.artifactId, false, 0, ethers.ZeroHash,
        { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      await f.ledger.connect(f.rando).executeRegistryChange(0);
      expect((await f.ledger.regs(f.artifactId)).admitted).to.equal(false);
    });

    it("a ratified quorum verdict (delist) executes itself",
       async function () {
      const f = await loadFixture(registeredFixture);
      const rec = ethers.keccak256(ethers.toUtf8Bytes("quorum-1"));
      await f.ledger.connect(f.rando).fileRegistryChange(
        1, f.artifactId, false, 0, rec, { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      await expect(f.ledger.connect(f.rando).executeRegistryChange(0))
        .to.emit(f.ledger, "Delisted").withArgs(f.artifactId, rec);
      expect((await f.ledger.regs(f.artifactId)).delisted).to.equal(true);
    });

    it("a false delist is refutable and the filer loses the bond",
       async function () {
      const f = await loadFixture(registeredFixture);
      // stranger files a delist; the operator refutes inside the window
      await f.ledger.connect(f.rando).fileRegistryChange(
        1, f.artifactId, false, 0, ethers.ZeroHash,
        { value: SLASH_BOND });
      await f.ledger.connect(f.operator).challengeRegistryChange(
        0, { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      // a challenged filing cannot be executed unilaterally
      await expect(f.ledger.connect(f.rando).executeRegistryChange(0))
        .to.be.revertedWithCustomError(f.ledger, "ChallengePending");
      // the quorum exonerates the artifact
      const rec = ethers.keccak256(ethers.toUtf8Bytes("v1"));
      await expect(f.ledger.connect(f.librarian)
        .resolveRegistryChange(0, false, rec))
        .to.emit(f.ledger, "RegistryChangeResolved")
        .withArgs(0, false, rec);
      expect((await f.ledger.regs(f.artifactId)).delisted).to.equal(false);
      // the false filer's bond is burned; the operator pulls its own
      expect(await f.ledger.registryBondsBurned()).to.equal(SLASH_BOND);
      await f.ledger.connect(f.operator).claimRegistryBond(0);
      expect(await f.ledger.registryBondHeld()).to.equal(0n);
      await expect(f.ledger.connect(f.rando).claimRegistryBond(0))
        .to.be.revertedWithCustomError(f.ledger, "NothingToClaim");
    });

    it("a ministerial freeze executes on its own confirmation " +
       "(unchallenged)", async function () {
      const f = await loadFixture(registeredFixture);
      const ev = ethers.keccak256(ethers.toUtf8Bytes("order-1"));
      await f.ledger.connect(f.rando).fileRegistryChange(
        2, f.artifactId, false, 2, ev, { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      await expect(f.ledger.connect(f.rando).executeRegistryChange(0))
        .to.emit(f.ledger, "Frozen")
        .withArgs(f.artifactId, ev, anyValue);
      expect(await f.ledger.isFrozen(f.artifactId)).to.equal(true);
      // the freeze follows the filing's window, not the filer's whim
      await advanceTime(15 * 24 * 3600);   // past 2 epochs
      expect(await f.ledger.isFrozen(f.artifactId)).to.equal(false);
    });

    it("a false freeze is refutable: the quorum lifts it and the " +
       "filer loses the bond", async function () {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.rando).fileRegistryChange(
        2, f.artifactId, false, 5, ethers.ZeroHash,
        { value: SLASH_BOND });
      await f.ledger.connect(f.operator).challengeRegistryChange(
        0, { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      await f.ledger.connect(f.librarian).resolveRegistryChange(
        0, false, ethers.keccak256(ethers.toUtf8Bytes("v2")));
      expect(await f.ledger.isFrozen(f.artifactId)).to.equal(false);
      expect(await f.ledger.registryBondsBurned()).to.equal(SLASH_BOND);
    });

    it("a freeze never shortens an existing longer freeze",
       async function () {
      const f = await loadFixture(registeredFixture);
      // the librarian files a long ministerial freeze (10 epochs)
      await f.ledger.connect(f.librarian).freezeArtifact(
        f.artifactId, ethers.keccak256(ethers.toUtf8Bytes("long")), 10);
      // a stranger files a 1-epoch freeze and it goes unchallenged
      await f.ledger.connect(f.rando).fileRegistryChange(
        2, f.artifactId, false, 1, ethers.ZeroHash,
        { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      await f.ledger.connect(f.rando).executeRegistryChange(0);
      // the longer window still governs
      await advanceTime(2 * 24 * 3600);    // past the short freeze
      expect(await f.ledger.isFrozen(f.artifactId)).to.equal(true);
    });

    it("a wrong kind, an unregistered artifact, and a zero-window " +
       "freeze are rejected at filing time", async function () {
      const f = await loadFixture(registeredFixture);
      await expect(f.ledger.connect(f.rando).fileRegistryChange(
        3, f.artifactId, false, 0, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "InvalidKind");
      const ghost = ethers.keccak256(ethers.toUtf8Bytes("ghost"));
      await expect(f.ledger.connect(f.rando).fileRegistryChange(
        1, ghost, false, 0, ethers.ZeroHash, { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "NotRegistered");
      await expect(f.ledger.connect(f.rando).fileRegistryChange(
        2, f.artifactId, false, 0, ethers.ZeroHash,
        { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "ZeroAmount");
    });

    it("a wrong bond is refused at filing and challenge time",
       async function () {
      const f = await loadFixture(registeredFixture);
      await expect(f.ledger.connect(f.rando).fileRegistryChange(
        1, f.artifactId, false, 0, ethers.ZeroHash, { value: 1n }))
        .to.be.revertedWithCustomError(f.ledger, "WrongBond");
      await f.ledger.connect(f.rando).fileRegistryChange(
        1, f.artifactId, false, 0, ethers.ZeroHash,
        { value: SLASH_BOND });
      await expect(f.ledger.connect(f.operator).challengeRegistryChange(
        0, { value: 2n }))
        .to.be.revertedWithCustomError(f.ledger, "WrongBond");
    });

    it("only the librarian resolves a challenged registry filing",
       async function () {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.rando).fileRegistryChange(
        1, f.artifactId, false, 0, ethers.ZeroHash,
        { value: SLASH_BOND });
      await f.ledger.connect(f.operator).challengeRegistryChange(
        0, { value: SLASH_BOND });
      await expect(f.ledger.connect(f.rando).resolveRegistryChange(
        0, false, ethers.ZeroHash))
        .to.be.revertedWithCustomError(f.ledger, "NotLibrarian");
    });

    it("an unchallenged filing cannot execute inside the window and " +
       "a challenge is refused after it closes", async function () {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.rando).fileRegistryChange(
        1, f.artifactId, false, 0, ethers.ZeroHash,
        { value: SLASH_BOND });
      await expect(f.ledger.connect(f.rando).executeRegistryChange(0))
        .to.be.revertedWithCustomError(f.ledger, "WindowOpen");
      await advanceTime(8 * 24 * 3600);
      await expect(f.ledger.connect(f.operator).challengeRegistryChange(
        0, { value: SLASH_BOND }))
        .to.be.revertedWithCustomError(f.ledger, "WindowClosed");
    });

    it("a filing resolves once and a second settlement reverts",
       async function () {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.rando).fileRegistryChange(
        1, f.artifactId, false, 0, ethers.ZeroHash,
        { value: SLASH_BOND });
      await advanceTime(8 * 24 * 3600);
      await f.ledger.connect(f.rando).executeRegistryChange(0);
      await expect(f.ledger.connect(f.rando).executeRegistryChange(0))
        .to.be.revertedWithCustomError(f.ledger, "AlreadyResolved");
    });
  });

  describe("librarian management", function () {
    it("owner sets and renounces the librarian", async function () {
      const { ledger, librarian, rando, owner } =
        await loadFixture(deployFixture);
      await expect(ledger.connect(rando).setLibrarian(rando.address))
        .to.be.reverted; // Ownable
      await expect(ledger.setLibrarian(rando.address))
        .to.emit(ledger, "LibrarianChanged")
        .withArgs(rando.address);
      expect(await ledger.librarian()).to.equal(rando.address);
      await expect(ledger.connect(owner).renounceLibrarian())
        .to.emit(ledger, "LibrarianChanged")
        .withArgs(ethers.ZeroAddress);
      expect(await ledger.librarian()).to.equal(ethers.ZeroAddress);
      await expect(ledger.connect(rando).renounceLibrarian())
        .to.be.reverted; // Ownable
    });
  });

  describe("librarian replacement survives retirement (M382, G53)",
           function () {
    it("reproduces the freeze: no governance, renounced owner, " +
       "librarian locked forever", async function () {
      // This is the defect G53 named. The registered endgame is that
      // the developer renounces ownership; setLibrarian was
      // onlyOwner; so the two-thirds earned-weight replacement vote
      // decided something with no execution path.
      const { ledger, rando } = await loadFixture(deployFixture);
      await ledger.renounceOwnership();
      expect(await ledger.governance()).to.equal(ethers.ZeroAddress);
      await expect(ledger.setLibrarian(rando.address))
        .to.be.revertedWithCustomError(ledger, "NotOwnerOrGovernance");
    });

    it("governance replaces the librarian after the owner is gone",
       async function () {
      const { ledger, owner, librarian, rando } =
        await loadFixture(deployFixture);
      await expect(ledger.setGovernance(rando.address))
        .to.emit(ledger, "GovernanceChanged").withArgs(rando.address);
      await ledger.setLibrarian(librarian.address);
      await ledger.renounceOwnership();

      // the developer's path is closed
      await expect(ledger.connect(owner).setLibrarian(owner.address))
        .to.be.revertedWithCustomError(ledger, "NotOwnerOrGovernance");
      // the replacement path is open: the deputy takes the role
      await expect(ledger.connect(rando).setLibrarian(rando.address))
        .to.emit(ledger, "LibrarianChanged").withArgs(rando.address);
      expect(await ledger.librarian()).to.equal(rando.address);
    });

    it("naming governance is a bootstrap act, owner-only",
       async function () {
      const { ledger, rando } = await loadFixture(deployFixture);
      await expect(ledger.connect(rando).setGovernance(rando.address))
        .to.be.reverted; // Ownable
    });

    it("governance hands itself on, so the role survives succession",
       async function () {
      const { ledger, contributor, rando } =
        await loadFixture(deployFixture);
      await ledger.setGovernance(rando.address);
      await ledger.renounceOwnership();
      await expect(ledger.transferGovernance(contributor.address))
        .to.be.revertedWithCustomError(ledger, "NotGovernance");
      await ledger.connect(rando).transferGovernance(
        contributor.address);
      expect(await ledger.governance()).to.equal(contributor.address);
      // the old executor is out, the new one is in
      await expect(ledger.connect(rando).setLibrarian(rando.address))
        .to.be.revertedWithCustomError(ledger, "NotOwnerOrGovernance");
      await ledger.connect(contributor).setLibrarian(rando.address);
      expect(await ledger.librarian()).to.equal(rando.address);
    });

    it("governance holds this one power and no other admin path",
       async function () {
      const { ledger, rando } = await loadFixture(deployFixture);
      await ledger.setGovernance(rando.address);
      await ledger.renounceOwnership();
      // it can replace the librarian ...
      await ledger.connect(rando).setLibrarian(rando.address);
      // ... and nothing else that the owner could do
      await expect(ledger.connect(rando).pause()).to.be.reverted;
      await expect(ledger.connect(rando).scheduleDevFundChange(
        rando.address)).to.be.reverted;
      await expect(ledger.connect(rando).scheduleRegistrationFee(1n))
        .to.be.reverted;
      await expect(ledger.connect(rando).setGovernance(rando.address))
        .to.be.reverted;
    });
  });

  describe("a freeze expires without anyone acting (M384, G54)",
           function () {
    it("an absent librarian cannot extend a freeze past its own " +
       "timestamp", async function () {
      // G54 listed liftFreeze as a librarian liveness risk. It is
      // not one for EXPIRY: isFrozen reads `frozenUntil >
      // block.timestamp`, so the window closes on its own clock.
      // Checked mechanically rather than repaired on the strength of
      // the description.
      const f = await loadFixture(registeredFixture);
      const evidence = ethers.keccak256(ethers.toUtf8Bytes("order-1"));
      await f.ledger.connect(f.librarian).freezeArtifact(
        f.artifactId, evidence, 1n);
      expect(await f.ledger.isFrozen(f.artifactId)).to.equal(true);

      // credits are gated while frozen
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 500n }]))
        .to.emit(f.ledger, "CreditSkipped")
        .withArgs(f.artifactId, f.contributor.address, 500n,
                  "frozen artifact (M323)");

      // the librarian now vanishes for good. Nobody calls liftFreeze.
      await advanceTime(31 * 24 * 60 * 60);
      expect(await f.ledger.isFrozen(f.artifactId)).to.equal(false);
      expect((await f.ledger.regs(f.artifactId)).frozenUntil)
        .to.be.greaterThan(0n);   // never cleared, and it does not matter
    });

    it("the only reader of the freeze is isFrozen, so a stale " +
       "frozenUntil blocks nothing", async function () {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.librarian).freezeArtifact(
        f.artifactId, ethers.keccak256(ethers.toUtf8Bytes("o2")), 1n);
      await advanceTime(31 * 24 * 60 * 60);
      // credits flow again with liftFreeze never called
      await f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 500n }]);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.be.greaterThan(0n);
    });
  });

  describe("attribution is a pull against a published root (M385, G54)",
           function () {
    // Minimal sorted-pair Merkle tree matching OZ MerkleProof.
    const leafOf = (epoch, artifactId, who, cumulative) =>
      ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(
        ["bytes32"],
        [ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(
          ["uint256", "bytes32", "address", "uint256"],
          [epoch, artifactId, who, cumulative]))]));

    const hashPair = (a, b) => {
      const [x, y] = BigInt(a) < BigInt(b) ? [a, b] : [b, a];
      return ethers.keccak256(ethers.concat([x, y]));
    };

    function buildTree(leaves) {
      let level = [...leaves];
      const levels = [level];
      while (level.length > 1) {
        const next = [];
        for (let i = 0; i < level.length; i += 2) {
          next.push(i + 1 < level.length
            ? hashPair(level[i], level[i + 1]) : level[i]);
        }
        level = next;
        levels.push(level);
      }
      return { root: level[0], levels };
    }

    function proofFor(levels, index) {
      const proof = [];
      let idx = index;
      for (let d = 0; d < levels.length - 1; d++) {
        const sib = idx ^ 1;
        if (sib < levels[d].length) proof.push(levels[d][sib]);
        idx = Math.floor(idx / 2);
      }
      return proof;
    }

    async function rootFixture() {
      const f = await loadFixture(registeredFixture);
      await f.ledger.connect(f.payer).deposit({ value: ethers.parseEther("1") });
      // close epoch 0 so a root may be posted for it
      await advanceTime(31 * 24 * 60 * 60);
      await f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 1n }]);      // rolls the epoch
      const leaves = [
        leafOf(0, f.artifactId, f.contributor.address, 5_000n),
        leafOf(0, f.artifactId, f.operator.address, 7_000n),
        leafOf(0, f.artifactId, f.rando.address, 3_000n),
      ];
      const tree = buildTree(leaves);
      await f.ledger.connect(f.librarian)
        .postAttributionRoot(0, tree.root);
      return { ...f, tree, leaves };
    }

    it("a payee is paid with the librarian absent from the payment " +
       "step", async function () {
      const f = await loadFixture(rootFixture);
      const before = await f.ledger.creditsOf(f.contributor.address);
      // the contributor draws its own credit; the librarian is not
      // the sender and is never consulted again
      await expect(f.ledger.connect(f.contributor).claimAttribution(
        0, f.artifactId, f.contributor.address, 5_000n,
        proofFor(f.tree.levels, 0)))
        .to.emit(f.ledger, "AttributionClaimed")
        .withArgs(0, f.artifactId, f.contributor.address, 5_000n);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(before + 5_000n);
    });

    it("no payee depends on being pushed to: a stranger may deliver " +
       "the proof and the credit still lands on the leaf's owner",
       async function () {
      const f = await loadFixture(rootFixture);
      await f.ledger.connect(f.rando).claimAttribution(
        0, f.artifactId, f.operator.address, 7_000n,
        proofFor(f.tree.levels, 1));
      expect(await f.ledger.creditsOf(f.operator.address))
        .to.equal(7_000n);
      expect(await f.ledger.creditsOf(f.rando.address)).to.equal(0n);
    });

    it("the librarian cannot pay one payee and strand the rest: the " +
       "root is write-once", async function () {
      const f = await loadFixture(rootFixture);
      const other = buildTree([leafOf(0, f.artifactId,
                                      f.contributor.address, 99_000n)]);
      await expect(f.ledger.connect(f.librarian)
        .postAttributionRoot(0, other.root))
        .to.be.revertedWithCustomError(f.ledger, "RootAlreadyPosted");
      // every leaf of the ORIGINAL tree is still drawable
      for (const [i, who] of [[0, f.contributor], [1, f.operator],
                              [2, f.rando]]) {
        await f.ledger.connect(f.rando).claimAttribution(
          0, f.artifactId, who.address, [5_000n, 7_000n, 3_000n][i],
          proofFor(f.tree.levels, i));
      }
      expect(await f.ledger.creditsOf(f.rando.address)).to.equal(3_000n);
    });

    it("an invented amount is refused", async function () {
      const f = await loadFixture(rootFixture);
      await expect(f.ledger.connect(f.contributor).claimAttribution(
        0, f.artifactId, f.contributor.address, 500_000n,
        proofFor(f.tree.levels, 0)))
        .to.be.revertedWithCustomError(f.ledger, "BadProof");
    });

    it("a claim is idempotent: cumulative leaves cannot be drawn " +
       "twice", async function () {
      const f = await loadFixture(rootFixture);
      const proof = proofFor(f.tree.levels, 0);
      await f.ledger.connect(f.contributor).claimAttribution(
        0, f.artifactId, f.contributor.address, 5_000n, proof);
      const after = await f.ledger.creditsOf(f.contributor.address);
      await f.ledger.connect(f.contributor).claimAttribution(
        0, f.artifactId, f.contributor.address, 5_000n, proof);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(after);
    });

    it("a withheld batch is provable: no root means no payment, and " +
       "the absence is on chain", async function () {
      const f = await loadFixture(registeredFixture);
      expect(await f.ledger.attributionRoot(0))
        .to.equal(ethers.ZeroHash);
      await expect(f.ledger.connect(f.contributor).claimAttribution(
        0, f.artifactId, f.contributor.address, 5_000n, []))
        .to.be.revertedWithCustomError(f.ledger, "NoAttributionRoot");
    });

    it("only the librarian posts a root, and only for a closed epoch",
       async function () {
      const f = await loadFixture(registeredFixture);
      const root = ethers.keccak256(ethers.toUtf8Bytes("r"));
      await expect(f.ledger.connect(f.rando).postAttributionRoot(0, root))
        .to.be.revertedWithCustomError(f.ledger, "NotLibrarian");
      await expect(f.ledger.connect(f.librarian)
        .postAttributionRoot(0, root))
        .to.be.revertedWithCustomError(f.ledger, "EpochNotClosed");
    });

    it("the root cannot buy an artifact past delisting",
       async function () {
      const f = await loadFixture(rootFixture);
      await f.ledger.connect(f.librarian).setDelisted(
        f.artifactId, ethers.keccak256(ethers.toUtf8Bytes("q")));
      await expect(f.ledger.connect(f.contributor).claimAttribution(
        0, f.artifactId, f.contributor.address, 5_000n,
        proofFor(f.tree.levels, 0)))
        .to.emit(f.ledger, "CreditSkipped")
        .withArgs(f.artifactId, f.contributor.address, 5_000n,
                  "delisted");
      // unchanged but for the 1 wei the fixture used to roll the epoch
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(1n);
    });
  });

  describe("dev fund timelock", function () {
    it("schedules and applies the dev fund change", async function () {
      const { ledger, rando, devFund } = await loadFixture(deployFixture);
      await expect(ledger.connect(rando).scheduleDevFundChange(
        rando.address))
        .to.be.reverted; // Ownable
      await expect(ledger.scheduleDevFundChange(ethers.ZeroAddress))
        .to.be.revertedWithCustomError(ledger, "ZeroAddress");
      await ledger.scheduleDevFundChange(rando.address);
      await expect(ledger.applyDevFundChange())
        .to.be.revertedWithCustomError(ledger, "ChangeTooSoon")
        .withArgs(anyValue, anyValue);
      await expect(ledger.scheduleDevFundChange(devFund.address))
        .to.be.revertedWithCustomError(ledger, "NoPendingChange");
      await advanceTime(CHANGE_DELAY + 1);
      await expect(ledger.applyDevFundChange())
        .to.emit(ledger, "DevFundChanged")
        .withArgs(rando.address);
      expect(await ledger.devFund()).to.equal(rando.address);
      await expect(ledger.applyDevFundChange())
        .to.be.revertedWithCustomError(ledger, "NoPendingChange");
    });
  });

  describe("pausing", function () {
    it("pause blocks deposit, register, recordCredits, claim, slash; " +
       "unpause restores", async function () {
      const f = await loadFixture(creditedFixture);
      await expect(f.ledger.connect(f.rando).pause())
        .to.be.reverted; // Ownable
      await f.ledger.pause();
      await expect(f.ledger.connect(f.payer).deposit({ value: 100n }))
        .to.be.reverted; // paused
      await expect(f.ledger.connect(f.operator).register(
        ethers.keccak256(ethers.toUtf8Bytes("z")),
        f.contributor.address, 1n, ethers.ZeroHash, { value: FEE }))
        .to.be.reverted; // paused
      await expect(f.ledger.connect(f.librarian).recordCredits(
        [f.payer.address],
        [{ artifactId: f.artifactId, who: f.contributor.address,
           amount: 1n }]))
        .to.be.reverted; // paused
      await expect(f.ledger.connect(f.contributor).claim())
        .to.be.reverted; // paused
      await expect(f.ledger.connect(f.librarian).slash(
        f.contributor.address, f.artifactId, 1n, 1, ethers.ZeroHash))
        .to.be.reverted; // paused
      await f.ledger.unpause();
      await expect(f.ledger.connect(f.rando).unpause())
        .to.be.reverted; // Ownable
      await f.ledger.connect(f.payer).deposit({ value: 100n });
    });
  });

  describe("reentrancy", function () {
    it("a fund that re-enters claimDevFund is stopped by the guard",
       async function () {
      const [owner] = await ethers.getSigners();
      const Mocks = await ethers.getContractFactory("ReentrantReceiver");
      const attacker = await Mocks.deploy();
      await attacker.waitForDeployment();
      const Ledger = await ethers.getContractFactory("CreditLedger");
      const ledger = await upgrades.deployProxy(Ledger, [
        await attacker.getAddress(), FEE,
      ]);
      await ledger.waitForDeployment();
      await ledger.setLibrarian(owner.address);
      const payer = (await ethers.getSigners())[2];
      await ledger.connect(payer).deposit({ value: 1000n });
      await attacker.setTarget(await ledger.getAddress());
      await expect(ledger.claimDevFund()).to.be.reverted;
      expect(await ledger.devFundShare()).to.equal(25n); // unclaimed
    });

    it("a beneficiary that re-enters claim is stopped by the guard",
       async function () {
      const [owner] = await ethers.getSigners();
      const Mocks = await ethers.getContractFactory("ReentrantClaimer");
      const claimant = await Mocks.deploy();
      await claimant.waitForDeployment();
      const Ledger = await ethers.getContractFactory("CreditLedger");
      const ledger = await upgrades.deployProxy(Ledger, [
        (await ethers.getSigners())[1].address, FEE,
      ]);
      await ledger.waitForDeployment();
      await ledger.setLibrarian(owner.address);
      const operator = (await ethers.getSigners())[4];
      const id = ethers.keccak256(ethers.toUtf8Bytes("arm-r"));
      await ledger.connect(operator).register(
        id, await claimant.getAddress(), 1n, ethers.ZeroHash,
        { value: FEE });
      await ledger.setAdmitted(id, true);
      const payer = (await ethers.getSigners())[2];
      await ledger.connect(payer).deposit({ value: 10000n });
      await ledger.recordCredits(
        [payer.address],
        [{ artifactId: id, who: await claimant.getAddress(),
           amount: 1000n }]);
      await claimant.setTarget(await ledger.getAddress());
      await advanceTime(5 * EPOCH);
      await ethers.provider.send("hardhat_impersonateAccount",
                                 [await claimant.getAddress()]);
      await ethers.provider.send("hardhat_setBalance", [
        await claimant.getAddress(), "0xDE0B6B3A7640000"]);
      const signer = await ethers.getSigner(
        await claimant.getAddress());
      await expect(ledger.connect(signer).claim()).to.be.reverted;
      expect(await ledger.claimedOf(await claimant.getAddress()))
        .to.equal(0n);
    });
  });

  describe("upgrades", function () {
    it("owner can upgrade the implementation", async function () {
      const { ledger, Ledger } = await loadFixture(deployFixture);
      await expect(upgrades.upgradeProxy(
        await ledger.getAddress(), Ledger)).to.not.be.reverted;
    });

    it("upgrade rehearsal preserves state", async function () {
      const f = await loadFixture(creditedFixture); // reg + credits
      const impl = await f.Ledger.deploy();
      await impl.waitForDeployment();
      await f.ledger.upgradeToAndCall(await impl.getAddress(), "0x");
      const r = await f.ledger.regs(f.artifactId);
      expect(r.payoutAddress).to.equal(f.contributor.address);
      expect(await f.ledger.creditsOf(f.contributor.address))
        .to.equal(2000n);
      expect(await f.ledger.attributable()).to.equal(97500n - 2000n);
    });

    it("a non-owner cannot authorize an upgrade", async function () {
      const { ledger, Ledger, rando } = await loadFixture(deployFixture);
      const impl = await Ledger.deploy();
      await impl.waitForDeployment();
      await expect(ledger.connect(rando).upgradeToAndCall(
        await impl.getAddress(), "0x"))
        .to.be.reverted; // Ownable inside _authorizeUpgrade
    });
  });

  describe("admin release", function () {
    it("a renounced owner closes every admin path", async function () {
      const { ledger, Ledger, rando } = await loadFixture(deployFixture);
      await ledger.renounceOwnership();
      expect(await ledger.owner()).to.equal(ethers.ZeroAddress);
      await expect(ledger.setLibrarian(rando.address))
        .to.be.reverted; // Ownable
      await expect(ledger.renounceLibrarian()).to.be.reverted;
      await expect(ledger.pause()).to.be.reverted;
      const impl = await Ledger.deploy();
      await impl.waitForDeployment();
      await expect(ledger.upgradeToAndCall(
        await impl.getAddress(), "0x")).to.be.reverted;
    });
  });
});
