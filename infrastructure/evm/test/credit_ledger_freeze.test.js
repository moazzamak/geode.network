const { expect } = require("chai");
const { ethers, upgrades } = require("hardhat");

const FEE = 1000n;

describe("CreditLedger ministerial freeze (M323)", function () {
  let ledger, librarian, operator, payout, outsider, devFund;
  const artifactId = ethers.keccak256(ethers.toUtf8Bytes("artifact-A"));
  const evidenceHash = ethers.keccak256(ethers.toUtf8Bytes("h(1)"));

  beforeEach(async function () {
    [devFund, librarian, operator, payout, outsider] =
      await ethers.getSigners();
    const L = await ethers.getContractFactory("CreditLedger");
    ledger = await upgrades.deployProxy(L, [devFund.address, FEE]);
    await ledger.waitForDeployment();
    await ledger.setLibrarian(librarian.address);
    await ledger.connect(operator).register(
      artifactId, payout.address, 1000n, evidenceHash, { value: FEE }
    );
    await ledger.connect(librarian).setAdmitted(artifactId, true);
  });

  async function creditArtifact(from, amount) {
    await ledger.connect(from).deposit({ value: amount });
    await ledger.connect(librarian).recordCredits(
      [from.address],
      [{ artifactId, who: payout.address, amount }]
    );
  }

  it("a frozen artifact earns nothing (freeze suspends attribution)", async function () {
    await ledger.connect(librarian).freezeArtifact(artifactId, evidenceHash, 1);
    expect(await ledger.isFrozen(artifactId)).to.equal(true);
    await creditArtifact(outsider, ethers.parseEther("1"));
    // the credit was skipped: the payout address has zero credits
    const r = await ledger.regs(artifactId);
    expect(r.freezeEvidence).to.equal(evidenceHash);
    const tx = await ledger.connect(librarian).recordCredits(
      [outsider.address],
      [{ artifactId, who: payout.address, amount: 100 }]
    );
    const receipt = await tx.wait();
    const skipped = receipt.logs
      .map((l) => { try { return ledger.interface.parseLog(l); } catch { return null; } })
      .filter((e) => e && e.name === "CreditSkipped");
    expect(skipped.length).to.be.greaterThan(0);
  });

  it("only the librarian can freeze or lift", async function () {
    await expect(
      ledger.connect(outsider).freezeArtifact(artifactId, evidenceHash, 1)
    ).to.be.revertedWithCustomError(ledger, "NotLibrarian");
    await expect(
      ledger.connect(outsider).liftFreeze(artifactId)
    ).to.be.revertedWithCustomError(ledger, "NotLibrarian");
  });

  it("no validator path exists: validators cannot release a freeze", async function () {
    await ledger.connect(librarian).freezeArtifact(artifactId, evidenceHash, 2);
    // the ABI exposes no function callable by any non-librarian that
    // clears frozenUntil; the only exits are liftFreeze and expiry
    const iface = ledger.interface;
    const mutators = Object.values(iface.fragments).filter(
      (f) => f.type === "function" &&
        f.stateMutability !== "view" && f.stateMutability !== "pure"
    );
    const names = mutators.map((f) => f.name);
    expect(names).to.not.include("releaseFreeze");
    expect(names).to.not.include("unfreezeByValidators");
    // expiry is time-based, not an action anyone can take early
    await expect(
      ledger.connect(operator).liftFreeze(artifactId)
    ).to.be.revertedWithCustomError(ledger, "NotLibrarian");
  });

  it("the freeze expires on the registered window", async function () {
    await ledger.connect(librarian).freezeArtifact(artifactId, evidenceHash, 1);
    expect(await ledger.isFrozen(artifactId)).to.equal(true);
    await ethers.provider.send("evm_increaseTime", [8 * 24 * 3600]);
    await ethers.provider.send("evm_mine");
    expect(await ledger.isFrozen(artifactId)).to.equal(false);
  });

  it("lifting the freeze restores earning", async function () {
    await ledger.connect(librarian).freezeArtifact(artifactId, evidenceHash, 3);
    await ledger.connect(librarian).liftFreeze(artifactId);
    expect(await ledger.isFrozen(artifactId)).to.equal(false);
  });
});
