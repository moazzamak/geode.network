const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("GovernanceFloors (M314/M315/M327/M328 EVM mirror)", function () {
  let floors, gov, outsider;

  beforeEach(async function () {
    [gov, outsider] = await ethers.getSigners();
    const F = await ethers.getContractFactory("GovernanceFloors");
    floors = await F.deploy(gov.address);
    await floors.waitForDeployment();
  });

  it("lowers are inexpressible: a proposal at or below current reverts", async function () {
    await expect(
      floors.connect(gov).proposeRaise(0, 50) // current is 50
    ).to.be.revertedWithCustomError(floors, "RaiseBelowCurrent");
    await expect(
      floors.connect(gov).proposeRaise(1, 3) // below the floor 4
    ).to.be.revertedWithCustomError(floors, "RaiseBelowCurrent");
  });

  it("only governance may propose", async function () {
    await expect(
      floors.connect(outsider).proposeRaise(0, 60)
    ).to.be.revertedWithCustomError(floors, "NotGovernance");
  });

  it("a raise applies only after the timelock", async function () {
    await floors.connect(gov).proposeRaise(0, 60);
    await expect(
      floors.connect(gov).executeRaise()
    ).to.be.revertedWithCustomError(floors, "TimelockNotElapsed");
    await ethers.provider.send("evm_increaseTime", [7 * 24 * 3600]);
    await ethers.provider.send("evm_mine");
    await floors.connect(gov).executeRaise();
    expect(await floors.probeRateBps()).to.equal(60);
  });

  it("the voting cap, quorum, and diversity constants are charter-fixed", async function () {
    expect(await floors.VOTING_CAP_BPS()).to.equal(2000);
    expect(await floors.QUORUM_BPS()).to.equal(6667);
    expect(await floors.DIVERSITY_MIN()).to.equal(3);
    expect(await floors.UNOPENED_FAIL_CLOSED_BPS()).to.equal(3334);
    // no setter exists: the ABI has no function that mutates them
    const iface = floors.interface;
    const mutators = Object.values(iface.fragments).filter(
      (f) => f.type === "function" && f.stateMutability !== "view" && f.stateMutability !== "pure"
    );
    const names = mutators.map((f) => f.name);
    expect(names).to.not.include("setVotingCap");
    expect(names).to.not.include("setQuorum");
  });

  it("diversityFloor matches the registered d = max(3, ceil(0.2*n))", async function () {
    expect(await floors.diversityFloor(1)).to.equal(3);
    expect(await floors.diversityFloor(10)).to.equal(3);
    expect(await floors.diversityFloor(16)).to.equal(4);
    expect(await floors.diversityFloor(100)).to.equal(20);
  });

  it("floor defaults mirror the registered values", async function () {
    expect(await floors.probeRateBps()).to.equal(50);
    expect(await floors.vestingEpochsFloor()).to.equal(4);
    expect(await floors.admissionSampleFloor()).to.equal(3);
    expect(await floors.referenceSampleFloor()).to.equal(2);
    expect(await floors.auditFractionBps()).to.equal(100);
    expect(await floors.takedownMinResponders()).to.equal(3);
  });
});
