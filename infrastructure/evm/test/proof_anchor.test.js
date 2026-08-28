// M214 — ProofAnchor contract tests: anchoring, retrieval,
// append-only no-op, and hash distinctness.
const { expect } = require("chai");
const { anyValue } = require(
  "@nomicfoundation/hardhat-chai-matchers/withArgs");
const { ethers } = require("hardhat");

async function deployAnchor() {
  const A = await ethers.getContractFactory("ProofAnchor");
  const a = await A.deploy();
  await a.waitForDeployment();
  return a;
}

describe("ProofAnchor (M214)", function () {
  it("anchors a proof and records its block", async function () {
    const a = await deployAnchor();
    const proof = "0x" + "ab".repeat(1024);
    await a.anchor(proof);
    const h = ethers.keccak256(proof);
    expect(await a.anchoredAt(h)).to.be.greaterThan(0n);
  });

  it("re-anchoring the same hash is a no-op", async function () {
    const a = await deployAnchor();
    const proof = "0x" + "cd".repeat(1024);
    await a.anchor(proof);
    const h = ethers.keccak256(proof);
    const first = await a.anchoredAt(h);
    await a.anchor(proof);
    expect(await a.anchoredAt(h)).to.equal(first);
  });

  it("distinct proofs get distinct hashes and records", async function () {
    const a = await deployAnchor();
    const p1 = "0x" + "ab".repeat(1024);
    const p2 = "0x" + "ac".repeat(1024);
    await a.anchor(p1);
    await a.anchor(p2);
    const h1 = ethers.keccak256(p1);
    const h2 = ethers.keccak256(p2);
    expect(h1).to.not.equal(h2);
    expect(await a.anchoredAt(h2)).to.be.greaterThan(0n);
  });

  it("rejects an empty proof", async function () {
    const a = await deployAnchor();
    await expect(a.anchor("0x"))
      .to.be.revertedWithCustomError(a, "NothingToAnchor");
  });

  it("emits the Anchored event", async function () {
    const a = await deployAnchor();
    const proof = "0x" + "ef".repeat(1024);
    const sender = (await ethers.getSigners())[0].address;
    await expect(a.anchor(proof))
      .to.emit(a, "Anchored")
      .withArgs(sender, ethers.keccak256(proof), anyValue);
  });
});
