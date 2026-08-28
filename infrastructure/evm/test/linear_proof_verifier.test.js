// M213 — LinearProofVerifier contract tests over the committed n=64
// fixture (built by the sealed Python runner): the same proof bytes
// verify on-chain, a tampered claim is rejected, and a malformed
// proof length reverts.
const { expect } = require("chai");
const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

const FIXTURE = path.join(__dirname, "..", "evidence",
                          "m213_fixture_n64.json");

function loadFixtureFile() {
  return JSON.parse(fs.readFileSync(FIXTURE, "utf8"));
}

async function deployVerifier() {
  const Verifier = await ethers.getContractFactory("LinearProofVerifier");
  const v = await Verifier.deploy();
  await v.waitForDeployment();
  return v;
}

describe("LinearProofVerifier (M213)", function () {
  it("verifies the Python-built n=64 proof bit-exactly", async function () {
    const v = await deployVerifier();
    const f = loadFixtureFile();
    const ok = await v.verify(f.proof, BigInt(f.claim),
                              f.w.map((x) => BigInt(x)));
    expect(ok).to.equal(true);
  });

  it("rejects a tampered claim", async function () {
    const v = await deployVerifier();
    const f = loadFixtureFile();
    const ok = await v.verify(f.proof,
                              (BigInt(f.claim) + 1n),
                              f.w.map((x) => BigInt(x)));
    expect(ok).to.equal(false);
  });

  it("rejects a tampered weight word", async function () {
    const v = await deployVerifier();
    const f = loadFixtureFile();
    const w = f.w.map((x) => BigInt(x));
    w[0] = w[0] + 1n;
    const ok = await v.verify(f.proof, BigInt(f.claim), w);
    expect(ok).to.equal(false);
  });

  it("reverts on a malformed proof length", async function () {
    const v = await deployVerifier();
    const f = loadFixtureFile();
    const short = "0x" + f.proof.slice(2, -64);
    await expect(
      v.verify(short, BigInt(f.claim), f.w.map((x) => BigInt(x))),
    ).to.be.revertedWithCustomError(v, "BadProofLength");
  });

  it("reverts on a non-power-of-two width", async function () {
    const v = await deployVerifier();
    const f = loadFixtureFile();
    await expect(
      v.verify(f.proof, BigInt(f.claim),
               f.w.slice(0, 63).map((x) => BigInt(x))),
    ).to.be.revertedWithCustomError(v, "NotPowerOfTwo");
  });
});
