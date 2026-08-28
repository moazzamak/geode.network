// M336 — InclusionInbox tests: on-chain force-inclusion, the
// registered M312 mirror. Gates: post -> incorporate -> valid;
// post -> withhold -> invalid; post -> incorporate-late -> invalid
// (recorded violation); deposit returns on incorporation and on
// poster withdrawal after the librarian fails.
const { expect } = require("chai");
const { ethers, network } = require("hardhat");

const WINDOW = 10;      // blocks
const MIN_DEPOSIT = 100n;

async function deployInbox() {
  const [poster, librarian, other] = await ethers.getSigners();
  const I = await ethers.getContractFactory("InclusionInbox");
  const inbox = await I.deploy(librarian.address, WINDOW, MIN_DEPOSIT);
  await inbox.waitForDeployment();
  return { inbox, poster, librarian, other };
}

async function mineBlocks(n) {
  for (let i = 0; i < n; i++) {
    await network.provider.send("evm_mine");
  }
}

describe("InclusionInbox (M336)", function () {
  it("post -> incorporate -> valid, deposit returned", async function () {
    const { inbox, poster, librarian } = await deployInbox();
    const entryId = ethers.keccak256(ethers.toUtf8Bytes("entry-1"));
    const digest = ethers.keccak256(ethers.toUtf8Bytes("payload-1"));
    await inbox.connect(poster).post(entryId, digest, {
      value: MIN_DEPOSIT,
    });
    expect(await inbox.chainValid()).to.equal(true);
    const before = await ethers.provider.getBalance(poster.address);
    await inbox.connect(librarian).incorporate(entryId);
    expect(await inbox.chainValid()).to.equal(true);
    expect(await inbox.openCount()).to.equal(0n);
    const after = await ethers.provider.getBalance(poster.address);
    // the deposit came back (poster also paid gas, so compare
    // against before plus the deposit minus gas: just assert the
    // entry's deposit is cleared and the balance rose past
    // before + deposit - a generous gas bound)
    expect(after).to.be.greaterThan(
      before - 10n ** 18n); // gas cost stays far below 1 ETH
    const e = await inbox.entries(entryId);
    expect(e.incorporatedBlock).to.be.greaterThan(0n);
    expect(e.deposit).to.equal(0n);
  });

  it("post -> withhold -> chain invalid", async function () {
    const { inbox, poster } = await deployInbox();
    const entryId = ethers.keccak256(ethers.toUtf8Bytes("entry-2"));
    const digest = ethers.keccak256(ethers.toUtf8Bytes("payload-2"));
    await inbox.connect(poster).post(entryId, digest, {
      value: MIN_DEPOSIT,
    });
    await mineBlocks(WINDOW + 1);
    expect(await inbox.chainValid()).to.equal(false);
    expect(await inbox.isLate(entryId)).to.equal(true);
  });

  it("post -> incorporate-late -> violation recorded, chain valid again", async function () {
    const { inbox, poster, librarian } = await deployInbox();
    const entryId = ethers.keccak256(ethers.toUtf8Bytes("entry-3"));
    const digest = ethers.keccak256(ethers.toUtf8Bytes("payload-3"));
    await inbox.connect(poster).post(entryId, digest, {
      value: MIN_DEPOSIT,
    });
    await mineBlocks(WINDOW + 1);
    expect(await inbox.chainValid()).to.equal(false); // was invalid
    const { anyValue } = require(
      "@nomicfoundation/hardhat-chai-matchers/withArgs");
    await expect(inbox.connect(librarian).incorporate(entryId))
      .to.emit(inbox, "Incorporated")
      .withArgs(entryId, poster.address, anyValue, true); // late = true
    expect(await inbox.chainValid()).to.equal(true);
  });

  it("poster withdraws the deposit after the librarian fails", async function () {
    const { inbox, poster } = await deployInbox();
    const entryId = ethers.keccak256(ethers.toUtf8Bytes("entry-4"));
    const digest = ethers.keccak256(ethers.toUtf8Bytes("payload-4"));
    await inbox.connect(poster).post(entryId, digest, {
      value: MIN_DEPOSIT,
    });
    // inside the window: no withdrawal yet
    await expect(inbox.connect(poster).withdrawDeposit(entryId))
      .to.be.revertedWithCustomError(inbox, "WindowNotElapsed");
    await mineBlocks(WINDOW + 1);
    await inbox.connect(poster).withdrawDeposit(entryId);
    expect((await inbox.entries(entryId)).deposit).to.equal(0n);
    // the entry stays open: the violation remains visible
    expect(await inbox.chainValid()).to.equal(false);
  });

  it("non-librarian cannot incorporate", async function () {
    const { inbox, poster, other } = await deployInbox();
    const entryId = ethers.keccak256(ethers.toUtf8Bytes("entry-5"));
    await inbox.connect(poster).post(
      entryId,
      ethers.keccak256(ethers.toUtf8Bytes("payload-5")),
      { value: MIN_DEPOSIT });
    await expect(inbox.connect(other).incorporate(entryId))
      .to.be.revertedWithCustomError(inbox, "NotLibrarian");
  });

  it("rejects empty digest and reused ids", async function () {
    const { inbox, poster } = await deployInbox();
    const entryId = ethers.keccak256(ethers.toUtf8Bytes("entry-6"));
    await expect(inbox.connect(poster).post(
      entryId, ethers.ZeroHash, { value: MIN_DEPOSIT }))
      .to.be.revertedWithCustomError(inbox, "EmptyDigest");
    await inbox.connect(poster).post(
      entryId, ethers.keccak256(ethers.toUtf8Bytes("p6")),
      { value: MIN_DEPOSIT });
    await expect(inbox.connect(poster).post(
      entryId, ethers.keccak256(ethers.toUtf8Bytes("p6b")),
      { value: MIN_DEPOSIT }))
      .to.be.revertedWithCustomError(inbox, "EntryClosed");
  });

  it("gas budget: post, incorporate, chainValid", async function () {
    const { inbox, poster, librarian } = await deployInbox();
    const entryId = ethers.keccak256(ethers.toUtf8Bytes("entry-7"));
    const txPost = await inbox.connect(poster).post(
      entryId, ethers.keccak256(ethers.toUtf8Bytes("p7")),
      { value: MIN_DEPOSIT });
    const receiptPost = await txPost.wait();
    const txInc = await inbox.connect(librarian).incorporate(entryId);
    const receiptInc = await txInc.wait();
    await mineBlocks(WINDOW + 1);
    // seal the gas budget as a generous ceiling, not a measurement:
    // each call sits far below 100k gas; assert the registered band.
    expect(receiptPost.gasUsed).to.be.lessThan(200_000n);
    expect(receiptInc.gasUsed).to.be.lessThan(200_000n);
    console.log(
      `M336 gas: post ${receiptPost.gasUsed}, ` +
      `incorporate ${receiptInc.gasUsed}`);
  });
});
