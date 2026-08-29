// M363 (G17): measure the librarian's per-operation EVM gas on the
// current contracts, for the operations-line cost model. Run:
//   npx hardhat run scripts/measure_operations_gas.js
const { ethers, upgrades } = require("hardhat");

async function main() {
  const [owner, devFund, payer, contributor, operator, librarian, rando] =
    await ethers.getSigners();

  // --- CreditLedger ------------------------------------------------
  const Ledger = await ethers.getContractFactory("CreditLedger");
  const ledger = await upgrades.deployProxy(Ledger, [devFund.address, 1000n]);
  await ledger.waitForDeployment();
  await ledger.setLibrarian(librarian.address);
  const aid = ethers.keccak256(ethers.toUtf8Bytes("arm-a"));
  await ledger.connect(operator).register(aid, contributor.address, 10n,
    ethers.ZeroHash, { value: 1000n });
  await ledger.connect(librarian).setAdmitted(aid, true);
  await ledger.connect(payer).deposit({ value: ethers.parseEther("1") });

  // --- InclusionInbox ----------------------------------------------
  const Mock = await ethers.getContractFactory("LibrarianSourceMock");
  const source = await Mock.deploy(librarian.address);
  await source.waitForDeployment();
  const Inbox = await ethers.getContractFactory("InclusionInbox");
  const WINDOW = 10, BOND = 100n, EPOCH_BLOCKS = 5000, BASE = 10n,
        FREE = 4, CAP = 8;
  const inbox = await Inbox.deploy(source.getAddress(), WINDOW, BOND,
    devFund.address, EPOCH_BLOCKS, BASE, FREE, CAP);
  await inbox.waitForDeployment();
  const digest = ethers.keccak256(ethers.toUtf8Bytes("d1"));

  const gas = {};
  const g = async (label, p) => {
    const tx = await p;
    const rc = await tx.wait();
    gas[label] = rc.gasUsed.toString();
  };

  // inbox: post + incorporate (two rounds)
  await g("inbox.post", inbox.connect(rando).post(
    ethers.keccak256(ethers.toUtf8Bytes("e0")), digest, {
      value: BOND + BASE }));
  await g("inbox.incorporate", inbox.connect(librarian).incorporate(
    ethers.keccak256(ethers.toUtf8Bytes("e0"))));

  // ledger: the librarian's settlement operations
  await g("ledger.recordCredits", ledger.connect(librarian).recordCredits(
    [payer.address],
    [{ artifactId: aid, who: contributor.address, amount: 1000n }]));
  // roll the epoch so a root can be posted for epoch 0
  await ethers.provider.send("evm_increaseTime", [31 * 24 * 3600]);
  await ethers.provider.send("evm_mine");
  await ledger.connect(librarian).recordCredits(
    [payer.address],
    [{ artifactId: aid, who: contributor.address, amount: 1n }]);
  const leaf = ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(
    ["bytes32"],
    [ethers.keccak256(ethers.AbiCoder.defaultAbiCoder().encode(
      ["uint256", "bytes32", "address", "uint256"],
      [0, aid, contributor.address, 1000n]))]));
  const root = ethers.keccak256(ethers.concat([leaf, leaf]));
  await g("ledger.postAttributionRoot",
          ledger.connect(librarian).postAttributionRoot(0, root));
  await g("ledger.claimAttribution",
          ledger.connect(contributor).claimAttribution(
            0, aid, contributor.address, 1000n, [leaf]));
  await g("ledger.claim", ledger.connect(contributor).claim());

  // propose-and-challenge slash + registry filings
  const B = ethers.parseEther("1");
  await ledger.connect(payer).deposit({ value: ethers.parseEther("2") });
  await g("ledger.fileSlash", ledger.connect(rando).fileSlash(
    contributor.address, aid, 100n, 1, ethers.ZeroHash,
    { value: B }));
  await ethers.provider.send("evm_increaseTime", [8 * 24 * 3600]);
  await ethers.provider.send("evm_mine");
  await g("ledger.executeSlash", ledger.connect(rando).executeSlash(0));

  await g("ledger.fileRegistryChange", ledger.connect(rando)
    .fileRegistryChange(1, aid, false, 0, ethers.ZeroHash,
                        { value: B }));

  // --- report -------------------------------------------------------
  console.log(JSON.stringify(gas, null, 2));
  const total = Object.values(gas).reduce((a, b) => a + BigInt(b), 0n);
  console.log("TOTAL_LIBRARIAN_SETTLEMENT_GAS", total.toString());
}

main().catch((e) => { console.error(e); process.exit(1); });
