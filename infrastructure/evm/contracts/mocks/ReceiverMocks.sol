// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

interface IDevFundClaimer {
    function claimDevFund() external;
}

interface IClaimable {
    function claim() external;
}

/// @notice Test-only receiver: rejects ETH, exercising the
/// SendFailed path of pull claims.
contract RejectingReceiver {
    error NoETH();

    receive() external payable {
        revert NoETH();
    }
}

/// @notice Test-only poster that reverts on ETH receive, proving a
/// reverting poster cannot jam the queue head (the M383 pull lesson
/// applied to the bond): incorporation credits the bond to the
/// poster's claimable balance instead of pushing it.
contract RejectingPoster {
    error NoETH();

    receive() external payable {
        revert NoETH();
    }

    function post(address inbox, bytes32 entryId, bytes32 digest)
        external payable {
        (bool ok, ) = inbox.call{value: msg.value}(
            abi.encodeWithSignature("post(bytes32,bytes32)",
                                    entryId, digest));
        require(ok, "post failed");
    }
}

/// @notice Test-only stand-in for the ledger's librarian address,
/// so InclusionInbox's M382 source lookup can be driven directly.
contract LibrarianSourceMock {
    address public librarian;

    constructor(address librarian_) {
        librarian = librarian_;
    }

    function setLibrarian(address newLibrarian) external {
        librarian = newLibrarian;
    }
}

/// @notice Test-only receiver: re-enters claimDevFund from its
/// receive hook, exercising the transient reentrancy guard.
contract ReentrantReceiver {
    address public target;

    function setTarget(address target_) external {
        target = target_;
    }

    receive() external payable {
        if (target != address(0)) {
            IDevFundClaimer(target).claimDevFund();
        }
    }
}

/// @notice Test-only beneficiary: re-enters claim from its receive
/// hook, exercising claim's transient reentrancy guard.
contract ReentrantClaimer {
    address public target;

    function setTarget(address target_) external {
        target = target_;
    }

    receive() external payable {
        if (target != address(0)) {
            IClaimable(target).claim();
        }
    }
}
