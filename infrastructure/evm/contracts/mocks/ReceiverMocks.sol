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
