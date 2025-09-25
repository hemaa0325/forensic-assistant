# Blockchain Setup Guide

## Current Status
Your blockchain integration is **properly configured** but **unfunded**:
- ✅ Web3 library installed
- ✅ Connected to Polygon Amoy Testnet
- ✅ Private key configured
- ✅ Account address: `0x0bBd39cbFd1180EcfAD76Fe4B1c07c4253c79CBB`
- ❌ Balance: 0 MATIC (insufficient for transactions)

## Step-by-Step Setup Instructions

### 1. Get Test MATIC Tokens
Visit the official Polygon faucet to get free test tokens:

1. Go to: https://faucet.polygon.technology/
2. Select "Amoy" testnet
3. Enter your wallet address: `0x0bBd39cbFd1180EcfAD76Fe4B1c07c4253c79CBB`
4. Click "Send me MATIC"
5. Wait for confirmation (usually instant)

### 2. Alternative Faucets
If the main faucet doesn't work, try these alternatives:
- https://mumbaifaucet.com/
- https://polygon-faucet.com/

### 3. Verify Your Tokens
After receiving tokens, run this command to verify:

```bash
python verify_blockchain_setup.py
```

You should see:
```
Balance: > 0 wei (> 0 MATIC)
Ready for Transactions: True
🎉 Blockchain is fully configured and ready!
```

## What Changes When Funded?

### Current State (Unfunded)
Reports show:
```
Blockchain Verification
• Status: BLOCKCHAIN READY
• Note: Blockchain system is properly configured
• Storage Status: Not stored due to Insufficient funds in test environment
• Network: Polygon Amoy Testnet
```

### After Funding (Ready)
Reports will show:
```
Blockchain Verification
• Status: VERIFIED ON BLOCKCHAIN
• Record ID: 12345
• File Hash: abc123...
• Transaction Hash: 0x123abc...
• Block Number: 1234567
• Verification: [View on Blockchain Explorer]
```

## Troubleshooting

### If You Don't Receive Tokens
1. Check that you selected "Amoy" testnet (not Mumbai)
2. Ensure you're using the correct wallet address
3. Try a different faucet
4. Wait a few minutes and try again

### If Verification Still Shows Unfunded
1. Restart your application
2. Check your internet connection
3. Run the verification script again

### If You Get Errors
Common errors and solutions:
- "Transaction failed": Likely gas price too low - we've already increased this
- "Insufficient funds": Wait for faucet transaction to confirm
- "Connection failed": Check internet connection

## Testing the Integration

After funding, test the full integration:
1. Run your forensic application
2. Upload any file for analysis
3. Check the generated report for blockchain verification
4. Click the "View on Blockchain Explorer" link to verify on PolygonScan

## Security Notes

- Your private key is stored in the `.env` file
- This is only for testnet - never use real funds
- The private key allows transactions on the testnet only
- Keep your `.env` file secure and don't share it

## Need Help?

If you encounter any issues:
1. Run `python debug_blockchain.py` for detailed diagnostics
2. Check the console output when running your application
3. Verify your environment variables in `.env`
4. Ensure you have the latest version of web3.py: `pip install web3 --upgrade`