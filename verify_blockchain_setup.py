"""
Script to verify blockchain setup after getting test tokens
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.blockchain_handler import blockchain_handler

def verify_blockchain_setup():
    print("=== Blockchain Setup Verification ===")
    print()
    
    # Check if web3 is available
    print("1. Web3 Availability:")
    print(f"   Web3 available: {blockchain_handler.web3_available}")
    
    if not blockchain_handler.web3_available:
        print("   ❌ Please install web3: pip install web3")
        return
    
    print()
    print("2. Connection Status:")
    try:
        if blockchain_handler.w3:
            connected = blockchain_handler.w3.is_connected()
            print(f"   Connected: {connected}")
            
            if not connected:
                print("   ❌ Cannot connect to Polygon Amoy network")
                print("   Check your internet connection")
                return
        else:
            print("   ❌ Web3 not initialized")
            return
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return
    
    print()
    print("3. Account Configuration:")
    print(f"   Account configured: {bool(blockchain_handler.account)}")
    
    if not blockchain_handler.account:
        print("   ❌ No account configured")
        print("   Check your BLOCKCHAIN_PRIVATE_KEY in .env file")
        return
    
    print(f"   Account address: {blockchain_handler.account.address}")
    
    print()
    print("4. Balance Check:")
    try:
        balance = blockchain_handler.w3.eth.get_balance(blockchain_handler.account.address)
        balance_eth = blockchain_handler.w3.from_wei(balance, 'ether')
        print(f"   Balance: {balance} wei ({balance_eth} MATIC)")
        
        if balance == 0:
            print("   ⚠️  Account has zero balance")
            print("   Please get test tokens from Polygon faucet:")
            print("   https://faucet.polygon.technology/")
            return
        elif balance < blockchain_handler.w3.to_wei(0.01, 'ether'):
            print("   ⚠️  Low balance - might not be enough for transactions")
        else:
            print("   ✅ Sufficient balance for transactions")
    except Exception as e:
        print(f"   ❌ Balance check error: {e}")
        return
    
    print()
    print("5. Contract Configuration:")
    print(f"   Contract configured: {bool(blockchain_handler.contract_address)}")
    
    if not blockchain_handler.contract_address:
        print("   ⚠️  No contract address configured")
        print("   Using demo contract address for testing")
    
    print()
    print("6. Overall Status:")
    status = blockchain_handler.get_blockchain_status()
    print(f"   Ready for transactions: {status.get('ready', False)}")
    
    if status.get('ready', False):
        print("   🎉 Blockchain is fully configured and ready!")
        print("   You can now store analysis results on the blockchain")
    else:
        print("   ⚠️  Blockchain not ready for transactions")
        if 'details' in status and 'message' in status['details']:
            print(f"   Reason: {status['details']['message']}")
        if 'details' in status and 'solution' in status['details']:
            print(f"   Solution: {status['details']['solution']}")

if __name__ == "__main__":
    verify_blockchain_setup()