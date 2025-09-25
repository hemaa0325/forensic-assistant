"""
Script to help get test tokens for blockchain integration
"""

def get_test_tokens_instructions():
    print("=== How to Get Test MATIC Tokens ===")
    print()
    print("1. Visit the official Polygon faucet:")
    print("   🔗 https://faucet.polygon.technology/")
    print()
    print("2. Select the 'Amoy' testnet")
    print()
    print("3. Enter your wallet address:")
    print("   📋 0x0bBd39cbFd1180EcfAD76Fe4B1c07c4253c79CBB")
    print()
    print("4. Click 'Send me MATIC'")
    print()
    print("5. Wait for the transaction to complete (usually instant)")
    print()
    print("6. After receiving tokens, restart your application")
    print()
    print("=== Alternative Faucets ===")
    print("• https://mumbaifaucet.com/")
    print("• https://faucet.polygon.technology/ (official)")
    print()
    print("=== Verification ===")
    print("After getting tokens, run the debug script again:")
    print("   python debug_blockchain.py")
    print()
    print("You should see:")
    print("   'funded': true")
    print("   'balance': > 0")
    print("   'ready': true")

if __name__ == "__main__":
    get_test_tokens_instructions()