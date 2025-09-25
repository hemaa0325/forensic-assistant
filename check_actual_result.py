"""
Check what's actually being passed to the reports template
"""

# Let's create a test to see what the actual analysis result looks like
import json

def check_analysis_result():
    print("=== Checking Actual Analysis Result Structure ===")
    
    # This is what a typical analysis result looks like
    typical_result = {
        "status": "Authentic",
        "confidence": "95",
        "explanation": "Test explanation",
        "data": {
            "suspicion_score": 0,
            "confidence_percentage": 95
        }
        # Note: blockchain_verified and blockchain_record might be missing!
    }
    
    print("1. Typical analysis result:")
    print(json.dumps(typical_result, indent=2))
    
    print("\n2. Checking if blockchain fields exist:")
    print(f"   'blockchain_verified' in result: {'blockchain_verified' in typical_result}")
    print(f"   'blockchain_record' in result: {'blockchain_record' in typical_result}")
    
    print("\n3. What the template condition checks:")
    blockchain_condition = 'blockchain_verified' in typical_result and typical_result.get('blockchain_verified') and 'blockchain_record' in typical_result
    print(f"   'blockchain_verified' in result AND result.blockchain_verified AND 'blockchain_record' in result: {blockchain_condition}")
    
    print("\n4. What happens in the template:")
    if blockchain_condition:
        print("   → Shows 'VERIFIED ON BLOCKCHAIN' section")
    else:
        print("   → Shows 'LOCAL STORAGE ONLY' section (this is what users see!)")
    
    print("\n=== Solution ===")
    print("The issue is that blockchain_verified and blockchain_record fields")
    print("are not being added to the analysis result when blockchain storage")
    print("is skipped due to insufficient funds.")

if __name__ == "__main__":
    check_analysis_result()