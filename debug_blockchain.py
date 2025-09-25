import os
import sys
import json
from utils.blockchain_handler import blockchain_handler

def debug_blockchain_integration():
    print("=== Debugging Blockchain Integration ===")
    
    # 1. Check blockchain status
    print("\n1. Checking blockchain status...")
    status = blockchain_handler.get_blockchain_status()
    print(f"   Full status: {json.dumps(status, indent=2)}")
    
    # 2. Check if blockchain is available
    print("\n2. Checking blockchain availability...")
    is_available = blockchain_handler.is_blockchain_available()
    print(f"   Blockchain available: {is_available}")
    
    # 3. Test file hash calculation
    print("\n3. Testing file hash calculation...")
    test_content = b"This is a test file for debugging blockchain integration"
    file_hash = blockchain_handler.calculate_file_hash(test_content)
    print(f"   File hash: {file_hash}")
    
    # 4. Test analysis data preparation
    print("\n4. Testing analysis data preparation...")
    test_analysis = {
        'filename': 'debug_test.jpg',
        'status': 'Authentic',
        'confidence_percentage': 95,
        'total_suspicion_score': 0
    }
    prepared_data = blockchain_handler.prepare_analysis_data(test_analysis)
    print(f"   Prepared data: {prepared_data}")
    
    # 5. Try to store on blockchain (this will likely fail in test environment)
    print("\n5. Attempting blockchain storage...")
    try:
        blockchain_record = blockchain_handler.store_analysis_on_blockchain(test_content, test_analysis)
        if blockchain_record:
            print(f"   ✓ Blockchain storage successful!")
            print(f"     Record ID: {blockchain_record.get('record_id')}")
            print(f"     File Hash: {blockchain_record.get('file_hash')}")
        else:
            print("   ✗ Blockchain storage failed or skipped")
    except Exception as e:
        print(f"   ✗ Blockchain storage error: {e}")
    
    # 6. Check what would be passed to reports
    print("\n6. Sample report data structure...")
    sample_report_data = {
        'status': 'Authentic',
        'confidence': '95',
        'explanation': 'Test analysis for debugging',
        'blockchain_verified': False,  # This is what gets set when blockchain fails
        'blockchain_record': None
    }
    print(f"   Sample report data: {json.dumps(sample_report_data, indent=2)}")
    
    # 7. Test with blockchain verified data
    print("\n7. Sample blockchain-verified report data...")
    sample_verified_data = {
        'status': 'Authentic',
        'confidence': '95',
        'explanation': 'Test analysis for debugging',
        'blockchain_verified': True,
        'blockchain_record': {
            'record_id': 12345,
            'file_hash': file_hash,
            'transaction_hash': '0x' + 'a' * 64,
            'block_number': 1234567
        }
    }
    print(f"   Sample verified data: {json.dumps(sample_verified_data, indent=2)}")

if __name__ == "__main__":
    debug_blockchain_integration()