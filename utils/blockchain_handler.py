"""
Blockchain handler for forensic analysis system
Provides immutable storage and verification of forensic analysis results
"""

import hashlib
import json
import time
from datetime import datetime
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional web3 import - blockchain features work only if available
try:
    from web3 import Web3  # type: ignore
    WEB3_AVAILABLE = True
except ImportError:
    Web3 = None  # type: ignore
    WEB3_AVAILABLE = False
    print("Web3 not available - blockchain features disabled")

class BlockchainHandler:
    def __init__(self):
        """Initialize blockchain connection and contract setup"""
        self.web3_available = WEB3_AVAILABLE
        self.w3 = None
        self.account = None
        self.contract_address = None
        self.contract_abi = self._get_contract_abi()
        
        if not self.web3_available:
            print("Blockchain functionality disabled - web3 not installed")
            return
        
        try:
            # Use Polygon Amoy testnet (Mumbai is deprecated)
            self.rpc_url = "https://rpc-amoy.polygon.technology"
            self.chain_id = 80002  # Polygon Amoy testnet
            
            # Initialize Web3 connection
            if Web3 is not None:
                self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            
            # Private key for signing (should be from environment in production)
            self.private_key = os.getenv('BLOCKCHAIN_PRIVATE_KEY', '')
            self.contract_address = os.getenv('BLOCKCHAIN_CONTRACT_ADDRESS', '')
            
            # For demo purposes, use a placeholder contract address if not configured
            if not self.contract_address or self.contract_address == '0x0000000000000000000000000000000000000000':
                # This is a demo address - in production, deploy the actual contract
                self.contract_address = '0x742d35Cc6634C0532925a3b8D24Ca0c717Ce7c64'
            
            # Ensure contract address is in proper checksum format
            if self.contract_address and self.w3:
                self.contract_address = self.w3.to_checksum_address(self.contract_address)
            
            if self.private_key and self.w3 is not None:
                self.account = self.w3.eth.account.from_key(self.private_key)
        except Exception as e:
            print(f"Blockchain initialization failed: {e}")
            self.web3_available = False
    
    def _get_contract_abi(self) -> List[Dict]:
        """Return the ABI for the forensic analysis smart contract"""
        return [
            {
                "inputs": [],
                "name": "totalRecords",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "string", "name": "_fileHash", "type": "string"},
                    {"internalType": "string", "name": "_analysisData", "type": "string"},
                    {"internalType": "uint8", "name": "_suspicionScore", "type": "uint8"},
                    {"internalType": "uint8", "name": "_confidenceLevel", "type": "uint8"}
                ],
                "name": "storeAnalysis",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "uint256", "name": "_recordId", "type": "uint256"}],
                "name": "getAnalysis",
                "outputs": [
                    {"internalType": "string", "name": "fileHash", "type": "string"},
                    {"internalType": "string", "name": "analysisData", "type": "string"},
                    {"internalType": "uint8", "name": "suspicionScore", "type": "uint8"},
                    {"internalType": "uint8", "name": "confidenceLevel", "type": "uint8"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "address", "name": "analyst", "type": "address"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "string", "name": "_fileHash", "type": "string"}],
                "name": "getAnalysisByHash",
                "outputs": [
                    {"internalType": "uint256", "name": "recordId", "type": "uint256"},
                    {"internalType": "string", "name": "analysisData", "type": "string"},
                    {"internalType": "uint8", "name": "suspicionScore", "type": "uint8"},
                    {"internalType": "uint8", "name": "confidenceLevel", "type": "uint8"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "address", "name": "analyst", "type": "address"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "uint256", "name": "recordId", "type": "uint256"},
                    {"indexed": True, "internalType": "string", "name": "fileHash", "type": "string"},
                    {"indexed": True, "internalType": "address", "name": "analyst", "type": "address"},
                    {"indexed": False, "internalType": "uint8", "name": "suspicionScore", "type": "uint8"},
                    {"indexed": False, "internalType": "uint8", "name": "confidenceLevel", "type": "uint8"}
                ],
                "name": "AnalysisStored",
                "type": "event"
            }
        ]
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    def prepare_analysis_data(self, analysis_result: Dict[str, Any]) -> str:
        """Prepare analysis data for blockchain storage"""
        # Extract key information for blockchain storage
        blockchain_data = {
            'filename': analysis_result.get('filename', 'unknown'),
            'file_type': analysis_result.get('file_type', 'unknown'),
            'analysis_type': analysis_result.get('analysis_type', 'unknown'),
            'criteria_results': analysis_result.get('criteria_results', {}),
            'total_suspicion_score': analysis_result.get('total_suspicion_score', 0),
            'confidence_percentage': analysis_result.get('confidence_percentage', 0),
            'status': analysis_result.get('status', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Convert to JSON string (limit size for blockchain efficiency)
        return json.dumps(blockchain_data, separators=(',', ':'))[:1000]  # Limit to 1KB
    
    def store_analysis_on_blockchain(self, file_content: bytes, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Store forensic analysis result on blockchain"""
        if not self.web3_available or self.w3 is None:
            print("Blockchain storage skipped - web3 not available")
            return None
            
        try:
            if not self.account or not self.contract_address:
                print("Blockchain not properly configured")
                return None
            
            # Check account balance
            if self.w3 and self.account:
                balance = self.w3.eth.get_balance(self.account.address)
                if balance == 0:
                    print("Blockchain storage skipped - insufficient funds (test environment)")
                    # Return a special status indicating blockchain is available but storage skipped
                    return {
                        'status': 'skipped',
                        'reason': 'insufficient_funds',
                        'message': 'Blockchain storage skipped due to insufficient funds (test environment)'
                    }
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_content)
            
            # Prepare analysis data
            analysis_data = self.prepare_analysis_data(analysis_result)
            
            # Get scores
            suspicion_score = min(255, analysis_result.get('total_suspicion_score', 0))
            confidence_level = min(255, analysis_result.get('confidence_percentage', 0))
            
            # Create contract instance
            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(self.contract_address),
                abi=self.contract_abi
            )
            
            # Build transaction
            transaction = contract.functions.storeAnalysis(
                file_hash,
                analysis_data,
                suspicion_score,
                confidence_level
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 300000,  # Increased gas limit
                'gasPrice': self.w3.to_wei('30', 'gwei'),  # Increased gas price
                'chainId': self.chain_id
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.private_key)
            # Handle different attribute names for raw transaction data
            raw_tx = getattr(signed_txn, 'rawTransaction', getattr(signed_txn, 'raw_transaction', None))
            if raw_tx is None:
                raise Exception('Unable to get raw transaction data from signed transaction')
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt['status'] == 1:
                # Get record ID from transaction logs
                record_id = self._extract_record_id_from_receipt(receipt)
                
                blockchain_record = {
                    'transaction_hash': tx_hash.hex(),
                    'block_number': receipt['blockNumber'],
                    'record_id': record_id,
                    'file_hash': file_hash,
                    'timestamp': time.time(),
                    'gas_used': receipt['gasUsed'],
                    'status': 'confirmed'
                }
                
                print(f"Analysis stored on blockchain - Record ID: {record_id}")
                return blockchain_record
            else:
                print("Transaction failed")
                return None
                
        except Exception as e:
            print(f"Blockchain storage error: {str(e)}")
            return None
    
    def _extract_record_id_from_receipt(self, receipt) -> Optional[int]:
        """Extract record ID from transaction receipt"""
        try:
            if receipt.logs:
                # Decode the first log (should be AnalysisStored event)
                record_id = int(receipt.logs[0]['topics'][1].hex(), 16)
                return record_id
        except:
            pass
        return None
    
    def verify_analysis_on_blockchain(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Verify analysis record exists on blockchain"""
        if not self.web3_available or self.w3 is None:
            print("Blockchain verification skipped - web3 not available")
            return None
            
        try:
            if not self.contract_address:
                return None
            
            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(self.contract_address),
                abi=self.contract_abi
            )
            
            # Get analysis from blockchain
            result = contract.functions.getAnalysis(record_id).call()
            
            verification_data = {
                'file_hash': result[0],
                'analysis_data': json.loads(result[1]),
                'suspicion_score': result[2],
                'confidence_level': result[3],
                'timestamp': result[4],
                'analyst_address': result[5],
                'verified': True,
                'blockchain_url': f"https://mumbai.polygonscan.com/tx/{record_id}"
            }
            
            return verification_data
            
        except Exception as e:
            print(f"Blockchain verification error: {str(e)}")
            return None
    
    def verify_file_hash_on_blockchain(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Verify if a file hash exists on blockchain"""
        if not self.web3_available or self.w3 is None:
            print("Blockchain hash verification skipped - web3 not available")
            return None
            
        try:
            if not self.contract_address:
                return None
            
            contract = self.w3.eth.contract(
                address=self.w3.to_checksum_address(self.contract_address),
                abi=self.contract_abi
            )
            
            # Get analysis by file hash
            result = contract.functions.getAnalysisByHash(file_hash).call()
            
            if result[0] > 0:  # Record ID > 0 means found
                verification_data = {
                    'record_id': result[0],
                    'analysis_data': json.loads(result[1]),
                    'suspicion_score': result[2],
                    'confidence_level': result[3],
                    'timestamp': result[4],
                    'analyst_address': result[5],
                    'verified': True,
                    'file_hash': file_hash
                }
                return verification_data
            
            return None
            
        except Exception as e:
            print(f"File hash verification error: {str(e)}")
            return None
    
    def generate_blockchain_proof(self, blockchain_record: Dict[str, Any]) -> Dict[str, Any]:
        """Generate blockchain proof for PDF reports"""
        proof = {
            'blockchain_network': 'Polygon Amoy Testnet',
            'transaction_hash': blockchain_record.get('transaction_hash', ''),
            'block_number': blockchain_record.get('block_number', 0),
            'record_id': blockchain_record.get('record_id', 0),
            'file_hash': blockchain_record.get('file_hash', ''),
            'verification_url': f"https://mumbai.polygonscan.com/tx/{blockchain_record.get('transaction_hash', '')}",
            'timestamp': datetime.fromtimestamp(blockchain_record.get('timestamp', 0)).isoformat(),
            'immutable': True,
            'tamper_proof': True
        }
        return proof
    
    def is_blockchain_available(self) -> bool:
        """Check if blockchain connection is available"""
        if not self.web3_available or self.w3 is None:
            return False
            
        try:
            return self.w3.is_connected() and bool(self.account) and bool(self.contract_address)
        except:
            return False
    
    def get_blockchain_status(self) -> Dict[str, Any]:
        """Get current blockchain connection status"""
        status = {
            'web3_available': self.web3_available,
            'connected': False,
            'network': 'Polygon Amoy Testnet',
            'account_configured': bool(self.account),
            'contract_configured': bool(self.contract_address),
            'ready': False,
            'details': {}
        }
        
        if not self.web3_available:
            status['error'] = 'Web3 library not installed - run: pip install web3'
            status['details'] = {
                'issue': 'missing_dependency',
                'solution': 'Install web3.py library'
            }
            return status
        
        try:
            if self.w3:
                status['connected'] = self.w3.is_connected()
                if status['connected'] and self.account:
                    status['account_address'] = self.account.address
                    balance = self.w3.eth.get_balance(self.account.address)
                    status['balance'] = balance
                    status['balance_eth'] = str(self.w3.from_wei(balance, 'ether'))
                    
                    # Check if balance is sufficient for transactions
                    if balance > 0:
                        status['funded'] = True
                        status['details'] = {
                            'issue': 'none',
                            'message': 'Blockchain fully operational'
                        }
                    else:
                        status['funded'] = False
                        status['details'] = {
                            'issue': 'insufficient_funds',
                            'message': 'Account configured but needs funding for transactions',
                            'solution': 'Add test MATIC to your wallet from Polygon faucet'
                        }
                else:
                    status['details'] = {
                        'issue': 'configuration_incomplete',
                        'message': 'Web3 connected but account/contract not configured properly'
                    }
            
            status['ready'] = (status['connected'] and 
                             status['account_configured'] and 
                             status['contract_configured'] and
                             status.get('funded', False))
        except Exception as e:
            status['error'] = str(e)
            status['details'] = {
                'issue': 'connection_error',
                'message': f'Failed to connect to blockchain: {str(e)}'
            }
        
        return status

# Global blockchain handler instance
blockchain_handler = BlockchainHandler()