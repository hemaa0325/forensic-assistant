// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ForensicAnalysisRegistry
 * @dev Smart contract for storing immutable forensic analysis records
 * @author Forensic Analysis System
 */
contract ForensicAnalysisRegistry {
    
    struct AnalysisRecord {
        string fileHash;           // SHA-256 hash of analyzed file
        string analysisData;       // JSON string of analysis results
        uint8 suspicionScore;      // Suspicion score (0-10)
        uint8 confidenceLevel;     // Confidence percentage (0-100)
        uint256 timestamp;         // Block timestamp when stored
        address analyst;           // Address of the analyst who performed analysis
        bool exists;               // Flag to check if record exists
    }
    
    // Mapping from record ID to analysis record
    mapping(uint256 => AnalysisRecord) public analysisRecords;
    
    // Mapping from file hash to record ID for quick lookup
    mapping(string => uint256) public fileHashToRecordId;
    
    // Counter for generating unique record IDs
    uint256 public totalRecords = 0;
    
    // Events
    event AnalysisStored(
        uint256 indexed recordId,
        string indexed fileHash,
        address indexed analyst,
        uint8 suspicionScore,
        uint8 confidenceLevel
    );
    
    event AnalysisVerified(
        uint256 indexed recordId,
        address indexed verifier,
        uint256 timestamp
    );
    
    /**
     * @dev Store a new forensic analysis record on blockchain
     * @param _fileHash SHA-256 hash of the analyzed file
     * @param _analysisData JSON string containing analysis details
     * @param _suspicionScore Suspicion score from 0-10
     * @param _confidenceLevel Confidence percentage from 0-100
     * @return recordId The unique ID assigned to this analysis record
     */
    function storeAnalysis(
        string memory _fileHash,
        string memory _analysisData,
        uint8 _suspicionScore,
        uint8 _confidenceLevel
    ) public returns (uint256) {
        require(bytes(_fileHash).length > 0, "File hash cannot be empty");
        require(bytes(_analysisData).length > 0, "Analysis data cannot be empty");
        require(_suspicionScore <= 10, "Suspicion score must be 0-10");
        require(_confidenceLevel <= 100, "Confidence level must be 0-100");
        
        // Increment record counter
        totalRecords++;
        uint256 recordId = totalRecords;
        
        // Store the analysis record
        analysisRecords[recordId] = AnalysisRecord({
            fileHash: _fileHash,
            analysisData: _analysisData,
            suspicionScore: _suspicionScore,
            confidenceLevel: _confidenceLevel,
            timestamp: block.timestamp,
            analyst: msg.sender,
            exists: true
        });
        
        // Map file hash to record ID for quick lookup
        fileHashToRecordId[_fileHash] = recordId;
        
        // Emit event
        emit AnalysisStored(
            recordId,
            _fileHash,
            msg.sender,
            _suspicionScore,
            _confidenceLevel
        );
        
        return recordId;
    }
    
    /**
     * @dev Get analysis record by record ID
     * @param _recordId The unique record ID
     * @return fileHash The SHA-256 hash of analyzed file
     * @return analysisData JSON string of analysis results
     * @return suspicionScore Suspicion score (0-10)
     * @return confidenceLevel Confidence percentage (0-100)
     * @return timestamp When the analysis was stored
     * @return analyst Address of the analyst
     */
    function getAnalysis(uint256 _recordId) public view returns (
        string memory fileHash,
        string memory analysisData,
        uint8 suspicionScore,
        uint8 confidenceLevel,
        uint256 timestamp,
        address analyst
    ) {
        require(_recordId > 0 && _recordId <= totalRecords, "Invalid record ID");
        require(analysisRecords[_recordId].exists, "Record does not exist");
        
        AnalysisRecord memory record = analysisRecords[_recordId];
        
        return (
            record.fileHash,
            record.analysisData,
            record.suspicionScore,
            record.confidenceLevel,
            record.timestamp,
            record.analyst
        );
    }
    
    /**
     * @dev Get analysis record by file hash
     * @param _fileHash SHA-256 hash of the file
     * @return recordId The unique record ID (0 if not found)
     * @return analysisData JSON string of analysis results
     * @return suspicionScore Suspicion score (0-10)
     * @return confidenceLevel Confidence percentage (0-100)
     * @return timestamp When the analysis was stored
     * @return analyst Address of the analyst
     */
    function getAnalysisByHash(string memory _fileHash) public view returns (
        uint256 recordId,
        string memory analysisData,
        uint8 suspicionScore,
        uint8 confidenceLevel,
        uint256 timestamp,
        address analyst
    ) {
        recordId = fileHashToRecordId[_fileHash];
        
        if (recordId == 0 || !analysisRecords[recordId].exists) {
            return (0, "", 0, 0, 0, address(0));
        }
        
        AnalysisRecord memory record = analysisRecords[recordId];
        
        return (
            recordId,
            record.analysisData,
            record.suspicionScore,
            record.confidenceLevel,
            record.timestamp,
            record.analyst
        );
    }
    
    /**
     * @dev Check if a file has been analyzed before
     * @param _fileHash SHA-256 hash of the file
     * @return exists True if analysis record exists
     * @return recordId The record ID if exists, 0 otherwise
     */
    function hasAnalysis(string memory _fileHash) public view returns (bool exists, uint256 recordId) {
        recordId = fileHashToRecordId[_fileHash];
        exists = recordId > 0 && analysisRecords[recordId].exists;
        return (exists, recordId);
    }
    
    /**
     * @dev Verify an analysis record (for audit trail)
     * @param _recordId The record ID to verify
     */
    function verifyAnalysis(uint256 _recordId) public {
        require(_recordId > 0 && _recordId <= totalRecords, "Invalid record ID");
        require(analysisRecords[_recordId].exists, "Record does not exist");
        
        emit AnalysisVerified(_recordId, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Get total number of analysis records
     * @return total The total number of records stored
     */
    function getTotalRecords() public view returns (uint256 total) {
        return totalRecords;
    }
    
    /**
     * @dev Get analysis records in a range (for pagination)
     * @param _startId Starting record ID
     * @param _count Number of records to return
     * @return recordIds Array of record IDs
     * @return fileHashes Array of file hashes
     * @return suspicionScores Array of suspicion scores
     * @return confidenceLevels Array of confidence levels
     * @return timestamps Array of timestamps
     * @return analysts Array of analyst addresses
     */
    function getAnalysisRange(uint256 _startId, uint256 _count) public view returns (
        uint256[] memory recordIds,
        string[] memory fileHashes,
        uint8[] memory suspicionScores,
        uint8[] memory confidenceLevels,
        uint256[] memory timestamps,
        address[] memory analysts
    ) {
        require(_startId > 0 && _startId <= totalRecords, "Invalid start ID");
        require(_count > 0 && _count <= 100, "Count must be 1-100");
        
        uint256 endId = _startId + _count - 1;
        if (endId > totalRecords) {
            endId = totalRecords;
        }
        
        uint256 actualCount = endId - _startId + 1;
        
        recordIds = new uint256[](actualCount);
        fileHashes = new string[](actualCount);
        suspicionScores = new uint8[](actualCount);
        confidenceLevels = new uint8[](actualCount);
        timestamps = new uint256[](actualCount);
        analysts = new address[](actualCount);
        
        for (uint256 i = 0; i < actualCount; i++) {
            uint256 recordId = _startId + i;
            AnalysisRecord memory record = analysisRecords[recordId];
            
            recordIds[i] = recordId;
            fileHashes[i] = record.fileHash;
            suspicionScores[i] = record.suspicionScore;
            confidenceLevels[i] = record.confidenceLevel;
            timestamps[i] = record.timestamp;
            analysts[i] = record.analyst;
        }
        
        return (recordIds, fileHashes, suspicionScores, confidenceLevels, timestamps, analysts);
    }
    
    /**
     * @dev Get contract metadata
     * @return name Contract name
     * @return version Contract version
     * @return description Contract description
     */
    function getContractInfo() public pure returns (string memory name, string memory version, string memory description) {
        return (
            "ForensicAnalysisRegistry",
            "1.0.0",
            "Immutable storage for forensic analysis records with tamper-proof verification"
        );
    }
}