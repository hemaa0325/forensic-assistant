import av
import numpy as np

# --- ANOMALY SCORING DICTIONARY ---
VIDEO_ANOMALY_SCORES = {
    "low_bitrate": {"penalty": 15, "finding": "unusually low bitrate detected"},
    "high_bitrate": {"penalty": 10, "finding": "unusually high bitrate detected"},
    "out_of_order_frames": {"penalty": 50, "finding": "contains duplicated or out-of-order frames"},
    "dropped_frames": {"penalty": 40, "finding": "potential frame drops detected"},
    "duplicated_frames": {"penalty": 40, "finding": "potential duplicated frames detected"}
}

def analyze(filepath):
    """
    Performs a multi-layered forensic analysis of a video file with dynamic scoring.
    """
    anomaly_keys = []
    details = {}

    try:
        with av.open(filepath) as container:
            video_stream = container.streams.video[0]
            details['Codec'] = video_stream.codec_context.name
            details['Resolution'] = f"{video_stream.codec_context.width}x{video_stream.codec_context.height}"
            details['Frame Rate (FPS)'] = round(float(video_stream.average_rate), 2)
            
            bit_rate = video_stream.codec_context.bit_rate
            if bit_rate:
                details['Bitrate (kbps)'] = round(bit_rate / 1000)
                if bit_rate < 100000: anomaly_keys.append("low_bitrate")
                if bit_rate > 20000000: anomaly_keys.append("high_bitrate")

            frame_timestamps = []
            for frame in container.decode(video=0):
                if frame.pts is not None:
                    frame_timestamps.append(frame.pts)
            
            if len(frame_timestamps) > 1:
                time_diffs = np.diff(frame_timestamps)
                median_diff = np.median(time_diffs)

                if median_diff <= 0:
                    anomaly_keys.append("out_of_order_frames")
                else:
                    outlier_threshold_upper = median_diff * 2.5
                    outlier_threshold_lower = median_diff * 0.5
                    
                    if np.sum(time_diffs > outlier_threshold_upper) > 2:
                        anomaly_keys.append("dropped_frames")
                    if np.sum(time_diffs < outlier_threshold_lower) > 2:
                        anomaly_keys.append("duplicated_frames")
        
        # --- DYNAMIC SCORING LOGIC ---
        final_confidence = 100
        findings = []
        for key in set(anomaly_keys):
            anomaly = VIDEO_ANOMALY_SCORES.get(key)
            if anomaly:
                final_confidence -= anomaly["penalty"]
                findings.append(anomaly["finding"])
        
        final_confidence = max(10, final_confidence)

        # --- Final Verdict ---
        if findings:
            explanation = "Potential tampering detected: " + ", ".join(sorted(findings)) + "."
            return {"status": "Suspicious", "confidence": str(final_confidence), "explanation": explanation, "data": details}
        else:
            explanation = "No major temporal or metadata anomalies detected in the video stream."
            return {"status": "Authentic", "confidence": "90", "explanation": explanation, "data": details}

    except Exception as e:
        return {"status": "Error", "confidence": "0", "explanation": f"Video analysis failed. File may be corrupt or an unsupported format. Error: {e}", "data": {}}