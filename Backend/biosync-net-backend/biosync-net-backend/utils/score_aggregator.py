from typing import Dict, Any
import numpy as np

def aggregate_scores(module_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine results from all 5 biometric modules into one final verdict.
    Weights (as per project spec):
        rPPG: 30%, Spectral: 20%, AV-Sync: 25%, Flash: 10%, Emotion: 15%
    """
    weights = {
        "rppg": 0.30,
        "spectral": 0.20,
        "av_sync": 0.25,
        "flash": 0.10,
        "emotion": 0.15
    }
    
    scores = []
    total_weight = 0.0
    
    for key, result in module_results.items():
        if isinstance(result, dict) and "score" in result:
            score = float(result["score"])
            weight = weights.get(key, 0.0)
            scores.append(score * weight)
            total_weight += weight
    
    if total_weight == 0:
        confidence_score = 0.5
    else:
        confidence_score = sum(scores) / total_weight
    
    # Final verdict logic
    if confidence_score >= 0.70:
        verdict = "REAL"
    elif confidence_score <= 0.40:
        verdict = "FAKE"
    else:
        verdict = "UNCERTAIN"
    
    return {
        "verdict": verdict,
        "confidence_score": round(float(confidence_score), 2),
        "details": module_results
    }

# Quick test (you can ignore)
if __name__ == "__main__":
    print("Score aggregator loaded successfully!")
