from typing import Dict, Any
import numpy as np
import time

def aggregate_scores(module_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine results from forensic modules with high-accuracy weighting.
    Priority:
        CNN (30%), Anti-Spoof (25%), Audio AI (15%), 
        rPPG (10%), AV-Sync (10%), Flash (5%), Emotion (5%)
    """
    weights = {
        "cnn": 0.30,
        "antispoof": 0.25,
        "audio_ai": 0.15,
        "rppg": 0.10,
        "av_sync": 0.10,
        "flash": 0.05,
        "emotion": 0.05,
    }
    
    weighted_sum = 0.0
    total_active_weight = 0.0
    active_modules = []
    
    # Priority handling: If CNN or AntiSpoof is extremely low, trigger FAKE immediately (Veto power)
    veto_fake = False
    
    for key, result in module_results.items():
        if isinstance(result, dict) and "score" in result:
            score = float(result["score"])
            weight = weights.get(key, 0.0)
            
            # Veto logic for high-confidence threats
            if key in ["cnn", "antispoof"] and score < 0.15:
                veto_fake = True
                
            weighted_sum += score * weight
            total_active_weight += weight
            active_modules.append(key)
    
    if total_active_weight == 0:
        confidence_score = 0.5
    else:
        confidence_score = weighted_sum / total_active_weight
    
    # Apply Veto
    if veto_fake:
        confidence_score = min(confidence_score, 0.2)

    # Dynamic Verdict Thresholds for 95%+ target
    # We prioritize low False Positives for "REAL"
    if confidence_score >= 0.85:
        verdict = "REAL"
    elif confidence_score <= 0.30:
        verdict = "FAKE"
    else:
        verdict = "UNCERTAIN"
    
    return {
        "verdict": verdict,
        "confidence_score": round(float(confidence_score), 4),
        "active_modules": active_modules,
        "total_modules": len(active_modules),
        "details": module_results,
        "timestamp": time.time()
    }

# Quick test (you can ignore)
if __name__ == "__main__":
    print("Score aggregator loaded successfully!")
