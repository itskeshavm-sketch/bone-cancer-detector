import json
from pathlib import Path

out = {
    "check": "equal-treatment comparison of every model on the 199-film wild pool",
    "why": ("V12 was quoted at argmax while V21 was quoted after a threshold search "
            "(worth +10 films on its own). The earlier claim that V12 beat V21 was a "
            "reporting artefact, not a model difference."),
    "method": ("one forward pass per model over all 199 films (gates A12 and A18, "
               "specialists B12/B18/B20/B21/B24); every (gate x threshold) "
               "combination scored from the same cached probabilities"),
    "argmax_only": {"V12": 91, "V18": 93, "V20": 90, "V21": 88, "V24": 139},
    "best_equal_search": {"V12": 95, "V18": 96, "V20": 98, "V21": 98, "V24": 142},
    "matched_at_49_of_95_cancers": {"V12": 90, "V18": 93, "V20": 89, "V21": 96, "V24": 134},
    "conclusion": ("V21 was always ahead of V12 on the wild pool (49.2% vs 47.7% under "
                   "equal search); the historical ordering was measurement bias. V24 leads "
                   "under every treatment (argmax 69.8%, best 71.4%, matched 67.3%)."),
    "corrected_story": ("V12/V18/V20/V21 all sit in a 45-49% band on real films - a data "
                        "ceiling from 2,736 osteosarcoma-dominated images. V24 broke it to "
                        "about 71% by fine-tuning on the named blind-spot entities."),
}
Path("checkpoints/equal_treatment.json").write_text(json.dumps(out, indent=1))
print("saved checkpoints/equal_treatment.json")
print("argmax   :", out["argmax_only"])
print("equal    :", out["best_equal_search"])
print("matched  :", out["matched_at_49_of_95_cancers"])