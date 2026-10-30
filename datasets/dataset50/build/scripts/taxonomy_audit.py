"""2026-10-29 audit of norm_cards/data/taxonomy.json against the dataset50 claims.

Run once; it edits the working taxonomy in place. Backed by the gpt-6-luna labels in
subfields_luna_v1.json: math/ML-theory claims had nowhere to go (SAM -> Image
Classification, Boolean functions -> Tabular Learning), vision robustness/backdoor papers fell
back to Image Classification, and Robotics took 19 of 20 robotics claims.
"""
import json

P = "norm_cards/data/taxonomy.json"
SRC = "manual_audit_2026-10-29"
tasks = json.load(open(P))["tasks"]
by = {t["task"]: t for t in tasks}

# 1. Placeholders with no keywords, no papers and no label in any dataset so far.
DROP = {"Test Task", "General", "Other", "Safety and Grounding", "Coding", "Classic Language Modeling"}

# 2. New entries. The first three keywords are the ones subfield_gather turns into queries.
NEW = [
    ("Deep Learning Optimization", "Math and ML Theory",
     ["optimizer", "learning rate schedule", "sharpness-aware minimization", "batch size", "adaptive optimizer",
      "adam", "adamw", "sgd", "momentum", "weight decay", "gradient descent", "step size", "line search", "muon"]),
    ("Training Dynamics", "Math and ML Theory",
     ["training dynamics", "neural collapse", "maximal update parameterization", "edge of stability",
      "loss landscape", "grokking", "implicit bias", "hyperparameter transfer", "mup", "feature learning",
      "catapult phase", "large learning rate"]),
    ("Learning Theory and Generalization", "Math and ML Theory",
     ["generalization", "inductive bias", "sample complexity", "simplicity bias", "flat minima",
      "generalization bound", "pac-bayes", "boolean functions", "parity learning", "benign overfitting",
      "statistical learning theory", "learnability"]),
    ("Scaling Laws", "Math and ML Theory",
     ["scaling law", "neural scaling laws", "data mixture", "compute-optimal", "data mixing law",
      "loss prediction", "scaling-law extrapolation", "chinchilla"]),
    ("Theorem Proving and Autoformalization", "Math and ML Theory",
     ["autoformalization", "theorem proving", "lean 4", "mathlib", "formal mathematics", "proof assistant",
      "minif2f", "formal verification", "neural theorem proving", "isabelle"]),
    ("Adversarial Robustness", "Robustness and Security",
     ["adversarial robustness", "adversarial training", "adversarial examples", "adversarial attack",
      "physical adversarial attack", "adversarial patch", "robust accuracy", "pgd", "autoattack",
      "certified robustness", "adversarial transferability"]),
    ("Backdoor Attacks and Defenses", "Robustness and Security",
     ["backdoor attack", "backdoor defense", "data poisoning", "trojan", "trigger inversion",
      "backdoor detection", "poisoned samples", "clean-label attack", "backdoor removal"]),
    ("Vision-Language-Action Models", "Robotics",
     ["vision-language-action", "vla", "vision language action model", "generalist robot policy",
      "robot foundation model", "openvla", "pi0"]),
    ("Robot Manipulation", "Robotics",
     ["robotic manipulation", "grasping", "dexterous manipulation", "pick and place",
      "non-prehensile manipulation", "diffusion policy", "behavior cloning", "bimanual manipulation"]),
    ("Embodied Navigation and Planning", "Robotics",
     ["vision-language navigation", "embodied task planning", "object navigation", "vln", "embodied agent",
      "r2r", "habitat", "task and motion planning"]),
    ("Sparse Autoencoders", "Interpretability",
     ["sparse autoencoder", "dictionary learning", "transcoder", "crosscoder", "sae features",
      "model diffing", "feature splitting", "dead latents"]),
    ("Constrained Decoding", "Language",
     ["constrained decoding", "grammar-constrained decoding", "structured output", "context-free grammar",
      "json schema", "token masking", "lexically constrained decoding", "regex-constrained generation"]),
]

# 3. Keyword fixes on existing entries.
#    Generic terms that misfire under word-boundary matching: "ai" (AI-generated images),
#    "adversarial"/"prompting" (adversarial robustness), "activation" (activation functions),
#    "editing"/"knowledge" (image editing, knowledge distillation).
REMOVE_KW = {
    "AI Safety and Alignment": {"ai"},
    "Jailbreaks and Adversarial Prompting": {"adversarial", "prompting"},
    "Activation Steering": {"activation"},
    "Knowledge Editing": {"editing", "knowledge"},
    # now their own entries
    "Code and Structured Generation": {"constrained decoding", "grammar-constrained generation"},
    "Verifiable-Reward Tasks": {"autoformalization", "theorem proving"},
}
SET_KW = {
    # was only VLA terms, which now have their own entry
    "Robotics": ["robot learning", "robot policy", "sim-to-real", "embodied ai", "humanoid", "locomotion",
                 "whole-body control", "legged robot", "cross-embodiment"],
}
ADD_KW = {
    "Deepfake and Forensics": ["ai-generated image detection", "synthetic image detection",
                               "image tampering localization", "image watermarking", "proactive forensics"],
    "Math": ["mathematical reasoning", "competition mathematics", "math word problems", "gsm8k", "olympiad"],
    "Long Context": ["long context", "long-context", "context length", "needle in a haystack", "long-context modeling"],
    "AI Safety and Alignment": ["over-refusal", "harmful fine-tuning", "safety alignment training"],
    "Anomaly Detection": ["failure detection", "runtime monitoring"],
}
# 4. Entries with an empty area were listed under "Other" in the prompt.
SET_AREA = {"Instruction Following": "Language", "Long Context": "Language", "Retrieval": "Language",
            "World Knowledge": "Language", "Math": "Math and ML Theory"}

out = [t for t in tasks if t["task"] not in DROP]
for name, kws in REMOVE_KW.items():
    by[name]["keywords"] = [k for k in by[name]["keywords"] if k not in kws]
for name, kws in SET_KW.items():
    by[name]["keywords"] = kws
for name, kws in ADD_KW.items():
    by[name]["keywords"] = list(dict.fromkeys(by[name]["keywords"] + kws))
for name, area in SET_AREA.items():
    by[name]["area"] = area
for name, area, kws in NEW:
    assert name not in by, name
    out.append({"task": name, "area": area, "keywords": kws, "datasets": [], "metrics": [],
                "card_type": "empirical", "source": SRC})
json.dump({"tasks": out}, open(P, "w"), indent=2)
print(f"{len(tasks)} -> {len(out)} entries (-{len(DROP)} placeholders, +{len(NEW)} new)")
