"""Keyless end-to-end test of the SOURCE/SEARCH/RANK stages.

The LLM stages (classify + query-gen) need a paid key, so here we stand in for
the LLM with hand-authored analyses + queries (Claude acting as the classifier)
for three representative sample claims. Everything downstream — OpenAlex, arXiv,
Semantic Scholar, Papers With Code — is keyless, so this pulls REAL papers with
no API keys configured.

    python -m norm_cards.test_gather            # default: claim 11
    python -m norm_cards.test_gather --case 9   # theoretical claim
    python -m norm_cards.test_gather --case 50  # forecast claim
    python -m norm_cards.test_gather --per_query 5 --top_k 15
"""

import argparse
import json

from .models import ClaimAnalysis
from .taxonomy import Taxonomy
from .sources import get_sources, DEFAULT_SOURCES
from . import pipeline


# Hand-authored stand-ins for the LLM output (analysis + query buckets).
CASES = {
    "11": {
        "claim": ("Ultralytics YOLO11n detector pretrained on COCO, evaluated on the "
                  "person class of COCO val2017 at 640x640, retains AP50 >= 0.90 under a "
                  "learned CVAE illumination PGD attack with Linf bound eps = 12/255."),
        "analysis": ClaimAnalysis(
            subfields=["Object Detection", "Adversarial Robustness", "Image Generation"],
            openalex_topic_ids=["T12549", "T11689", "T10775"],
            openalex_topic_names=["Image and Object Detection Techniques",
                                  "Adversarial Robustness in Machine Learning",
                                  "Generative Adversarial Networks and Image Synthesis"],
            claim_type="empirical", temporal_mode="static",
            entities={"models": ["YOLO11n", "CVAE"],
                      "datasets": ["COCO", "COCO val2017"],
                      "metrics": ["AP50", "IoU"],
                      "thresholds": ["AP50 >= 0.90", "<= 5% degradation", "eps = 12/255"]},
            confidence=0.9, method="hand"),
        "queries": {
            "survey": ["object detection survey", "adversarial robustness survey",
                       "robustness of object detectors review"],
            "benchmark_dataset": ["COCO object detection benchmark",
                                  "object detection evaluation metrics",
                                  "adversarial robustness benchmark"],
            "canonical_method": ["YOLO object detection", "projected gradient descent attack",
                                 "state of the art object detection"],
            "sota_leaderboard": ["COCO detection leaderboard",
                                 "state of the art adversarial robustness"],
            "entity_anchored": ["YOLO11n COCO", "AP50 evaluation object detection",
                                "CVAE adversarial perturbation"],
            "intersection": ["adversarial robustness object detection",
                             "object detection under adversarial attack"],
        },
    },
    "9": {
        "claim": ("For an unknown discrete-time robotic system, there exists a learning "
                  "procedure that outputs a dynamics model and a non-vacuous error bound "
                  "holding with confidence at least 1-delta (delta=0.05)."),
        "analysis": ClaimAnalysis(
            subfields=["System Identification", "Statistical Learning Theory",
                       "Model-based Reinforcement Learning"],
            openalex_topic_ids=["T11236", "T10040", "T10243"],
            openalex_topic_names=["Control Systems and Identification",
                                  "Adaptive Control of Nonlinear Systems",
                                  "Statistical Methods and Bayesian Inference"],
            claim_type="theoretical", temporal_mode="static",
            entities={"models": [], "datasets": [],
                      "metrics": ["error bound", "coverage probability"],
                      "thresholds": ["confidence 1-delta", "delta=0.05", "probability >= 0.95"]},
            confidence=0.85, method="hand"),
        "queries": {
            "survey": ["system identification survey", "learning dynamical systems review",
                       "statistical learning theory bounds survey"],
            "benchmark_dataset": ["learned dynamics model evaluation",
                                  "system identification benchmark"],
            "canonical_method": ["learning Lipschitz dynamics", "Gaussian process dynamics model",
                                 "model-based reinforcement learning dynamics"],
            "theory": ["non-vacuous generalization bounds", "PAC bounds dynamical systems",
                       "high-probability error bounds learned dynamics",
                       "concentration inequalities system identification"],
        },
    },
    "50": {
        "claim": ("By January 2027, synthetic media generators will defeat forensic, "
                  "frequency-domain, and representation-learning detectors on benchmarks "
                  "derived from FaceForensics++, DFDC, and Synthbuster."),
        "analysis": ClaimAnalysis(
            subfields=["Deepfake and Forensics", "Image generation"],
            openalex_topic_ids=["T12357", "T10775", "T10388"],
            openalex_topic_names=["Digital Media Forensic Detection",
                                  "Generative Adversarial Networks and Image Synthesis",
                                  "Advanced Steganography and Watermarking Techniques"],
            claim_type="empirical", temporal_mode="forecast",
            entities={"models": [], "datasets": ["FaceForensics++", "DFDC", "Synthbuster"],
                      "metrics": ["AUC", "detection accuracy"], "thresholds": ["near chance"]},
            confidence=0.8, method="hand"),
        "queries": {
            "survey": ["deepfake detection survey", "synthetic media detection review",
                       "media forensics survey"],
            "benchmark_dataset": ["FaceForensics++ benchmark", "DFDC deepfake benchmark",
                                  "deepfake detection evaluation metrics"],
            "canonical_method": ["frequency domain deepfake detection",
                                 "representation learning deepfake detection"],
            "sota_leaderboard": ["state of the art deepfake detection"],
            "trend": ["measuring progress deepfake detection",
                      "deepfake detection arms race", "generator detector co-evolution",
                      "deepfake benchmark over time"],
        },
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="11", choices=list(CASES))
    ap.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES))
    ap.add_argument("--per_query", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=15)
    args = ap.parse_args()

    case = CASES[args.case]
    taxonomy = Taxonomy.load()
    sources = get_sources([s.strip() for s in args.sources.split(",") if s.strip()])
    print(f"Case {args.case}: {case['analysis'].subfields} "
          f"({case['analysis'].claim_type}/{case['analysis'].temporal_mode})")
    print(f"Sources: {[s.name for s in sources]}\n")

    result = pipeline.gather_from_analysis(
        case["analysis"], case["queries"], taxonomy, sources,
        per_query=args.per_query, top_k=args.top_k)

    print(f"\nStats: {json.dumps(result['stats'])}")
    print(f"\nTop {len(result['papers'])} papers:")
    for i, p in enumerate(result["papers"], 1):
        print(f"\n{i:>2}. [{p['score']:.2f}] {p['title']}  ({p['year']}, "
              f"cites={p['citations']}, type={p['paper_type']})")
        print(f"    sources={p['sources']} buckets={p['buckets']}")
        print(f"    {p['url']}")


if __name__ == "__main__":
    main()
