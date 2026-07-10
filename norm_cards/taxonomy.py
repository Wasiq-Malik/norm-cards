"""Subfield taxonomy: Papers With Code task list as a controlled vocabulary.

The taxonomy is a *living* set: PwC tasks (the backbone) plus custom subfields we
append when a claim doesn't map to anything existing. New labels are deduped
against the current vocabulary (token-overlap here; embedding-based in a later
pass) so we don't accumulate near-duplicates ("MT" vs "machine translation").

Files:
  data/pwc_tasks_seed.json  - curated offline seed (committed)
  data/taxonomy.json        - working taxonomy (seed + live PwC + custom); generated

Run `python -m norm_cards.taxonomy --bootstrap` to fetch the live PwC task list
and merge it into data/taxonomy.json.
"""

import json
import os
import re
from typing import List, Dict, Optional

from . import config

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_SEED_PATH = os.path.join(_DATA_DIR, "pwc_tasks_seed.json")
_WORKING_PATH = os.path.join(_DATA_DIR, "taxonomy.json")

_STOP = set("a an the of for to in on with and or via using under over from this that".split())


def _tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in toks if t not in _STOP and len(t) > 1}


def _contains(text_low: str, term: str) -> bool:
    """Word-boundary containment (so 'ap'/'rl'/'map' don't match inside words)."""
    if not term:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])",
                     text_low) is not None


class Taxonomy:
    def __init__(self, tasks: List[Dict]):
        self.tasks = tasks
        self._by_name = {t["task"].lower(): t for t in tasks}

    # ---- loading -------------------------------------------------------
    @classmethod
    def load(cls) -> "Taxonomy":
        """Load working taxonomy if present, else the committed seed."""
        path = _WORKING_PATH if os.path.exists(_WORKING_PATH) else _SEED_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["tasks"])

    def save(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_WORKING_PATH, "w", encoding="utf-8") as f:
            json.dump({"tasks": self.tasks}, f, indent=2)

    # ---- lookup --------------------------------------------------------
    def names(self) -> List[str]:
        return [t["task"] for t in self.tasks]

    def get(self, task_name: str) -> Optional[Dict]:
        return self._by_name.get((task_name or "").lower())

    def heuristic_match(self, text: str, top_k: int = 3) -> List[Dict]:
        """Score tasks against free text by keyword/name token overlap.

        Returns a ranked list of {task, area, score, card_type, ...}. Used as the
        offline classifier and as a prior for the LLM classifier.
        """
        text_l = (text or "").lower()
        text_toks = _tokens(text)
        scored = []
        for t in self.tasks:
            score = 0.0
            # Strong signal: a keyword phrase appears verbatim (word-boundary).
            for kw in t.get("keywords", []):
                if _contains(text_l, kw):
                    score += 2.0 + 0.2 * len(kw.split())
            # Weaker signal: token overlap with task name + keywords.
            task_toks = _tokens(t["task"])
            for kw in t.get("keywords", []):
                task_toks |= _tokens(kw)
            overlap = text_toks & task_toks
            score += 0.5 * len(overlap)
            # Dataset / metric name hits are decisive for norm purposes.
            for ds in t.get("datasets", []):
                if _contains(text_l, ds):
                    score += 1.5
            for m in t.get("metrics", []):
                if _contains(text_l, m):
                    score += 0.75
            if score > 0:
                scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, t in scored[:top_k]:
            d = dict(t)
            d["score"] = round(score, 3)
            out.append(d)
        return out

    # ---- extension -----------------------------------------------------
    def find_near_duplicate(self, label: str, threshold: float = 0.6) -> Optional[str]:
        """Return an existing task name if `label` is essentially the same."""
        lt = _tokens(label)
        if not lt:
            return None
        best, best_name = 0.0, None
        for t in self.tasks:
            et = _tokens(t["task"])
            for kw in t.get("keywords", []):
                et |= _tokens(kw)
            if not et:
                continue
            jacc = len(lt & et) / len(lt | et)
            if jacc > best:
                best, best_name = jacc, t["task"]
        return best_name if best >= threshold else None

    def add_custom(self, label: str, area: str = "Custom", card_type: str = "empirical") -> str:
        """Add a new subfield (deduped). Returns the canonical task name used."""
        existing = self.find_near_duplicate(label)
        if existing:
            return existing
        entry = {
            "task": label,
            "area": area,
            "keywords": sorted(_tokens(label)),
            "datasets": [],
            "metrics": [],
            "card_type": card_type,
            "source": "custom",
        }
        self.tasks.append(entry)
        self._by_name[label.lower()] = entry
        return label


# ----------------------------------------------------------------------
# Live PwC bootstrap (run manually; needs network).
# ----------------------------------------------------------------------
PWC_API = "https://paperswithcode.co/api/v1"  # verified live 2026-06-28


def _pwc_get(session, path, **params):
    r = session.get(f"{PWC_API}/{path}", params=params, timeout=config.HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def bootstrap_from_pwc(max_pages: int = 20) -> int:
    """Fetch the live paperswithcode.co taxonomy and (re)write the seed file.

    Pulls /areas/ (id->name) and all /tasks/ (paginated), with each task's real
    curated keywords, description, area, parent/child level, and paper count.
    Overwrites data/pwc_tasks_seed.json with the authoritative snapshot, and
    refreshes data/taxonomy.json while preserving any source=custom entries.
    """
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT})

    areas = {a["id"]: a["name"] for a in _pwc_get(session, "areas/").get("results", [])}

    tasks, page = [], 1
    while page <= max_pages:
        data = _pwc_get(session, "tasks/", page=page)
        results = data.get("results", [])
        if not results:
            break
        for it in results:
            kw = [k.strip().lower() for k in (it.get("keywords") or "").split(",") if k.strip()]
            tasks.append({
                "task": it["name"],
                "area": areas.get(it.get("area_id"), ""),
                "area_id": it.get("area_id"),
                "parent_id": it.get("parent_id"),
                "level": it.get("level"),
                "paper_count": it.get("paper_count"),
                "slug": it.get("slug"),
                "keywords": kw,
                "datasets": [],
                "metrics": [],
                "card_type": "empirical",
                "source": "pwc_co_api",
                "pwc_id": it.get("id"),
            })
        if len(tasks) >= data.get("count", 0):
            break
        page += 1

    # Write the authoritative seed snapshot.
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_SEED_PATH, "w", encoding="utf-8") as f:
        json.dump({"source_url": "https://paperswithcode.co/api/v1",
                   "tasks": tasks}, f, indent=2)

    # Refresh working taxonomy, preserving custom additions.
    customs = []
    if os.path.exists(_WORKING_PATH):
        with open(_WORKING_PATH, "r", encoding="utf-8") as f:
            customs = [t for t in json.load(f).get("tasks", []) if t.get("source") == "custom"]
    Taxonomy(tasks + customs).save()
    print(f"[taxonomy] fetched {len(tasks)} PwC.co tasks ({len(areas)} areas), "
          f"+{len(customs)} custom -> {_SEED_PATH}")
    return len(tasks)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Norm-cards subfield taxonomy utilities")
    ap.add_argument("--bootstrap", action="store_true", help="Fetch live PwC tasks and merge")
    ap.add_argument("--match", type=str, default="", help="Heuristically match free text to subfields")
    args = ap.parse_args()
    if args.bootstrap:
        bootstrap_from_pwc()
    if args.match:
        for m in Taxonomy.load().heuristic_match(args.match):
            print(f"{m['score']:6.2f}  {m['task']}  ({m['area']}, {m['card_type']})")
