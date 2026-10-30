# -*- coding: utf-8 -*-
"""
Keyword/regex instruction parsing -- no LLM. The clinician vocabulary for
this research tool is small and foreseeable (proportion/heatmap/episodes/
latency), so a dispatch table is enough; an instruction can fire more than
one kind at once ("percent... and a heatmap"). Every parsed instruction is
logged so keyword-table gaps surface from real usage instead of guesswork.

# ponytail: substring/regex intent matching, not NLU -- upgrade to an LLM
# parser only if clinicians start phrasing things this table can't catch
# (check codex_outputs/instruction_log.csv for unsupported/no-target rows first).
"""
import csv
import re
from collections import namedtuple
from pathlib import Path

Intent = namedtuple("Intent", ["kind", "target_phrase"])

KIND_KEYWORDS = {
    "proportion": [r"\bpercent\b", r"\bproportion\b", r"%", r"\bhow much time\b"],
    "heatmap": [r"\bheatmap\b", r"\bheat map\b", r"\bdensity\b", r"\bwhere did (?:they|he|she) look\b"],
    "episodes": [r"\bepisode\b", r"\bhow many times\b", r"\brecheck\b", r"\bnumber of (?:times|checks|looks)\b"],
    "timestamps": [r"\bwhen\b", r"\bwhat time\b", r"\btimestamp", r"\bmoments?\b", r"\blist the times\b"],
    "latency": [r"\blatency\b", r"\bhow fast\b", r"\borienting\b", r"\bhow quickly\b", r"\breaction time\b"],
    "unsupported": [r"\bthreat aversion\b", r"\bsocial avoidance\b", r"\bta\b", r"\bea\b", r"\benvironmental anchoring\b"],
}

TARGET_PATTERNS = [
    re.compile(r"(?:look(?:ed|ing)?|gaz(?:e|ed|ing)|attention)\s+(?:at|on|towards|to)\s+(?:the\s+)?"
               r"([a-z][a-z \-]*?)(?=[,.]|\s+(?:and|give|after|when|while|during)\b|$)"),
    re.compile(r"(?:re)?check(?:ed)?\s+(?:the\s+)?"
               r"([a-z][a-z \-]*?)(?=[,.]|\s+(?:and|give|after|when|while|during)\b|$)"),
]

from gaze_matching import LOCAL_OUT_DIR

LOG_PATH = LOCAL_OUT_DIR / "instruction_log.csv"


def extract_target_phrase(text_lower):
    for pat in TARGET_PATTERNS:
        m = pat.search(text_lower)
        if m:
            return m.group(1).strip()
    return None


def parse_instruction(text):
    text_lower = text.lower()
    target = extract_target_phrase(text_lower)
    kinds = [kind for kind, patterns in KIND_KEYWORDS.items() if any(re.search(p, text_lower) for p in patterns)]
    if not kinds:
        kinds = ["unsupported"]
    intents = [Intent(kind, target) for kind in kinds]
    _log(text, intents)
    return intents


def _log(text, intents):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["instruction", "matched_kinds", "target_phrase"])
        writer.writerow([text, ";".join(i.kind for i in intents), intents[0].target_phrase])


if __name__ == "__main__":
    tests = [
        ("what percent of time did they look at people, give me a heatmap", {"proportion", "heatmap"}, "people"),
        ("how many times did they recheck the dining table", {"episodes"}, "dining table"),
        ("how fast did they look at the window after it appeared", {"latency"}, "window"),
        ("when did they look at the person, give me timestamps", {"timestamps"}, "person"),
        ("give me their threat aversion score", {"unsupported"}, None),
    ]
    for text, expected_kinds, expected_target in tests:
        intents = parse_instruction(text)
        got_kinds = {i.kind for i in intents}
        got_target = intents[0].target_phrase
        print(f"{text!r} -> kinds={got_kinds} target={got_target!r}")
        assert got_kinds == expected_kinds, f"kinds: expected {expected_kinds}, got {got_kinds}"
        assert got_target == expected_target, f"target: expected {expected_target!r}, got {got_target!r}"
    print("OK")
