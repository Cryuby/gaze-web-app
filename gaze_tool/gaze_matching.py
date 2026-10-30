# -*- coding: utf-8 -*-
"""
Consolidated gaze/bounding-box matching primitives, deduped from the
near-identical copies in gaze_object_proportions_both_groups.py,
person_gaze_episodes.py, person_gaze_episode_dynamics.py, and
person_orienting_latency.py. No logic changes from those originals.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent  # YOLO/
VR_PHASE_DIR = BASE / "VR Autism Project" / "Analysis Data" / "VR Phase"
SCENE_DIR = BASE / "VR Autism Project" / "DePaul loop images, videos" / "1"
CODEX_OUT = BASE / "codex_outputs"

# Every file this tool WRITES (results, caches, logs) goes under the user's own home
# directory, never under BASE -- BASE lives in a shared/synced Drive folder that a
# recipient (e.g. a clinician) may only have read access to, and a write there would
# fail outright for them even though it works fine for whoever owns the folder.
LOCAL_OUT_DIR = Path.home() / "GazeToolOutputs"

VIDEO_W, VIDEO_H = 5760.0, 2880.0
GAZE_RADIUS_PX = 30.0

# old-cohort hand-merged box CSVs (scene stem -> dotted scene name)
EXISTING_MERGED = {"1A": "1.A", "1D": "1.D", "1E": "1.E", "21B": "2.1.B", "21C": "2.1.C", "21D": "2.1.D"}

# 13 scenes both the autism (Round 1) and non-autism (Round 2) cohorts viewed
SHARED_SCENES = {"1.A", "1.B", "1.C", "1.D", "1.E", "2.1.A", "2.1.B", "2.1.C", "2.1.D",
                  "2.A", "2.B", "2.C", "2.D"}


@lru_cache(maxsize=1)
def load_box_table():
    """All known (closed-vocab) object boxes across every scene we have pre-computed.
    Cached (no args, same result every call): these CSVs live on a Google-Drive-mounted
    path and re-reading them from scratch on every analysis request was the dominant
    cost (a minute-plus for a single click) -- this makes every call after the first
    one in the same process instant.

    Missing source files (e.g. someone copied only the code, not our project data) are
    skipped rather than raising -- this just means 0 known closed-vocab classes, so
    every target falls through to the open-vocab YOLO-World path, which is correct:
    there's nothing closed-vocab to match against without our pre-computed boxes."""
    cols = ["scene", "frame_id", "box_x1", "box_y1", "box_x2", "box_y2", "box_confidence", "box_class"]
    frames = []
    for old_stem, dot_stem in EXISTING_MERGED.items():
        path = SCENE_DIR / f"{old_stem}_merged_data.csv"
        if not path.exists():
            continue  # our pre-computed data isn't bundled on this machine -- fine, see docstring
        df = pd.read_csv(path)
        df = df[df["source"] == "box_in_frame_with_gaze"].dropna(subset=["box_class"])
        df = df.drop_duplicates(subset=["frame_id", "box_x1", "box_y1", "box_x2", "box_y2", "box_class"])
        df["scene"] = dot_stem
        frames.append(df[cols])
    new_path = CODEX_OUT / "scene_object_boxes_new.csv"
    if new_path.exists():
        new_df = pd.read_csv(new_path).rename(columns={"video_stem": "scene"})
        frames.append(new_df[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    boxes = pd.concat(frames, ignore_index=True)
    boxes["frame_id"] = boxes["frame_id"].astype(int)
    return boxes


def build_index(boxes):
    """boxes -> (frame_index: {scene: sorted frame_ids}, by_frame: {(scene,frame_id): [(x1,y1,x2,y2,conf,class),...]})"""
    frame_index = {scene: np.sort(g["frame_id"].unique()) for scene, g in boxes.groupby("scene")}
    by_frame = {}
    for (scene, fid), g in boxes.groupby(["scene", "frame_id"]):
        by_frame[(scene, fid)] = list(zip(g.box_x1, g.box_y1, g.box_x2, g.box_y2, g.box_confidence, g.box_class))
    return frame_index, by_frame


def nearest_frame_id(sorted_ids, query):
    if len(sorted_ids) == 0:
        return None
    pos = np.searchsorted(sorted_ids, query)
    cands = [sorted_ids[p] for p in (pos, pos - 1) if 0 <= p < len(sorted_ids)]
    return min(cands, key=lambda fr: abs(fr - query)) if cands else None


def point_in_box(px, py, x1, y1, x2, y2, r=GAZE_RADIUS_PX):
    return (px >= x1 - r) and (px <= x2 + r) and (py >= y1 - r) and (py <= y2 + r)


def label_sample(px, py, scene, fid, by_frame, frame_index):
    """Highest-confidence box class gazed at, among ALL overlapping boxes regardless of class
    (winner-take-all -- matches gaze_object_proportions_both_groups.py's `label()`), or None.

    NOTE: this does NOT restrict the candidate search to a target class. Two of this session's
    original per-script versions disagreed on this point -- gaze_object_proportions_both_groups.py
    picked the single best class across ALL overlapping boxes (winner-take-all), while
    person_gaze_episodes.py / person_orienting_latency.py's is_person()/gazed_on_person() checked
    only person-class boxes, ignoring any higher-confidence competing box. Winner-take-all is the
    one adopted here (it's the "object-grounded, best single competing label per point" convention
    documented for this project) -- so results from this module differ slightly (sub-1pp in
    practice) from the two episode/latency scripts' original numbers, which used the narrower check.
    """
    nf = nearest_frame_id(frame_index.get(scene, np.array([])), fid)
    if nf is None:
        return None
    best_conf, best_cls = -1.0, None
    for (x1, y1, x2, y2, conf, cls) in by_frame.get((scene, nf), []):
        if point_in_box(px, py, x1, y1, x2, y2) and conf > best_conf:
            best_conf, best_cls = conf, cls
    return best_cls


def in_target(px, py, scene, fid, by_frame, frame_index, target_classes):
    return label_sample(px, py, scene, fid, by_frame, frame_index) in target_classes


@lru_cache(maxsize=8)
def load_gaze(path):
    """Eye-tracker CSV -> DataFrame with parsed ts/scene, restricted to valid rows.
    Cached per path for the same reason as load_box_table -- a clinician re-running
    several instructions against the same participant shouldn't re-read it from
    Google Drive every time."""
    df = pd.read_csv(path, usecols=["timestamp", "filename", "currentFrame", "FrameX", "FrameY"])
    ts = pd.to_datetime(df["timestamp"].str.slice(0, 19), format="%Y-%m-%d_%H-%M-%S", errors="coerce")
    df["ts"] = ts
    df["scene"] = df["filename"].str.extract(r"/([^/]+)\.mp4")[0]
    return df.dropna(subset=["ts", "scene", "FrameX", "FrameY"])


if __name__ == "__main__":
    boxes = load_box_table()
    frame_index, by_frame = build_index(boxes)
    print(f"Loaded {len(boxes)} box rows across {boxes['scene'].nunique()} scenes")
    print(f"Classes present: {sorted(boxes['box_class'].unique())}")
    assert len(boxes) > 0
    assert "person" in set(boxes["box_class"])
    print("OK")
