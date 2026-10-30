# -*- coding: utf-8 -*-
"""
Generalized versions of the per-script analysis functions written this
session: proportion-of-gaze-time, episode (contiguous-run) stats, and
orienting latency -- each now parameterized on an arbitrary target_classes
set instead of being hardcoded to "person".
"""
import numpy as np

from gaze_matching import VIDEO_W, VIDEO_H, in_target, label_sample


def label_gaze_df(gdf, by_frame, frame_index):
    """Adds a 'gazed_object' column: best-matching box class per row (winner-take-all
    across all classes), 'background' if none."""
    labels = []
    for scene, sub in gdf.groupby("scene"):
        for row in sub.itertuples():
            px, py = row.FrameX * VIDEO_W, row.FrameY * VIDEO_H
            cls = label_sample(px, py, scene, row.currentFrame, by_frame, frame_index)
            labels.append(cls if cls is not None else "background")
    out = gdf.copy()
    out["gazed_object"] = labels
    return out


def proportion_by_class(gdf, by_frame, frame_index):
    """% of every gaze sample landing on each object class (or 'background')."""
    labeled = label_gaze_df(gdf, by_frame, frame_index)
    return labeled["gazed_object"].value_counts(normalize=True)


def target_proportion(gdf, by_frame, frame_index, target_classes):
    """% of gaze samples landing specifically within target_classes."""
    n = len(gdf)
    if n == 0:
        return dict(pct=np.nan, n_samples=0)
    hits = sum(
        in_target(row.FrameX * VIDEO_W, row.FrameY * VIDEO_H, row.scene, row.currentFrame,
                  by_frame, frame_index, target_classes)
        for row in gdf.itertuples()
    )
    return dict(pct=100.0 * hits / n, n_samples=n)


def extract_episodes(gdf, by_frame, frame_index, target_classes):
    """Contiguous runs of gaze-on-target, scene by scene (sorted by ts within each scene).
    Returns a list of dicts: scene, start_ts, end_ts, duration_s, gap_before_s."""
    episodes = []
    for scene, sub in gdf.groupby("scene"):
        sub = sub.sort_values("ts")
        flags = [in_target(row.FrameX * VIDEO_W, row.FrameY * VIDEO_H, scene, row.currentFrame,
                            by_frame, frame_index, target_classes)
                  for row in sub.itertuples()]
        ts_list = sub["ts"].tolist()
        scene_eps = []
        i = 0
        while i < len(flags):
            if flags[i]:
                j = i
                while j + 1 < len(flags) and flags[j + 1]:
                    j += 1
                scene_eps.append((ts_list[i], ts_list[j]))
                i = j + 1
            else:
                i += 1
        for k, (s, e) in enumerate(scene_eps):
            gap = (s - scene_eps[k - 1][1]).total_seconds() if k > 0 else np.nan
            episodes.append(dict(scene=scene, start_ts=s, end_ts=e,
                                  duration_s=(e - s).total_seconds(), gap_before_s=gap))
    return episodes


def episode_summary(gdf, episodes):
    """Aggregate stats over an episode list, normalized by gdf's total exposure time."""
    total_exposure_s = sum(
        (sub["ts"].max() - sub["ts"].min()).total_seconds() for _, sub in gdf.groupby("scene")
    )
    durations = np.array([e["duration_s"] for e in episodes])
    gaps = np.array([e["gap_before_s"] for e in episodes if not np.isnan(e["gap_before_s"])])
    n = len(episodes)
    exposure_min = total_exposure_s / 60.0
    return dict(
        n_episodes=n,
        mean_duration_s=float(np.mean(durations)) if n else np.nan,
        median_duration_s=float(np.median(durations)) if n else np.nan,
        episodes_per_min=n / exposure_min if exposure_min else np.nan,
        mean_gap_s=float(np.mean(gaps)) if len(gaps) else np.nan,
        median_gap_s=float(np.median(gaps)) if len(gaps) else np.nan,
        pct_micro_episodes=float((durations == 0).mean()) * 100 if n else np.nan,
    )


def target_first_frame_by_scene(boxes, target_classes, scenes=None):
    """Per scene, the earliest frame_id at which a target-class box appears."""
    b = boxes[boxes["box_class"].isin(target_classes)]
    if scenes is not None:
        b = b[b["scene"].isin(scenes)]
    return {scene: int(g["frame_id"].min()) for scene, g in b.groupby("scene")}


def orienting_latency(gdf, by_frame, frame_index, target_first_frame, target_classes):
    """Per scene containing the target: time from the target's first on-screen frame to
    the participant's gaze first landing on it. Returns aggregate mean latency + hit-rate."""
    hits, misses = [], 0
    for scene, min_frame in target_first_frame.items():
        sub = gdf[gdf["scene"] == scene].sort_values("ts")
        sub = sub[sub["currentFrame"] >= min_frame]
        if sub.empty:
            continue
        t_appear = sub["ts"].iloc[0]
        found_ts = None
        for row in sub.itertuples():
            px, py = row.FrameX * VIDEO_W, row.FrameY * VIDEO_H
            if in_target(px, py, scene, row.currentFrame, by_frame, frame_index, target_classes):
                found_ts = row.ts
                break
        if found_ts is None:
            misses += 1
        else:
            hits.append((found_ts - t_appear).total_seconds())
    n_scenes = len(target_first_frame)
    return dict(mean_latency_s=float(np.mean(hits)) if hits else np.nan,
                hit_rate=len(hits) / n_scenes if n_scenes else np.nan,
                n_hits=len(hits), n_misses=misses)


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    from gaze_matching import load_box_table, build_index, load_gaze, VR_PHASE_DIR, SHARED_SCENES, CODEX_OUT

    boxes = load_box_table()
    frame_index, by_frame = build_index(boxes)
    gdf = load_gaze(VR_PHASE_DIR / "1" / "Participant 1 Eye tracking data.csv")
    gdf = gdf[gdf["scene"].isin(SHARED_SCENES) & gdf["FrameX"].between(0, 1) & gdf["FrameY"].between(0, 1)]

    # golden-number check against codex_outputs/gaze_object_proportions_both_groups.csv (P1 row).
    # This one MUST match exactly: that original script used the same winner-take-all semantics
    # this module now standardizes on.
    prop = target_proportion(gdf, by_frame, frame_index, {"person"})
    golden = pd.read_csv(CODEX_OUT / "gaze_object_proportions_both_groups.csv").set_index("participant").loc["P1"]
    print(f"person%% new={prop['pct']:.4f} golden={golden['person']*100:.4f}")
    assert abs(prop["pct"] - golden["person"] * 100) < 1e-6, "proportion mismatch vs golden CSV"

    # person_gaze_episodes.py / person_orienting_latency.py used a NARROWER check (only looked at
    # person-class boxes, ignoring any higher-confidence competing box at the same point) --
    # this module intentionally standardizes on winner-take-all instead (see gaze_matching.label_sample
    # docstring), so these two don't match exactly, only approximately. Reporting both, not asserting
    # equality, with a sanity bound instead.
    episodes = extract_episodes(gdf, by_frame, frame_index, {"person"})
    summary = episode_summary(gdf, episodes)
    golden_ep = pd.read_csv(CODEX_OUT / "person_gaze_episodes.csv").set_index("participant").loc["P1"]
    print(f"n_episodes new={summary['n_episodes']} golden(narrower-check)={golden_ep['n_episodes']} "
          f"[expected to differ slightly -- different box-matching rule, see module docstring]")
    assert summary["n_episodes"] > 0

    tff = target_first_frame_by_scene(boxes, {"person"}, scenes=SHARED_SCENES)
    lat = orienting_latency(gdf, by_frame, frame_index, tff, {"person"})
    golden_lat = pd.read_csv(CODEX_OUT / "person_orienting_latency.csv").set_index("participant").loc["P1"]
    print(f"mean_latency new={lat['mean_latency_s']:.4f} golden(narrower-check)={golden_lat['mean_latency_s']:.4f} "
          f"[expected to differ slightly, same reason]")
    assert lat["n_hits"] > 0

    print("OK -- proportion matches exactly; episodes/latency differ as expected (documented, not a bug)")
