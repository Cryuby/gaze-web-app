# -*- coding: utf-8 -*-
"""
CLI entrypoint: feed in a VR scene video + gaze CSV + a free-text
instruction, get an analysis back.

IMPORTANT: run with `python`, not `python3` -- on this machine `python` is
the conda env with ultralytics/torch installed; `python3` is the bare
system Python without them.

Example:
    python analyze.py \
      --video "VR Autism Project/DePaul loop images, videos/1/2.1.C.mp4" \
      --gaze  "VR Autism Project/Analysis Data/VR Phase/eyegaze_2026-04-27_13-19-47-398.csv" \
      --scene 2.1.C \
      --instruction "what percent of time did they look at people, give me a heatmap"
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaze_matching import load_box_table, build_index, load_gaze, LOCAL_OUT_DIR
from grounding import resolve_target, run_yolo_world
from instruction import parse_instruction
from metrics import target_proportion, extract_episodes, episode_summary, target_first_frame_by_scene, orienting_latency
from viz import render_heatmap
from scene_description import describe_scene, prepare_video_for_analysis

OUT_DIR = LOCAL_OUT_DIR / "results"

def _fmt_offset(td):
    """pandas Timedelta -> 'M:SS' relative to the scene's first gaze sample."""
    total_s = int(td.total_seconds())
    return f"{total_s // 60}:{total_s % 60:02d}"


UNSUPPORTED_MSG = (
    "Could not act on this instruction. If you're asking for Threat-Aversion or "
    "Environmental-Anchoring scoring: not implemented in v1 (the existing experimental "
    "model in run_yolo_framework_best_batch.py is weak, F1<0.4 -- reference only, not wired in). "
    "Otherwise: try phrasing with one of -- percent/proportion, heatmap, "
    "how many times/recheck, how fast/latency -- plus 'look at <object>'."
)


def resolve_boxes(target_phrase, scene, video_path, box_table, global_frame_index, global_by_frame):
    """Returns (classes, frame_index, by_frame, boxes_for_scene, source) for this target.
    boxes_for_scene is the box DataFrame actually used -- reused for latency's first-appearance
    lookup so an open-vocab target only ever runs YOLO-World once per call."""
    classes, source = resolve_target(target_phrase, scene, box_table)
    if source == "closed_vocab":
        boxes_for_scene = box_table[(box_table["scene"] == scene) & (box_table["box_class"].isin(classes))]
        return classes, global_frame_index, global_by_frame, boxes_for_scene, source
    print(f"  '{target_phrase}' not in known classes -- running YOLO-World open-vocab detection "
          f"(1 frame/sec) over {video_path.name}...")
    new_boxes = run_yolo_world(video_path, target_phrase, scene)
    frame_index, by_frame = build_index(new_boxes) if len(new_boxes) else ({}, {})
    return {target_phrase}, frame_index, by_frame, new_boxes, source


def run_analysis(video, gaze, scene, instruction):
    """Core entrypoint, usable from the CLI or directly from a notebook. Returns the results dict
    (also written to codex_outputs/gaze_tool_outputs/result_<scene>_<timestamp>.json)."""
    video_path, gaze_path = Path(video), Path(gaze)
    # Large uploads (multi-GB originals) get downscaled+cached once here, before anything
    # touches the video -- both describe_scene() and the open-vocab YOLO-World path (the
    # two places that actually decode the video frame-by-frame) benefit automatically.
    video_path = prepare_video_for_analysis(video_path)
    box_table = load_box_table()
    frame_index, by_frame = build_index(box_table)

    gdf = load_gaze(gaze_path)
    gdf = gdf[(gdf["scene"] == scene) & gdf["FrameX"].between(0, 1) & gdf["FrameY"].between(0, 1)]
    if gdf.empty:
        msg = f"No gaze samples found for scene '{scene}' in {gaze_path.name}. Check scene matches the video."
        print(msg)
        return {"error": msg}

    intents = parse_instruction(instruction)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {"instruction": instruction, "scene": scene, "n_samples": len(gdf), "results": []}

    # Always-on scene + gaze summary (no instruction needed for this one): what happened in
    # the scene (TwelveLabs, if configured -- None otherwise) + where the participant looked
    # (our own object-grounded metrics, always available). See scene_description.py docstring
    # for the data-privacy tradeoff of enabling the TwelveLabs half.
    print("\n[scene_summary]")
    scene_desc = describe_scene(video_path)
    default_prop = target_proportion(gdf, by_frame, frame_index, {"person"})
    gaze_summary = (f"{default_prop['pct']:.1f}% of gaze time was on people in this scene "
                     f"(n={default_prop['n_samples']} samples).")
    if scene_desc:
        print(f"  Scene: {scene_desc}")
    else:
        print("  Scene description unavailable (TwelveLabs API key not configured -- see scene_description.py)")
    print(f"  Gaze: {gaze_summary}")
    results["results"].append({"kind": "scene_summary", "scene_description": scene_desc, "gaze_summary": gaze_summary})

    for intent in intents:
        print(f"\n[{intent.kind}] target={intent.target_phrase!r}")

        if intent.kind == "unsupported":
            print(f"  {UNSUPPORTED_MSG}")
            results["results"].append({"kind": "unsupported", "message": UNSUPPORTED_MSG})
            continue

        if intent.target_phrase is None and intent.kind != "heatmap":
            msg = "Couldn't figure out what object you meant -- try phrasing like 'look at <object>'."
            print(f"  {msg}")
            results["results"].append({"kind": intent.kind, "message": msg})
            continue

        if intent.kind == "heatmap":
            out_png = OUT_DIR / f"heatmap_{scene}_{stamp}.png"
            ok = render_heatmap(scene, gdf["FrameX"].to_numpy(), gdf["FrameY"].to_numpy(), out_png, video_path=video_path)
            if ok:
                print(f"  Wrote heatmap -> {out_png}")
                results["results"].append({"kind": "heatmap", "output_png": str(out_png)})
            else:
                print("  Not enough gaze samples (or no scene image) to render a heatmap.")
                results["results"].append({"kind": "heatmap", "message": "insufficient data"})
            continue

        classes, tf_idx, tf_by_frame, boxes_for_scene, source = resolve_boxes(
            intent.target_phrase, scene, video_path, box_table, frame_index, by_frame
        )

        if intent.kind == "proportion":
            r = target_proportion(gdf, tf_by_frame, tf_idx, classes)
            print(f"  {r['pct']:.1f}% of gaze time (n={r['n_samples']} samples) was on '{intent.target_phrase}' "
                  f"in scene {scene} [{source}]")
            results["results"].append({"kind": "proportion", "target": intent.target_phrase, "source": source, **r})

        elif intent.kind == "episodes":
            episodes = extract_episodes(gdf, tf_by_frame, tf_idx, classes)
            s = episode_summary(gdf, episodes)
            print(f"  {s['n_episodes']} episodes ({s['episodes_per_min']:.1f}/min) looking at "
                  f"'{intent.target_phrase}', mean duration {s['mean_duration_s']:.2f}s, "
                  f"mean gap between episodes {s['mean_gap_s']:.2f}s [{source}]")
            results["results"].append({"kind": "episodes", "target": intent.target_phrase, "source": source, **s})

        elif intent.kind == "timestamps":
            episodes = extract_episodes(gdf, tf_by_frame, tf_idx, classes)
            scene_start = gdf["ts"].min()
            # Single-sample "episodes" (duration 0.0s) are mostly tracking jitter at a box's
            # edge, not a real look -- same noise the episode_summary() pct_micro_episodes stat
            # already flags. Drop them here so the list a clinician reads is actual moments of
            # looking, not noise; n_filtered keeps the full count honest.
            # ponytail: fixed 0.1s threshold -- revisit if real brief-but-genuine glances turn
            # out to matter for some instruction (then make this a per-instruction parameter).
            MIN_MOMENT_DURATION_S = 0.1
            all_moments = [{"start": _fmt_offset(e["start_ts"] - scene_start),
                            "end": _fmt_offset(e["end_ts"] - scene_start),
                            "duration_s": round(e["duration_s"], 2)} for e in episodes]
            moments = [m for m in all_moments if m["duration_s"] >= MIN_MOMENT_DURATION_S]
            n_filtered = len(all_moments) - len(moments)
            if moments:
                print(f"  {len(moments)} moments looking at '{intent.target_phrase}' [{source}]"
                      f" ({n_filtered} briefer than {MIN_MOMENT_DURATION_S}s omitted as likely tracking noise):")
                for m in moments[:10]:
                    print(f"    {m['start']}-{m['end']} ({m['duration_s']}s)")
                if len(moments) > 10:
                    print(f"    ... and {len(moments) - 10} more (see full result JSON)")
            else:
                print(f"  Never looked at '{intent.target_phrase}' in this scene "
                      f"(for longer than {MIN_MOMENT_DURATION_S}s).")
            results["results"].append({"kind": "timestamps", "target": intent.target_phrase, "source": source,
                                        "moments": moments, "n_filtered_as_noise": n_filtered})

        elif intent.kind == "latency":
            tff = target_first_frame_by_scene(boxes_for_scene, classes, scenes={scene})
            r = orienting_latency(gdf, tf_by_frame, tf_idx, tff, classes)
            if r["n_hits"] == 0:
                print(f"  Never oriented to '{intent.target_phrase}' in this scene.")
            else:
                print(f"  Mean latency to first look at '{intent.target_phrase}': {r['mean_latency_s']:.2f}s "
                      f"(hit rate {r['hit_rate']*100:.0f}%) [{source}]")
            results["results"].append({"kind": "latency", "target": intent.target_phrase, "source": source, **r})

    out_json = OUT_DIR / f"result_{scene}_{stamp}.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull result written to {out_json}")
    results["output_json"] = str(out_json)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to the scene video (e.g. .../2.1.C.mp4)")
    ap.add_argument("--gaze", required=True, help="Path to the participant's eye-tracking CSV")
    ap.add_argument("--scene", required=True, help="Scene stem, e.g. 2.1.C (must match the video's scene)")
    ap.add_argument("--instruction", required=True, help="Free-text instruction")
    args = ap.parse_args()
    run_analysis(args.video, args.gaze, args.scene, args.instruction)


if __name__ == "__main__":
    main()
