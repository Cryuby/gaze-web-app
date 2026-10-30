# -*- coding: utf-8 -*-
"""
Optional TwelveLabs integration: given the scene video, generates a plain-
language description of what happens in it -- no prompt needed from the
clinician (we supply a fixed default prompt behind the scenes, since
TwelveLabs' API requires one even though the UI never shows it).

This is combined in analyze.py with our own object-grounded gaze metrics to
produce one "what happened + where they looked" summary automatically,
every run -- TwelveLabs only understands the video, it has no idea where the
participant looked, so the gaze half always comes from our own pipeline.

Requires a TwelveLabs API key (https://twelvelabs.io -- free tier, no credit
card). Reads it from the TWELVELABS_API_KEY env var, or ~/.twelvelabs/api_key.
If neither is present, describe_scene() returns None (not an error) -- this
feature is optional and OFF by default on a machine with no key configured.

IMPORTANT: enabling this sends the scene video to TwelveLabs' cloud. That is
a data-privacy tradeoff the rest of this tool deliberately avoids (everything
else runs fully local). Only set up an API key for this deployment once
that's been cleared (IRB / data agreement) -- this is a research-policy
decision, not a technical one.
"""
import os
import time
from pathlib import Path

# TwelveLabs' "direct" asset-upload method caps files at 200MB -- our raw VR scene
# videos are multi-GB, so anything over this gets downscaled first. Also reused by
# grounding.run_yolo_world() for the same reason (a multi-GB upload is slow to decode
# frame-by-frame) -- the scene video is the shared stimulus (not participant-specific),
# so the downscaled copy is cached and reused across every participant/instruction
# that analyzes the same scene.
MAX_UPLOAD_MB = 180
DOWNSCALE_SIZE = (1280, 640)  # matches the resolution already validated to give good Pegasus descriptions

DEFAULT_PROMPT = (
    "Describe in detail what happens in this scene: the setting, the people or "
    "objects present, and the sequence of events. Be specific and concrete."
)


def _load_api_key():
    key = os.environ.get("TWELVELABS_API_KEY")
    if key:
        return key
    key_file = Path.home() / ".twelvelabs" / "api_key"
    if key_file.exists():
        return key_file.read_text().strip()
    return None


def prepare_video_for_analysis(video_path):
    """Returns a path safe/fast to process (<= MAX_UPLOAD_MB). Our raw VR scene videos
    (and anything a clinician uploads) can be multi-GB/high-res, so anything over that
    gets downscaled to DOWNSCALE_SIZE first -- cached per source video (by name) since
    the same stimulus video is reused across every participant/instruction that analyzes
    that scene. Used both for TwelveLabs upload and for grounding.run_yolo_world()."""
    from gaze_matching import LOCAL_OUT_DIR

    video_path = Path(video_path)
    size_mb = video_path.stat().st_size / 1e6
    if size_mb <= MAX_UPLOAD_MB:
        return video_path

    cache_dir = LOCAL_OUT_DIR / "downscaled_video_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{video_path.stem}_small.mp4"
    if cached.exists():
        return cached

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(cached), fourcc, fps, DOWNSCALE_SIZE)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.resize(frame, DOWNSCALE_SIZE))
    finally:
        cap.release()
        writer.release()
    return cached


def describe_scene(video_path, prompt=DEFAULT_PROMPT, model_name="pegasus1.5", timeout_s=300):
    """Uploads video_path to TwelveLabs and returns its scene description, or None if
    this optional feature isn't usable right now (no API key, package not installed,
    network/API error, or processing didn't finish within timeout_s) -- this must never
    raise, since the rest of the tool's core (local, no-cloud) analysis has to keep
    working even when this integration can't."""
    api_key = _load_api_key()
    if not api_key:
        return None

    try:
        from twelvelabs import TwelveLabs
        client = TwelveLabs(api_key=api_key)

        upload_path = prepare_video_for_analysis(video_path)
        with open(upload_path, "rb") as f:
            asset = client.assets.create(method="direct", file=f, filename=upload_path.name)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            detail = client.assets.retrieve(asset.id)
            if detail.status not in ("processing", "pending", "uploading", "queued"):
                break
            time.sleep(3)
        else:
            return None  # timed out waiting

        if detail.status != "ready":
            return None

        resp = client.analyze(model_name=model_name, video={"type": "asset_id", "asset_id": asset.id},
                               prompt=prompt, max_tokens=512)
        return resp.data
    except Exception as e:
        print(f"  (scene description skipped -- TwelveLabs call failed: {e})")
        return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python scene_description.py <video_path>")
        sys.exit(1)
    desc = describe_scene(sys.argv[1])
    print(desc or "(no API key configured, or processing failed/timed out -- skipped)")
