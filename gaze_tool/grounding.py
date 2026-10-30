# -*- coding: utf-8 -*-
"""
Resolves a clinician's free-text target phrase ("people", "the window") to a
set of bounding boxes: closed-vocab first (reuse the box tables we already
computed this project -- zero inference), open-vocab fallback via
Ultralytics' YOLO-World (already installed, no new dependency, runs on the
local Apple-Silicon GPU) for anything outside our known classes.

CLASS_ALIASES is seeded from run_yolo_framework_best_batch.py's
SOCIAL_KEYWORDS/INANIMATE_KEYWORDS (that project's own taxonomy), mapped to
the box_class strings that actually appear in our detection tables.
"""
import cv2
import pandas as pd

CLASS_ALIASES = {
    "person": "person", "people": "person", "human": "person", "man": "person", "woman": "person",
    "boy": "person", "girl": "person", "face": "person", "head": "person", "body": "person",
    "table": "dining table", "dining table": "dining table",
    "tv": "tv", "screen": "tv", "monitor": "tv", "display": "tv",
    "chair": "chair", "stool": "chair",
    "bench": "bench",
    "bag": "backpack", "backpack": "backpack",
    "handbag": "handbag", "purse": "handbag",
    "umbrella": "umbrella",
    "door": "revolving door", "revolving door": "revolving door",
    "cup": "cup", "laptop": "laptop", "microwave": "microwave", "bottle": "bottle",
    "skateboard": "skateboard", "elevator": "elevator",
    "car": "car", "bus": "bus", "suitcase": "suitcase", "skis": "skis",
}


def resolve_target(phrase, scene, box_table):
    """(box_classes, source) -- box_classes is a set if resolved to the closed-vocab table
    (source="closed_vocab"), or None if an open-vocab YOLO-World run is needed (source="open_vocab")."""
    norm = phrase.strip().lower()
    aliased = CLASS_ALIASES.get(norm, norm)
    scene_classes = set(box_table[box_table["scene"] == scene]["box_class"].unique())
    if aliased in scene_classes:
        return {aliased}, "closed_vocab"
    return None, "open_vocab"


def run_yolo_world(video_path, phrase, scene, fps_stride_s=1.0, model_name="yolov8s-world.pt"):
    """Open-vocabulary detection: samples the video at ~1 frame/fps_stride_s, runs YOLO-World
    prompted with `phrase`, returns a box table in the same shape as gaze_matching.load_box_table().

    video_path can be any resolution (the full-res original, or a downscaled copy from
    scene_description.prepare_video_for_analysis -- large uploads get downscaled before this
    runs, since decoding a multi-GB video frame-by-frame is the slow part). Box coordinates are
    rescaled to gaze_matching.VIDEO_W/VIDEO_H's fixed space either way, since that's the space
    gaze points get converted into for matching (metrics.py: `FrameX * VIDEO_W`)."""
    from ultralytics import YOLO
    from gaze_matching import VIDEO_W, VIDEO_H
    model = YOLO(model_name)
    model.set_classes([phrase])

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or VIDEO_W
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or VIDEO_H
    scale_x, scale_y = VIDEO_W / frame_w, VIDEO_H / frame_h
    step = max(1, round(fps * fps_stride_s))

    rows = []
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % step == 0:
            result = model.predict(frame, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append(dict(scene=scene, frame_id=frame_id,
                                  box_x1=x1 * scale_x, box_y1=y1 * scale_y,
                                  box_x2=x2 * scale_x, box_y2=y2 * scale_y,
                                  box_confidence=float(box.conf[0]), box_class=phrase))
        frame_id += 1
    cap.release()
    return pd.DataFrame(rows, columns=["scene", "frame_id", "box_x1", "box_y1", "box_x2", "box_y2",
                                        "box_confidence", "box_class"])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gaze_matching import load_box_table

    boxes = load_box_table()
    classes, source = resolve_target("people", "2.1.C", boxes)
    print(f"'people' in scene 2.1.C -> classes={classes} source={source}")
    assert source == "closed_vocab" and classes == {"person"}

    classes, source = resolve_target("the window", "2.1.C", boxes)
    print(f"'the window' in scene 2.1.C -> classes={classes} source={source}")
    assert source == "open_vocab" and classes is None
    print("OK")
