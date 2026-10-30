# -*- coding: utf-8 -*-
"""
Heatmap rendering, lifted from generate_asd_heatmaps.py's render_heatmap(),
generalized to take an arbitrary (xs, ys) gaze subset instead of always a
full participant-scene stream.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

BASE = Path(__file__).resolve().parent.parent  # YOLO/
SCENES_DIR = BASE / "VR Autism Project" / "DePaul loop images, videos" / "1" / "Scenes jpg"

FIGSIZE = (18, 9)
DPI = 300
CMAP = "coolwarm"
ALPHA = 0.55
N_LEVELS = 60
DENSITY_THRESH = 0.04
GRID_N = 300j
MAX_POINTS = 20000
MIN_POINTS = 30


def _grab_frame(video_path):
    """Mid-video frame as a PIL Image, for scenes with no bundled background JPG
    (e.g. a clinician's own uploaded video) -- None if the video can't be read."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)  # mid-video, not a black/blank intro frame
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def render_heatmap(scene, xs_norm, ys_norm, out_path, video_path=None):
    """xs_norm/ys_norm: 0-1 normalized FrameX/FrameY arrays. Background is the bundled scene
    JPG if we have one (known scenes), else a frame grabbed straight from video_path (needed
    for an uploaded video we have no pre-rendered image for). Returns False if too few points
    or no image source is available."""
    if len(xs_norm) < MIN_POINTS:
        return False

    img_path = SCENES_DIR / f"{scene}.jpg"
    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
    elif video_path is not None:
        img = _grab_frame(video_path)
        if img is None:
            return False
    else:
        return False
    img_w, img_h = img.size
    xs, ys = np.asarray(xs_norm) * img_w, np.asarray(ys_norm) * img_h

    if len(xs) > MAX_POINTS:
        idx = np.random.RandomState(0).choice(len(xs), MAX_POINTS, replace=False)
        xs, ys = xs[idx], ys[idx]

    kde = gaussian_kde(np.vstack([xs, ys]))
    xx, yy = np.mgrid[0:img_w:GRID_N, 0:img_h:GRID_N]
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    density = density / density.max()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.imshow(np.asarray(img))
    ax.contourf(xx, yy, density, levels=np.linspace(DENSITY_THRESH, 1.0, N_LEVELS), cmap=CMAP, alpha=ALPHA,
                extend="neither")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gaze_matching import load_gaze, VR_PHASE_DIR

    gdf = load_gaze(VR_PHASE_DIR / "1" / "Participant 1 Eye tracking data.csv")
    sub = gdf[gdf["scene"] == "2.1.C"]
    out = BASE / "codex_outputs" / "gaze_tool_smoke_heatmap.png"
    ok = render_heatmap("2.1.C", sub["FrameX"].to_numpy(), sub["FrameY"].to_numpy(), out)
    print(f"render_heatmap -> {ok}, wrote {out if ok else '(skipped, too few points)'}")
    assert ok and out.stat().st_size > 10_000
    print("OK")
