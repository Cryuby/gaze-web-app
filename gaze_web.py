# -*- coding: utf-8 -*-
"""
Browser UI for the Gaze Analysis Tool -- cloud-hosted version (Streamlit Community
Cloud). This is the same run_analysis() pipeline as the local desktop version, but
deployed so anyone with the link can use it with no install. Uploaded video + gaze
CSVs ARE sent to this server to run the analysis (unlike the local version, which
never leaves the clinician's own computer) -- see the in-app notice below.
"""
import os
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "gaze_tool"))

import streamlit as st

# Streamlit Cloud secrets (set in the app's dashboard, never committed to the repo)
# aren't OS env vars by default -- bridge it so scene_description._load_api_key()
# (which checks TWELVELABS_API_KEY) picks it up unchanged, same as the local version.
try:
    if "TWELVELABS_API_KEY" in st.secrets:
        os.environ["TWELVELABS_API_KEY"] = st.secrets["TWELVELABS_API_KEY"]
except Exception:
    pass

from analyze import run_analysis
from gaze_matching import SCENE_DIR, VR_PHASE_DIR, load_box_table, load_gaze

UPLOAD_DIR = Path(tempfile.gettempdir()) / "gaze_tool_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def save_upload(uploaded_file):
    """Writes an st.file_uploader result to a real path on disk (run_analysis needs a
    path, not an in-memory buffer) -- keyed by name+size so re-running doesn't rewrite
    a multi-GB video on every widget interaction."""
    if uploaded_file is None:
        return None
    dest = UPLOAD_DIR / f"{uploaded_file.size}_{uploaded_file.name}"
    if not dest.exists():
        with st.spinner(f"Saving {uploaded_file.name}..."):
            dest.write_bytes(uploaded_file.getvalue())
    return dest

PARTICIPANTS = {
    "Participant 1 (ASD group)": VR_PHASE_DIR / "1" / "Participant 1 Eye tracking data.csv",
    "Participant 21 (comparison group)": VR_PHASE_DIR / "eyegaze_2026-04-27_13-19-47-398.csv",
}
EXAMPLE_INSTRUCTION = "what percent of time did they look at people, give me a heatmap, and when did they look at people"


@st.cache_resource(show_spinner="Loading known scenes (first time only -- this reads from Google Drive and can take a minute)...")
def available_scenes():
    """[] if our bundled project data isn't present on this machine (e.g. someone only
    copied the code) -- that just means "known example" mode is unavailable, not a
    crash; "upload your own files" mode doesn't need this data at all."""
    try:
        boxes = load_box_table()
        return sorted(boxes["scene"].unique())
    except Exception:
        return []


def render_result(r):
    kind = r.get("kind")
    if kind == "scene_summary":
        with st.container(border=True):
            st.markdown("**What happened in the scene**")
            if r.get("scene_description"):
                st.write(r["scene_description"])
            else:
                st.caption("Scene description unavailable (TwelveLabs API key not configured on this machine).")
            st.markdown("**Where they looked**")
            st.write(r["gaze_summary"])
    elif kind == "heatmap":
        if "output_png" in r:
            st.image(r["output_png"], caption="Gaze heatmap")
        else:
            st.warning(r.get("message", "Could not render heatmap."))
    elif kind == "proportion":
        st.metric(f"Gaze time on '{r['target']}'", f"{r['pct']:.1f}%", help=f"n={r['n_samples']} samples [{r['source']}]")
    elif kind == "episodes":
        c1, c2, c3 = st.columns(3)
        c1.metric("Episodes", r["n_episodes"])
        c2.metric("Per minute", f"{r['episodes_per_min']:.1f}")
        c3.metric("Mean duration", f"{r['mean_duration_s']:.2f}s")
        st.caption(f"Looking at '{r['target']}', mean gap between episodes {r['mean_gap_s']:.2f}s [{r['source']}]")
    elif kind == "timestamps":
        if r.get("moments"):
            st.write(f"{len(r['moments'])} moments looking at '{r['target']}' [{r['source']}]")
            st.dataframe(r["moments"], width="stretch", hide_index=True)
            if r.get("n_filtered_as_noise"):
                st.caption(f"({r['n_filtered_as_noise']} briefer glances omitted as likely tracking noise)")
        else:
            st.warning(f"Never looked at '{r['target']}' in this scene.")
    elif kind == "latency":
        if r.get("n_hits", 0) == 0:
            st.warning(f"Never oriented to '{r['target']}' in this scene.")
        else:
            st.metric(f"Time to first look at '{r['target']}'", f"{r['mean_latency_s']:.2f}s",
                      help=f"hit rate {r['hit_rate']*100:.0f}% [{r['source']}]")
    elif kind == "unsupported":
        st.warning(r["message"])
    else:
        st.info(r.get("message", str(r)))


st.set_page_config(page_title="Gaze Analysis Tool", page_icon="👁️", layout="wide")

st.title("👁️ Gaze Analysis Tool")
st.caption("⚠️ Prototype hosted online: the video and gaze file you upload are sent to this "
           "server to run the analysis (not kept local like the desktop version). Only use "
           "de-identified data, or data already cleared for this kind of processing.")
st.divider()

scenes = available_scenes()
has_known_data = bool(scenes)

left, right = st.columns([1, 1.4], gap="large")

with left:
    with st.container(border=True):
        st.subheader("1. Choose what to analyze")
        if has_known_data:
            source_mode = st.radio("Video + gaze data source", ["Use a known example", "Upload your own files"],
                                    horizontal=True, label_visibility="collapsed")
        else:
            source_mode = "Upload your own files"
            st.caption("No bundled example data found on this machine -- upload your own video + gaze CSV below.")

        video_path, gaze_path, scene = None, None, None
        if source_mode == "Use a known example":
            participant_label = st.selectbox("Participant", list(PARTICIPANTS.keys()))
            scene = st.selectbox("Scene", scenes, index=scenes.index("2.1.C") if "2.1.C" in scenes else 0)
            video_path, gaze_path = SCENE_DIR / f"{scene}.mp4", PARTICIPANTS[participant_label]
        else:
            video_upload = st.file_uploader("Scene video", type=["mp4", "mov", "avi", "mkv"])
            gaze_upload = st.file_uploader("Gaze CSV", type=["csv"])
            if video_upload is not None:
                video_path = save_upload(video_upload)
            if gaze_upload is not None:
                gaze_path = save_upload(gaze_upload)
                detected_scenes = sorted(load_gaze(gaze_path)["scene"].dropna().unique())
                if detected_scenes:
                    scene = st.selectbox("Scene (detected from your gaze file)", detected_scenes)
                else:
                    st.warning("Couldn't detect a scene name in this gaze file's 'filename' column.")

        st.subheader("2. What do you want to know?")
        instruction = st.text_area("Instruction", EXAMPLE_INSTRUCTION, height=90,
                                    help="Say what to pay attention to (e.g. 'look at the window') and what "
                                         "output you want (percent, heatmap, how many times, how fast).")
        ready = video_path is not None and gaze_path is not None and scene is not None
        run_clicked = st.button("▶ Run analysis", type="primary", width="stretch", disabled=not ready)
        if not ready and source_mode == "Upload your own files":
            st.caption("Upload both a video and a gaze CSV to continue.")

with right:
    st.subheader("Result")
    if run_clicked:
        with st.spinner("Analyzing..."):
            results = run_analysis(str(video_path), str(gaze_path), scene, instruction)
        if "error" in results:
            st.error(results["error"])
        else:
            for r in results["results"]:
                render_result(r)
            # output_json is a path on the server's own disk -- useless to the clinician as
            # text, so offer its contents as an actual download instead of just naming it.
            with open(results["output_json"], "rb") as f:
                st.download_button("⬇ Download full result (JSON)", f.read(),
                                    file_name=Path(results["output_json"]).name, mime="application/json")
    else:
        st.info("Pick a participant and scene, enter an instruction, and click **Run analysis**.")
