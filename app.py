"""
app.py
------
Vision → Descriptor → LLM Pipeline
Streamlit demo app.

Shows the full pipeline with side-by-side comparison:
LEFT:  Naive (raw labels → LLM) — demonstrates the Granularity Gap
RIGHT: Full pipeline (descriptor layer → LLM) — the solution

Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from core.pipeline import VisionLLMPipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision-LLM Pipeline",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stMetric label { font-size: 0.78rem; color: #888; }
  .stage-box {
    background: #1a1a2e;
    border: 1px solid #2d2d44;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-family: monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
    color: #e0e0e0;
    max-height: 320px;
    overflow-y: auto;
  }
  .naive-box {
    background: #1f0a0a;
    border: 1px solid #5c1a1a;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-family: monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
    color: #f0d0d0;
    max-height: 320px;
    overflow-y: auto;
  }
  .full-box {
    background: #0a1f0f;
    border: 1px solid #1a5c2a;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-family: monospace;
    font-size: 0.82rem;
    white-space: pre-wrap;
    color: #d0f0d8;
    max-height: 320px;
    overflow-y: auto;
  }
  .badge-naive { background:#5c1a1a; color:#f0d0d0; padding:2px 10px; border-radius:20px; font-size:0.78rem; }
  .badge-full  { background:#1a5c2a; color:#d0f0d8; padding:2px 10px; border-radius:20px; font-size:0.78rem; }
  .badge-stage { background:#1a2a5c; color:#d0d8f0; padding:2px 10px; border-radius:20px; font-size:0.78rem; }
  h1 { font-size: 1.6rem !important; }
  h2 { font-size: 1.2rem !important; }
  h3 { font-size: 1.0rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔭 Vision → Descriptor → LLM Pipeline")
st.caption(
    "Full pipeline: YOLOv8 detection → descriptor layer → Groq LLM reasoning. "
    "Demonstrates the **Granularity Gap** and how the descriptor layer solves it."
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )

    st.divider()

    yolo_model = st.selectbox(
        "YOLO Model",
        ["yolov8n", "yolov8s", "yolov8m"],
        index=0,
        help="n=fastest (nano), s=small, m=medium"
    )

    conf_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1, max_value=0.9,
        value=0.35, step=0.05
    )

    proximity_pct = st.slider(
        "Proximity Threshold (% of diagonal)",
        min_value=5, max_value=30,
        value=15, step=1,
        help="Objects within this % of image diagonal are flagged as 'near' each other"
    )

    st.divider()
    st.markdown("**Pipeline stages:**")
    st.markdown("1️⃣ YOLOv8 Detection")
    st.markdown("2️⃣ Descriptor Layer ← *the missing piece*")
    st.markdown("3️⃣ Groq LLM Reasoning")
    st.divider()
    st.markdown("**Compare:**")
    st.markdown('<span class="badge-naive">❌ Naive</span> Raw labels → LLM', unsafe_allow_html=True)
    st.markdown('<span class="badge-full">✅ Full</span> Descriptor → LLM', unsafe_allow_html=True)


# ── Pipeline init (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline(api_key, model):
    return VisionLLMPipeline(groq_api_key=api_key, yolo_model=model)


# ── Image upload ───────────────────────────────────────────────────────────────
st.subheader("📤 Upload Image")

uploaded = st.file_uploader(
    "Upload any image (street, office, indoor, outdoor...)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Try busy scenes with multiple objects for best demo results"
)

col_preview, col_info = st.columns([2, 1])
with col_preview:
    if uploaded:
        st.image(uploaded, caption="Input image", use_container_width=True)
    else:
        st.info("👆 Upload an image to begin. Try a street scene, office, or any photo with multiple objects.")

with col_info:
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        w, h = img_pil.size
        st.metric("Width", f"{w}px")
        st.metric("Height", f"{h}px")
        st.metric("Mode", img_pil.mode)


# ── Run button ─────────────────────────────────────────────────────────────────
st.divider()

run_disabled = not uploaded or not groq_api_key
if not groq_api_key:
    st.warning("⚠️ Enter your Groq API key in the sidebar to enable pipeline execution.")

run_clicked = st.button(
    "▶️ Run Full Pipeline",
    disabled=run_disabled,
    use_container_width=True,
    type="primary"
)


# ── Pipeline execution ─────────────────────────────────────────────────────────
if run_clicked and uploaded and groq_api_key:

    img_pil = Image.open(uploaded).convert("RGB")
    img_np  = np.array(img_pil)

    pipeline = get_pipeline(groq_api_key, yolo_model)
    # Update proximity threshold live
    pipeline.descriptor.proximity_threshold_pct = proximity_pct

    with st.spinner("Running pipeline... (Stage 1: YOLO → Stage 2: Descriptor → Stage 3: Groq)"):
        results = pipeline.run(img_np, conf_threshold=conf_threshold)

    # ── Stage 1: Detection results ─────────────────────────────────────────
    st.subheader("🟦 Stage 1 — YOLOv8 Detection")

    col1, col2 = st.columns([2, 1])
    with col1:
        ann_img = Image.fromarray(results["annotated_image"])
        st.image(ann_img, caption="Annotated detections", use_container_width=True)

    with col2:
        st.metric("Objects Found", results["context"]["object_count"])
        st.metric("Detection Time", f"{results['detection_ms']} ms")
        st.metric("Scene Type", results["context"]["scene_type"].split("(")[0].strip())

        st.markdown("**Raw labels (what LLM sees naive):**")
        st.markdown(f'<div class="naive-box">{results["raw_labels"]}</div>', unsafe_allow_html=True)

    # Detection table
    with st.expander("📋 Full detection data", expanded=False):
        if results["detections"]:
            det_data = []
            for d in results["detections"]:
                det_data.append({
                    "Label": d["label"],
                    "Confidence": f"{d['confidence']:.0%}",
                    "Zone": d.get("spatial_zone", "—"),
                    "Size": d.get("size_descriptor", "—"),
                    "Area %": d["area_pct"],
                    "BBox": str(d["bbox"]),
                })
            import pandas as pd
            st.dataframe(det_data, use_container_width=True)
        else:
            st.info("No objects detected above confidence threshold.")

    st.divider()

    # ── Stage 2: Descriptor layer output ──────────────────────────────────
    st.subheader("🟨 Stage 2 — Descriptor Layer (The Missing Piece)")
    st.caption("This is the intermediate layer that converts raw YOLO output into structured temporal context before the LLM sees it.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Spatial layout:**")
        layout = results["context"]["spatial_layout"]
        for zone, objs in layout.items():
            if objs:
                st.markdown(f"- **{zone}**: {', '.join(objs)}")

        if results["context"]["relationships"]:
            st.markdown("**Proximity relationships:**")
            for rel in results["context"]["relationships"]:
                st.markdown(f"- {rel}")
        else:
            st.markdown("*No close-proximity relationships detected.*")

    with col_b:
        st.markdown("**Enriched objects:**")
        for obj in results["context"]["objects"]:
            risk = ", ".join(obj["risk_flags"]) if obj["risk_flags"] else "—"
            st.markdown(
                f"- **{obj['label']}** → *{obj['role']}* | "
                f"{obj['spatial_zone']} | {obj['size_descriptor']} | risks: {risk}"
            )

    with st.expander("📋 Full context prompt sent to LLM", expanded=False):
        st.markdown(
            f'<div class="stage-box">{results["context_prompt"]}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Stage 3: LLM Comparison ────────────────────────────────────────────
    st.subheader("🟥🟩 Stage 3 — LLM Reasoning Comparison")
    st.caption("Same LLM (Groq llama-3.3-70b), same scene. Different inputs. See the quality difference.")

    col_naive, col_full = st.columns(2)

    with col_naive:
        st.markdown('<span class="badge-naive">❌ NAIVE — Raw Labels → LLM</span>', unsafe_allow_html=True)
        st.caption(f"Input: `{results['raw_labels'][:80]}...`")

        naive = results["naive_reasoning"]
        if naive["success"]:
            st.markdown(
                f'<div class="naive-box">{naive["response"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.error(naive["response"])

        st.metric("Latency", f"{naive['latency_ms']} ms")
        if naive.get("tokens_used"):
            st.metric("Tokens", naive["tokens_used"])

    with col_full:
        st.markdown('<span class="badge-full">✅ FULL PIPELINE — Descriptor → LLM</span>', unsafe_allow_html=True)
        st.caption("Input: Structured temporal context block (see Stage 2)")

        full = results["full_reasoning"]
        if full["success"]:
            st.markdown(
                f'<div class="full-box">{full["response"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.error(full["response"])

        st.metric("Latency", f"{full['latency_ms']} ms")
        if full.get("tokens_used"):
            st.metric("Tokens", full["tokens_used"])

    st.divider()

    # ── Summary metrics ────────────────────────────────────────────────────
    st.subheader("📊 Pipeline Metrics")
    total_ms = results["detection_ms"] + full["latency_ms"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("YOLO Detection", f"{results['detection_ms']} ms")
    m2.metric("Descriptor Build", "< 5 ms", help="Pure Python, negligible overhead")
    m3.metric("Groq Inference", f"{full['latency_ms']} ms")
    m4.metric("Total Pipeline", f"{total_ms:.0f} ms")
    m5.metric("Objects Enriched", results["context"]["object_count"])


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "Built by **Kartik Patil** · "
    "Pipeline: YOLOv8 + Custom Descriptor Layer + Groq LLaMA 3.3 · "
    "[GitHub](https://github.com) · "
    "[LinkedIn](https://linkedin.com)",
    unsafe_allow_html=False
)
