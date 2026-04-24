# Vision-LLM Pipeline


> ⚠️ **Status: Demo / Work in Progress**  
> This is an early-stage demo built to validate the descriptor layer architecture 
> described in [LinkedIn post : https://www.linkedin.com/feed/update/urn:li:activity:7453489963859329024/](#). Core pipeline works end-to-end. 
> Webcam/video support, expanded semantic maps, and performance benchmarks are in progress.




**YOLOv8 + Descriptor Layer + Groq LLM — Full scene reasoning pipeline**

Built as the implementation behind the LinkedIn post:  
*"Why LLM Reasoning Alone Wasn't Enough for My Computer Vision Pipeline"*

---

## The Problem This Solves

A naive CV + LLM pipeline looks like this:

```

![Vision LLM Demo](src/vision_Ilm_demo.png)

Image → YOLO → "person, bottle, chair" → LLM → shallow reasoning
```

The LLM gets raw labels with no spatial context, no relationships,  
no scene type, no semantic enrichment. This is the **Granularity Gap**.

## The Solution: Descriptor Layer

```
Image → YOLO → [Descriptor Layer] → Structured Context → LLM → deep reasoning
```

The descriptor layer converts raw detections into:
- Spatial zones (where each object is in the frame)
- Proximity relationships (which objects are near each other)
- Semantic roles (what each object *means*, not just what it *is*)
- Scene classification (traffic scene, office, indoor, etc.)
- Temporal state (what's new, persistent, disappeared across frames)

## Architecture

```
core/
  detector.py    — YOLOv8 detection wrapper
  descriptor.py  — The descriptor / context builder (key contribution)
  reasoner.py    — Groq LLM integration
  pipeline.py    — Orchestrator
app.py           — Streamlit demo UI
```

## Setup

```bash
pip install -r requirements.txt
```

Get a free Groq API key: https://console.groq.com

## Run

```bash
streamlit run app.py
```

Enter your Groq API key in the sidebar, upload any image, click Run.

The app shows a **side-by-side comparison**:
- ❌ Naive: raw labels sent directly to LLM
- ✅ Full pipeline: descriptor layer output sent to LLM

## The Three Bottlenecks (from the post)

| Bottleneck | Problem | Solution in this code |
|---|---|---|
| Granularity Gap | "device" vs "status" — labels aren't rich enough | `descriptor.py` semantic map + role enrichment |
| Latency | LLM inference killed FPS | Decoupled: YOLO runs every frame, LLM on demand |
| Temporal Blindness | No state across frames | `_frame_history` + temporal summary in descriptor |

## Model

- YOLO: `yolov8n` by default (fastest, CPU-friendly)
- LLM: `llama-3.3-70b-versatile` via Groq (fast, free tier available)

## Author

Kartik Patil · AI & ML Engineer  
LinkedIn: [linkedin.com/in/kartikpatil](https://linkedin.com/in/kartikpatil24)
