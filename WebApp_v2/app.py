"""
OCT Glaucoma Screening Demo — Hugging Face Spaces (Gradio)

Three 3D CNN architectures for feature-agnostic glaucoma detection from OCT volumes.
Accepts .npy files of shape (64, 128, 64) uint8.
"""

import os
import numpy as np
import tensorflow as tf
import gradio as gr
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_volumes")

MODELS_INFO = {
    "Sequential CNN (AUC 0.88)": {
        "file": "model_1.keras",
        "description": "5× Conv3D blocks → GlobalAvgPool → Dense. Replicates Maetschke et al. (2019).",
    },
    "ResNet-Like (AUC 0.94)": {
        "file": "model_2.keras",
        "description": "Residual skip connections between conv blocks. Best overall performance.",
    },
    "Attention Network (AUC 0.93)": {
        "file": "model_3.keras",
        "description": "SE (channel) + spatial attention. Recalibrates feature importance.",
    },
}

EXPECTED_SHAPE = (64, 128, 64)

# ---------------------------------------------------------------------------
# Model loading (lazy, cached)
# ---------------------------------------------------------------------------

_model_cache = {}


def load_model(model_name: str) -> tf.keras.Model:
    """Load and cache a model by display name."""
    if model_name not in _model_cache:
        info = MODELS_INFO[model_name]
        path = os.path.join(MODEL_DIR, info["file"])
        _model_cache[model_name] = tf.keras.models.load_model(path)
    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_volume(volume: np.ndarray, model_name: str) -> dict:
    """
    Run inference on a single OCT volume.

    Parameters
    ----------
    volume : np.ndarray, shape (64, 128, 64), dtype uint8
    model_name : str
        Key into MODELS_INFO.

    Returns
    -------
    dict with keys: probability, prediction, threshold, model_description
    """
    model = load_model(model_name)

    # Preprocess: add batch + channel dims, normalize to [0, 1]
    x = volume.astype(np.float32)[np.newaxis, ..., np.newaxis] / 255.0

    prob = float(model.predict(x, verbose=0)[0])

    # Youden's J threshold (precomputed from validation; fallback 0.5)
    threshold = 0.5
    label = "Glaucoma" if prob >= threshold else "Normal"

    return {
        "probability": prob,
        "prediction": label,
        "threshold": threshold,
        "model_description": MODELS_INFO[model_name]["description"],
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def build_slice_viewer(volume: np.ndarray, axis: int = 1, slice_idx: int = 0) -> go.Figure:
    """
    Build a Plotly figure showing a single slice of the volume.

    Parameters
    ----------
    volume : np.ndarray, shape (64, 128, 64)
    axis : int (0, 1, or 2)
    slice_idx : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if axis == 0:
        img = volume[slice_idx, :, :]
        n_slices = volume.shape[0]
    elif axis == 1:
        img = volume[:, slice_idx, :]
        n_slices = volume.shape[1]
    else:
        img = volume[:, :, slice_idx]
        n_slices = volume.shape[2]

    fig = go.Figure(
        data=go.Heatmap(
            z=img,
            colorscale="Gray",
            showscale=False,
            hovertemplate="x: %{x}<br>y: %{y}<br>intensity: %{z}<extra></extra>",
        )
    )

    h, w = img.shape
    fig.update_layout(
        yaxis=dict(autorange="reversed", scaleanchor="x", constrain="domain"),
        xaxis=dict(constrain="domain"),
        width=max(400, w * 3),
        height=max(300, h * 3),
        margin=dict(l=20, r=20, t=40, b=20),
        title=f"Axis {axis} — Slice {slice_idx}/{n_slices - 1}",
    )

    return fig


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------


def load_volume_from_file(file_obj):
    """Load a .npy file and return the volume + metadata string."""
    if file_obj is None:
        return None, "No file uploaded."

    path = file_obj.name if hasattr(file_obj, "name") else file_obj
    volume = np.load(path)

    if volume.shape != EXPECTED_SHAPE:
        return None, f"❌ Shape mismatch: got {volume.shape}, expected {EXPECTED_SHAPE}"

    info = (
        f"✅ Loaded: {os.path.basename(path)}\n"
        f"Shape: {volume.shape} | dtype: {volume.dtype}\n"
        f"Range: [{volume.min()}, {volume.max()}]"
    )
    return volume, info


def load_sample_volume(sample_name):
    """Load one of the bundled sample volumes."""
    if not sample_name:
        return None, "No sample selected."

    path = os.path.join(SAMPLE_DIR, sample_name)
    volume = np.load(path)
    info = (
        f"✅ Loaded sample: {sample_name}\n"
        f"Shape: {volume.shape} | dtype: {volume.dtype}\n"
        f"Range: [{volume.min()}, {volume.max()}]"
    )
    return volume, info


def update_slice_viewer(volume, axis, slice_idx):
    """Update the slice plot when axis or slider changes."""
    if volume is None:
        return go.Figure()
    axis = int(axis)
    slice_idx = int(slice_idx)
    n_slices = volume.shape[axis]
    slice_idx = min(slice_idx, n_slices - 1)
    return build_slice_viewer(volume, axis, slice_idx)


def get_max_slices(volume, axis):
    """Return the max slider value for the selected axis."""
    if volume is None:
        return 63
    return volume.shape[int(axis)] - 1


def run_screening(volume, model_name):
    """Run inference and format results."""
    if volume is None:
        return "No volume loaded.", "", "", ""

    result = predict_volume(volume, model_name)

    prediction_display = (
        f"🔴 **{result['prediction']}**"
        if result["prediction"] == "Glaucoma"
        else f"🟢 **{result['prediction']}**"
    )

    prob_pct = f"{result['probability'] * 100:.1f}%"
    details = (
        f"**Model**: {model_name}\n"
        f"**Architecture**: {result['model_description']}\n"
        f"**Threshold**: {result['threshold']}"
    )

    return prediction_display, prob_pct, f"{result['probability']:.6f}", details


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------


def create_app():
    """Construct and return the Gradio Blocks app."""

    # Discover sample volumes
    sample_files = []
    if os.path.isdir(SAMPLE_DIR):
        sample_files = sorted(
            [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".npy")]
        )

    with gr.Blocks(
        title="OCT Glaucoma Screening",
        theme=gr.themes.Soft(primary_hue="indigo"),
    ) as app:

        # --- Header ---
        gr.Markdown(
            """
            # 👁️ OCT Glaucoma Screening
            **Feature-agnostic glaucoma detection from 3D OCT volumes using deep learning**

            Upload a preprocessed `.npy` volume (shape 64×128×64, uint8) or select a sample below.
            """
        )

        # Hidden state for loaded volume
        volume_state = gr.State(None)

        with gr.Row():
            # ===== LEFT COLUMN: Input & Visualization =====
            with gr.Column(scale=2):
                gr.Markdown("### 📁 Load Volume")

                with gr.Tab("Upload File"):
                    file_input = gr.File(
                        label="Upload .npy volume",
                        file_types=[".npy"],
                        type="filepath",
                    )

                with gr.Tab("Sample Volumes"):
                    sample_dropdown = gr.Dropdown(
                        choices=sample_files,
                        label="Select sample",
                        info="Pre-loaded OCT volumes for testing",
                    )

                volume_info = gr.Textbox(
                    label="Volume Info",
                    interactive=False,
                    lines=3,
                )

                gr.Markdown("### 🔬 Slice Viewer")

                with gr.Row():
                    axis_radio = gr.Radio(
                        choices=["0", "1", "2"],
                        value="1",
                        label="Slicing Axis",
                        info="0=depth, 1=height (B-scan), 2=width",
                    )
                    slice_slider = gr.Slider(
                        minimum=0,
                        maximum=127,
                        step=1,
                        value=64,
                        label="Slice Index",
                    )

                slice_plot = gr.Plot(label="OCT Slice")

            # ===== RIGHT COLUMN: Model & Results =====
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Model Selection")

                model_dropdown = gr.Dropdown(
                    choices=list(MODELS_INFO.keys()),
                    value="ResNet-Like (AUC 0.94)",
                    label="Architecture",
                    info="Choose which trained model to use for screening",
                )

                submit_btn = gr.Button(
                    "🔍 Run Screening",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("### 📊 Results")

                prediction_md = gr.Markdown("*Awaiting scan...*")
                probability_text = gr.Textbox(
                    label="Glaucoma Probability", interactive=False
                )
                raw_prob = gr.Textbox(label="Raw Score", interactive=False)
                details_md = gr.Markdown("")

                gr.Markdown(
                    """
                    ---
                    ⚠️ **Disclaimer**: This is a research demo, not a medical device.
                    Predictions are not clinical diagnoses. Always consult a qualified
                    ophthalmologist for glaucoma screening.
                    """
                )

        # --- Event wiring ---

        # Upload file → load volume
        file_input.change(
            fn=load_volume_from_file,
            inputs=[file_input],
            outputs=[volume_state, volume_info],
        ).then(
            fn=update_slice_viewer,
            inputs=[volume_state, axis_radio, slice_slider],
            outputs=[slice_plot],
        )

        # Sample dropdown → load volume
        sample_dropdown.change(
            fn=load_sample_volume,
            inputs=[sample_dropdown],
            outputs=[volume_state, volume_info],
        ).then(
            fn=update_slice_viewer,
            inputs=[volume_state, axis_radio, slice_slider],
            outputs=[slice_plot],
        )

        # Axis change → update max slider + re-render
        axis_radio.change(
            fn=get_max_slices,
            inputs=[volume_state, axis_radio],
            outputs=[slice_slider],
        ).then(
            fn=update_slice_viewer,
            inputs=[volume_state, axis_radio, slice_slider],
            outputs=[slice_plot],
        )

        # Slice slider → re-render
        slice_slider.release(
            fn=update_slice_viewer,
            inputs=[volume_state, axis_radio, slice_slider],
            outputs=[slice_plot],
        )

        # Run screening button
        submit_btn.click(
            fn=run_screening,
            inputs=[volume_state, model_dropdown],
            outputs=[prediction_md, probability_text, raw_prob, details_md],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
