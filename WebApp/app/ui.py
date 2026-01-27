import gradio as gr
import os
import numpy as np

def create_ui(inference_logic_fn):
    with gr.Blocks(title="Glaucoma Detection") as ui:
        gr.Markdown("# Glaucoma Detection - 3D CNN")
        gr.Markdown("Upload an optic-nerve head centered OCT Volume Scan.")
        
        volume_state = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload .npy Volume", file_types=[".npy"])
                submit_btn = gr.Button("Run Screening", variant="primary")
                
                # Results Display
                out_pred = gr.Textbox(label="Prediction")
                out_prob = gr.Number(label="Probability")
                out_thresh = gr.Number(label="Threshold")
                out_filename = gr.Textbox(label="Filename")
                
        # Run Inference
        gr.Markdown('---')
        submit_btn.click(
            fn=inference_logic_fn,
            inputs=file_input,
            outputs=[out_filename, out_prob, out_pred, out_thresh]
        )
        
    return ui

