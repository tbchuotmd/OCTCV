import gradio as gr
import os
import numpy as np
import plotly.graph_objects as go

SAMPLE_DATA = os.path.join(os.path.dirname(__file__),'../sample_volumes/')

def sliceSliderPlot(datapath,axis=0):
    if datapath is None:
        return
    path = datapath.name if hasattr(datapath, 'name') else datapath
    data = np.load(path)
    
    fig = go.Figure()

    match axis:
        case 0: 
            zdata = lambda data, i: data[i, :, :]
        case 1: 
            zdata = lambda data, i: data[:, i, :]
        case 2: 
            zdata = lambda data, i: data[:, :, i]
        case _: raise ValueError('Invalid axis.')

    # Get Slice Dimensions
    h,w = zdata(data, axis).shape
    hwratio = h/w
    adjhwr = hwratio * 0.75
    if hwratio > 1:
        W = 300
        H = int(W*adjhwr)
    elif hwratio < 1:
        H = 300
        W = int(H/hwratio)
    else:
        W = 400
        H = 400
    
    for i in range(data.shape[axis]):
        fig.add_trace(
            go.Heatmap(
                visible=False,
                z=zdata(data, i),
                colorscale='Viridis',
                showscale=False
            )
        )
    
    fig.data[0].visible = True
    
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Slice: {i}"}
                  ],
            label=f"{i}"
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    fig.update_layout(
        sliders=[dict(active=0, steps=steps)],
        yaxis=dict(autorange='reversed'),
        width=W,
        height=H
    )
    
    return fig # Crucial for Gradio

def create_ui(inference_logic_fn):
    with gr.Blocks(title="Glaucoma Detection") as ui:
        gr.Markdown("# Glaucoma Detection - 3D CNN")
        gr.Markdown("Upload an optic-nerve head centered OCT Volume Scan.")

        volume_state = gr.State(None)
        
        local_files = [f for f in os.listdir(SAMPLE_DATA) if f.endswith(".npy")]
        local_files = [os.path.join(SAMPLE_DATA,f) for f in local_files]
        local_files = [os.path.relpath(f,SAMPLE_DATA) for f in local_files]

        with gr.Row():
            with gr.Column(scale=1):
                file_dropdown = gr.Dropdown(
                    choices=local_files, 
                    label="Select from Sample Volumes", 
                    info="Choose a pre-loaded volume"
                )               
                
                file_input = gr.File(
                    label="Upload .npy Volume",
                    file_types=[".npy"]
                )
                
                axis_input = gr.Radio(choices=[0, 1, 2], label="Select Slicing Axis", value=0)
                
                plot_output = gr.Plot()
                
            with gr.Column(scale=1):
                submit_btn = gr.Button("Run Screening", variant="primary")
                out_pred = gr.Textbox(label="Prediction")
                out_prob = gr.Number(label="Probability")
                out_thresh = gr.Number(label="Threshold")
                out_filename = gr.Textbox(label="Filename")

        # --- Event Listeners ---

        # Trigger 1: Auto-plot when file is uploaded
        file_input.change(
            fn=sliceSliderPlot,
            inputs=[file_input, axis_input],
            outputs=plot_output
        )

        # Trigger 2: Re-plot if user switches axis
        axis_input.change(
            fn=sliceSliderPlot,
            inputs=[file_input, axis_input],
            outputs=plot_output
        )

        # Trigger 3: Existing Inference logic
        submit_btn.click(
            fn=inference_logic_fn,
            inputs=file_input,
            outputs=[out_filename, out_prob, out_pred, out_thresh, volume_state]
        )

    return ui

