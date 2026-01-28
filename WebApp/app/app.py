from fastapi import FastAPI,HTTPException,File,UploadFile
import tensorflow as tf
import numpy as np
import io,os,sys
import gradio as gr

sys.path.append(os.path.dirname(__file__))
from ui import create_ui

app = FastAPI()

THIS_FOLDER = os.path.dirname(__file__)
MODEL_PATH = os.path.join(THIS_FOLDER,'../models/model-2.keras')
model = tf.keras.models.load_model(MODEL_PATH)

THRESHOLD = 0.5 #0.7815

@app.post("/")
async def predict(file: UploadFile = File(...)):
    # 1. Validate file extension
    if not file.filename.endswith('.npy'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .npy file.")

    try:
        # 2. Read the file bytes into memory
        contents = await file.read()
        
        # 3. Load the numpy array
        # Use io.BytesIO to treat the byte string as a file-like object
        data = np.load(io.BytesIO(contents))

        # 4. Validate input shape
        if data.shape != (64, 128, 64):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid input shape. Expected (64, 128, 64), got {data.shape}"
            )

        # 5. Preprocessing: Reshape from (64, 128, 64) to (1, 64, 128, 64, 1)
        # The model expects [Batch, Depth, Height, Width, Channels]
        processed_data = data[np.newaxis, ..., np.newaxis]
        
        # Normalize Pixel Intensities
        if max(processed_data.flatten()) > 1:
            processed_data = processed_data / 255

        # 6. Run Inference
        prediction = model.predict(processed_data)
        probability = float(prediction[0])
        
        # 7. Apply Threshold Logic
        label = "Glaucoma" if probability >= THRESHOLD else "Normal"

        return {
            "filename": file.filename,
            "probability": probability,
            "prediction": label,
            "threshold_used": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# --- Gradio Wrapper Logic ---
async def gradio_wrapper(file_obj, progress=gr.Progress()):
    if file_obj is None:
        return None, None, None, None, None

    progress(0.2, desc="Loading data...")

    # Load volume for UI
    volume = np.load(file_obj.name)

    with open(file_obj.name, "rb") as f:
        from starlette.datastructures import UploadFile as StarletteUploadFile
        mock_file = StarletteUploadFile(
            filename=os.path.basename(file_obj.name),
            file=f
        )

        progress(0.5, desc="Analyzing OCT Scan...")
        result = await predict(mock_file)

    progress(1.0, desc="Complete")

    return (
        result["filename"],
        result["probability"],
        result["prediction"],
        result["threshold_used"],
        volume            
    )


# Mount the UI
glaucoma_ui = create_ui(gradio_wrapper)
app = gr.mount_gradio_app(app, glaucoma_ui, path="/")

