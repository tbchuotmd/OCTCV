# Deployment to Hugging Face Spaces

## Quick Start

1. **Create a Hugging Face Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `oct-glaucoma-screening` (or similar)
   - SDK: **Gradio**
   - Hardware: **CPU Basic** (free tier works — models are small)
   - Visibility: Public

2. **Push this directory**:
   ```bash
   # Install git-lfs (required for model/npy files)
   git lfs install

   # Clone the empty space
   git clone https://huggingface.co/spaces/YOUR_USERNAME/oct-glaucoma-screening
   cd oct-glaucoma-screening

   # Copy all WebApp_v2 contents here
   cp -r /path/to/OCTCV/WebApp_v2/* .
   cp /path/to/OCTCV/WebApp_v2/.gitattributes .

   # Track large files with LFS
   git lfs track "*.keras" "*.npy"

   # Commit and push
   git add .
   git commit -m "Initial deployment: OCT Glaucoma Screening demo"
   git push
   ```

3. **Wait for build** — HF will install requirements and start the app automatically.

## Alternative: Push via huggingface_hub CLI

```bash
pip install huggingface_hub
huggingface-cli login

# Upload directory directly
huggingface-cli upload YOUR_USERNAME/oct-glaucoma-screening ./WebApp_v2 . --repo-type space
```

## Local Testing

```bash
cd WebApp_v2
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860
```

## Notes

- **CPU is sufficient**: The models are ~1.5–2.7 MB each. Inference takes <2s on CPU.
- **Free tier**: CPU Basic (2 vCPU, 16GB RAM) handles this easily.
- **Cold start**: First load after inactivity may take 30–60s while TensorFlow initializes.
- **Model files**: Must be tracked with Git LFS (`.gitattributes` handles this).
