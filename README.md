# FrameForensics

Deepfake detection on faces, running on CPU: MediaPipe face cropping, a MobileNetV3-Small
classifier exported to ONNX (macro F1 0.977), frame sampling at 2 FPS and temporal smoothing.
The web app has two tabs: analyse an uploaded video, or a live webcam stream.

## Run the app

You need **Python 3.10 – 3.13** ([python.org](https://www.python.org/downloads/); on Windows,
tick *"Add python.exe to PATH"* during setup). Then:

| OS | How to start |
|----|--------------|
| Windows | Double-click **`start.bat`** |
| macOS | Double-click **`start.command`** (or run `./start.sh` in Terminal) |
| Linux | `./start.sh` |

The first start creates `frameforensics-backend/.venv` and installs the dependencies, which takes
a few minutes. Later starts are fast. The app opens in your browser at
<http://127.0.0.1:7860>. Close the window (or press Ctrl+C) to stop it.

> macOS: if Finder says the file can't be opened, run `chmod +x start.sh start.command` once,
> or right-click → Open.

### Manual start

```bash
cd frameforensics-backend
python -m venv .venv
.venv/bin/pip install -r requirements-app.txt     # Windows: .venv\Scripts\pip ...
.venv/bin/python app/app.py                        # Windows: .venv\Scripts\python app\app.py
```

Optional environment variables: `FF_MODEL_PATH` (another ONNX model), `FF_PORT` (default 7860).

## Repository layout

- `frameforensics-backend/app/app.py`: Gradio app
- `frameforensics-backend/app/face_crop.py`: face crop + preprocessing, identical to training
- `frameforensics-backend/models/`: ONNX classifier and MediaPipe BlazeFace detector
- `process_data.ipynb`, `model_training.ipynb`, `export_to_onnx.ipynb`: data prep, training and
  ONNX export (training dependencies: `frameforensics-backend/requirements.txt`)
