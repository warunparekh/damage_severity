# Damage Severity Demo UI

This small Flask app provides a simple, professional UI to upload an image and get a damage severity prediction from the trained PyTorch model in this repo.

Files added:
- `app.py` — Flask application that loads the model and serves the upload/prediction pages.
- `templates/index.html` — single-page upload + result UI.
- `requirements.txt` — minimal Python dependencies.

Quick run (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in a browser. The app will load the checkpoint `accident_severity_model.pth` and `label_map.json` from the repo root; ensure those files exist (they already do in this repo).

Notes:
- The app loads the model once at startup. If your GPU is available, the app will use it. For demo machines without GPU, it will run on CPU.
- For a short demo, using small batches and single-image inference is fast; keep images reasonably sized for quicker responses.

If you want, I can:
- Add Dockerfile to containerize the demo.
- Add a Windows shortcut or script to launch the demo.
- Improve the frontend styling or add progress indicators.
