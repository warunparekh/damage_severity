from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F

from infer import make_transform, load_label_map
from model import get_model
from utils import get_device

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(PROJECT_ROOT, "accident_severity_model.pth")
LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "label_map.json")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "static", "uploads")
IMG_SIZE = 224

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model and label map once at startup
device = get_device()
class_names = load_label_map(LABEL_MAP_PATH)
num_classes = len(class_names)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
backbone = ckpt.get("backbone", "resnet18")
model = get_model(backbone, num_classes=num_classes, pretrained=False, freeze_backbone=False)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.to(device).eval()

tfm = make_transform(IMG_SIZE)


def predict_pil_image(pil_img):
    img = pil_img.convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
    topv, topi = probs.topk(1, dim=1)
    prob = float(topv.squeeze(0).cpu().item())
    idx = int(topi.squeeze(0).cpu().item())
    return class_names[idx], prob


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Validate file in request
    if "image" not in request.files:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.accept_json:
            return {"error": "No file part in the request"}, 400
        return render_template("index.html", error="No file part in the request")

    file = request.files["image"]
    if file.filename == "":
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.accept_json:
            return {"error": "No file selected"}, 400
        return render_template("index.html", error="No file selected")

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Open and predict
    pil_img = Image.open(save_path)
    label, prob = predict_pil_image(pil_img)

    img_url = url_for("static", filename=f"uploads/{filename}")

    # If request came from fetch/ajax, return JSON so the client can render inline
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.accept_json:
        return {"label": label, "prob": prob, "img_url": img_url}

    # Fallback: render the main page with result embedded so non-JS clients still see output
    return render_template("index.html", label=label, prob=prob, img_url=img_url)


if __name__ == "__main__":
    # Run in production mode for demo (no debug)
    app.run(host="0.0.0.0", port=5000)
