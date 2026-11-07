"""Project 2 API: serves the Week 6 crop-disease image model and Week 7 text evidence."""
import io
import os
import re
import time
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms

MODEL_PATH = os.getenv('MODEL_PATH', 'week6_model.pt')
# PlantVillage has 38 classes, so "the class you must not miss" is "any diseased class".
# Leave POSITIVE empty to use every class that is not healthy/background, or set a
# comma-separated list, e.g. POSITIVE="Potato___Late_blight,Tomato___Late_blight"
POSITIVE = [c.strip() for c in os.getenv('POSITIVE', '').split(',') if c.strip()]
THRESHOLD = float(os.getenv('THRESHOLD', '0.5'))
API_KEY = os.getenv('API_KEY')                      # set it when you deploy
MAX_BYTES = 5 * 1024 * 1024                         # reject uploads over 5 MB
# Farmer-note warning words (prefix match, so 'spot' also catches 'spots')
URGENT = ['blight', 'spot', 'wilt', 'yellow', 'mold', 'mildew', 'rot',
          'spreading', 'dying', 'curl', 'lesion', 'rust', 'whitefly', 'aphid']
PATTERNS = {
    'cnic': r'\b\d{5}-\d{7}-\d\b',
    'mobile': r'(?:\+92[- ]?|\b0)3\d{2}[- ]?\d{7}\b',
    'amount': r'(?:PKR|Rs\.?)\s?([\d,]+(?:\.\d{1,2})?)',
    'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
}
state = {}


def rebuild(ck):
    name, k = ck['backbone'], len(ck['classes'])
    if name.startswith('resnet'):
        m = getattr(models, name)(weights=None)
        m.fc = nn.Linear(m.fc.in_features, k)
    else:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, k)
    m.load_state_dict(ck['state_dict'])
    return m.eval()


def positive_classes(classes):
    if POSITIVE:
        return [c for c in classes if c in POSITIVE]
    return [c for c in classes
            if not c.lower().endswith('healthy') and 'background' not in c.lower()]


@asynccontextmanager
async def lifespan(app):
    ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    state['ck'], state['model'] = ck, rebuild(ck)       # load ONCE
    state['positive'] = positive_classes(ck['classes'])
    state['tf'] = transforms.Compose(
        ([transforms.Grayscale(3)] if ck['grayscale'] else []) +
        [transforms.Resize((ck['img_size'],) * 2), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
    yield
    state.clear()


app = FastAPI(title='Project 2 Vision + Text API', version='1.0',
              lifespan=lifespan)


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]
    latency_ms: float


class CaseRecord(BaseModel):
    prediction: Prediction
    note_masked: str
    fields: dict[str, list[str]]
    urgent_terms: list[str]
    decision: str
    reasons: list[str]
    disclaimer: str


def check_key(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, 'Invalid or missing API key')


async def read_image(file: UploadFile):
    if file.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(415, 'Upload a JPEG or PNG image')
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, 'Image larger than 5 MB')
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, 'File is not a readable image')


def classify(img):
    ck, t0 = state['ck'], time.perf_counter()
    x = state['tf'](img.convert('L' if ck['grayscale'] else 'RGB'))
    with torch.inference_mode():
        p = F.softmax(state['model'](x.unsqueeze(0)), dim=1)[0].tolist()
    probs = {c: round(v, 4) for c, v in zip(ck['classes'], p)}
    return Prediction(label=max(probs, key=probs.get), probabilities=probs,
                      latency_ms=round((time.perf_counter() - t0) * 1000, 1))


def mask_pii(text):
    for key, tag in [('cnic', '[CNIC]'), ('mobile', '[PHONE]')]:
        text = re.sub(PATTERNS[key], tag, text)
    return text


def find_urgent(text):
    low = text.lower()
    return [w for w in URGENT if re.search(r'\b' + re.escape(w), low)]


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': 'model' in state,
            'classes': state['ck']['classes'], 'positive': state['positive']}


@app.post('/predict', response_model=Prediction)
async def predict(file: UploadFile = File(...),
                  x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    img = await read_image(file)
    return await run_in_threadpool(classify, img)       # keep the loop free


@app.post('/analyze', response_model=CaseRecord)
async def analyze(file: UploadFile = File(...), note: str = Form(...),
                  x_api_key: str | None = Header(None)):
    check_key(x_api_key)
    if len(note) > 2000:
        raise HTTPException(422, 'Note longer than 2,000 characters')
    pred = await run_in_threadpool(classify, await read_image(file))
    safe = mask_pii(note)                               # mask FIRST
    fields = {k: re.findall(p, safe, flags=re.IGNORECASE)
              for k, p in PATTERNS.items()}
    urgent = find_urgent(safe)
    reasons = []
    p_disease = sum(pred.probabilities.get(c, 0) for c in state['positive'])
    if p_disease >= THRESHOLD:
        reasons.append(f'image: P(diseased) = {p_disease:.2f} (top: {pred.label})')
    if urgent:
        reasons.append('note mentions: ' + ', '.join(urgent))
    return CaseRecord(
        prediction=pred, note_masked=safe,
        fields={k: v for k, v in fields.items() if v},
        urgent_terms=urgent, decision='REFER' if reasons else 'ROUTINE',
        reasons=reasons,
        disclaimer='Decision support only; a qualified agronomist decides.')
