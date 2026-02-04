import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

DATASET_DIR = "dataset"
OUTPUT_FILE = "embeddings.pkl"

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

embeddings = []
names = []

for student in os.listdir(DATASET_DIR):
    student_path = os.path.join(DATASET_DIR, student)
    if not os.path.isdir(student_path):
        continue

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)
if faces:
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.embedding
    emb = emb / np.linalg.norm(emb)
    embeddings.append(emb)
    names.append(student)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump((embeddings, names), f)

print("✅ Face registration completed")


