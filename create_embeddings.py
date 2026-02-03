import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

DATASET_DIR = "dataset"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []
names = []

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_path):
        continue

    print(f"📸 Processing {person}")

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)

        embeddings.append(emb)
        names.append(person)

        print(f"  ✔ {img_name} added as {person}")

with open("embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, names), f)

print("✅ Embeddings recreated successfully")
