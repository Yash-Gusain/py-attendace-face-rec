import cv2
import numpy as np
import pickle
import csv
from datetime import datetime
from insightface.app import FaceAnalysis

# ---------- LOAD EMBEDDINGS ----------
with open("embeddings.pkl", "rb") as f:
    known_embeddings, known_names = pickle.load(f)

known_embeddings = np.array(known_embeddings)

# ---------- INSIGHTFACE ----------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- ATTENDANCE ----------
marked = set()
csv_file = "attendance.csv"

# ---------- VISUAL CONFIRMATION ----------
confirmation_text = ""
confirmation_time = 0

def mark_attendance(name):
    global confirmation_text, confirmation_time

    if name in marked:
        return

    time_now = datetime.now().strftime("%H:%M:%S")
    date_today = datetime.now().strftime("%Y-%m-%d")

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date_today, time_now])

    marked.add(name)

    confirmation_text = f"✔ Attendance Recorded: {name}"
    confirmation_time = cv2.getTickCount()

    print(f"✔ Attendance marked for {name}")

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("📷 Camera started. Press 'q' to quit.")

# ---------- LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        # ---- STEP 3: NORMALIZED COSINE SIMILARITY ----
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        similarities = np.dot(known_embeddings, emb)
        idx = np.argmax(similarities)
        best_score = similarities[idx]

        name = "Unknown"

        if best_score > 0.35:
            name = known_names[idx]
            mark_attendance(name)

        # ---- DRAW FACE BOX ----
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ---- VISUAL CONFIRMATION OVERLAY ----
    if confirmation_text:
        elapsed = (cv2.getTickCount() - confirmation_time) / cv2.getTickFrequency()
        if elapsed < 2:
            cv2.putText(
                frame,
                confirmation_text,
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
        else:
            confirmation_text = ""

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------- CLEANUP ----------
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera closed")
print("✅ Attendance session ended")
