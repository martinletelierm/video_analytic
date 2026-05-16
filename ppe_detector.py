# pip install ultralytics opencv-python

import sys
import sqlite3
import datetime
import cv2
from ultralytics import YOLO

DB_PATH = "ppe_violations.db"


def load_model():
    try:
        model = YOLO("keremberke/yolov8m-hard-hat-detection")
        print("Model loaded: keremberke/yolov8m-hard-hat-detection")
        return model
    except Exception as e:
        print(f"Warning: Could not load keremberke model: {e}")
        print("Falling back to yolov8n.pt")
        try:
            model = YOLO("yolov8n.pt")
            print("Fallback model loaded: yolov8n.pt")
            return model
        except Exception as e2:
            print(f"Error: Could not load fallback model: {e2}")
            sys.exit(1)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            frame_number INTEGER,
            violation_type TEXT,
            confidence REAL,
            bbox TEXT
        )
    """)
    conn.commit()
    return conn


def classify_model_classes(model_names):
    """
    Parse model.names dict into semantic buckets.

    Returns six sets of integer class IDs:
      person_ids       – generic person / worker
      hardhat_ids      – hardhat / helmet (present)
      vest_ids         – safety vest (present)
      no_hardhat_ids   – person explicitly labelled without hardhat
      no_vest_ids      – person explicitly labelled without vest
      no_ppe_ids       – person explicitly labelled without any PPE
    """
    person_ids = set()
    hardhat_ids = set()
    vest_ids = set()
    no_hardhat_ids = set()
    no_vest_ids = set()
    no_ppe_ids = set()

    for idx, raw in model_names.items():
        name = raw.lower().strip()

        # Explicit negative classes first (check before positive ones)
        if name in ("no-hardhat", "no hardhat", "without hardhat",
                    "no-helmet", "no helmet", "without helmet"):
            no_hardhat_ids.add(idx)
        elif name in ("no-vest", "no vest", "no-safety vest",
                      "no safety vest", "without vest"):
            no_vest_ids.add(idx)
        elif name in ("no-ppe", "no ppe", "without ppe", "no safety equipment"):
            no_ppe_ids.add(idx)
        # Positive equipment classes
        elif any(t in name for t in ("hardhat", "hard hat", "helmet")):
            hardhat_ids.add(idx)
        elif any(t in name for t in ("vest", "safety vest")):
            vest_ids.add(idx)
        # Person / worker
        elif name in ("person", "worker", "human", "people", "man", "woman"):
            person_ids.add(idx)

    return person_ids, hardhat_ids, vest_ids, no_hardhat_ids, no_vest_ids, no_ppe_ids


def overlap_ratio(person_box, equip_box):
    """
    Return what fraction of equip_box overlaps with person_box.
    Used to associate equipment with the person beneath it.
    """
    px1, py1, px2, py2 = person_box
    ex1, ey1, ex2, ey2 = equip_box

    ix1, iy1 = max(px1, ex1), max(py1, ey1)
    ix2, iy2 = min(px2, ex2), min(py2, ey2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    equip_area = max((ex2 - ex1) * (ey2 - ey1), 1)
    return inter / equip_area


def save_violation(conn, frame_number, violation_type, confidence, bbox):
    try:
        conn.execute(
            "INSERT INTO violations (timestamp, frame_number, violation_type, confidence, bbox) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.datetime.now().isoformat(),
                frame_number,
                violation_type,
                round(confidence, 4),
                str(bbox),
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"Warning: DB write failed (frame {frame_number}): {e}")


def draw_box(frame, bbox, color, label):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def process_frame(frame, results, conn, frame_number, class_buckets):
    person_ids, hardhat_ids, vest_ids, no_hardhat_ids, no_vest_ids, no_ppe_ids = class_buckets

    annotated = frame.copy()
    violations = 0

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return annotated, violations

    # ── Bucket raw detections ──────────────────────────────────────────────
    persons = []       # {'bbox': tuple, 'conf': float}
    hardhats = []
    vests = []
    explicit_violations = []  # {'bbox', 'conf', 'vtype'}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        bbox = (x1, y1, x2, y2)

        if cls_id in person_ids:
            persons.append({"bbox": bbox, "conf": conf})
        elif cls_id in hardhat_ids:
            hardhats.append({"bbox": bbox, "conf": conf})
        elif cls_id in vest_ids:
            vests.append({"bbox": bbox, "conf": conf})
        elif cls_id in no_hardhat_ids:
            explicit_violations.append({"bbox": bbox, "conf": conf, "vtype": "no_hardhat"})
        elif cls_id in no_vest_ids:
            explicit_violations.append({"bbox": bbox, "conf": conf, "vtype": "no_vest"})
        elif cls_id in no_ppe_ids:
            explicit_violations.append({"bbox": bbox, "conf": conf, "vtype": "no_ppe"})

    has_vest_class = bool(vest_ids)

    # ── Explicit violation classes (model already labelled them) ───────────
    for v in explicit_violations:
        x1, y1, x2, y2 = v["bbox"]
        label = {"no_hardhat": "No Hardhat",
                 "no_vest": "No Vest",
                 "no_ppe": "No PPE"}[v["vtype"]]
        draw_box(annotated, v["bbox"], (0, 0, 255), f"{label} {v['conf']:.2f}")
        save_violation(conn, frame_number, v["vtype"], v["conf"], v["bbox"])
        violations += 1

    # ── Equipment-only yellow boxes (draw before person assessment) ─────────
    for h in hardhats:
        draw_box(annotated, h["bbox"], (0, 255, 255), f"Hardhat {h['conf']:.2f}")
    for v in vests:
        draw_box(annotated, v["bbox"], (0, 255, 255), f"Vest {v['conf']:.2f}")

    # ── Person assessment via overlap ──────────────────────────────────────
    for p in persons:
        pbbox = p["bbox"]
        has_hardhat = any(overlap_ratio(pbbox, h["bbox"]) > 0.15 for h in hardhats)
        has_vest = any(overlap_ratio(pbbox, v["bbox"]) > 0.15 for v in vests)

        compliant = has_hardhat and (has_vest or not has_vest_class)

        if compliant:
            draw_box(annotated, pbbox, (0, 255, 0), f"Compliant {p['conf']:.2f}")
        else:
            if not has_hardhat and (not has_vest and has_vest_class):
                vtype = "no_ppe"
                label = "No PPE"
            elif not has_hardhat:
                vtype = "no_hardhat"
                label = "No Hardhat"
            else:
                vtype = "no_vest"
                label = "No Vest"

            draw_box(annotated, pbbox, (0, 0, 255), f"{label} {p['conf']:.2f}")
            save_violation(conn, frame_number, vtype, p["conf"], pbbox)
            violations += 1

    return annotated, violations


def main():
    if len(sys.argv) < 2:
        print("Usage: python ppe_detector.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        sys.exit(1)

    model = load_model()

    print(f"Model classes: {model.names}")
    class_buckets = classify_model_classes(model.names)
    p_ids, h_ids, v_ids, nh_ids, nv_ids, np_ids = class_buckets
    print(f"  persons={p_ids}  hardhats={h_ids}  vests={v_ids}")
    print(f"  no_hardhat={nh_ids}  no_vest={nv_ids}  no_ppe={np_ids}")

    conn = init_db()

    frame_number = 0
    total_violations = 0

    cv2.namedWindow("PPE Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        results = model(frame, verbose=False)

        annotated, frame_violations = process_frame(
            frame, results, conn, frame_number, class_buckets
        )
        total_violations += frame_violations

        cv2.imshow("PPE Detection", annotated)

        if frame_number % 50 == 0:
            print(f"Frame {frame_number} | Violations so far: {total_violations}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit requested by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

    print(f"\nTotal frames:     {frame_number}")
    print(f"Total violations: {total_violations}")
    print(f"Violations saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
