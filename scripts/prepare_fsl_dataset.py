#!/usr/bin/env python3
"""
scripts/prepare_fsl_dataset.py
Ekstraksi crop udang berpenyakit/sehat dari dataset deteksi objek untuk benchmark Few-Shot Learning (FSL).
"""

import os
import glob
import cv2
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

DATASET_ROOT = "dataset/roboflow/combined_v4_multiclass_null"
OUTPUT_DIR = "dataset/FSL_CROPS"

CLASS_NAMES = ['IMNV', 'WFD', 'blackgill', 'healthy', 'wssv', 'wssv_bg', 'yellowhead']

def extract_crops(padding_ratio=0.10):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for c in CLASS_NAMES:
        os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

    counts = defaultdict(int)
    splits = ["train", "valid", "test"]

    print("✂️ Memulai ekstraksi crop udang beranotasi...")
    for split in splits:
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        lbl_dir = os.path.join(DATASET_ROOT, split, "labels")

        img_files = glob.glob(os.path.join(img_dir, "*.*"))
        for img_path in tqdm(img_files, desc=f"Split: {split}"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")

            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            if not lines:
                continue # Null image

            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            for idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                if cls_id >= len(CLASS_NAMES):
                    continue
                cls_name = CLASS_NAMES[cls_id]

                cx, cy, bw, bh = map(float, parts[1:])
                # Konversi ke absolut koordinat piksel
                x_c, y_c = cx * w_img, cy * h_img
                box_w, box_h = bw * w_img, bh * h_img

                # Tambahkan padding
                pad_w = box_w * padding_ratio
                pad_h = box_h * padding_ratio

                x1 = max(0, int(x_c - box_w / 2 - pad_w))
                y1 = max(0, int(y_c - box_h / 2 - pad_h))
                x2 = min(w_img, int(x_c + box_w / 2 + pad_w))
                y2 = min(h_img, int(y_c + box_h / 2 + pad_h))

                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue # Abaikan crop terlalu kecil

                crop = img[y1:y2, x1:x2]
                save_filename = f"{split}_{base_name}_crop{idx}.jpg"
                save_path = os.path.join(OUTPUT_DIR, cls_name, save_filename)
                cv2.imwrite(save_path, crop)
                counts[cls_name] += 1

    print("\n✅ Ekstraksi Selesai! Ringkasan Crop per Kelas:")
    total_crops = 0
    for c in CLASS_NAMES:
        print(f"  - {c:12s} : {counts[c]:5d} crops")
        total_crops += counts[c]
    print(f"  Total Seluruh Crop: {total_crops} crops")

if __name__ == "__main__":
    extract_crops()
