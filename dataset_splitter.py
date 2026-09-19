#!/usr/bin/env python3
"""
Dataset Splitter for Negative Images
Based on `agent.md` guidelines.

Allows splitting scraped negative images into:
1. Dedicated OOD Test Dataset (`OOD_TEST/`) for objective model comparison
2. YOLO Training Negatives (`YOLO_NEGATIVES/` or directly into training set)
"""

import argparse
import random
import shutil
from pathlib import Path

DEFAULT_OOD_QUOTAS = {
    "human": 50,
    "hand": 50,
    "fish": 50,
    "crustacean": 50,
    "water_pond": 50,
    "rock_shell": 50,
    "equipment": 50,
    "random_objects": 50,
    "seafood_food": 0,    # 100% for Roboflow training/val/test
    "plants_debris": 0,   # 100% for Roboflow training/val/test
}


def split_dataset(
    source_dir: str = "dataset/scrap/negative_images",
    ood_dir: str = "dataset/OOD_TEST",
    train_dir: str = "dataset/YOLO_NEGATIVES",
    ood_quota_per_cat: int = 50,
    seed: int = 42,
    create_yolo_txt_for_train: bool = True,
    clean_dest: bool = True,
):
    source_path = Path(source_dir)
    ood_path = Path(ood_dir)
    train_path = Path(train_dir)

    if not source_path.exists():
        print(f"Error: Source directory {source_path} does not exist.")
        return

    random.seed(seed)
    if clean_dest:
        if ood_path.exists():
            shutil.rmtree(ood_path)
        if train_path.exists():
            shutil.rmtree(train_path)

    ood_path.mkdir(parents=True, exist_ok=True)
    train_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Splitting Negative Images into OOD_TEST and YOLO_NEGATIVES ---")
    print(f"Source   : {source_path.resolve()}")
    print(f"OOD Test : {ood_path.resolve()}")
    print(f"Train Neg: {train_path.resolve()}")
    print(f"Seed     : {seed}\n")

    total_ood = 0
    total_train = 0

    for cat_folder in sorted(source_path.iterdir()):
        if not cat_folder.is_dir():
            continue

        cat_name = cat_folder.name
        images = sorted(list(cat_folder.glob("*.jpg")))
        random.shuffle(images)

        target_ood = ood_quota_per_cat if ood_quota_per_cat is not None else DEFAULT_OOD_QUOTAS.get(cat_name, 0)
        ood_images = images[:target_ood]
        train_images = images[target_ood:]

        # Copy to OOD
        target_ood_cat = ood_path / cat_name
        target_ood_cat.mkdir(parents=True, exist_ok=True)
        for img in ood_images:
            shutil.copy2(img, target_ood_cat / img.name)
        total_ood += len(ood_images)

        # Copy to Train Negatives
        target_train_cat = train_path / cat_name
        target_train_cat.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            dest_img = target_train_cat / img.name
            shutil.copy2(img, dest_img)
            if create_yolo_txt_for_train:
                txt_file = target_train_cat / f"{img.stem}.txt"
                txt_file.touch()
        total_train += len(train_images)

        print(
            f"[{cat_name:15}] Total: {len(images):3d} -> OOD_TEST: {len(ood_images):2d}, TRAIN: {len(train_images):2d}"
        )

    print("\nSplit Complete!")
    print(f"Total OOD Test Images     : {total_ood}")
    print(f"Total Train Negative Images: {total_train}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Split negative images for OOD testing and training")
    parser.add_argument("--source-dir", type=str, default="dataset/scrap/negative_images")
    parser.add_argument("--ood-dir", type=str, default="dataset/OOD_TEST", help="Folder for dedicated 400 OOD benchmark test images")
    parser.add_argument("--roboflow-dir", "--train-dir", dest="train_dir", type=str, default="dataset/ROBOFLOW_NEGATIVES", help="Folder for 460 Roboflow negative images (train/val/test)")
    parser.add_argument("--ood-quota", type=int, default=None, help="Override quota per category for OOD test set (default: 50 per category)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-yolo-txt", action="store_true", help="Do not generate empty .txt label files")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean target directories before copying")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(
        source_dir=args.source_dir,
        ood_dir=args.ood_dir,
        train_dir=args.train_dir,
        ood_quota_per_cat=args.ood_quota,
        seed=args.seed,
        create_yolo_txt_for_train=not args.no_yolo_txt,
        clean_dest=not args.no_clean,
    )
