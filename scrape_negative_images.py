#!/usr/bin/env python3
"""
Scraper Negative Images for YOLO Shrimp Detection
Based on recommendations in `agent.md`.

Fetches non-shrimp images (hard negatives & backgrounds) to prevent false-positive
shrimp detections in YOLO models.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from bing_image_downloader import downloader

# Presets for target quotas based on agent.md
TARGET_PRESETS = {
    "combined": {  # 860 images total = 460 Roboflow (Train/Val/Test) + 400 OOD_TEST
        "human": 110,         # 60 Roboflow + 50 OOD
        "hand": 110,          # 60 Roboflow + 50 OOD
        "fish": 110,          # 60 Roboflow + 50 OOD
        "crustacean": 110,    # 60 Roboflow + 50 OOD
        "rock_shell": 100,    # 50 Roboflow + 50 OOD
        "water_pond": 100,    # 50 Roboflow + 50 OOD
        "equipment": 90,      # 40 Roboflow + 50 OOD
        "seafood_food": 30,   # 30 Roboflow +  0 OOD
        "plants_debris": 25,  # 25 Roboflow +  0 OOD
        "random_objects": 75, # 25 Roboflow + 50 OOD
    },
    "roboflow_only": {  # 460 images for Roboflow training/val/test only
        "human": 60,
        "hand": 60,
        "fish": 60,
        "crustacean": 60,
        "rock_shell": 50,
        "water_pond": 50,
        "equipment": 40,
        "seafood_food": 30,
        "plants_debris": 25,
        "random_objects": 25,
    },
    "ood_only": {  # 400 images for dedicated OOD benchmark test only
        "human": 50,
        "hand": 50,
        "fish": 50,
        "crustacean": 50,
        "rock_shell": 50,
        "water_pond": 50,
        "equipment": 50,
        "seafood_food": 0,
        "plants_debris": 0,
        "random_objects": 50,
    },
}

# Rich query collection to ensure variety without running out of search results
CATEGORY_QUERIES = {
    "human": [
        "person standing outdoors portrait",
        "fisherman aquaculture pond portrait",
        "farmer working pond portrait",
        "full body human photo outdoor",
        "people walking outdoor daytime",
        "fisherman holding net portrait",
        "man standing near water pond",
        "farmer portrait countryside",
    ],
    "hand": [
        "empty human hand palm open",
        "farmer hand holding mud",
        "human fingers close up holding stone",
        "holding thermometer in water hand",
        "human hand gesture outdoors",
        "hand holding plastic cup close up",
        "fingers touching water surface",
        "hand holding water sensor probe",
    ],
    "fish": [
        "tilapia fish pond underwater",
        "milkfish swimming pond",
        "freshwater catfish aquaculture",
        "small fish swimming in pond",
        "carp fish freshwater pond",
        "gourami fish swimming",
        "freshwater fish aquaculture tank",
        "bandeng fish swimming pond",
    ],
    "crustacean": [
        "mud crab aquaculture pond",
        "freshwater crayfish river",
        "blue swimming crab close up",
        "fiddler crab on mud flat",
        "freshwater lobster crayfish claws",
        "mangrove crab close up mud",
        "king crab underwater",
        "land crab walking gravel",
    ],
    "rock_shell": [
        "gravel stones aquaculture pond bottom",
        "empty river snail shell",
        "coral rock stones close up",
        "pond bottom sediment rocks",
        "river pebble stones texture",
        "clam shells on beach sand",
        "coarse aquarium gravel rocks",
        "limestone rocks water edge",
    ],
    "water_pond": [
        "empty shrimp pond green water surface",
        "murky brackish water pond ripples",
        "paddle wheel aerator water foam splashing",
        "pond water texture surface outdoor",
        "clean freshwater pond surface reflection",
        "turbid brown aquaculture water",
        "green algae water surface texture",
        "water ripple waves pond surface",
    ],
    "equipment": [
        "aquaculture scoop net mesh empty",
        "plastic bucket aquaculture pond",
        "paddlewheel aerator shrimp pond equipment",
        "water dissolved oxygen meter aquaculture",
        "feeding tray aquaculture empty",
        "water pump aquaculture pond",
        "plastic drum barrel aquaculture",
        "floating aquaculture cage net",
    ],
    "seafood_food": [
        "grilled tilapia fish plate",
        "fried calamari squid rings dish",
        "steamed crab seafood restaurant",
        "cooked fish fillet vegetables plate",
        "fried gurame fish indonesian food",
        "grilled gurame fish sweet soy",
    ],
    "plants_debris": [
        "aquatic weeds duckweed pond surface",
        "algae moss pond water",
        "mangrove plant roots muddy water",
        "fallen dried leaves floating on pond",
        "water hyacinth pond plant leaves",
        "floating plant duckweed texture",
    ],
    "random_objects": [
        "plastic bottle floating pond water debris",
        "white styrofoam box aquaculture dock",
        "bamboo poles pond fence",
        "plastic water pipe aquaculture installation",
        "discarded plastic bag in pond water",
        "rubber boot pond dock",
        "wooden dock plank texture water",
        "floating pvc pipe floatation",
    ],
}


def get_image_hash(filepath: Path) -> Optional[str]:
    """Calculate MD5 hash of image content for deduplication."""
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def validate_and_clean_image(
    source_path: Path, dest_path: Path, min_size: int = 200
) -> bool:
    """Validate image integrity, check minimum resolution, convert to clean JPEG."""
    try:
        with Image.open(source_path) as img:
            img.verify()  # Check for corruption

        with Image.open(source_path) as img:
            width, height = img.size
            if width < min_size or height < min_size:
                return False

            # Convert RGBA / P / CMYK modes to RGB JPEG
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dest_path, "JPEG", quality=92, optimize=True)
            return True
    except Exception:
        return False


def scrape_category(
    category: str,
    queries: List[str],
    target_dir: Path,
    target_count: int,
    min_size: int,
    seen_hashes: set,
    create_yolo_txt: bool = False,
) -> int:
    """Scrapes images for a single category until target_count valid images are collected."""
    category_dir = target_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    current_count = len(list(category_dir.glob(f"{category}_*.jpg")))
    needed = target_count - current_count

    if needed <= 0:
        print(f"[{category}] Already has {current_count} images (target: {target_count}). Skipping.")
        return current_count

    print(f"\n=======================================================")
    print(f"[{category.upper()}] Target: {target_count} | Needed: {needed}")
    print(f"Available Queries: {len(queries)}")
    print(f"=======================================================")

    images_per_query = max(int(needed / len(queries) * 1.5) + 5, 12)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for query in queries:
            if current_count >= target_count:
                break

            print(f"-> Searching: '{query}' (requesting {images_per_query} candidates)...")
            try:
                downloader.download(
                    query,
                    limit=images_per_query,
                    output_dir=str(temp_path),
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=15,
                    verbose=False,
                )
            except Exception as e:
                print(f"   Warning: Download failed for '{query}': {e}")
                continue

            # Process downloaded candidates from temp folder
            query_folder = temp_path / query
            if not query_folder.exists():
                continue

            for raw_file in sorted(query_folder.iterdir()):
                if current_count >= target_count:
                    break
                if not raw_file.is_file():
                    continue

                file_hash = get_image_hash(raw_file)
                if not file_hash or file_hash in seen_hashes:
                    continue

                filename = f"{category}_{current_count + 1:04d}.jpg"
                final_img_path = category_dir / filename

                if validate_and_clean_image(raw_file, final_img_path, min_size=min_size):
                    seen_hashes.add(file_hash)
                    current_count += 1

                    # Optional: generate empty .txt file for YOLO negative annotation
                    if create_yolo_txt:
                        txt_path = category_dir / f"{category}_{current_count:04d}.txt"
                        txt_path.touch()

                    print(f"   ✓ [{current_count}/{target_count}] Saved {filename}")

    return current_count


def run_scraper(
    output_dir: str = "dataset/scrap/negative_images",
    mode: str = "combined",
    limit_per_cat: Optional[int] = None,
    categories: Optional[List[str]] = None,
    min_size: int = 200,
    create_yolo_txt: bool = False,
    dry_run: bool = False,
):
    """Main function to run scraping pipeline across categories."""
    output_path = Path(output_dir)

    if mode not in TARGET_PRESETS:
        print(f"Error: Unknown mode '{mode}'. Choose from: {list(TARGET_PRESETS.keys())}")
        sys.exit(1)

    preset_targets = TARGET_PRESETS[mode]
    selected_categories = categories if categories else list(preset_targets.keys())

    for cat in selected_categories:
        if cat not in CATEGORY_QUERIES:
            print(f"Error: Unknown category '{cat}'. Available: {list(CATEGORY_QUERIES.keys())}")
            sys.exit(1)

    print("\n--- Negative Images Scraper (SMARTAMBAK YOLO Project) ---")
    print(f"Output Directory : {output_path.resolve()}")
    print(f"Mode             : {mode} ({'860 total: 460 Roboflow + 400 OOD' if mode == 'combined' else mode})")
    print(f"Categories       : {', '.join(selected_categories)}")
    print(f"Min Size         : {min_size}x{min_size} px")
    print(f"Create YOLO .txt : {'Yes (0-byte background labels)' if create_yolo_txt else 'No'}")
    print(f"Dry Run          : {dry_run}\n")

    if dry_run:
        print("Planned scraping quotas:")
        total_target = 0
        for cat in selected_categories:
            target = limit_per_cat if limit_per_cat is not None else preset_targets.get(cat, 0)
            total_target += target
            print(f" - {cat:15}: {target} images")
        print(f"Total Target: {total_target} images")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize seen hashes from existing images in output directory
    seen_hashes = set()
    for existing_file in output_path.glob("*/*.jpg"):
        h = get_image_hash(existing_file)
        if h:
            seen_hashes.add(h)
    print(f"Found {len(seen_hashes)} existing images in destination directory.")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "total_images": 0,
        "categories": {},
    }

    for cat in selected_categories:
        target = limit_per_cat if limit_per_cat is not None else preset_targets.get(cat, 0)
        if target <= 0:
            continue

        count = scrape_category(
            category=cat,
            queries=CATEGORY_QUERIES[cat],
            target_dir=output_path,
            target_count=target,
            min_size=min_size,
            seen_hashes=seen_hashes,
            create_yolo_txt=create_yolo_txt,
        )
        summary["categories"][cat] = count
        summary["total_images"] += count

    # Save summary metadata
    summary_file = output_path / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print("SCRAPING COMPLETED SUCCESSFULLY!")
    print(f"Total Images: {summary['total_images']}")
    for cat, count in summary["categories"].items():
        print(f" - {cat:15}: {count} images")
    print(f"Summary log : {summary_file}")
    print("=======================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape negative/background images for YOLO training and OOD test based on agent.md"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/scrap/negative_images",
        help="Directory to store scraped images (default: dataset/scrap/negative_images)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["combined", "roboflow_only", "ood_only"],
        default="combined",
        help="Preset mode: 'combined' (860 images: 460 Roboflow + 400 OOD_TEST), 'roboflow_only' (460 images), or 'ood_only' (400 images)",
    )
    parser.add_argument(
        "--limit-per-cat",
        type=int,
        default=None,
        help="Override target count for every category (e.g. --limit-per-cat 10 for testing)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to scrape (e.g. --categories human hand fish)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=200,
        help="Minimum image width and height in pixels (default: 200)",
    )
    parser.add_argument(
        "--create-yolo-txt",
        action="store_true",
        help="Generate empty .txt label files for YOLO negative training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display target quota and categories without downloading",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_scraper(
        output_dir=args.output_dir,
        mode=args.mode,
        limit_per_cat=args.limit_per_cat,
        categories=args.categories,
        min_size=args.min_size,
        create_yolo_txt=args.create_yolo_txt,
        dry_run=args.dry_run,
    )
