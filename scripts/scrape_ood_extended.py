#!/usr/bin/env python3
"""
Scraper Negative Images Extended for OOD (Out-of-Distribution) Testing
SMARTAMBAK Project

Mengumpulkan dataset OOD baru di luar kelas yang sudah ada, berfokus pada:
1. Perikanan (Ikan polikultur, moluska, cumi, bivalvia, alga bloom, pelet pakan)
2. Infrastruktur Tambak (Terpal geomembran HDPE, pipa PVC, anco, kincir air, alat lab, lumpur hitam)
3. Luar Tambak & Lingkungan (Burung pemangsa, biawak, APD boots/sarung tangan, keranjang panen, styrofoam es, vegetasi mangrove)
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

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str | Path) -> Path:
    """Resolve path relative to cwd if exists, otherwise relative to PROJECT_ROOT."""
    path = Path(p)
    if path.is_absolute():
        return path
    if (Path.cwd() / path).exists():
        return (Path.cwd() / path).resolve()
    return (PROJECT_ROOT / path).resolve()


# 15 Kategori OOD Baru yang sangat spesifik dan realistis
EXTENDED_QUERIES: Dict[str, List[str]] = {
    # --- 1. PERIKANAN & ORGANISME AIR NON-UDANG ---
    "ikan_polikultur": [
        "freshwater tilapia fish oreochromis niloticus aquaculture",
        "clarias catfish lele close up pond water",
        "milkfish chanos chanos ikan bandeng fresh",
        "mugil cephalus mullet fish brackish water",
        "barramundi sea bass juvenile fish pond",
    ],
    "moluska_sefalopoda": [
        "fresh raw calamari squid tentacles close up",
        "cuttlefish sepia underwater marine",
        "small common octopus seafood wet market",
        "green mussel perna viridis shell cluster",
        "blood clam anadara granosa raw shell",
    ],
    "krustasea_non_udang": [
        "mud crab scylla serrata mangrove mud pond",
        "blue swimming crab portunus pelagicus fresh",
        "freshwater crayfish redclaw cherax quadricarinatus",
        "small fiddler crab uca on wet mud flat",
        "hermit crab in shell on wet sand",
    ],
    "pakan_pelet_anco": [
        "shrimp feed pellets floating on water surface",
        "aquaculture shrimp feeding pellets in hand",
        "shrimp feed crumble feed tray anco",
        "feed tray mesh submerged aquaculture pond",
        "aquaculture sinking fish pellet texture",
    ],
    "alga_plankton_water": [
        "filamentous green algae mat pond surface",
        "green water microalgae bloom aquaculture pond",
        "gracilaria seaweed red algae brackish water",
        "thick yellow brown algae foam water edge",
        "duckweed lemna minor floating water pond",
    ],

    # --- 2. INFRASTRUKTUR & PERALATAN TAMBAK ---
    "terpal_geomembrane": [
        "black hdpe geomembrane pond liner texture",
        "black plastic geomembrane liner wrinkles folds",
        "shrimp pond geomembrane welding seam close up",
        "wet black geomembrane sheet outdoor reflection",
        "geomembrane liner installation shrimp pond",
    ],
    "pipa_saluran_pvc": [
        "white pvc pipe drain pipe aquaculture pond",
        "pvc elbow fitting valve water pipeline pond",
        "central drainage siphon pipe shrimp pond",
        "submerged aeration perforated pvc pipe water",
        "orange pvc conduit pipe outdoor water line",
    ],
    "kincir_aerator": [
        "paddle wheel aerator yellow impeller blades",
        "paddlewheel aerator blue plastic float pontoon",
        "electric motor paddle wheel aerator splash water",
        "aquaculture aerator shaft stainless steel close up",
        "paddle wheel splashing bubbles foam close up",
    ],
    "alat_ukur_lab": [
        "optical salinity refractometer aquaculture tool",
        "handheld digital dissolved oxygen meter probe",
        "digital ph pen meter water testing aquaculture",
        "secchi disk water transparency measurement pond",
        "water sampling beaker glass pond testing",
    ],
    "lumpur_dasar_monik": [
        "black anaerobic sediment pond bottom sludge",
        "cracked dry clay soil shrimp pond bottom",
        "concrete sluice gate monik shrimp pond inlet",
        "wooden drainage board sluice gate aquaculture",
        "wire mesh screen sluice gate filter pond",
    ],

    # --- 3. LINGKUNGAN LUAR TAMBAK, PREDATOR & APD PEKERJA ---
    "burung_pemangsa": [
        "little egret bird stalking water pond edge",
        "pond heron blekok standing near fish pond",
        "common kingfisher bird perched near water",
        "little cormorant bird swimming pond water",
        "white heron bird hunting pond reflection",
    ],
    "hewan_liar_pesisir": [
        "water monitor lizard varanus salvator swimming water",
        "monitor lizard biawak on mud pond dike",
        "wetland rat rodent near pond water bank",
        "stray domestic cat walking near fish pond",
        "freshwater turtle basking near aquaculture pond",
    ],
    "apd_pekerja": [
        "muddy black rubber boots farmer feet pond",
        "yellow rubber safety boots walking wet mud",
        "blue heavy duty rubber gloves wet aquaculture",
        "orange rubber gloves holding net muddy",
        "farmer traditional woven bamboo conical hat",
    ],
    "logistik_panen": [
        "plastic perforated harvest basket aquaculture red blue",
        "plastic crate container for shrimp harvesting",
        "white styrofoam box for ice fish packing",
        "crushed ice block for seafood preservation box",
        "hanging digital scale weighing seafood basket",
    ],
    "mangrove_vegetasi": [
        "rhizophora mangrove stilt roots wet mud swamp",
        "mangrove pneumatophores pencil roots mud",
        "fallen mangrove leaves floating brackish water",
        "mangrove forest coastal muddy shoreline",
        "salicornia coastal salt marsh vegetation",
    ],
}


def get_image_md5(file_path: Path) -> str:
    """Compute MD5 hash of image content for deduplication."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def validate_and_clean_image(
    src_path: Path, dst_path: Path, min_size: int = 150
) -> bool:
    """
    Validate image integrity, check minimum resolution, and convert to clean RGB JPEG.
    """
    try:
        with Image.open(src_path) as img:
            img.verify()
        with Image.open(src_path) as img:
            w, h = img.size
            if w < min_size or h < min_size:
                return False
            # Convert RGBA/Palette/Grayscale to standard RGB
            rgb_img = img.convert("RGB")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_img.save(dst_path, "JPEG", quality=90)
            return True
    except Exception:
        return False


def scrape_category(
    category: str,
    queries: List[str],
    target_count: int,
    output_dir: Path,
    known_hashes: set,
    min_size: int = 150,
) -> int:
    """
    Scrapes images for a single category until target_count valid images are collected.
    """
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images in category directory
    existing_files = list(cat_dir.glob("*.jpg"))
    current_count = 0
    for f in existing_files:
        try:
            h = get_image_md5(f)
            known_hashes.add(h)
            current_count += 1
        except Exception:
            pass

    if current_count >= target_count:
        print(f"[{category}] Sudah lengkap: {current_count}/{target_count} gambar.")
        return current_count

    needed = target_count - current_count
    print(f"\n>>> [{category}] Mengunduh {needed} gambar (saat ini: {current_count}/{target_count})...")

    with tempfile.TemporaryDirectory() as temp_download_dir:
        temp_dir_path = Path(temp_download_dir)

        for query_idx, query in enumerate(queries):
            if current_count >= target_count:
                break

            batch_limit = min(max(needed * 2, 20), 40)
            print(f"    - Query [{query_idx + 1}/{len(queries)}]: '{query}' (limit={batch_limit})")

            try:
                downloader.download(
                    query,
                    limit=batch_limit,
                    output_dir=str(temp_dir_path),
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=15,
                    verbose=False,
                )
            except Exception as e:
                print(f"      [Warning] Download error on query '{query}': {e}")
                continue

            query_folder = temp_dir_path / query
            if not query_folder.exists():
                continue

            downloaded_files = sorted(query_folder.glob("*.*"))
            for raw_file in downloaded_files:
                if current_count >= target_count:
                    break

                try:
                    file_hash = get_image_md5(raw_file)
                    if file_hash in known_hashes:
                        continue
                except Exception:
                    continue

                filename = f"{category}_{current_count + 1:04d}.jpg"
                final_path = cat_dir / filename

                if validate_and_clean_image(raw_file, final_path, min_size=min_size):
                    known_hashes.add(file_hash)
                    current_count += 1
                    needed = target_count - current_count
                    if current_count % 5 == 0 or current_count == target_count:
                        print(f"      Progress [{category}]: {current_count}/{target_count}")

            # Clean query temp folder to save disk space
            shutil.rmtree(query_folder, ignore_errors=True)

    print(f"[{category}] Selesai: {current_count}/{target_count} gambar tersimpan di {cat_dir.relative_to(PROJECT_ROOT)}.")
    return current_count


def main():
    parser = argparse.ArgumentParser(
        description="Scraper OOD Extended Dataset untuk SMARTAMBAK (Perikanan, Tambak, Luar Tambak)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/OOD_EXTENDED",
        help="Direktori target penyimpanan dataset OOD (default: dataset/OOD_EXTENDED)",
    )
    parser.add_argument(
        "--limit-per-cat",
        type=int,
        default=30,
        help="Target jumlah gambar per kategori (default: 30 gambar, total 15 x 30 = 450 gambar)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(EXTENDED_QUERIES.keys()),
        help="Pilih kategori spesifik yang ingin di-scrape (default: seluruh kategori)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=150,
        help="Resolusi minimum gambar (lebar & tinggi pixel, default: 150)",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Tampilkan daftar seluruh kategori OOD yang tersedia lalu keluar",
    )

    args = parser.parse_args()

    if args.list_categories:
        print("=== DAFTAR KATEGORI OOD EXTENDED SMARTAMBAK ===")
        for idx, (cat, q_list) in enumerate(EXTENDED_QUERIES.items(), 1):
            print(f"{idx:2d}. {cat:22s} ({len(q_list)} variasi query pencarian)")
        sys.exit(0)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🦐 SMARTAMBAK: OOD EXTENDED SCRAPER")
    print("=" * 60)
    print(f"Target Direktori : {output_dir}")
    print(f"Target Per Kelas : {args.limit_per_cat} gambar")
    print(f"Jumlah Kategori  : {len(args.categories)}")
    print(f"Estimasi Total   : {len(args.categories) * args.limit_per_cat} gambar")
    print("=" * 60)

    known_hashes = set()
    stats = {}

    for cat in args.categories:
        if cat not in EXTENDED_QUERIES:
            print(f"[Warning] Kategori '{cat}' tidak dikenali. Melewati...")
            continue

        queries = EXTENDED_QUERIES[cat]
        collected = scrape_category(
            category=cat,
            queries=queries,
            target_count=args.limit_per_cat,
            output_dir=output_dir,
            known_hashes=known_hashes,
            min_size=args.min_size,
        )
        stats[cat] = collected

    # Simpan metadata hasil scraping
    meta_path = output_dir / "metadata.json"
    meta_data = {
        "timestamp": datetime.now().isoformat(),
        "total_images": sum(stats.values()),
        "target_per_category": args.limit_per_cat,
        "categories_stats": stats,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)

    print("\n" + "=" * 60)
    print("🎉 SCRAPING OOD EXTENDED SELESAI!")
    print(f"Total Gambar Valid Terkumpul: {sum(stats.values())}")
    print(f"Metadata tersimpan di: {meta_path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
