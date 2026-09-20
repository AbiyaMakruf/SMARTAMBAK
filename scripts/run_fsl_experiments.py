#!/usr/bin/env python3
"""
scripts/run_fsl_experiments.py
Eksperimen Komprehensif Few-Shot Learning (FSL) untuk Diagnosis Penyakit Udang.
Mencakup:
1. Ekstraksi Fitur (DINOv2, ConvNeXt-Tiny, ResNet-50)
2. Episodic Meta-Evaluation (N-way K-shot, K=1, 5, 10) dengan Prototypical Networks (Euclidean vs Cosine)
3. Baseline Comparison: Conventional Linear Fine-Tuning pada Support Set
4. Evaluasi Stage 1 Generic Detector & Rejection OOD
5. Ekspor data kuantitatif ke reports/fsl_experiment_results.json & reports/fsl_summary_table.csv
"""

import os
import glob
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# Pastikan seed tetap untuk reproduktibilitas
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CROPS_DIR = "dataset/FSL_CROPS"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Transformasi standar untuk backbone (224x224, ImageNet Normalization)
STANDARD_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ShrimpCropDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = self.labels[idx]
        with Image.open(path).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)
        return img, label, path

def collect_crops(max_per_class=500):
    """Mengumpulkan path gambar per kelas dari dataset/FSL_CROPS"""
    class_dirs = sorted([d for d in os.listdir(CROPS_DIR) if os.path.isdir(os.path.join(CROPS_DIR, d))])
    data_by_class = {}
    
    for c in class_dirs:
        files = sorted(glob.glob(os.path.join(CROPS_DIR, c, "*.jpg")))
        random.seed(42)
        random.shuffle(files)
        if max_per_class and len(files) > max_per_class:
            files = files[:max_per_class]
        data_by_class[c] = files
        print(f"Loaded class '{c}': {len(files)} samples")
        
    return data_by_class

def extract_features_for_model(model, dataloader, device):
    """Ekstraksi fitur embedding dari seluruh batch dataset"""
    model.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls, _ in tqdm(dataloader, desc="Extracting features"):
            imgs = imgs.to(device)
            feats = model(imgs)
            # Normalisasi L2 untuk kestabilan metrik
            feats = F.normalize(feats, p=2, dim=1)
            all_feats.append(feats.cpu().numpy())
            all_labels.extend(lbls.numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.array(all_labels)
    return all_feats, all_labels

def get_backbone(name):
    """Menginisialisasi arsitektur backbone"""
    print(f"\n📦 Loading backbone: {name} ...")
    if name == "dinov2_vits14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        feature_dim = 384
    elif name == "convnext_tiny":
        base = models.convnext_tiny(weights="DEFAULT")
        base.classifier[2] = nn.Identity() # Buang classifier head
        model = base
        feature_dim = 768
    elif name == "resnet50":
        base = models.resnet50(weights="DEFAULT")
        base.fc = nn.Identity() # Buang fc head
        model = base
        feature_dim = 2048
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    model = model.to(DEVICE)
    model.eval()
    return model, feature_dim

def evaluate_few_shot_episodes(feats_by_class, n_way=5, k_shot=5, q_query=15, n_episodes=100, metric="euclidean"):
    """
    Evaluasi N-way K-shot episodic meta-testing menggunakan Prototypical Networks.
    """
    classes = list(feats_by_class.keys())
    assert len(classes) >= n_way, f"Jumlah kelas ({len(classes)}) < N-way ({n_way})"
    
    accuracies = []
    
    for ep in range(n_episodes):
        # Sample N kelas
        selected_classes = random.sample(classes, n_way)
        
        prototypes = []
        query_feats_list = []
        query_labels_list = []
        
        for c_idx, c in enumerate(selected_classes):
            c_feats = feats_by_class[c]
            n_samples = len(c_feats)
            if n_samples < (k_shot + q_query):
                # Jika sampel kelas sedikit, ambil semaksimal mungkin untuk query
                actual_q = max(1, n_samples - k_shot)
            else:
                actual_q = q_query
                
            perm = np.random.permutation(n_samples)
            support_idx = perm[:k_shot]
            query_idx = perm[k_shot:k_shot + actual_q]
            
            support_feats = c_feats[support_idx] # [K, D]
            proto = np.mean(support_feats, axis=0) # [D]
            if metric == "cosine":
                proto = proto / (np.linalg.norm(proto) + 1e-8)
            prototypes.append(proto)
            
            query_feats_list.append(c_feats[query_idx])
            query_labels_list.extend([c_idx] * len(query_idx))
            
        prototypes = np.stack(prototypes, axis=0) # [N, D]
        all_queries = np.concatenate(query_feats_list, axis=0) # [Total_Q, D]
        true_labels = np.array(query_labels_list)
        
        if metric == "euclidean":
            # Hitung jarak Euclidean: ||q - p||^2
            dists = np.linalg.norm(all_queries[:, np.newaxis, :] - prototypes[np.newaxis, :, :], axis=2)
            preds = np.argmin(dists, axis=1)
        elif metric == "cosine":
            # Cosine similarity: q . p / (||q|| ||p||)
            sims = np.dot(all_queries, prototypes.T)
            preds = np.argmax(sims, axis=1)
            
        acc = np.mean(preds == true_labels) * 100.0
        accuracies.append(acc)
        
    mean_acc = np.mean(accuracies)
    ci95 = 1.96 * (np.std(accuracies) / np.sqrt(n_episodes))
    return mean_acc, ci95

def evaluate_linear_fine_tuning(feats_by_class, n_way=5, k_shot=5, q_query=15, n_episodes=50, feature_dim=384):
    """
    Baseline komparasi: Conventional Linear Fine-Tuning pada Support Set (Transfer Learning).
    Melatih linear layer pada feature representations selama 40 epochs.
    """
    classes = list(feats_by_class.keys())
    accuracies = []
    
    for ep in range(n_episodes):
        selected_classes = random.sample(classes, n_way)
        
        train_x, train_y = [], []
        test_x, test_y = [], []
        
        for c_idx, c in enumerate(selected_classes):
            c_feats = feats_by_class[c]
            n_samples = len(c_feats)
            actual_q = min(q_query, max(1, n_samples - k_shot))
            perm = np.random.permutation(n_samples)
            
            support_idx = perm[:k_shot]
            query_idx = perm[k_shot:k_shot + actual_q]
            
            train_x.append(c_feats[support_idx])
            train_y.extend([c_idx] * k_shot)
            
            test_x.append(c_feats[query_idx])
            test_y.extend([c_idx] * len(query_idx))
            
        train_x = torch.tensor(np.concatenate(train_x, axis=0), dtype=torch.float32).to(DEVICE)
        train_y = torch.tensor(train_y, dtype=torch.long).to(DEVICE)
        test_x = torch.tensor(np.concatenate(test_x, axis=0), dtype=torch.float32).to(DEVICE)
        test_y = np.array(test_y)
        
        classifier = nn.Linear(feature_dim, n_way).to(DEVICE)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        classifier.train()
        for _ in range(35):
            optimizer.zero_grad()
            out = classifier(train_x)
            loss = criterion(out, train_y)
            loss.backward()
            optimizer.step()
            
        classifier.eval()
        with torch.no_grad():
            preds = classifier(test_x).argmax(dim=1).cpu().numpy()
            
        acc = np.mean(preds == test_y) * 100.0
        accuracies.append(acc)
        
    mean_acc = np.mean(accuracies)
    ci95 = 1.96 * (np.std(accuracies) / np.sqrt(n_episodes))
    return mean_acc, ci95

def main():
    print("🚀 Memulai Rangkaian Eksperimen Few-Shot Learning SMARTAMBAK...")
    print(f"Device: {DEVICE}")
    
    # 1. Kumpulkan seluruh data crop
    data_by_class = collect_crops(max_per_class=400)
    all_img_paths = []
    all_labels = []
    class_to_idx = {c: i for i, c in enumerate(data_by_class.keys())}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    for c, paths in data_by_class.items():
        for p in paths:
            all_img_paths.append(p)
            all_labels.append(class_to_idx[c])
            
    crop_dataset = ShrimpCropDataset(all_img_paths, all_labels, transform=STANDARD_TRANSFORM)
    dataloader = DataLoader(crop_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 2. Ekstraksi dan evaluasi tiap backbone
    backbones = ["dinov2_vits14", "convnext_tiny", "resnet50"]
    cached_feats = {}
    
    for b_name in backbones:
        model, dim = get_backbone(b_name)
        start_time = time.time()
        feats, labels = extract_features_for_model(model, dataloader, DEVICE)
        elapsed = time.time() - start_time
        print(f"-> Selesai ekstraksi {b_name} dalam {elapsed:.2f} detik ({len(feats)/elapsed:.1f} FPS)")
        
        # Simpan fitur per kelas
        feats_by_c = {}
        for c in data_by_class.keys():
            c_mask = (labels == class_to_idx[c])
            feats_by_c[c] = feats[c_mask]
        cached_feats[b_name] = (feats_by_c, dim, feats, labels)
        
        # Bersihkan VRAM
        del model
        torch.cuda.empty_cache()

    # 3. Jalankan Eksperimen Few-Shot (Ablation & Benchmark)
    print("\n" + "="*70)
    print("📊 MENJALANKAN BENCHMARK FEW-SHOT LEARNING (100 EPISODES PER RUN)")
    print("="*70)
    
    results = []
    k_shots = [1, 5, 10]
    n_way = 5 # 5-way classification task across shrimp health/disease conditions
    
    # Simpan fitur DINOv2 untuk pembuatan visualisasi t-SNE & Confusion Matrix nanti
    dino_feats_by_c, dino_dim, dino_raw_feats, dino_raw_labels = cached_feats["dinov2_vits14"]
    np.savez_compressed("reports/dinov2_features_cache.npz", 
                        feats=dino_raw_feats, 
                        labels=dino_raw_labels, 
                        classes=list(data_by_class.keys()))
    print("💾 DINOv2 feature embeddings disimpan ke: reports/dinov2_features_cache.npz")
    
    # A. Evaluasi Tiap Backbone dengan ProtoNet (Euclidean & Cosine)
    for b_name in backbones:
        feats_by_c, dim, _, _ = cached_feats[b_name]
        for metric in ["euclidean", "cosine"]:
            for k in k_shots:
                acc, ci = evaluate_few_shot_episodes(feats_by_c, n_way=n_way, k_shot=k, q_query=15, n_episodes=100, metric=metric)
                res_entry = {
                    "Method": f"ProtoNet ({metric.capitalize()})",
                    "Backbone": b_name,
                    "N-Way": n_way,
                    "K-Shot": k,
                    "Metric": metric,
                    "Accuracy": round(acc, 2),
                    "CI95": round(ci, 2)
                }
                results.append(res_entry)
                print(f"[{b_name:14s}] ProtoNet ({metric:9s}) | {n_way}-way {k:2d}-shot : {acc:5.2f}% ± {ci:.2f}%")

    # B. Evaluasi Baseline: Conventional Fine-Tuning pada DINOv2
    print("\n--- Baseline: Conventional Fine-Tuning (Transfer Learning) ---")
    for k in k_shots:
        acc, ci = evaluate_linear_fine_tuning(dino_feats_by_c, n_way=n_way, k_shot=k, q_query=15, n_episodes=50, feature_dim=dino_dim)
        res_entry = {
            "Method": "Conventional Fine-Tuning",
            "Backbone": "dinov2_vits14",
            "N-Way": n_way,
            "K-Shot": k,
            "Metric": "Cross-Entropy",
            "Accuracy": round(acc, 2),
            "CI95": round(ci, 2)
        }
        results.append(res_entry)
        print(f"[dinov2_vits14 ] Fine-Tuning (Linear)   | {n_way}-way {k:2d}-shot : {acc:5.2f}% ± {ci:.2f}%")

    # 4. Ekspor Hasil Eksperimen
    df_results = pd.DataFrame(results)
    df_results.to_csv("reports/fsl_summary_table.csv", index=False)
    with open("reports/fsl_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*70)
    print("✅ SELURUH EKSPERIMEN FEW-SHOT SELESAI!")
    print(f"📁 Hasil disimpan di:")
    print("   - reports/fsl_summary_table.csv")
    print("   - reports/fsl_experiment_results.json")
    print("="*70)

if __name__ == "__main__":
    main()
