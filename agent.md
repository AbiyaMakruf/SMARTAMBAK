# Project Context: Shrimp Detection with YOLO

## Current Problem

I am developing a shrimp detection system using YOLO for deployment in a mobile application.

Currently, my YOLO model has **2 classes**:

* `Healthy Shrimp`
* `Sick Shrimp`

The current dataset contains:

* **2,301 images**
* **5,176 annotated shrimp objects**

The main issue is that the model produces **false-positive detections on non-shrimp objects**.

For example, when I provide an image of a human, the model can still detect the human as either:

* `Healthy Shrimp`
* `Sick Shrimp`

In some cases, the confidence score is extremely high, reaching around:

```text
0.90 or 90%
```

Because of this, simply increasing the YOLO confidence threshold is not sufficient.

Example:

```text
Input:
Human image

YOLO output:
Sick Shrimp
Confidence: 0.92
```

Even though there is clearly no shrimp in the image.

---

# Desired Behavior

The desired system behavior is:

```text
If shrimp exists:
    Detect the shrimp
    Draw bounding box
    Predict Healthy or Sick

If there is no shrimp:
    Do not display any bounding box
    Do not classify anything as Healthy or Sick
```

In other words, the model must become more robust to **non-shrimp / out-of-distribution inputs**.

The main concern is not only classification accuracy between Healthy and Sick shrimp, but also:

```text
Can the system correctly reject images or objects that are not shrimp?
```

---

# Current Hypothesis

The current model only learns two target classes:

```text
Healthy Shrimp
Sick Shrimp
```

Although YOLO technically learns objectness/background information, the training dataset contains relatively limited variation of explicit non-shrimp examples.

As a result, some unfamiliar objects can produce features similar enough to shrimp features that YOLO generates a high-confidence false-positive detection.

This can be considered an **out-of-distribution / open-set detection problem**.

---

# Planned Experiments

I plan to conduct three separate experiments.

The purpose is to compare different approaches for reducing false-positive detections on non-shrimp objects.

The experiments should preferably use the same evaluation dataset so the results can be compared fairly.

---

## Experiment 1 — Add Negative Images Without Bounding Boxes

Keep the current YOLO architecture:

```text
Input Image
    ↓
YOLO
    ↓
Healthy Shrimp / Sick Shrimp
```

However, add non-shrimp images to the YOLO training dataset.

Examples:

```text
Human
Hand
Fish
Crab
Lobster
Rock
Shell
Aquarium
Water
Net
Bucket
Plants
Food
Random objects
```

These images should contain **no shrimp** and therefore should have:

```text
No bounding boxes
No additional class
```

The class list remains:

```yaml
0: Healthy
1: Sick
```

There should NOT be a third class such as:

```yaml
2: Non-Shrimp
```

for this experiment.

The purpose is to explicitly teach YOLO:

> Images such as humans, fish, hands, objects, or empty backgrounds should produce zero detections.

### Initial Recommendation

Current shrimp dataset:

```text
2,301 images
5,176 shrimp objects
```

Proposed negative training images:

```text
~460 images
```

Approximately 20% of the current image count.

Suggested composition:

| Negative Category                      | Approx. Images |
| -------------------------------------- | -------------: |
| Human / person                         |             60 |
| Hand / arm / finger                    |             60 |
| Fish                                   |             60 |
| Crab / lobster / other crustaceans     |             60 |
| Rock / shell / gravel / coral          |             50 |
| Water / aquarium / pond without shrimp |             50 |
| Net / bucket / equipment               |             40 |
| Seafood / cooked food                  |             30 |
| Plants / leaves / debris               |             25 |
| Random objects                         |             25 |
| **Total**                              |        **460** |

Priority should be given to **hard negatives**, especially objects that are visually or contextually similar to shrimp.

Examples:

```text
Fish
Crab
Lobster
Shell
Aquatic animals
Hand
Net
Water reflections
Aquarium background
```

---

## Experiment 2 — Shrimp Detector → Healthy/Sick Classifier

Change the architecture into a two-stage system.

### Stage 1

Use YOLO only to detect whether an object is a shrimp.

Only one detection class:

```yaml
0: shrimp
```

The existing classes:

```text
Healthy Shrimp
Sick Shrimp
```

will be merged into:

```text
Shrimp
```

for the detector.

Architecture:

```text
Input Image
    ↓
YOLO Shrimp Detector
    ↓
Bounding Box of Shrimp
    ↓
Crop Detection
    ↓
Healthy/Sick Classifier
```

Stage 1 answers:

```text
Is this region actually a shrimp?
```

Stage 2 answers:

```text
If it is shrimp, is it Healthy or Sick?
```

The negative images from Experiment 1 can also be used for the Stage 1 detector without bounding boxes.

### Stage 2 Dataset

The existing **5,176 annotated shrimp objects** can be cropped based on their bounding boxes.

Each crop becomes one classifier image.

Example:

```text
Original image
    ↓
Shrimp bounding box
    ↓
Crop
    ↓
Healthy/
or
Sick/
```

Approximately:

```text
5,176 object crops
```

can potentially be generated for classifier training.

The exact number depends on annotation quality and class balance.

Possible lightweight classifiers for mobile deployment include:

```text
MobileNetV3
EfficientNet-Lite
MobileViT
```

The goal is to separate:

```text
Object localization
```

from:

```text
Disease classification
```

instead of forcing a single detector to learn both simultaneously.

---

## Experiment 3 — OOD Rejection Using Feature Embeddings

The third experiment will add an explicit **out-of-distribution rejection mechanism**.

Current YOLO confidence alone is not reliable enough.

Example:

```text
Human
    ↓
YOLO
    ↓
Sick Shrimp
confidence = 0.92
```

A high confidence score does not necessarily mean that the object truly belongs to the training distribution.

The idea is to extract a feature embedding from the model.

Example:

```text
Detection
    ↓
Feature Extractor
    ↓
Embedding Vector
    ↓
Compare with Shrimp Feature Distribution
```

Shrimp images should form feature distributions such as:

```text
Healthy shrimp cluster
Sick shrimp cluster
```

A non-shrimp object may be far from these distributions.

Example:

```text
YOLO:
Healthy Shrimp = 0.94

Embedding distance:
Very far from shrimp distribution

Final decision:
REJECT
```

Possible OOD methods to investigate:

```text
Cosine Distance / Similarity
Euclidean Distance
Mahalanobis Distance
k-Nearest Neighbors
Prototype Distance
Energy Score
One-Class SVM
Deep SVDD
```

Initial preference is to start with a relatively simple feature-distance-based method before testing more complex OOD approaches.

---

# Evaluation Dataset

A dedicated OOD test dataset should be created and kept separate from training.

Suggested structure:

```text
OOD_TEST/
├── human/
├── hand/
├── fish/
├── crustacean/
├── aquatic_background/
├── rock_shell/
├── equipment/
└── random_objects/
```

Recommended size:

```text
400 images
```

Example distribution:

| Category             |  Images |
| -------------------- | ------: |
| Human                |      50 |
| Hand / Arm           |      50 |
| Fish                 |      50 |
| Crab / Lobster       |      50 |
| Water / Aquarium     |      50 |
| Rock / Shell / Coral |      50 |
| Equipment            |      50 |
| Random Objects       |      50 |
| **Total**            | **400** |

These images should **never be used for training**.

They should be used consistently to compare all experiments.

---

# Important Evaluation Metrics

The experiment should not be evaluated only using:

```text
mAP
Precision
Recall
F1-score
```

A specific metric for non-shrimp rejection is required.

Important metrics include:

```text
Shrimp Detection Recall
Healthy/Sick Macro-F1
False Positive Rate on OOD Images
Number of OOD Images Producing at Least One Detection
Average False Detections per OOD Image
Maximum False-Positive Confidence
```

Example comparison table:

| Model                        | mAP50 | Shrimp Recall | OOD Images | OOD Images with False Detection | OOD FPR |
| ---------------------------- | ----: | ------------: | ---------: | ------------------------------: | ------: |
| Baseline YOLO                |     - |             - |        400 |                               - |       - |
| + Negative Images            |     - |             - |        400 |                               - |       - |
| Shrimp Detector → Classifier |     - |             - |        400 |                               - |       - |
| + Feature OOD Rejection      |     - |             - |        400 |                               - |       - |

A useful definition is:

```text
OOD False Positive Rate
=
Number of non-shrimp images producing ≥1 shrimp detection
----------------------------------------------------------
Total number of non-shrimp test images
```

---


# Main Research Question

The overall objective is to determine which method most effectively reduces high-confidence false-positive detections on non-shrimp images while maintaining shrimp detection and Healthy/Sick classification performance.

The three approaches to compare are:

1. **YOLO + Negative Images**
2. **Shrimp Detector → Healthy/Sick Classifier**
3. **YOLO / Detector + Feature-Embedding-Based OOD Rejection**

The final solution should also remain practical for deployment in a **mobile application**, so inference latency, model size, and computational complexity should be considered.
