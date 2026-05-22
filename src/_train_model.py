# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
YOLO Training Setup Script
Interactively configures and launches Ultralytics YOLO training.
"""

import os
import sys
import json
import yaml
import shutil
from pathlib import Path


# ---------------------------------------------------------
# STEP 1 - Dataset Discovery
# ---------------------------------------------------------

def get_dataset_root():
    print("\n+======================================+")
    print("|      YOLO Training Setup Script      |")
    print("+======================================+\n")
    while True:
        root = input("Enter the dataset root directory: ").strip()
        path = Path(root).expanduser().resolve()
        if path.exists():
            return path
        print("  [ERROR] Directory not found: {}. Try again.".format(path))


def parse_classes(dataset_root):
    classes_txt = dataset_root / "classes.txt"
    notes_json  = dataset_root / "notes.json"

    if classes_txt.exists():
        names = [l.strip() for l in classes_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        print("  [OK] Loaded {} class(es) from classes.txt: {}".format(len(names), names))
        return names

    if notes_json.exists():
        data  = json.loads(notes_json.read_text(encoding="utf-8"))
        names = [c["name"] for c in sorted(data["categories"], key=lambda x: x["id"])]
        print("  [OK] Loaded {} class(es) from notes.json: {}".format(len(names), names))
        return names

    raise FileNotFoundError("No classes.txt or notes.json found in dataset root.")


def detect_splits(dataset_root):
    img_root = dataset_root / "images"
    splits   = {}

    for split in ("train", "val", "test"):
        candidate = img_root / split
        if candidate.exists() and any(candidate.iterdir()):
            splits[split] = str(candidate)

    if splits:
        print("  [OK] Detected splits: {}".format(list(splits.keys())))
        return splits

    # Flat layout - write path-list txt files (no symlinks, no copying)
    all_imgs = sorted([
        p for p in img_root.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ])

    if not all_imgs:
        raise FileNotFoundError("No images found under images/")

    cutoff     = int(len(all_imgs) * 0.8)
    train_imgs = all_imgs[:cutoff]
    val_imgs   = all_imgs[cutoff:]

    train_txt = dataset_root / "train.txt"
    val_txt   = dataset_root / "val.txt"

    train_txt.write_text("\n".join(str(p.resolve()) for p in train_imgs), encoding="utf-8")
    val_txt.write_text(  "\n".join(str(p.resolve()) for p in val_imgs),   encoding="utf-8")

    print("  [OK] Auto-split: {} train / {} val images".format(len(train_imgs), len(val_imgs)))
    print("       train list -> {}".format(train_txt))
    print("       val list   -> {}".format(val_txt))

    return {"train": str(train_txt), "val": str(val_txt)}


# ---------------------------------------------------------
# STEP 2 - Write data.yaml
# ---------------------------------------------------------

def write_data_yaml(dataset_root, splits, class_names):
    yaml_path = dataset_root / "data.yaml"

    config = {
        "path"  : str(dataset_root),
        "train" : splits.get("train", ""),
        "val"   : splits.get("val",   ""),
        "nc"    : len(class_names),
        "names" : class_names,
    }
    if "test" in splits:
        config["test"] = splits["test"]

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("\n  [OK] data.yaml written -> {}".format(yaml_path))
    print("       nc   : {}".format(config["nc"]))
    print("       names: {}".format(config["names"]))
    return yaml_path


# ---------------------------------------------------------
# STEP 3 - Model Selection
# ---------------------------------------------------------

MODELS = {
    "n": ("yolo26n.pt", "~2.6M  params - fastest, least accurate"),
    "s": ("yolo26s.pt", "~9.4M  params - fast, good for small objects"),
    "m": ("yolo26m.pt", "~20M   params - balanced  [DEFAULT]"),
    "l": ("yolo26l.pt", "~43M   params - accurate, slower"),
    "x": ("yolo26x.pt", "~68M   params - most accurate, slowest"),
}

def choose_model():
    print("\n-- Model Size -----------------------------")
    for key, (pt, desc) in MODELS.items():
        marker = " <-- default" if key == "m" else ""
        print("  [{}]  {:18s}  {}{}".format(key, pt, desc, marker))

    choice = input("\nPick model size [n/s/m/l/x] (Enter = m): ").strip().lower() or "m"
    if choice not in MODELS:
        print("  Invalid choice, defaulting to m.")
        choice = "m"

    pt_file = MODELS[choice][0]
    print("  [OK] Selected: {}".format(pt_file))
    return pt_file


# ---------------------------------------------------------
# STEP 4 - GPU Detection
# ---------------------------------------------------------

def detect_gpus():
    print("\n-- GPU Detection --------------------------")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [WARN] No CUDA GPUs detected. Training will use CPU.")
            return None, "cpu"

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram  = props.total_memory / (1024 ** 3)
            gpus.append((i, props.name, vram))
            print("  [{}] {}  --  {:.1f} GB VRAM".format(i, props.name, vram))

        if len(gpus) == 1:
            choice = input("\nUse GPU 0? [Y/n]: ").strip().lower()
            device = "0" if choice != "n" else "cpu"
        else:
            print("\n  Enter GPU id(s) to use, comma-separated (e.g. 0,1)")
            raw    = input("  GPUs [default=0]: ").strip() or "0"
            device = raw.replace(" ", "")

        print("  [OK] Device set to: {}".format(device))
        return gpus, device

    except ImportError:
        print("  [WARN] PyTorch not installed. Defaulting to cpu.")
        return None, "cpu"


# ---------------------------------------------------------
# STEP 5 - Training Config
# ---------------------------------------------------------

TRAIN_DEFAULTS = {
    "epochs"  : 100,
    "imgsz"   : 640,
    "batch"   : 16,
    "lr0"     : 0.01,
    "patience": 50,
    "workers" : 8,
    "project" : "runs/train",
    "name"    : "exp",
}

def ask_train_config():
    print("\n-- Training Configuration -----------------")
    print("  (Press Enter to keep default value)\n")

    cfg = {}
    for key, default in TRAIN_DEFAULTS.items():
        raw = input("  {:10s} [{}]: ".format(key, default)).strip()
        if raw == "":
            cfg[key] = default
        else:
            try:
                cfg[key] = type(default)(raw)
            except ValueError:
                cfg[key] = raw

    print("\n  [OK] Training config:")
    for k, v in cfg.items():
        print("       {:10s}: {}".format(k, v))
    return cfg


# ---------------------------------------------------------
# STEP 6 - Launch Training
# ---------------------------------------------------------

def launch_training(model_pt, yaml_path, device, train_cfg, dry_run=False):
    print("\n-- Launching Training ---------------------")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    kwargs = {
        "data"  : str(yaml_path),
        "device": device,
        **train_cfg,
    }

    if dry_run:
        print("\n  *** DRY RUN - no actual training will occur ***\n")
        print("  model = YOLO(\"{}\")".format(model_pt))
        print("  model.train(")
        for k, v in kwargs.items():
            print("      {}={},".format(k, repr(v)))
        print("  )")
        return

    model   = YOLO(model_pt)
    print("\n  Starting training...")
    results = model.train(**kwargs)
    print("\n  [OK] Training complete! Results -> {}/{}".format(train_cfg["project"], train_cfg["name"]))
    return results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main(dry_run=False):
    # 1. Dataset
    dataset_root = get_dataset_root()
    print("\n-- Dataset: {} --".format(dataset_root))
    class_names  = parse_classes(dataset_root)
    splits       = detect_splits(dataset_root)

    # 2. data.yaml
    yaml_path = write_data_yaml(dataset_root, splits, class_names)

    # 3. Model
    model_pt = choose_model()

    # 4. GPU
    _, device = detect_gpus()

    # 5. Config
    train_cfg = ask_train_config()

    # 6. Confirm & launch
    print("\n+======================================+")
    print("|          Ready to Train              |")
    print("+======================================+")
    print("  Model  : {}".format(model_pt))
    print("  Data   : {}".format(yaml_path))
    print("  Device : {}".format(device))
    print("  Epochs : {}".format(train_cfg["epochs"]))

    go = input("\n>> Start training? [Y/n]: ").strip().lower()
    if go == "n":
        print("  Aborted.")
        return

    launch_training(model_pt, yaml_path, device, train_cfg, dry_run=dry_run)


if __name__ == "__main__":
    DRY_RUN = "--dry-run" in sys.argv
    main(dry_run=DRY_RUN)