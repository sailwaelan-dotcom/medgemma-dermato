"""Train Edge-native System 1 ConvNeXt-Tiny on multi-center ISIC data and export to ONNX."""

import argparse
import json
import math
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from medgemma_dermato.data.isic_ingest import ISICRecord, load_records_jsonl
from medgemma_dermato.eval.calibration import (
    RiskScoreCase,
    apply_temperature,
    compute_auprc,
    compute_auroc,
    fit_temperature_scaling,
    select_threshold_for_sensitivity,
)
from medgemma_dermato.schema import map_isic_to_entity_category
from medgemma_dermato.system1 import (
    ENTITY_LABELS,
    ENTITY_TO_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    System1Config,
    build_tta4_views_pil,
    logits_to_system1_probs,
    preprocess_batch_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multi-center System 1 ConvNeXt-Tiny and export ONNX + TTA scores."
    )
    parser.add_argument("--train-manifest", type=Path, default=Path("data/system1_v1/train.jsonl"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/system1_v1/val.jsonl"))
    parser.add_argument("--hiba-val-manifest", type=Path, default=Path("data/isic_v1/val.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/isic_v1/test.jsonl"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/system1_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/system1_fusion"))
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny.fb_in22k_ft_in1k_384",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--drop-path-rate", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--skip-onnx-export", action="store_true")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _download_one(
    record: ISICRecord,
    image_dir: Path,
    timeout: float = 30.0,
) -> tuple[str, bool, str | None]:
    target = image_dir / f"{record.isic_id}.jpg"
    if target.exists() and target.stat().st_size > 100:
        return record.isic_id, True, None
    if not record.image_url:
        return record.isic_id, False, "missing image_url"
    last_err = "unknown"
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(record.image_url)
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                tmp = image_dir / f".{record.isic_id}.part"
                tmp.write_bytes(resp.content)
                tmp.replace(target)
                return record.isic_id, True, None
        except Exception as exc:
            last_err = str(exc)
            time.sleep(0.8 * (attempt + 1))
    return record.isic_id, False, last_err


def prefetch_images(
    records: list[ISICRecord],
    image_dir: Path,
    max_workers: int = 24,
) -> dict[str, Path]:
    """Concurrently download all unique ISIC images and verify readability."""
    image_dir.mkdir(parents=True, exist_ok=True)
    unique: dict[str, ISICRecord] = {}
    for rec in records:
        unique.setdefault(rec.isic_id, rec)

    missing = [
        rec
        for rec in unique.values()
        if not ((image_dir / f"{rec.isic_id}.jpg").exists()
                and (image_dir / f"{rec.isic_id}.jpg").stat().st_size > 100)
    ]
    print(
        f"Prefetching images: total_unique={len(unique)}, "
        f"already_cached={len(unique) - len(missing)}, to_download={len(missing)}",
        flush=True,
    )
    t0 = time.perf_counter()
    failures: list[tuple[str, str]] = []
    if missing:
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_download_one, rec, image_dir) for rec in missing]
            for fut in as_completed(futs):
                isic_id, ok, err = fut.result()
                done += 1
                if not ok:
                    failures.append((isic_id, err or "failed"))
                if done % 500 == 0 or done == len(missing):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"  downloaded {done}/{len(missing)} "
                        f"({len(failures)} failed) in {elapsed:.1f}s",
                        flush=True,
                    )

    valid_paths: dict[str, Path] = {}
    for isic_id in unique:
        p = image_dir / f"{isic_id}.jpg"
        if p.exists() and p.stat().st_size > 100:
            valid_paths[isic_id] = p
    if failures:
        print(
            f"WARNING: {len(failures)} images failed download (first 5: {failures[:5]})",
            flush=True,
        )
    return valid_paths


class RandomQuarterTurn:
    """Randomly rotate dermoscopic image by 0, 90, 180, or 270 degrees."""

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        if k == 1:
            return img.transpose(Image.Transpose.ROTATE_90)
        if k == 2:
            return img.transpose(Image.Transpose.ROTATE_180)
        if k == 3:
            return img.transpose(Image.Transpose.ROTATE_270)
        return img


class DermoscopySystem1Dataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset returning augmented dermoscopic tensors and dual-head labels."""

    def __init__(
        self,
        records: list[ISICRecord],
        image_paths: dict[str, Path],
        transform: Any,
    ) -> None:
        self.samples: list[tuple[ISICRecord, Path, int, float, float, float]] = []
        for rec in records:
            p = image_paths.get(rec.isic_id)
            if p is None:
                continue
            entity = map_isic_to_entity_category(rec.diagnosis_1, rec.diagnosis_2, rec.diagnosis_3)
            ent_idx = ENTITY_TO_INDEX[entity]
            if rec.is_malignant is True:
                bin_label = 1.0
                bin_mask = 1.0
                sample_w = 1.5 if entity in ("melanoma", "squamous_cell_carcinoma") else 1.0
            elif rec.is_malignant is False:
                bin_label = 0.0
                bin_mask = 1.0
                sample_w = 1.0
            else:
                bin_label = 0.0
                bin_mask = 0.0
                sample_w = 0.0
            self.samples.append((rec, p, ent_idx, bin_label, bin_mask, sample_w))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec, path, ent_idx, bin_label, bin_mask, sample_w = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            tensor = self.transform(rgb)
        return {
            "isic_id": rec.isic_id,
            "pixel_values": tensor,
            "entity_label": torch.tensor(ent_idx, dtype=torch.long),
            "binary_label": torch.tensor(bin_label, dtype=torch.float32),
            "binary_mask": torch.tensor(bin_mask, dtype=torch.float32),
            "sample_weight": torch.tensor(sample_w, dtype=torch.float32),
        }


class DualHeadConvNeXt(nn.Module):
    """Edge-ready ConvNeXt backbone with binary malignancy head + 9-class entity head."""

    def __init__(
        self,
        backbone_name: str,
        num_entities: int = len(ENTITY_LABELS),
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = int(self.backbone.num_features)
        self.head_drop = nn.Dropout(p=0.2)
        self.malignancy_head = nn.Linear(feat_dim, 1)
        self.entity_head = nn.Linear(feat_dim, num_entities)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.head_drop(self.backbone(pixel_values))
        mal_logit = self.malignancy_head(feats).squeeze(-1)
        ent_logits = self.entity_head(feats)
        return mal_logit, ent_logits


@torch.no_grad()
def evaluate_single_view(
    model: DualHeadConvNeXt,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    use_bf16: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run fast single-view validation epoch and return AUROC & AUPRC."""
    model.eval()
    y_true: list[bool] = []
    p_mal_list: list[float] = []
    p_risk_list: list[float] = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        px = batch["pixel_values"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            mal_logits, ent_logits = model(px)
        mal_np = mal_logits.float().cpu().numpy()
        ent_np = ent_logits.float().cpu().numpy()
        masks = batch["binary_mask"].numpy()
        labels = batch["binary_label"].numpy()
        for i in range(len(mal_np)):
            if masks[i] < 0.5:
                continue
            probs = logits_to_system1_probs(mal_np[i : i + 1], ent_np[i : i + 1])
            y_true.append(bool(labels[i] > 0.5))
            p_mal_list.append(probs["p_malignant"])
            p_risk_list.append(probs["p_risk"])

    if len(set(y_true)) < 2:
        return {"auroc_risk": 0.5, "auprc_risk": 0.5, "auroc_mal": 0.5}
    return {
        "auroc_risk": compute_auroc(y_true, p_risk_list),
        "auprc_risk": compute_auprc(y_true, p_risk_list),
        "auroc_mal": compute_auroc(y_true, p_mal_list),
    }


@torch.no_grad()
def score_manifest_tta4(
    model: DualHeadConvNeXt,
    records: list[ISICRecord],
    image_paths: dict[str, Path],
    image_size: int,
    device: torch.device,
    use_bf16: bool,
    limit: int | None = None,
) -> list[RiskScoreCase]:
    """Run exact 4-view TTA inference matching Edge ONNX preprocessing and output RiskScoreCase."""
    model.eval()
    target_recs = records[:limit] if limit is not None else records
    cases: list[RiskScoreCase] = []
    for rec in target_recs:
        path = image_paths.get(rec.isic_id)
        if path is None:
            continue
        t0 = time.perf_counter()
        with Image.open(path) as img:
            views = build_tta4_views_pil(img)
        batch_np = preprocess_batch_numpy(
            views,
            image_size=image_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        px = torch.from_numpy(batch_np).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            mal_logits, ent_logits = model(px)
        probs = logits_to_system1_probs(
            mal_logits.float().cpu().numpy(),
            ent_logits.float().cpu().numpy(),
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        gt_entity = map_isic_to_entity_category(rec.diagnosis_1, rec.diagnosis_2, rec.diagnosis_3)
        cases.append(
            RiskScoreCase(
                isic_id=rec.isic_id,
                ground_truth_malignant=rec.is_malignant,
                ground_truth_is_melanoma=(gt_entity == "melanoma"),
                ground_truth_entity=gt_entity,
                melanocytic_probs={},
                entity_probs=probs["entity_probs"],
                predicted_entity_argmax=probs["predicted_entity"],
                p_malignant=probs["p_malignant"],
                p_indeterminate=probs["p_indeterminate"],
                p_risk=probs["p_risk"],
                p_melanoma=probs["p_melanoma"],
                latency_ms=latency_ms,
            )
        )
    return cases


def save_score_cases(cases: list[RiskScoreCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")


def export_and_verify_onnx(
    model: DualHeadConvNeXt,
    onnx_path: Path,
    image_size: int,
    sample_image_paths: list[Path],
) -> dict[str, Any]:
    """Export FP32 ONNX graph (single file) and assert PyTorch vs ONNX Runtime parity."""
    import onnxruntime as ort

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.to("cpu").eval()
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    export_kwargs: dict[str, Any] = {
        "input_names": ["pixel_values"],
        "output_names": ["malignancy_logit", "entity_logits"],
        "dynamic_axes": {
            "pixel_values": {0: "batch"},
            "malignancy_logit": {0: "batch"},
            "entity_logits": {0: "batch"},
        },
        "opset_version": 17,
        "do_constant_folding": True,
    }
    try:
        torch.onnx.export(cpu_model, (dummy,), str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(cpu_model, (dummy,), str(onnx_path), **export_kwargs)

    # Verify numerical parity against ONNX Runtime CPUExecutionProvider on real images
    imgs: list[Image.Image] = []
    for p in sample_image_paths[:4]:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    batch_np = preprocess_batch_numpy(imgs, image_size=image_size)
    with torch.no_grad():
        pt_mal, pt_ent = cpu_model(torch.from_numpy(batch_np))
    pt_mal_np = pt_mal.numpy()
    pt_ent_np = pt_ent.numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_mal_np, ort_ent_np = sess.run(None, {"pixel_values": batch_np})

    max_diff_mal = float(np.max(np.abs(pt_mal_np - ort_mal_np)))
    max_diff_ent = float(np.max(np.abs(pt_ent_np - ort_ent_np)))
    size_mb = round(onnx_path.stat().st_size / (1024 * 1024), 2)
    print(
        f"ONNX export verified: {onnx_path} ({size_mb} MB), "
        f"max_diff_mal={max_diff_mal:.6f}, max_diff_ent={max_diff_ent:.6f}",
        flush=True,
    )
    if max(max_diff_mal, max_diff_ent) >= 1e-2:
        raise RuntimeError(
            f"ONNX parity check failed: max_diff_mal={max_diff_mal}, max_diff_ent={max_diff_ent}"
        )
    return {
        "onnx_path": str(onnx_path),
        "onnx_size_mb": size_mb,
        "max_abs_diff_malignancy_logit": round(max_diff_mal, 6),
        "max_abs_diff_entity_logits": round(max_diff_ent, 6),
        "parity_passed": True,
    }


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    t_start = time.perf_counter()

    train_records = load_records_jsonl(args.train_manifest)
    val_records = load_records_jsonl(args.val_manifest)
    hiba_val_records = load_records_jsonl(args.hiba_val_manifest)
    test_records = load_records_jsonl(args.test_manifest)

    # Anti-leakage verification right before training
    train_ids = {r.isic_id for r in train_records}
    val_ids = {r.isic_id for r in val_records}
    test_ids = {r.isic_id for r in test_records}
    assert not (train_ids & test_ids), "FATAL: train/test overlap detected!"
    assert not (val_ids & test_ids), "FATAL: val/test overlap detected!"
    assert not (train_ids & val_ids), "FATAL: train/val overlap detected!"

    all_records = train_records + val_records + hiba_val_records + test_records
    image_paths = prefetch_images(
        all_records,
        image_dir=args.image_dir,
        max_workers=args.download_workers,
    )

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.image_size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomQuarterTurn(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = DermoscopySystem1Dataset(train_records, image_paths, train_tf)
    val_ds = DermoscopySystem1Dataset(val_records, image_paths, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(len(train_ds) >= args.batch_size),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Class balancing statistics strictly from training split
    n_pos = sum(1 for r in train_records if r.isic_id in image_paths and r.is_malignant is True)
    n_neg = sum(1 for r in train_records if r.isic_id in image_paths and r.is_malignant is False)
    pos_weight_val = float(np.clip(n_neg / max(n_pos, 1), 0.5, 3.0))

    ent_counts = Counter(
        ENTITY_TO_INDEX[map_isic_to_entity_category(r.diagnosis_1, r.diagnosis_2, r.diagnosis_3)]
        for r in train_records
        if r.isic_id in image_paths
    )
    ent_weights = []
    mean_cnt = sum(ent_counts.values()) / float(len(ENTITY_LABELS))
    for i in range(len(ENTITY_LABELS)):
        cnt = max(ent_counts.get(i, 1), 1)
        ent_weights.append(float(np.clip(math.sqrt(mean_cnt / cnt), 0.5, 4.0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(
        f"Training System 1: backbone={args.backbone}, device={device}, bf16={use_bf16}, "
        f"train={len(train_ds)}, val={len(val_ds)}, pos_weight={pos_weight_val:.3f}",
        flush=True,
    )

    model = DualHeadConvNeXt(
        backbone_name=args.backbone,
        num_entities=len(ENTITY_LABELS),
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    pos_weight_tensor = torch.tensor(pos_weight_val, dtype=torch.float32, device=device)
    ent_weight_tensor = torch.tensor(ent_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = (
        min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, min(steps_per_epoch, int(0.1 * total_steps)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.checkpoint_dir / "best_system1.pt"

    history: list[dict[str, Any]] = []
    best_val_auroc = -1.0
    best_epoch = -1
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_steps = 0

        for b_idx, batch in enumerate(train_loader):
            if args.max_batches is not None and b_idx >= args.max_batches:
                break
            px = batch["pixel_values"].to(device, non_blocking=True)
            ent_y = batch["entity_label"].to(device, non_blocking=True)
            bin_y = batch["binary_label"].to(device, non_blocking=True)
            bin_m = batch["binary_mask"].to(device, non_blocking=True)
            sw = batch["sample_weight"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                mal_logits, ent_logits = model(px)
                bce_raw = F.binary_cross_entropy_with_logits(
                    mal_logits,
                    bin_y,
                    pos_weight=pos_weight_tensor,
                    reduction="none",
                )
                denom = torch.clamp(torch.sum(bin_m * sw), min=1.0)
                loss_bce = torch.sum(bce_raw * bin_m * sw) / denom
                loss_ent = F.cross_entropy(
                    ent_logits,
                    ent_y,
                    weight=ent_weight_tensor,
                    label_smoothing=0.05,
                )
                loss = loss_bce + 0.5 * loss_ent

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            n_steps += 1
            global_step += 1

        val_metrics = evaluate_single_view(
            model,
            val_loader,
            device=device,
            use_bf16=use_bf16,
            max_batches=args.max_batches,
        )
        ep_sec = round(time.perf_counter() - ep_t0, 1)
        avg_loss = round(running_loss / max(n_steps, 1), 4)
        lr_now = scheduler.get_last_lr()[0]
        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auroc_risk": val_metrics["auroc_risk"],
            "val_auroc_mal": val_metrics["auroc_mal"],
            "val_auprc_risk": val_metrics["auprc_risk"],
            "lr": round(lr_now, 7),
            "elapsed_s": ep_sec,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | loss={avg_loss:.4f} | "
            f"val_auroc_risk={val_metrics['auroc_risk']:.4f} | "
            f"val_auroc_mal={val_metrics['auroc_mal']:.4f} | "
            f"val_auprc={val_metrics['auprc_risk']:.4f} | {ep_sec}s",
            flush=True,
        )
        if val_metrics["auroc_risk"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc_risk"]
            best_epoch = epoch
            torch.save(
                {
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_auroc_risk": best_val_auroc,
                    "state_dict": model.state_dict(),
                },
                best_ckpt,
            )

    # Reload best checkpoint for 4-view TTA scoring and ONNX export
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    print(
        f"Loaded best checkpoint from epoch {best_epoch} "
        f"(val_auroc_risk={best_val_auroc:.4f})",
        flush=True,
    )

    eval_limit = args.max_batches * args.batch_size if args.max_batches else None
    val_cases = score_manifest_tta4(
        model, val_records, image_paths, args.image_size, device, use_bf16, limit=eval_limit
    )
    hiba_val_cases = score_manifest_tta4(
        model, hiba_val_records, image_paths, args.image_size, device, use_bf16, limit=eval_limit
    )
    test_cases = score_manifest_tta4(
        model, test_records, image_paths, args.image_size, device, use_bf16, limit=eval_limit
    )

    save_score_cases(val_cases, args.output_dir / "val_scores_sys1.jsonl")
    save_score_cases(hiba_val_cases, args.output_dir / "hiba_val_scores_sys1.jsonl")
    save_score_cases(test_cases, args.output_dir / "test_scores_sys1.jsonl")

    # Fit temperature scaling & sensitivity thresholds strictly on multi-center val
    val_bin = [c for c in val_cases if c.ground_truth_malignant is not None]
    y_val = [bool(c.ground_truth_malignant) for c in val_bin]
    raw_risk_val = [c.p_risk for c in val_bin]
    raw_mal_val = [c.p_malignant for c in val_bin]

    temp_risk = fit_temperature_scaling(y_val, raw_risk_val) if len(set(y_val)) == 2 else 1.0
    temp_mal = fit_temperature_scaling(y_val, raw_mal_val) if len(set(y_val)) == 2 else 1.0
    cal_risk_val = apply_temperature(raw_risk_val, temp_risk)
    cal_mal_val = apply_temperature(raw_mal_val, temp_mal)

    sys1_cfg = System1Config(
        backbone_name=args.backbone,
        image_size=args.image_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        entity_labels=ENTITY_LABELS,
        temperature_malignant=temp_mal,
        temperature_risk=temp_risk,
        tau_mc_val_90_risk=select_threshold_for_sensitivity(y_val, cal_risk_val, 0.90)
        if any(y_val)
        else 0.5,
        tau_mc_val_95_risk=select_threshold_for_sensitivity(y_val, cal_risk_val, 0.95)
        if any(y_val)
        else 0.5,
        tau_mc_val_90_malignant=select_threshold_for_sensitivity(y_val, cal_mal_val, 0.90)
        if any(y_val)
        else 0.5,
        tau_mc_val_95_malignant=select_threshold_for_sensitivity(y_val, cal_mal_val, 0.95)
        if any(y_val)
        else 0.5,
    )
    cal_path = args.checkpoint_dir / "system1_calibration.json"
    cal_path.write_text(sys1_cfg.model_dump_json(indent=2) + "\n", encoding="utf-8")

    # Quick summary of TTA4 AUROC on val / hiba_val / test
    def _quick_auroc(cs: list[RiskScoreCase]) -> dict[str, float]:
        b = [c for c in cs if c.ground_truth_malignant is not None]
        y = [bool(c.ground_truth_malignant) for c in b]
        if len(set(y)) < 2:
            return {"auroc_risk": 0.5, "auroc_mal": 0.5}
        return {
            "auroc_risk": compute_auroc(y, [c.p_risk for c in b]),
            "auroc_mal": compute_auroc(y, [c.p_malignant for c in b]),
            "auprc_risk": compute_auprc(y, [c.p_risk for c in b]),
        }

    tta_summary = {
        "multicenter_val_tta4": _quick_auroc(val_cases),
        "hiba_val_tta4": _quick_auroc(hiba_val_cases),
        "mskcc_test_tta4": _quick_auroc(test_cases),
    }
    print("TTA4 summary:", json.dumps(tta_summary, indent=2), flush=True)

    onnx_info: dict[str, Any] = {"skipped": True}
    if not args.skip_onnx_export:
        onnx_path = args.checkpoint_dir / "system1_convnext_tiny_fp32.onnx"
        sample_paths = [
            image_paths[r.isic_id] for r in test_records if r.isic_id in image_paths
        ][:4]
        onnx_info = export_and_verify_onnx(model, onnx_path, args.image_size, sample_paths)

    provenance = {
        "backbone": args.backbone,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "hiba_val_count": len(hiba_val_cases),
        "test_count": len(test_cases),
        "pos_weight": round(pos_weight_val, 4),
        "best_epoch": best_epoch,
        "best_val_auroc_risk_1view": best_val_auroc,
        "tta4_summary": tta_summary,
        "calibration": sys1_cfg.model_dump(),
        "onnx": onnx_info,
        "history": history,
        "total_elapsed_s": round(time.perf_counter() - t_start, 1),
    }
    prov_path = args.checkpoint_dir / "system1_training_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Done in {provenance['total_elapsed_s']}s. Provenance -> {prov_path}", flush=True)


if __name__ == "__main__":
    main()
