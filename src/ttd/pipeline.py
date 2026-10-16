from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .baselines import (
    diffusion_synthetic,
    gradient_matching,
    total_variation,
    trajectory_matching,
)
from .config import ExperimentConfig, dump_config
from .data import (
    TensorDataset,
    build_datasets,
    class_indices,
    dataset_spec,
    denormalize,
    normalized_bounds,
)
from .diffusion import DiffusionSchedule
from .models import LatentDiffusion, classifier
from .topology import diagram_distance, persistence_signature
from .trajectory import feature_statistics, trajectory_embeddings, trajectory_noise_shape
from .utils import (
    environment_info,
    fingerprint,
    resolve_device,
    save_json,
    seed_everything,
    tensor_fingerprint,
)


def build_backbone(cfg: ExperimentConfig, device: torch.device) -> LatentDiffusion:
    classes, _ = dataset_spec(cfg.dataset.name)
    model = LatentDiffusion(
        cfg.backbone.latent_channels, cfg.backbone.width, cfg.backbone.embed_dim, classes
    ).to(device)
    checkpoint = cfg.backbone.checkpoint
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        stats_path = Path(checkpoint).with_name("latent_stats.pt")
        if not stats_path.exists():
            raise FileNotFoundError(f"required latent statistics artifact is missing: {stats_path}")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        actual_fingerprint = tensor_fingerprint(stats["mean"], stats["std"])
        if actual_fingerprint != payload.get("latent_stats_fingerprint"):
            raise ValueError("latent statistics fingerprint does not match backbone checkpoint")
        if cfg.backbone.use_ema and payload.get("ema_denoiser"):
            model.denoiser.load_state_dict(payload["ema_denoiser"])
    return model


def _artifact_meta(cfg: ExperimentConfig, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "config": asdict(cfg),
        "config_fingerprint": fingerprint(asdict(cfg)),
        "environment": environment_info(),
        "created_unix": time.time(),
    }


def _target_contract(cfg: ExperimentConfig) -> dict[str, Any]:
    checkpoint = cfg.backbone.checkpoint
    return {
        "dataset": cfg.dataset.name,
        "image_size": cfg.dataset.image_size,
        "backbone_checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
        "diffusion_steps": cfg.trajectory.diffusion_steps,
        "timesteps": cfg.trajectory.timesteps,
        "space": cfg.trajectory.space,
        "direction": cfg.trajectory.direction,
        "conditioning": cfg.trajectory.conditioning,
        "source": cfg.trajectory.source,
        "representation": cfg.trajectory.representation,
        "inference_steps": cfg.trajectory.inference_steps,
        "metric": cfg.topology.metric,
        "signature_size": cfg.topology.signature_size,
        "max_points_per_class": cfg.topology.max_points_per_class,
    }


def _training_split(cfg: ExperimentConfig):
    augmented, _ = build_datasets(cfg.dataset)
    clean, _ = build_datasets(cfg.dataset, train_augment=False)
    generator = torch.Generator().manual_seed(cfg.seed)
    order = torch.randperm(len(clean), generator=generator).tolist()
    validation_count = max(1, round(len(clean) * cfg.backbone.validation_fraction))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    return Subset(augmented, training_indices), Subset(clean, training_indices), Subset(
        clean, validation_indices
    )


def _training_loaders(cfg: ExperimentConfig, device: torch.device):
    train_augmented, train_clean, validation = _training_split(cfg)
    common = {
        "batch_size": cfg.backbone.batch_size,
        "num_workers": cfg.dataset.num_workers,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_augmented, shuffle=True, **common),
        DataLoader(train_clean, shuffle=False, **common),
        DataLoader(validation, shuffle=False, **common),
    )


def _new_model(cfg: ExperimentConfig, device: torch.device) -> LatentDiffusion:
    classes, _ = dataset_spec(cfg.dataset.name)
    return LatentDiffusion(
        cfg.backbone.latent_channels, cfg.backbone.width, cfg.backbone.embed_dim, classes
    ).to(device)


def _load_autoencoder_into(model: LatentDiffusion, path: str | Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "autoencoder" not in payload:
        raise ValueError(f"{path} is not an autoencoder checkpoint")
    model.autoencoder.load_state_dict(payload["autoencoder"])


def _load_latent_stats_into(
    model: LatentDiffusion, path: str | Path, device: torch.device
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.set_latent_stats(payload["mean"].to(device), payload["std"].to(device))
    return payload


def _load_json_if_exists(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def _fit_latent_statistics(model: LatentDiffusion, loader: DataLoader, device: torch.device):
    channel_sum = torch.zeros(model.latent_mean.shape[1], device=device, dtype=torch.float64)
    channel_square_sum = torch.zeros_like(channel_sum)
    count = 0
    model.autoencoder.eval()
    for images, _ in tqdm(loader, desc="latent statistics", leave=False):
        latent = model.encode_raw(images.to(device)).double()
        channel_sum += latent.sum(dim=(0, 2, 3))
        channel_square_sum += latent.square().sum(dim=(0, 2, 3))
        count += latent.shape[0] * latent.shape[2] * latent.shape[3]
    mean = channel_sum / count
    variance = channel_square_sum / count - mean.square()
    std = variance.clamp_min(1e-12).sqrt()
    model.set_latent_stats(mean.float(), std.float())
    return mean.float(), std.float(), count


@torch.no_grad()
def _validate_autoencoder(model, loader, dataset_name: str, device: torch.device):
    squared_error = count = 0
    model.autoencoder.eval()
    for images, _ in loader:
        images = images.to(device)
        reconstruction = model.autoencoder(images)
        pixel_images = denormalize(images, dataset_name)
        pixel_reconstruction = denormalize(reconstruction, dataset_name)
        squared_error += float((pixel_images - pixel_reconstruction).square().sum())
        count += pixel_images.numel()
    mse = squared_error / count
    return {"pixel_mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12))}


@torch.no_grad()
def _validate_denoiser(model, loader, schedule, cfg, device):
    totals: dict[int, dict[str, float]] = {
        timestep: {
            "random": 0.0,
            "zero": 0.0,
            "unconditional": 0.0,
            "class": 0.0,
            "count": 0,
        }
        for timestep in cfg.trajectory.timesteps
    }
    model.eval()
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 404)
    for batch_index, (images, labels) in enumerate(loader):
        if batch_index >= cfg.backbone.validation_batches:
            break
        images, labels = images.to(device), labels.to(device)
        z0 = model.encode(images)
        noise = torch.randn(z0.shape, generator=generator, device=device)
        unconditional = model.unconditional_labels(len(images), device)
        for timestep in cfg.trajectory.timesteps:
            t = torch.full((len(images),), timestep, dtype=torch.long, device=device)
            zt = schedule.q_sample(z0, t, noise)
            predictions = {
                "unconditional": model.denoiser(zt, t, unconditional),
                "class": model.denoiser(zt, t, labels),
            }
            random_prediction = torch.randn(noise.shape, generator=generator, device=device)
            totals[timestep]["random"] += float((random_prediction - noise).square().sum())
            totals[timestep]["zero"] += float(noise.square().sum())
            totals[timestep]["unconditional"] += float(
                (predictions["unconditional"] - noise).square().sum()
            )
            totals[timestep]["class"] += float((predictions["class"] - noise).square().sum())
            totals[timestep]["count"] += noise.numel()
    by_time = {
        str(timestep): {
            key: values[key] / values["count"]
            for key in ("random", "zero", "unconditional", "class")
        }
        for timestep, values in totals.items()
    }
    aggregate = {
        key: sum(values[key] for values in by_time.values()) / len(by_time)
        for key in ("random", "zero", "unconditional", "class")
    }
    return {"timesteps": by_time, "aggregate": aggregate}


def _require_training_checkpoint_null(cfg: ExperimentConfig) -> None:
    if cfg.backbone.checkpoint:
        raise ValueError("training stages expect backbone.checkpoint=null; resume is not implemented")


def train_autoencoder(cfg: ExperimentConfig) -> Path:
    """Stage 1: train the compact autoencoder without touching the denoiser."""
    _require_training_checkpoint_null(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    train_loader, _, validation_loader = _training_loaders(cfg, device)
    model = _new_model(cfg, device)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.autoencoder.parameters(), lr=cfg.backbone.ae_lr, weight_decay=1e-4
    )
    history = []
    for epoch in range(cfg.backbone.ae_epochs):
        model.autoencoder.train()
        total = count = 0
        for images, _ in tqdm(train_loader, desc=f"AE {epoch + 1}", leave=False):
            images = images.to(device)
            loss = F.mse_loss(model.autoencoder(images), images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
            count += len(images)
        history.append({"epoch": epoch + 1, "reconstruction": total / count})

    validation = _validate_autoencoder(model, validation_loader, cfg.dataset.name, device)
    out = cfg.run_dir / "autoencoder.pt"
    torch.save(
        {"autoencoder": model.autoencoder.state_dict(), "meta": _artifact_meta(cfg, "autoencoder")},
        out,
    )
    save_json(history, cfg.run_dir / "ae_history.json")
    save_json(validation, cfg.run_dir / "autoencoder_validation.json")
    dump_config(cfg, cfg.run_dir / "resolved_config.yaml")
    return out


def fit_latent_statistics(cfg: ExperimentConfig) -> Path:
    """Stage 2: fit and freeze channel-wise latent statistics."""
    _require_training_checkpoint_null(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    _, clean_loader, _ = _training_loaders(cfg, device)
    autoencoder_path = cfg.run_dir / "autoencoder.pt"
    if not autoencoder_path.exists():
        raise FileNotFoundError(
            f"autoencoder checkpoint not found: {autoencoder_path}; run train-autoencoder first"
        )
    model = _new_model(cfg, device)
    _load_autoencoder_into(model, autoencoder_path, device)
    mean, std, statistic_count = _fit_latent_statistics(model, clean_loader, device)
    latent_stats = {
        "mean": mean.cpu(),
        "std": std.cpu(),
        "channels": cfg.backbone.latent_channels,
        "sample_values_per_channel": statistic_count,
        "meta": _artifact_meta(cfg, "latent_statistics"),
    }
    latent_stats["fingerprint"] = tensor_fingerprint(latent_stats["mean"], latent_stats["std"])
    out = cfg.run_dir / "latent_stats.pt"
    torch.save(latent_stats, out)
    save_json(
        {
            "raw_mean": mean.tolist(),
            "raw_std": std.tolist(),
            "normalized_mean": [0.0] * len(mean),
            "normalized_std": [1.0] * len(std),
            "sample_values_per_channel": statistic_count,
            "fingerprint": latent_stats["fingerprint"],
        },
        cfg.run_dir / "latent_statistics.json",
    )
    dump_config(cfg, cfg.run_dir / "resolved_config.yaml")
    return out


def train_diffusion(cfg: ExperimentConfig) -> Path:
    """Stage 3: train the denoiser on the frozen, normalized latent space."""
    _require_training_checkpoint_null(cfg)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    train_loader, _, validation_loader = _training_loaders(cfg, device)
    autoencoder_path = cfg.run_dir / "autoencoder.pt"
    stats_path = cfg.run_dir / "latent_stats.pt"
    if not autoencoder_path.exists() or not stats_path.exists():
        raise FileNotFoundError(
            f"expected {autoencoder_path} and {stats_path}; run train-autoencoder and "
            "fit-latent-stats first"
        )
    model = _new_model(cfg, device)
    _load_autoencoder_into(model, autoencoder_path, device)
    latent_stats = _load_latent_stats_into(model, stats_path, device)
    for parameter in model.autoencoder.parameters():
        parameter.requires_grad_(False)
    model.autoencoder.eval()

    classes, _ = dataset_spec(cfg.dataset.name)
    aux_head = nn.Linear(cfg.backbone.embed_dim, classes).to(device)
    diffusion_parameters = list(model.denoiser.parameters()) + list(model.features.parameters())
    diffusion_parameters += list(aux_head.parameters())
    optimizer = torch.optim.AdamW(
        diffusion_parameters, lr=cfg.backbone.diffusion_lr, weight_decay=1e-4
    )
    schedule = DiffusionSchedule(
        cfg.trajectory.diffusion_steps,
        cfg.trajectory.beta_start,
        cfg.trajectory.beta_end,
        device,
    )
    ema = {key: value.detach().clone() for key, value in model.denoiser.state_dict().items()}
    diffusion_history = []
    best_score = float("inf")
    best_model_state = None
    best_ema = None
    for epoch in range(cfg.backbone.diffusion_epochs):
        model.denoiser.train()
        model.features.train()
        totals = {"loss": 0.0, "diffusion": 0.0, "semantic": 0.0}
        count = 0
        for images, labels in tqdm(train_loader, desc=f"diffusion {epoch + 1}", leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                z0 = model.encode(images)
            noise = torch.randn_like(z0)
            t = torch.randint(schedule.steps, (len(images),), device=device)
            zt = schedule.q_sample(z0, t, noise)
            denoiser_labels = labels.clone()
            drop_mask = torch.rand(len(labels), device=device) < cfg.backbone.condition_dropout
            denoiser_labels[drop_mask] = model.denoiser.null_label
            prediction = model.denoiser(zt, t, denoiser_labels)
            embeddings = model.embed(z0)
            diffusion_loss = F.mse_loss(prediction, noise)
            semantic_loss = F.cross_entropy(aux_head(embeddings), labels)
            loss = diffusion_loss + cfg.backbone.semantic_weight * semantic_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion_parameters, 1.0)
            optimizer.step()
            with torch.no_grad():
                for key, value in model.denoiser.state_dict().items():
                    ema[key].lerp_(value, 1 - cfg.backbone.ema_decay)
            batch = len(images)
            count += batch
            for key, value in {
                "loss": loss,
                "diffusion": diffusion_loss,
                "semantic": semantic_loss,
            }.items():
                totals[key] += float(value.detach()) * batch
        current_denoiser = {
            key: value.detach().clone() for key, value in model.denoiser.state_dict().items()
        }
        model.denoiser.load_state_dict(ema)
        validation_report = _validate_denoiser(model, validation_loader, schedule, cfg, device)
        score = validation_report["aggregate"]["unconditional"]
        if score < best_score:
            best_score = score
            best_model_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_ema = {key: value.detach().cpu().clone() for key, value in ema.items()}
        model.denoiser.load_state_dict(current_denoiser)
        diffusion_history.append(
            {
                "epoch": epoch + 1,
                **{key: value / count for key, value in totals.items()},
                "validation": validation_report,
            }
        )
    if best_model_state is None or best_ema is None:
        raise RuntimeError("diffusion training produced no checkpoint")
    model.load_state_dict(best_model_state)
    final_validation = _validate_denoiser(model, validation_loader, schedule, cfg, device)
    frozen = not any(parameter.requires_grad for parameter in model.autoencoder.parameters())
    aggregate = final_validation["aggregate"]
    validation_report = {
        "autoencoder": _load_json_if_exists(cfg.run_dir / "autoencoder_validation.json"),
        "latent_statistics": {
            "raw_mean": latent_stats["mean"].tolist(),
            "raw_std": latent_stats["std"].tolist(),
            "normalized_mean": [0.0] * len(latent_stats["mean"]),
            "normalized_std": [1.0] * len(latent_stats["std"]),
        },
        "denoiser": final_validation,
        "gates": {
            "autoencoder_frozen": frozen,
            "learned_better_than_random": aggregate["unconditional"] < aggregate["random"],
            "learned_better_than_zero": aggregate["unconditional"] < aggregate["zero"],
            "noise_mse_below_target": aggregate["unconditional"]
            < cfg.backbone.noise_mse_target,
        },
    }
    out = cfg.run_dir / "backbone.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema_denoiser": best_ema,
            "aux_head": aux_head.state_dict(),
            "latent_stats_artifact": "latent_stats.pt",
            "latent_stats_fingerprint": latent_stats["fingerprint"],
            "meta": _artifact_meta(cfg, "two_stage_backbone"),
        },
        out,
    )
    save_json(diffusion_history, cfg.run_dir / "diffusion_history.json")
    save_json(validation_report, cfg.run_dir / "backbone_validation.json")
    dump_config(cfg, cfg.run_dir / "resolved_config.yaml")
    return out


def train_backbone(cfg: ExperimentConfig) -> Path:
    """Run the three frozen Phase 1-B training stages in order."""
    train_autoencoder(cfg)
    fit_latent_statistics(cfg)
    return train_diffusion(cfg)


@torch.no_grad()
def extract_targets(cfg: ExperimentConfig) -> Path:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    train_set, _ = build_datasets(cfg.dataset, train_augment=False)
    classes, _ = dataset_spec(cfg.dataset.name)
    model = build_backbone(cfg, device).eval()
    schedule = DiffusionSchedule(
        cfg.trajectory.diffusion_steps,
        cfg.trajectory.beta_start,
        cfg.trajectory.beta_end,
        device,
    )
    indices = class_indices(train_set, classes)
    generator = torch.Generator().manual_seed(cfg.seed)
    meta = _artifact_meta(cfg, "targets")
    meta["target_contract"] = _target_contract(cfg)
    target: dict[str, Any] = {"classes": {}, "meta": meta}
    for label in tqdm(range(classes), desc="real trajectory targets"):
        order = torch.randperm(len(indices[label]), generator=generator)
        chosen = [indices[label][i] for i in order[: cfg.topology.max_points_per_class].tolist()]
        images = torch.stack([train_set[i][0] for i in chosen]).to(device)
        # Fixed per-class noise makes target extraction deterministic.
        noise_gen = torch.Generator(device=device).manual_seed(cfg.seed * 10_000 + label)
        noise_shape = trajectory_noise_shape(model, images, cfg.trajectory.space)
        noise = torch.randn((len(images), *noise_shape), generator=noise_gen, device=device)
        embedding = trajectory_embeddings(
            model,
            images,
            cfg.trajectory.timesteps,
            schedule,
            noise=noise,
            space=cfg.trajectory.space,
            direction=cfg.trajectory.direction,
            labels=torch.full((len(images),), label, device=device, dtype=torch.long),
            conditioning=cfg.trajectory.conditioning,
            source=cfg.trajectory.source,
            representation=cfg.trajectory.representation,
            inference_steps=cfg.trajectory.inference_steps,
        )
        mean, variance = feature_statistics(embedding)
        signatures = torch.stack(
            [
                persistence_signature(
                    embedding[:, time_index],
                    cfg.topology.signature_size,
                    cfg.topology.metric,
                )
                for time_index in range(len(cfg.trajectory.timesteps))
            ]
        )
        target["classes"][label] = {
            "mean": mean.cpu(),
            "variance": variance.cpu(),
            "topology": signatures.cpu(),
        }
    # Class-balanced global point cloud: inter-class manifold target. With IPC=1
    # intra-class H0 topology is degenerate (one point per class), so the
    # topology loss relies on this entry to constrain inter-class structure.
    global_per_class = max(2, cfg.topology.max_points_per_class // classes)
    global_images = []
    global_labels: list[int] = []
    for label in range(classes):
        order = torch.randperm(len(indices[label]), generator=generator)
        chosen = [indices[label][i] for i in order[:global_per_class].tolist()]
        global_images.append(torch.stack([train_set[i][0] for i in chosen]))
        global_labels.extend([label] * len(chosen))
    global_inputs = torch.cat(global_images).to(device)
    noise_gen = torch.Generator(device=device).manual_seed(cfg.seed * 10_000 + classes)
    noise = torch.randn(
        (len(global_inputs), *trajectory_noise_shape(model, global_inputs, cfg.trajectory.space)),
        generator=noise_gen,
        device=device,
    )
    global_embedding = trajectory_embeddings(
        model,
        global_inputs,
        cfg.trajectory.timesteps,
        schedule,
        noise=noise,
        space=cfg.trajectory.space,
        direction=cfg.trajectory.direction,
        labels=torch.as_tensor(global_labels, device=device),
        conditioning=cfg.trajectory.conditioning,
        source=cfg.trajectory.source,
        representation=cfg.trajectory.representation,
        inference_steps=cfg.trajectory.inference_steps,
    )
    target["global"] = {
        "samples": len(global_inputs),
        "per_class": int(global_per_class),
        "topology": torch.stack(
            [
                persistence_signature(
                    global_embedding[:, time_index],
                    cfg.topology.signature_size,
                    cfg.topology.metric,
                )
                for time_index in range(len(cfg.trajectory.timesteps))
            ]
        ).cpu(),
    }
    out = cfg.run_dir / "targets.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target, out)
    return out


def _permute_class_targets(classes_map: dict[Any, Any]) -> dict[Any, Any]:
    """Cyclic class-label permutation of topology targets (attribution control).

    Every synthetic class is matched to the topology target of the next class,
    so the regularizer's form and magnitude are preserved but the class-specific
    structural content is wrong. The global entry is intentionally untouched
    (class-agnostic), which the control report must state.
    """
    keys = sorted(classes_map)
    if len(keys) < 2:
        raise ValueError("permutation control requires at least two classes")
    return {c: classes_map[keys[(i + 1) % len(keys)]] for i, c in enumerate(keys)}


def _initialize_synthetic(cfg: ExperimentConfig, device: torch.device):
    train_set, _ = build_datasets(cfg.dataset, train_augment=False)
    classes, _ = dataset_spec(cfg.dataset.name)
    labels = torch.arange(classes, device=device).repeat_interleave(cfg.distill.ipc)
    if cfg.distill.init == "noise":
        images = torch.randn(
            classes * cfg.distill.ipc,
            3,
            cfg.dataset.image_size,
            cfg.dataset.image_size,
            device=device,
        )
    elif cfg.distill.init == "real":
        indices = class_indices(train_set, classes)
        generator = torch.Generator().manual_seed(cfg.seed)
        values = []
        for c in range(classes):
            order = torch.randperm(len(indices[c]), generator=generator)[: cfg.distill.ipc]
            values.extend(train_set[indices[c][int(i)]][0] for i in order)
        images = torch.stack(values).to(device)
    else:
        raise ValueError(f"unsupported initialization: {cfg.distill.init}")
    return images, labels


@torch.no_grad()
def _selection_baseline(cfg: ExperimentConfig, model: LatentDiffusion, method: str, device):
    images, labels = _initialize_synthetic(cfg, device)
    if method == "random":
        return images, labels
    train_set, _ = build_datasets(cfg.dataset, train_augment=False)
    classes, _ = dataset_spec(cfg.dataset.name)
    all_indices = class_indices(train_set, classes)
    selected_images = []
    for c in tqdm(range(classes), desc="herding"):
        idx = all_indices[c]
        candidates = torch.stack([train_set[i][0] for i in idx]).to(device)
        features = model.embed(model.encode(candidates))
        mean = features.mean(0)
        remaining = torch.ones(len(candidates), dtype=torch.bool, device=device)
        running = torch.zeros_like(mean)
        for k in range(cfg.distill.ipc):
            scores = torch.norm((running[None] + features) / (k + 1) - mean[None], dim=1)
            scores[~remaining] = float("inf")
            choice = int(scores.argmin())
            selected_images.append(candidates[choice].cpu())
            running += features[choice]
            remaining[choice] = False
    return torch.stack(selected_images).to(device), labels


def distill(cfg: ExperimentConfig, targets_path: str | Path | None = None) -> Path:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    classes, _ = dataset_spec(cfg.dataset.name)
    model = build_backbone(cfg, device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if cfg.distill.method in {"random", "herding"}:
        images, labels = _selection_baseline(cfg, model, cfg.distill.method, device)
        history: list[dict[str, float]] = []
    elif cfg.distill.method in {
        "gradient_matching",
        "trajectory_matching",
        "trajectory_matching_tv",
        "diffusion",
    }:
        images, labels = _initialize_synthetic(cfg, device)
        if cfg.distill.method == "gradient_matching":
            images, history = gradient_matching(cfg, images, labels, device)
        elif cfg.distill.method == "trajectory_matching":
            images, history = trajectory_matching(cfg, images, labels, device)
        elif cfg.distill.method == "trajectory_matching_tv":
            images, history = trajectory_matching(
                cfg, images, labels, device, aux_loss_fn=total_variation
            )
        else:
            images, history = diffusion_synthetic(cfg, model, labels, images, device)
    elif cfg.distill.method == "trajectory_matching_topology":
        targets_path = Path(targets_path or cfg.run_dir / "targets.pt")
        target = torch.load(targets_path, map_location=device, weights_only=False)
        if target.get("meta", {}).get("target_contract") != _target_contract(cfg):
            raise ValueError(
                "Target artifact is incompatible with the current dataset/backbone/trajectory/topology "
                "configuration. Re-run extract-targets with these settings."
            )
        if "global" not in target:
            raise ValueError(
                "Target artifact has no global topology entry; re-run extract-targets."
            )
        if cfg.distill.permute_class_targets:
            target["classes"] = _permute_class_targets(target["classes"])
        images, labels = _initialize_synthetic(cfg, device)
        schedule = DiffusionSchedule(
            cfg.trajectory.diffusion_steps,
            cfg.trajectory.beta_start,
            cfg.trajectory.beta_end,
            device,
        )
        latent_shape = trajectory_noise_shape(model, images, cfg.trajectory.space)
        noise_gen = torch.Generator(device=device).manual_seed(cfg.seed + 17)
        fixed_noise = torch.randn((len(images), *latent_shape), generator=noise_gen, device=device)
        topology_pairs = [
            (c, ti) for c in range(classes) for ti in range(len(cfg.trajectory.timesteps))
        ]
        pair_generator = torch.Generator().manual_seed(cfg.seed + 29)

        def topology_loss_fn(current_images: torch.Tensor) -> torch.Tensor:
            embedding = trajectory_embeddings(
                model,
                current_images,
                cfg.trajectory.timesteps,
                schedule,
                fixed_noise,
                space=cfg.trajectory.space,
                direction=cfg.trajectory.direction,
                labels=labels,
                conditioning=cfg.trajectory.conditioning,
                source=cfg.trajectory.source,
                representation=cfg.trajectory.representation,
                inference_steps=cfg.trajectory.inference_steps,
            )
            pair_count = min(cfg.topology.pairs_per_step, len(topology_pairs))
            chosen = torch.randperm(len(topology_pairs), generator=pair_generator)[:pair_count]
            loss = current_images.new_zeros(())
            terms = 0
            for pair_index in chosen.tolist():
                c, ti = topology_pairs[pair_index]
                subset = embedding[labels == c, ti]
                if len(subset) < 2:
                    continue
                if len(subset) > cfg.topology.max_points_per_class:
                    start = int(torch.randint(len(subset), (1,), generator=pair_generator).item())
                    point_indices = torch.arange(
                        start, start + cfg.topology.max_points_per_class, device=device
                    )
                    subset = subset[point_indices % len(subset)]
                loss = loss + diagram_distance(
                    persistence_signature(subset, cfg.topology.signature_size, cfg.topology.metric),
                    target["classes"][c]["topology"][ti],
                )
                terms += 1
            for ti in range(len(cfg.trajectory.timesteps)):
                global_subset = embedding[:, ti]
                if len(global_subset) > cfg.topology.max_points_per_class:
                    start = int(
                        torch.randint(len(global_subset), (1,), generator=pair_generator).item()
                    )
                    point_indices = torch.arange(
                        start, start + cfg.topology.max_points_per_class, device=device
                    )
                    global_subset = global_subset[point_indices % len(global_subset)]
                loss = loss + diagram_distance(
                    persistence_signature(
                        global_subset, cfg.topology.signature_size, cfg.topology.metric
                    ),
                    target["global"]["topology"][ti],
                )
                terms += 1
            return loss / max(terms, 1)

        images, history = trajectory_matching(
            cfg, images, labels, device, aux_loss_fn=topology_loss_fn
        )
    else:
        targets_path = Path(targets_path or cfg.run_dir / "targets.pt")
        target = torch.load(targets_path, map_location=device, weights_only=False)
        if target.get("meta", {}).get("target_contract") != _target_contract(cfg):
            raise ValueError(
                "Target artifact is incompatible with the current dataset/backbone/trajectory/topology "
                "configuration. Re-run extract-targets with these settings."
            )
        images, labels = _initialize_synthetic(cfg, device)
        images.requires_grad_(True)
        optimizer = torch.optim.Adam([images], lr=cfg.distill.lr)
        schedule = DiffusionSchedule(
            cfg.trajectory.diffusion_steps,
            cfg.trajectory.beta_start,
            cfg.trajectory.beta_end,
            device,
        )
        latent_shape = trajectory_noise_shape(model, images, cfg.trajectory.space)
        noise_gen = torch.Generator(device=device).manual_seed(cfg.seed + 17)
        fixed_noise = torch.randn((len(images), *latent_shape), generator=noise_gen, device=device)
        low, high = normalized_bounds(cfg.dataset.name, device)
        history = []
        topology_pairs = [
            (c, ti) for c in range(classes) for ti in range(len(cfg.trajectory.timesteps))
        ]
        pair_generator = torch.Generator().manual_seed(cfg.seed + 29)
        for step in tqdm(range(cfg.distill.steps), desc=f"distill:{cfg.distill.method}"):
            embedding = trajectory_embeddings(
                model,
                images,
                cfg.trajectory.timesteps,
                schedule,
                fixed_noise,
                space=cfg.trajectory.space,
                direction=cfg.trajectory.direction,
                labels=labels,
                conditioning=cfg.trajectory.conditioning,
                source=cfg.trajectory.source,
                representation=cfg.trajectory.representation,
                inference_steps=cfg.trajectory.inference_steps,
            )
            feature_loss = images.new_zeros(())
            topology_loss = images.new_zeros(())
            for c in range(classes):
                subset = embedding[labels == c]
                real = target["classes"][c]
                mean, variance = feature_statistics(subset)
                feature_loss = feature_loss + F.mse_loss(mean, real["mean"])
                if cfg.distill.feature_covariance:
                    feature_loss = feature_loss + F.mse_loss(variance, real["variance"])
            feature_loss = feature_loss / classes
            pair_count = min(cfg.topology.pairs_per_step, len(topology_pairs))
            chosen = torch.randperm(len(topology_pairs), generator=pair_generator)[:pair_count]
            for pair_index in chosen.tolist():
                c, ti = topology_pairs[pair_index]
                subset = embedding[labels == c, ti]
                if len(subset) > cfg.topology.max_points_per_class:
                    start = (step * 17 + c * 7 + ti) % len(subset)
                    point_indices = torch.arange(
                        start, start + cfg.topology.max_points_per_class, device=device
                    )
                    subset = subset[point_indices % len(subset)]
                syn_sig = persistence_signature(
                    subset, cfg.topology.signature_size, cfg.topology.metric
                )
                topology_loss = topology_loss + diagram_distance(
                    syn_sig, target["classes"][c]["topology"][ti]
                )
            topology_loss = topology_loss / pair_count
            if cfg.distill.method == "feature":
                loss = cfg.distill.lambda_feature * feature_loss
            elif cfg.distill.method == "topology":
                loss = cfg.distill.lambda_topology * topology_loss
            else:
                loss = (
                    cfg.distill.lambda_feature * feature_loss
                    + cfg.distill.lambda_topology * topology_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                images.clamp_(low, high)
            if step % cfg.distill.log_every == 0 or step == cfg.distill.steps - 1:
                history.append(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "feature": float(feature_loss.detach()),
                        "topology": float(topology_loss.detach()),
                    }
                )
    out = cfg.run_dir / f"synthetic_{cfg.distill.method}_ipc{cfg.distill.ipc}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "images": images.detach().cpu(),
            "labels": labels.cpu(),
            "meta": _artifact_meta(cfg, "synthetic"),
        },
        out,
    )
    save_json(history, out.with_suffix(".history.json"))
    dump_config(cfg, cfg.run_dir / "resolved_config.yaml")
    return out


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += int((model(images).argmax(1) == labels).sum())
            total += len(labels)
    return correct / max(total, 1)


def evaluate(cfg: ExperimentConfig, synthetic_path: str | Path) -> Path:
    device = resolve_device(cfg.device)
    payload = torch.load(synthetic_path, map_location="cpu", weights_only=False)
    source_dataset = payload.get("meta", {}).get("config", {}).get("dataset", {})
    if (
        source_dataset.get("name") != cfg.dataset.name
        or source_dataset.get("image_size") != cfg.dataset.image_size
    ):
        raise ValueError("Synthetic artifact dataset/image size does not match evaluation config")
    synthetic = TensorDataset(payload["images"], payload["labels"], augment=True)
    _, test_set = build_datasets(cfg.dataset)
    classes, image_size = dataset_spec(cfg.dataset.name)
    results = []
    for architecture in cfg.evaluation.architectures:
        for seed in cfg.evaluation.seeds:
            seed_everything(seed)
            model = classifier(architecture, classes, image_size).to(device)
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.evaluation.lr,
                momentum=0.9,
                weight_decay=cfg.evaluation.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.evaluation.epochs)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                synthetic,
                batch_size=min(cfg.evaluation.batch_size, len(synthetic)),
                shuffle=True,
                generator=generator,
                num_workers=cfg.dataset.num_workers,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=cfg.evaluation.batch_size,
                shuffle=False,
                num_workers=cfg.dataset.num_workers,
            )
            for _ in tqdm(
                range(cfg.evaluation.epochs), desc=f"eval:{architecture}:{seed}", leave=False
            ):
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    loss = F.cross_entropy(model(images), labels)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            results.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "top1": _accuracy(model, test_loader, device),
                }
            )
    out = Path(synthetic_path).with_suffix(".evaluation.json")
    previous = _load_json_if_exists(out)
    if isinstance(previous, dict):
        new_keys = {(r["architecture"], r["seed"]) for r in results}
        results = [
            r
            for r in previous.get("runs", [])
            if (r.get("architecture"), r.get("seed")) not in new_keys
        ] + results
    summary = []
    for architecture in sorted({r["architecture"] for r in results}):
        values = [r["top1"] for r in results if r["architecture"] == architecture]
        summary.append(
            {
                "architecture": architecture,
                "mean": sum(values) / len(values),
                "std": math.sqrt(
                    sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values)
                ),
                "n": len(values),
            }
        )
    save_json({"runs": results, "summary": summary}, out)
    return out
