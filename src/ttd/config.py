from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    root: str = "data"
    image_size: int = 32
    num_workers: int = 4
    download: bool = True
    fake_train_size: int = 500
    fake_test_size: int = 200


@dataclass
class BackboneConfig:
    latent_channels: int = 8
    width: int = 64
    embed_dim: int = 128
    ae_epochs: int = 50
    diffusion_epochs: int = 100
    ae_lr: float = 2e-4
    diffusion_lr: float = 2e-4
    batch_size: int = 128
    checkpoint: str | None = None
    condition_dropout: float = 0.1
    semantic_weight: float = 0.1
    ema_decay: float = 0.999
    validation_fraction: float = 0.1
    validation_batches: int = 10
    noise_mse_target: float = 0.8
    use_ema: bool = True


@dataclass
class TrajectoryConfig:
    diffusion_steps: int = 1000
    timesteps: list[int] = field(default_factory=lambda: [999, 749, 499, 249, 0])
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    space: str = "latent"
    direction: str = "reverse"
    conditioning: str = "unconditional"
    source: str = "inversion"
    representation: str = "predicted_clean"
    inference_steps: int = 50
    noise_repeats: int = 1


@dataclass
class TopologyConfig:
    max_points_per_class: int = 256
    signature_size: int = 32
    metric: str = "cosine"
    homology_dim: int = 0
    pairs_per_step: int = 4


@dataclass
class DistillConfig:
    method: str = "ttd"
    ipc: int = 10
    steps: int = 2000
    lr: float = 0.05
    lambda_feature: float = 1.0
    lambda_topology: float = 1.0
    feature_covariance: bool = True
    init: str = "real"
    log_every: int = 50
    permute_class_targets: bool = False


@dataclass
class EvalConfig:
    architectures: list[str] = field(default_factory=lambda: ["resnet18"])
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    weight_decay: float = 5e-4
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    probe_views: int = 8
    probe_max_iter: int = 500


@dataclass
class FeasibilityConfig:
    sample_count: int = 10_000
    train_fraction: float = 0.8
    split_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    topology_neighbors: int = 16
    topology_jobs: int = 1
    dimension_matched_probe: bool = True
    visualization_samples: int = 500
    exact_ph_max_dim: int = 1
    require_backbone_gate: bool = True
    probe_estimator: str = "ridge"
    ridge_alpha: float = 1.0
    topology_v2: bool = False
    topology_curve_bins: int = 50
    compression_budgets: list[int] = field(
        default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 4000, 8000]
    )


@dataclass
class PilotAConfig:
    """Confound-clearance probe battery over the saved compression artifact."""

    budgets: list[int] = field(
        default_factory=lambda: [50, 100, 200, 500, 1000, 2000, 4000, 8000]
    )
    alphas: list[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    )
    h1_budgets: list[int] = field(default_factory=lambda: [100, 200, 500])
    random_dim: int = 250
    knn_profile_dims: int = 16
    localcov_dims: int = 16
    laplacian_dims: int = 8
    chunk_size: int = 512
    selection_folds: int = 3


@dataclass
class ExperimentConfig:
    experiment: str = "ttd"
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    pilot_a: PilotAConfig = field(default_factory=PilotAConfig)

    def validate(self) -> None:
        valid_data = {"cifar10", "cifar100", "tiny_imagenet", "imagewoof", "imagenette", "fake"}
        if self.dataset.name not in valid_data:
            raise ValueError(f"dataset.name must be one of {sorted(valid_data)}")
        if not self.trajectory.timesteps:
            raise ValueError("trajectory.timesteps cannot be empty")
        ts = self.trajectory.timesteps
        if any(t < 0 or t >= self.trajectory.diffusion_steps for t in ts):
            raise ValueError("trajectory timesteps must lie inside the diffusion schedule")
        if ts != sorted(set(ts), reverse=True):
            raise ValueError("trajectory timesteps must be unique and descending")
        if self.trajectory.space not in {"pixel", "latent", "feature"}:
            raise ValueError("trajectory.space must be pixel, latent, or feature")
        if self.trajectory.direction not in {"forward", "reverse"}:
            raise ValueError("trajectory.direction must be forward or reverse")
        if self.trajectory.conditioning not in {"unconditional", "class"}:
            raise ValueError("trajectory.conditioning must be unconditional or class")
        if self.trajectory.source not in {"inversion", "q_sample"}:
            raise ValueError("trajectory.source must be inversion or q_sample")
        if self.trajectory.representation not in {"raw", "predicted_clean"}:
            raise ValueError("trajectory.representation must be raw or predicted_clean")
        if self.trajectory.direction == "reverse" and self.trajectory.space != "latent":
            raise ValueError("reverse denoising trajectory currently requires latent space")
        if not 0 <= self.backbone.condition_dropout <= 1:
            raise ValueError("backbone.condition_dropout must lie in [0, 1]")
        if not 0 < self.backbone.validation_fraction < 1:
            raise ValueError("backbone.validation_fraction must lie in (0, 1)")
        if self.backbone.ae_epochs < 1 or self.backbone.diffusion_epochs < 1:
            raise ValueError("both AE and diffusion must train for at least one epoch")
        if self.trajectory.inference_steps < 2:
            raise ValueError("trajectory.inference_steps must be at least two")
        if self.topology.homology_dim != 0:
            raise ValueError(
                "Optimization currently supports differentiable H0 only; use analyze-ph for H1/H2."
            )
        if self.topology.max_points_per_class < 2 or self.topology.pairs_per_step < 1:
            raise ValueError("topology requires at least two points and one pair per step")
        if self.distill.ipc < 1 or self.distill.steps < 1:
            raise ValueError("distill.ipc and distill.steps must be positive")
        if not 0 < self.feasibility.train_fraction < 1:
            raise ValueError("feasibility.train_fraction must lie in (0, 1)")
        if self.feasibility.topology_neighbors < 2:
            raise ValueError("feasibility.topology_neighbors must be at least two")
        if self.feasibility.probe_estimator not in {"ridge", "sgd"}:
            raise ValueError("feasibility.probe_estimator must be ridge or sgd")
        if self.feasibility.ridge_alpha <= 0:
            raise ValueError("feasibility.ridge_alpha must be positive")
        if self.feasibility.topology_curve_bins < 4:
            raise ValueError("feasibility.topology_curve_bins must be at least four")
        if any(budget < 1 for budget in self.feasibility.compression_budgets):
            raise ValueError("feasibility.compression_budgets must be positive")
        if any(budget < 1 for budget in self.pilot_a.budgets):
            raise ValueError("pilot_a.budgets must be positive")
        if any(alpha <= 0 for alpha in self.pilot_a.alphas):
            raise ValueError("pilot_a.alphas must be positive")
        if len(self.pilot_a.alphas) < 1:
            raise ValueError("pilot_a.alphas cannot be empty")
        if self.pilot_a.random_dim < 1:
            raise ValueError("pilot_a.random_dim must be positive")
        if self.pilot_a.selection_folds < 2:
            raise ValueError("pilot_a.selection_folds must be at least two")
        if self.distill.method not in {
            "ttd",
            "feature",
            "topology",
            "random",
            "herding",
            "gradient_matching",
            "trajectory_matching",
            "trajectory_matching_topology",
            "trajectory_matching_tv",
            "diffusion",
        }:
            raise ValueError("unsupported distillation method")

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _construct(raw: dict[str, Any]) -> ExperimentConfig:
    fields = {
        "dataset": DatasetConfig,
        "backbone": BackboneConfig,
        "trajectory": TrajectoryConfig,
        "topology": TopologyConfig,
        "distill": DistillConfig,
        "evaluation": EvalConfig,
        "feasibility": FeasibilityConfig,
        "pilot_a": PilotAConfig,
    }
    top = {k: v for k, v in raw.items() if k not in fields}
    for name, cls in fields.items():
        top[name] = cls(**raw.get(name, {}))
    cfg = ExperimentConfig(**top)
    cfg.validate()
    return cfg


@dataclass
class AuditBenchE0Config:
    """Standalone config for the AuditBench-DD E0 external-bank stage.

    Kept separate from ExperimentConfig so the E0 schema can evolve without
    touching the historical experiment pipeline.
    """

    dataset: str = "cifar10"
    stage: str = "auditbench_e0"
    num_classes: int = 10
    image_size: int = 32
    manifest_path: str = "manifests/cifar10_external_sources.yaml"
    raw_dir: str = "external/cifar10/raw"
    out_dir: str = "external/cifar10"
    report_path: str = "runs/cifar10/auditbench_e0/e0_report.json"
    e1_protocol_path: str = "runs/cifar10/auditbench_e0/e1_protocol_frozen.json"
    resolved_manifest_path: str = "runs/cifar10/auditbench_e0/manifest_resolved.yaml"
    fetch_enabled: bool = True
    fetch_timeout_sec: float = 120.0
    fetch_retries: int = 1
    primary_ipcs: list[int] = field(default_factory=lambda: [10, 50])
    extreme_ipcs: list[int] = field(default_factory=lambda: [1])
    gate_thresholds: dict[str, Any] = field(
        default_factory=lambda: {
            "pass_strong": {
                "min_lineages": 6,
                "min_families": 6,
                "min_dual_ipc_lineages": 4,
                "min_primary_entries": 12,
            },
            "pass_minimum": {
                "min_lineages": 4,
                "min_families": 4,
                "min_dual_ipc_lineages": 3,
                "min_primary_entries": 8,
            },
        }
    )

    def validate(self) -> None:
        if self.dataset not in ("cifar10", "cifar100"):
            raise ValueError("AuditBench bank builds support cifar10 and cifar100 only")
        if not self.primary_ipcs:
            raise ValueError("primary_ipcs cannot be empty")
        for tier in ("pass_strong", "pass_minimum"):
            if tier not in self.gate_thresholds:
                raise ValueError(f"gate_thresholds must define {tier}")


def load_auditbench_config(
    path: str | Path, overrides: list[str] | None = None
) -> AuditBenchE0Config:
    raw = asdict(AuditBenchE0Config())
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = _merge(raw, yaml.safe_load(handle) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        cursor = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(value)
    cfg = AuditBenchE0Config(**raw)
    cfg.validate()
    return cfg

def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = asdict(ExperimentConfig())
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = _merge(raw, yaml.safe_load(handle) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        cursor = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(value)
    return _construct(raw)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False, allow_unicode=True)
