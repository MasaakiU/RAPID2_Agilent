"""Explicit settings; all alignment RTs and RT windows use seconds."""

from dataclasses import dataclass, field, replace
import math


@dataclass(frozen=True)
class BinningConfig:
    mz_min: float = 50.0
    mz_max: float = 1700.0
    ppm: float = 10.0
    aggregation: str = "sum"

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.mz_min, self.mz_max, self.ppm)):
            raise ValueError("m/z limits and ppm must be finite")
        if not 0 < self.mz_min < self.mz_max or self.ppm <= 0:
            raise ValueError("Require 0 < mz_min < mz_max and ppm > 0")
        if self.aggregation not in {"sum", "mean", "max"}:
            raise ValueError("aggregation must be sum, mean or max")


@dataclass(frozen=True)
class TransformConfig:
    method: str = "asinh"
    scale: float | None = None

    def __post_init__(self) -> None:
        if self.method not in {"none", "sqrt", "log1p", "asinh"}:
            raise ValueError("Unknown intensity transform")
        if self.scale is not None and (not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("scale must be finite and positive")

    def validate_for_use(self) -> None:
        if self.method == "asinh" and self.scale is None:
            raise ValueError("asinh requires an explicit scale; use suggest_scales to inspect candidates")


@dataclass(frozen=True)
class ComputeConfig:
    metric: str = "cosine"
    device: str = "auto"  # auto: CUDA if available, otherwise NumPy CPU
    chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.metric not in {"cosine", "pearson"}:
            raise ValueError("metric must be cosine or pearson")
        if self.device not in {"auto", "cpu", "torch_cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, torch_cpu or cuda")
        if not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")


@dataclass(frozen=True)
class DTWConfig:
    diagonal_weight: float = 2.0
    horizontal_weight: float = 1.0
    vertical_weight: float = 1.0
    gap_penalty: float = 0.1
    rt_band_seconds: float | None = None

    def __post_init__(self) -> None:
        weights = (self.diagonal_weight, self.horizontal_weight, self.vertical_weight)
        if any(not math.isfinite(x) or x <= 0 for x in weights):
            raise ValueError("DTW step weights must be finite and positive")
        if not math.isfinite(self.gap_penalty) or self.gap_penalty < 0:
            raise ValueError("gap_penalty must be finite and nonnegative")
        if self.rt_band_seconds is not None and (
            not math.isfinite(self.rt_band_seconds) or self.rt_band_seconds < 0
        ):
            raise ValueError("RT band must be finite and nonnegative")


@dataclass(frozen=True)
class AnchorConfig:
    stride: int = 1
    max_anchors: int | None = None
    preserve_endpoints: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.stride, int) or self.stride < 1:
            raise ValueError("anchor stride must be a positive integer")
        if self.max_anchors is not None and (
            not isinstance(self.max_anchors, int) or self.max_anchors < 2
        ):
            raise ValueError("max_anchors must be at least 2")


@dataclass(frozen=True)
class ScanSelection:
    ms_level: int = 1
    polarity: str | None = None
    reader_rt_unit: str = "minutes"
    rt_range_seconds: tuple[float, float] | None = None
    stride: int = 1
    max_scans: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ms_level, int) or self.ms_level < 1 or self.polarity not in {None, "positive", "negative"}:
            raise ValueError("Choose a positive MS level and a valid polarity")
        if self.reader_rt_unit not in {"seconds", "minutes"}:
            raise ValueError("reader_rt_unit must be seconds or minutes")
        if not isinstance(self.stride, int) or self.stride < 1:
            raise ValueError("scan stride must be positive")
        if self.max_scans is not None and (not isinstance(self.max_scans, int) or self.max_scans < 2):
            raise ValueError("max_scans must be at least 2")
        if self.rt_range_seconds is not None:
            lo, hi = self.rt_range_seconds
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                raise ValueError("RT range must have finite increasing bounds")


@dataclass(frozen=True)
class AlignmentConfig:
    binning: BinningConfig = field(default_factory=BinningConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    dtw: DTWConfig = field(default_factory=DTWConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    selection: ScanSelection = field(default_factory=ScanSelection)
    solver: str = "obiwarp"
    extrapolation: str = "nan"

    def __post_init__(self) -> None:
        if self.extrapolation not in {"nan", "linear", "error"}:
            raise ValueError("extrapolation must be nan, linear or error")


def config_from_dict(values: dict) -> AlignmentConfig:
    """Reject unknown options rather than silently ignoring misspelled settings."""
    values = dict(values)
    try:
        for name, cls in (("binning", BinningConfig), ("transform", TransformConfig),
                          ("compute", ComputeConfig), ("dtw", DTWConfig),
                          ("anchors", AnchorConfig), ("selection", ScanSelection)):
            if name in values:
                values[name] = cls(**values[name])
        return AlignmentConfig(**values)
    except TypeError as error:
        raise ValueError(f"Invalid configuration: {error}") from error


# The flat options a command line or a tool call names, and the field each one
# sets. The one place such a name becomes a configuration: the alignment CLI's
# flags, `rapid2-agilent-cli` and the MCP tools all pass these names.
OPTIONS = {
    "ppm": ("binning", "ppm"),
    "aggregation": ("binning", "aggregation"),
    "transform": ("transform", "method"),
    "scale": ("transform", "scale"),
    "metric": ("compute", "metric"),
    "device": ("compute", "device"),
    "chunk_size": ("compute", "chunk_size"),
    "polarity": ("selection", "polarity"),
    "reader_rt_unit": ("selection", "reader_rt_unit"),
    "max_scans": ("selection", "max_scans"),
    "scan_stride": ("selection", "stride"),
    "rt_band_seconds": ("dtw", "rt_band_seconds"),
    "anchor_stride": ("anchors", "stride"),
}


def with_options(config: AlignmentConfig, **options) -> AlignmentConfig:
    """`config` with the named OPTIONS replaced; None leaves an option as it is."""
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise ValueError(f"Unknown alignment option(s): {', '.join(unknown)}")
    changes: dict[str, dict] = {}
    for name, value in options.items():
        if value is not None:
            group, field_name = OPTIONS[name]
            changes.setdefault(group, {})[field_name] = value
    for group, fields in changes.items():
        config = replace(config, **{group: replace(getattr(config, group), **fields)})
    return config
