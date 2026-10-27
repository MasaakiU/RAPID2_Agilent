import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import cut, make_item, validation_set  # noqa: E402
from decode import average_precision, decode, match  # noqa: E402
from features import features  # noqa: E402
from losses import losses  # noqa: E402
from models import build  # noqa: E402
from simulator import Config, Peak, simulate  # noqa: E402
from targets import make_targets  # noqa: E402

HERE = Path(__file__).resolve().parent


def _peak(apex, label="positive", fwhm=10.0, snr=50.0):
    return Peak(apex, 100.0, fwhm, 8.0, 12.0, "gauss", snr, label)


@pytest.mark.parametrize("stride", [1, 32])
def test_an_apex_comes_back_from_its_targets(stride):
    targets = make_targets([_peak(100.3)], 512, stride)
    heat, offset, log_widths = targets["heat"], targets["offset"], targets["log_widths"]
    (found,) = decode(np.where(heat >= 1, 0.99, heat * 0.5), offset, log_widths, stride)
    assert found["apex"] == pytest.approx(100.3, abs=1e-4)
    assert (found["left"], found["right"]) == pytest.approx((8.0, 12.0))


def test_maxima_within_a_tenth_of_a_likelier_ones_widths_are_of_its_peak():
    heat = np.zeros(300)
    heat[[100, 104, 150]] = [0.9, 0.6, 0.7]  # widths of 50: 104 within 5 of 100, 150 not
    found = decode(heat, np.full(300, 0.5), np.log(np.full((2, 300), 50.0)), 1)
    assert [(apex["apex"], apex["confidence"]) for apex in found] == [(100.0, 0.9), (150.0, 0.7)]


def test_peaks_not_scored_and_the_padding_weigh_nothing():
    valid = np.ones(512, dtype=bool)
    valid[400:] = False
    targets = make_targets([_peak(100.0), _peak(200.0, "ignore"), _peak(300.0, "negative")], 512, 1, valid)
    weight, heat = targets["weight"], targets["heat"]
    assert weight[200] == 0 and weight[400:].sum() == 0 and weight[300] == 1
    assert heat[100] == 1 and heat[300] == 0


def test_a_window_cut_of_a_sample_moves_its_peaks_and_scores_none_at_its_ends():
    sample = SimpleNamespace(signal=np.arange(1000.0), clean=np.arange(1000.0), missing=np.zeros(1000, bool),
                             peaks=[_peak(450.0), _peak(503.0), _peak(990.0)], meta={})

    class Start:
        def integers(self, low, high):
            return 500

    window = cut(sample, 400, Start())
    assert window["signal"][0] == 500
    # 450 is out of it, 503 by its start (less than half its FWHM in), 990 out at the far end
    assert [(peak.apex, peak.label) for peak in window["peaks"]] == [(3.0, "ignore")]


def test_the_clean_channels_are_of_the_noisy_ones_scaling():
    rng = np.random.default_rng(0)
    clean = 100 * np.exp(-((np.arange(500) - 250) / 10) ** 2 / 2) + 50
    noisy = clean + rng.normal(0, 1, 500)
    x, scaling = features(noisy)
    clean_x, _ = features(clean, scaling=scaling)
    assert x.shape == (3, 500) and x.dtype == np.float32
    assert np.abs(x[0] - clean_x[0]).max() < 0.1
    assert x[0].max() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", ["transformer", "unet", "kensert"])
@pytest.mark.parametrize("length", [1000, 2048])
def test_each_model_gives_its_heads_of_its_cells(name, length):
    options = {"transformer": {"width": 16, "middle": 24, "d_model": 32, "layers": 1, "heads": 2, "feedforward": 64},
               "unet": {"channels": [8, 16, 32, 32], "dilations": [1, 2]}, "kensert": {"filters": [8, 8, 8, 16, 16]}}
    model = build(name, **options[name]).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, length))
    cells = -(-length // model.stride)
    assert out["heat"].shape == (2, cells) and out["log_widths"].shape == (2, 2, cells)
    assert ("denoised" in out) == (name != "kensert")
    if "denoised" in out:
        assert out["denoised"].shape == (2, 2, length)


def test_matching_counts_what_is_found_by_the_rule():
    peaks = [_peak(100.0), _peak(200.0), _peak(300.0, "ignore"), _peak(400.0, "negative")]
    found = [{"apex": 101.0, "confidence": 0.9}, {"apex": 103.0, "confidence": 0.8},  # the second of 100: false
             {"apex": 301.0, "confidence": 0.7}, {"apex": 400.0, "confidence": 0.6}]  # nothing; false
    records, matched = match(found, peaks)
    assert records == [(0.9, True), (0.8, False), (0.6, False)]
    assert [peak.apex for peak in matched] == [100.0]
    assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    assert average_precision([(0.9, False), (0.8, True)], 2) == pytest.approx(0.25)


def test_a_small_model_learns_a_batch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build("transformer", width=16, middle=24, d_model=32, layers=1, heads=2, feedforward=64, dropout=0.0)
    items = [make_item(simulate(rng), 512, model.stride, rng)[0] for _ in range(4)]
    batch = {key: torch.from_numpy(np.stack([item[key] for item in items])) for key in items[0]}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        total = losses(model(batch["x"]), batch)["total"]
        first = first or total.item()
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    assert total.item() < 0.5 * first


def test_training_checking_exporting_and_real_data_run_through(tmp_path):
    import evaluate_real
    import export
    import train

    assert train.main(["--config", str(HERE / "configs" / "cpu_tiny.json"), "--out", str(tmp_path / "run"),
                       "--device", "cpu", "--set", "steps=4", "validation=4", "eval_every=4", "log_every=2"]) == 0
    lines = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert any("check" in line for line in lines)
    made = json.loads((tmp_path / "run" / "config.json").read_text())["provenance"]
    assert made["simulator_version"] >= 2 and set(made) >= {"git_commit", "git_dirty", "started", "host"}
    export.export(str(tmp_path / "run" / "last.pt"), str(tmp_path / "model.npz"))
    with np.load(tmp_path / "model.npz") as arrays:
        exported = json.loads(str(arrays["config"]))
        assert exported["model"] == "transformer" and exported["provenance"] == made
    # a labeled dataset as tools/peak_labeling writes it: one trace, two peaks and a top unsure, done
    rt = np.linspace(0, 5, 300)
    intensity = 1000 * np.exp(-((rt - 2) / 0.05) ** 2 / 2) + 500 * np.exp(-((rt - 3.5) / 0.05) ** 2 / 2) + 10
    meta = [{"id": "0000", "run": "/x.d", "kind": "target", "name": "x", "mz_window": None, "polarity": 1,
             "ms_level": 1, "start": 0, "end": 300}]
    np.savez(tmp_path / "xics.npz", meta=json.dumps(meta), rt=rt, intensity=intensity)
    labels = {"traces": {"0000": {"candidates": [{"rt": 2.0, "source": "kensert", "confidence": 1.0, "state": "taken"},
                                                 {"rt": 4.2, "source": "scipy", "confidence": None, "state": "unsure"}],
                                  "added": [3.5], "added_unsure": [], "done": True, "unusable": False, "note": ""}}}
    (tmp_path / "xics.labels.json").write_text(json.dumps(labels))
    assert evaluate_real.main(["--checkpoint", str(tmp_path / "run" / "last.pt"), "--dataset",
                               str(tmp_path / "xics.npz"), "--device", "cpu", "--out", str(tmp_path / "real.json"),
                               "--details", str(tmp_path / "found.json")]) == 0
    summary = json.loads((tmp_path / "real.json").read_text())
    assert summary["all"]["traces_peaks"] == 2 and summary["by_dataset"]["xics"]["traces"] == 1
    (trace,) = json.loads((tmp_path / "found.json").read_text())
    assert [(peak["rt"], peak["unsure"]) for peak in trace["peaks"]] == [(2.0, False), (3.5, False), (4.2, True)]
    assert {apex["outcome"] for apex in trace["found"]} <= {"true", "false", "ignored"}


def test_a_real_peak_is_matched_within_half_its_width_measured():
    import evaluate_real

    rt = np.arange(0, 10, 1 / 60)  # a scan a second
    sigma = 0.15  # min: FWHM 0.353
    rng = np.random.default_rng(0)
    # a broad peak, its top noisy, beside a narrow one of its own
    intensity = (1000 * np.exp(-((rt - 4) / sigma) ** 2 / 2) * (1 + 0.05 * rng.standard_normal(len(rt)))
                 + 800 * np.exp(-((rt - 4.9) / 0.03) ** 2 / 2) + 10)
    broad, narrow = evaluate_real.half_widths(rt, intensity, [4.02, 4.9])
    assert broad == pytest.approx(2.3548 * sigma, rel=0.1)
    assert narrow == pytest.approx(2.3548 * 0.03, rel=0.2)
    peaks = [SimpleNamespace(apex=4.02 * 120, fwhm=broad * 120, label="positive")]  # in points of 0.5 s
    # 6 s off the label: within half the broad peak's width, not within 3 s
    assert match([{"apex": 4.12 * 120, "confidence": 0.9}], peaks, tolerance=(6, 0.5))[0] == [(0.9, True)]
    assert match([{"apex": 4.12 * 120, "confidence": 0.9}], peaks, tolerance=(6, 0.0))[0] == [(0.9, False)]


def test_the_validation_set_is_the_same_each_time():
    first, second = validation_set(3, 512, 1), validation_set(3, 512, 1)
    for (a, _), (b, _) in zip(first, second):
        assert torch.equal(a["x"], b["x"]) and torch.equal(a["heat"], b["heat"])
