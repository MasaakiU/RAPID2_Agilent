import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deconvolve import heights, present, weighed  # noqa: E402
from deconvolve_data import PER_CANDIDATE, collate, make_item  # noqa: E402
from deconvolve_losses import losses, solve  # noqa: E402
from deconvolve_models import build, deconvolve, run, unimodal  # noqa: E402
from simulator import simulate  # noqa: E402

HERE = Path(__file__).resolve().parent
TINY = HERE / "configs" / "deconvolution_cpu_tiny.json"


def _model(seed=0):
    torch.manual_seed(seed)
    return build(**json.loads(TINY.read_text())["model_options"])


def _items(count, length=512, seed=0):
    rng = np.random.default_rng(seed)
    return [make_item(simulate(rng), length, rng) for _ in range(count)]


def test_points_of_a_part_not_measured_hold_no_peak_and_zeros_weigh_nothing():
    signal = np.array([5, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 4], dtype=float)
    missing = np.zeros(12, dtype=bool)
    valid = np.ones(12, dtype=bool)
    valid[-1] = False
    assert present(signal, missing, valid).tolist() == [True] * 3 + [False] * 4 + [True] * 4 + [False]
    assert weighed(signal, missing, valid).tolist() == [s != 0 for s in signal[:-1]] + [False]


def test_the_targets_make_up_the_clean_signal():
    for item, window, candidates in _items(30, 600):
        base, top = item["scaling"][:2]
        made = (item["baseline"].astype(np.float64) + item["others"]
                + (item["height"][:, None].astype(np.float64) * item["shape"]).sum(axis=0)) * top + base
        valid = window["valid"]
        clean = window["clean"].astype(np.float64)
        np.testing.assert_allclose(made[valid], clean[valid], atol=1e-5 * np.abs(clean[valid]).max())
        # of each candidate matched, a shape of top 1; of the others, none, and a realness of 0 or none
        matched = item["matched"] > 0
        np.testing.assert_allclose(item["shape"][matched].max(axis=1), 1.0)
        assert not item["shape"][~matched].any() and not item["real"][~matched].any()
        assert len(item["candidates"]) == len(candidates)


def test_a_unimodal_shape_is_of_top_1_up_then_down_and_of_any_depth_each_side():
    torch.manual_seed(0)
    where = torch.linspace(-6, 6, 2001).expand(50, -1)
    log_shape = unimodal(torch.randn(50, 64) / 2, torch.randn(50, 64) / 2, 3 * torch.randn(50, 64), where, 4.0)
    torch.testing.assert_close(log_shape.amax(-1), torch.zeros(50), atol=0.05, rtol=0)
    assert (log_shape <= 0).all()
    for steps in torch.diff(log_shape, dim=-1):
        down = torch.nonzero(steps < 0).flatten()
        assert not len(down) or not (steps[down[0]:] > 0).any()
    # steep on the left, shallow on the right: an exponential tail
    log_shape = unimodal(torch.full((1, 64), np.log(30.0)), torch.full((1, 64), np.log(3.0)), torch.zeros(1, 64),
                         torch.tensor([[-1.0, 1.0]]), 4.0)
    torch.testing.assert_close(log_shape, torch.tensor([[-30.0, -3.0]]), rtol=1e-4, atol=1e-4)


def test_shapes_at_first_are_down_to_5_percent_at_the_widths_given_and_on_past_the_grid():
    model = _model().eval()
    with torch.no_grad():
        model.shape_out.weight.zero_()  # the slopes of the bias alone: 3 up and 3 down, of exp(-3 |z|)
        out = model(torch.zeros(1, 3, 400), torch.tensor([[[150.0, 1.0, 10.0, 20.0]]]), torch.ones(1, 1, dtype=bool),
                    torch.ones(1, 400), torch.tensor([[0.0, 100.0, 1.0]]))
    shape = out["shape"][0].double().numpy()
    for point, z in ((150, 0.0), (140, -1.0), (170, 1.0), (190, 2.0), (250, 5.0), (90, -6.0)):
        assert shape[point] == pytest.approx(np.exp(-3 * abs(z)), rel=1e-3)


def test_heights_solved_are_those_of_least_squares_and_carry_gradients():
    rng = np.random.default_rng(0)
    x = np.arange(200.0)
    shapes = np.array([np.exp(-0.5 * ((x - center) / 8) ** 2) for center in (80, 95, 130)])
    weights = rng.uniform(0.5, 2.0, 200)
    residual = np.array([3.0, 1.0, -0.3]) @ shapes + rng.normal(0, 0.05, 200)  # the third below 0 alone
    prior = np.array([2.5, 1.5, 0.5])
    expected, _, _ = heights(residual, shapes, weights, prior=prior, strength=0.01)
    assert expected[2] == 0
    tensor = torch.tensor(shapes[None], dtype=torch.float32, requires_grad=True)

    def solved(include):
        return solve(torch.tensor(residual[None], dtype=torch.float32), tensor,
                     torch.tensor(weights[None], dtype=torch.float32), torch.tensor([include]),
                     torch.tensor(prior[None], dtype=torch.float32), 0.01)[0]

    found = solved([True, True, True])
    np.testing.assert_allclose(found.detach().numpy(), expected, rtol=1e-4, atol=1e-5)
    found[:2].sum().backward()
    assert torch.isfinite(tensor.grad).all() and tensor.grad[0, :2].abs().sum() > 0
    assert solved([True, False, True])[1] == 0


def test_the_model_takes_windows_of_any_candidates_and_its_losses_run_back():
    model = _model()
    items = [item for item, _window, _candidates in _items(3)]
    items[1] = items[1] | {key: items[1][key][:0] for key in PER_CANDIDATE + ("shape",)}  # one of no candidates
    batch = collate(items)
    out = run(model, batch, "cpu")
    count = int(batch["candidate_mask"].sum())
    assert out["shape"].shape == (count, 512) and out["real"].shape == batch["candidate_mask"].shape
    assert all(torch.isfinite(value).all() for value in out.values())
    torch.testing.assert_close(out["shape"].amax(-1), torch.ones(count))
    values = losses(out, batch)
    values["total"].backward()
    assert all(torch.isfinite(value) for value in values.values())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_a_small_model_learns_a_batch():
    model = _model()
    batch = collate([item for item, _window, _candidates in _items(4)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        values = losses(run(model, batch, "cpu"), batch)
        first = first or {name: value.item() for name, value in values.items()}
        optimizer.zero_grad()
        values["total"].backward()
        optimizer.step()
    assert values["area"].item() < 0.4 * first["area"] and values["shape"].item() < 0.75 * first["shape"]


def test_the_later_stage_on_a_window_gives_each_candidate_and_the_parts():
    x = np.arange(300.0)
    signal = 10 + 1000 * np.exp(-0.5 * ((x - 100) / 5) ** 2) + 300 * np.exp(-0.5 * ((x - 200) / 8) ** 2)
    signal = signal + np.random.default_rng(0).normal(0, 2, 300)
    candidates = [{"apex": 100.0, "confidence": 0.9, "left": 12.0, "right": 12.0},
                  {"apex": 200.0, "confidence": 0.8, "left": 20.0, "right": 20.0}]
    found = deconvolve(_model(), signal, np.zeros(300, bool), np.ones(300, bool), candidates, real=0.0)
    assert [peak["apex"] for peak in found["peaks"]] == [100.0, 200.0]
    for peak in found["peaks"]:
        assert peak["real"] and peak["area"] >= 0 and peak["shape"].shape == (300,)
    for name in ("baseline", "others", "sigma", "residual"):
        assert found[name].shape == (300,) and np.isfinite(found[name]).all()


def test_training_checking_and_evaluating_run_through(tmp_path):
    import evaluate_deconvolution
    import train_deconvolution

    assert train_deconvolution.main(["--config", str(TINY), "--out", str(tmp_path / "run"), "--device", "cpu",
                                      "--set", "steps=4", "validation=4", "eval_every=4", "log_every=2"]) == 0
    lines = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    (checked,) = [line["check"] for line in lines if "check" in line]
    assert {"objective", "area log error", "false area", "real ap", "by category", "loss"} <= set(checked)
    made = json.loads((tmp_path / "run" / "config.json").read_text())["provenance"]
    assert made["simulator_version"] >= 6 and set(made) >= {"git_commit", "git_dirty", "started", "host"}
    assert evaluate_deconvolution.main(["--method", "model", "--checkpoint", str(tmp_path / "run" / "last.pt"),
                                        "--candidates", "perturbed", "--count", "3", "--length", "512",
                                        "--out", str(tmp_path / "model.json")]) == 0
    summary = json.loads((tmp_path / "model.json").read_text())
    assert summary["peaks"] > 0 and summary["real"] == 0.5 and summary["strength"] == 0.01
