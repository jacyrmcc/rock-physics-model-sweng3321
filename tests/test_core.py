"""Smoke / contract tests mapped to the requirements. Run: pytest -q"""
import numpy as np
import pytest

from mlrpm import (EvaluationReport, FeatureEngineer, ForwardModel, ModelEvaluator,
                   RockPhysicsPipeline, SyntheticWellGenerator, VelocityPredictor,
                   WellLogDataset, WellLogRepository, create_predictor, load_config, register)
from mlrpm.entities import SCHEMA


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def fm(cfg):
    return ForwardModel.from_config(cfg)


@pytest.fixture(scope="module")
def syn(fm):
    return SyntheticWellGenerator(fm, n_samples=200, random_state=1).generate(4)


@pytest.fixture
def data_dir(tmp_path, syn):
    for name, w in syn.items():
        w.to_dataframe().to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


# FR-1 / FR-2 / NFR-3 ---------------------------------------------------------
def test_dataset_rejects_bad_shape_and_nan():
    with pytest.raises(ValueError):
        WellLogDataset("x", np.zeros((5, 9)))
    a = np.zeros((5, 10)); a[0, 0] = np.nan
    with pytest.raises(ValueError):
        WellLogDataset("x", a)


def test_repository_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        WellLogRepository(tmp_path).load("nope")


def test_repository_column_map(tmp_path, syn):
    df = next(iter(syn.values())).to_dataframe().rename(columns={"POR": "PHIT", "RHO": "RHOB"})
    df.to_csv(tmp_path / "W.csv", index=False)
    w = WellLogRepository(tmp_path, {"PHIT": "POR", "RHOB": "RHO"}).load("W")
    assert w.array.shape[1] == len(SCHEMA)


def test_repository_missing_column(tmp_path, syn):
    next(iter(syn.values())).to_dataframe().drop(columns="SW").to_csv(tmp_path / "W.csv", index=False)
    with pytest.raises(ValueError, match="SW"):
        WellLogRepository(tmp_path).load("W")


# FR-4 / NFR-2 ----------------------------------------------------------------
def test_forward_model_physical_and_fast(fm, syn):
    import time
    w = SyntheticWellGenerator(fm, n_samples=2000).generate(1)["SYN-001"]
    t = time.perf_counter(); vp = fm.predict(w, "VP"); vs = fm.predict(w, "VS")
    assert time.perf_counter() - t < 2.0
    assert np.all((vp > 1.0) & (vp < 7.0)) and np.all(vs < vp)


def test_forward_model_bad_target(fm, syn):
    with pytest.raises(ValueError):
        fm.predict(next(iter(syn.values())), "RHO")


def test_porosity_above_critical_raises(fm, syn):
    a = next(iter(syn.values())).array.copy(); a[:, 4] = 0.9
    with pytest.raises(ValueError, match="porosity"):
        fm.predict(WellLogDataset("hi", a), "VP")


# FR-5 / NFR-11 ---------------------------------------------------------------
def test_synthetic_reproducible(fm):
    g = lambda: SyntheticWellGenerator(fm, n_samples=50, random_state=3).generate(2)
    a, b = g(), g()
    assert list(a) == ["SYN-001", "SYN-002"]
    assert np.array_equal(a["SYN-001"].array, b["SYN-001"].array)


# FR-6 / NFR-9 ----------------------------------------------------------------
def test_factory_and_registry():
    with pytest.raises(KeyError, match="valid options"):
        create_predictor("xgb")
    with pytest.raises(RuntimeError):
        create_predictor("rf").predict(np.zeros((2, 6)))
    with pytest.raises(ValueError):
        create_predictor("svr", C=-1).fit(np.zeros((4, 6)), np.zeros(4))

    from sklearn.linear_model import LinearRegression

    @register("lin")
    class LinearPredictor(VelocityPredictor):
        display_name = "LIN"
        def _build_model(self, **hp):
            return LinearRegression()

    X = np.random.default_rng(0).normal(size=(20, 6))
    assert create_predictor("lin").fit(X, X @ np.ones(6)).predict(X).shape == (20,)


# NFR-12 ----------------------------------------------------------------------
def test_blind_well_excluded(syn):
    real = dict(list(syn.items())[:3]); extra = dict(list(syn.items())[3:])
    fs = FeatureEngineer().build(real, "SYN-001", "VP", extra)
    n_train_total = len(fs.y_train) + len(fs.y_test)
    assert n_train_total == sum(len(w) for n, w in syn.items() if n != "SYN-001")
    with pytest.raises(KeyError):
        FeatureEngineer().build(real, "SYN-004", "VP", extra)


# FR-8 ------------------------------------------------------------------------
def test_evaluator_perfect_and_pivot():
    ev = ModelEvaluator()
    r = ev.evaluate(np.array([2.0, 3.0]), np.array([2.0, 3.0]), "W", "RPM", "VP")
    assert r.r2 == 1.0 and r.rmse == 0.0 and r.accuracy_pct == pytest.approx(100.0)
    rep = EvaluationReport([r]); assert rep.pivot("r2").loc["W", "RPM"] == 1.0
    with pytest.raises(ValueError):
        rep.pivot("bogus")


# FR-12 -----------------------------------------------------------------------
def test_pipeline_end_to_end(data_dir, cfg):
    cfg = dict(cfg, model_keys=["rf", "svr"])
    res = RockPhysicsPipeline.from_config(data_dir, cfg).run(target_name="VS", n_synthetic=1)
    table = res.report.pivot("r2")
    assert set(table.columns) == {"RPM", "RF", "SVR"} and len(table) == 4
    # data were generated by the same RPM, so RPM must be (near) perfect
    assert (table["RPM"] > 0.999).all()

# FR-1: .npy input -------------------------------------------------------------
def test_repository_npy(tmp_path, syn):
    w = next(iter(syn.values()))
    np.save(tmp_path / "N1.npy", w.array)
    repo = WellLogRepository(tmp_path)
    assert "N1" in repo.available_wells()
    assert np.array_equal(repo.load("N1").array, w.array)
    # custom column order
    order = ["RHO"] + SCHEMA[:-1]
    np.save(tmp_path / "N2.npy", w.to_dataframe()[order].to_numpy())
    assert np.array_equal(WellLogRepository(tmp_path, npy_column_order=order).load("N2").array, w.array)
    # wrong width -> clear error
    np.save(tmp_path / "N3.npy", w.array[:, :9])
    with pytest.raises(ValueError, match="columns"):
        repo.load("N3")
        