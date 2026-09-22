# LimiX-2 on MLX (Apple silicon)

Native MLX port of [stable-ai/LimiX-2](https://huggingface.co/stable-ai/LimiX-2)
(400M-param tabular foundation model: classification, regression,
missing-value imputation), plus a drop-in sklearn-style predictor.

Same pattern as the TabFM MLX port: identical module/parameter names, so
weight conversion is mechanical and parity is verifiable layer by layer.

## Layout

| File | Purpose |
|---|---|
| `lxm_model.py` | MLX core blocks: RMSNorm, SoftmaxScalingMLP, sample attention (separate X/Y), decoupled structural-task attention (DStI), gated SiLU MLP, pre-norm `smf` layers |
| `lxm_top.py` | MLX encoders (mask-emb/numeric/fusion, cls/reg Y), parameter-free preprocessing, decoders, adapters, `FeaturesTransformer`, config-driven `build_model` |
| `convert.py` | torch `.ckpt` → `.npz` (no key remapping; `np.savez` — `mx.savez` caps kwargs at 1024) |
| `loader.py` | `build_model(config)` + `load()` with strict 1496-key check; `__main__` key-coverage smoke test |
| `api.py` | `LimiXMLXPredictor`: subclasses the reference `LimiXPredictor`, reuses ALL preprocessing/ensemble/decode logic, swaps only the forward backend to MLX |
| `shim.py` | `nvtx`/`triton` stubs so the reference torch code imports on Mac CPU (Triton norms already fall back to torch ops off-CUDA) |
| `parity.py` | raw-forward parity vs torch CPU (controls the per-forward pos-emb noise) |
| `e2e.py` | sklearn-dataset smoke test (breast cancer + diabetes) |
| `e2e_parity.py` | end-to-end torch-CPU vs MLX agreement |
| `bench_mps_vs_mlx.py` | head-to-head torch/MPS vs MLX timing (mirrors `bench_tabfm_mps_vs_mlx.py`) |
| `baseline_torch.py`, `introspect*.py` | dev utilities |

## Results (M4, this repo)

- Forward parity vs torch CPU: cls 9.7e-05, reg 1.8e-05, imputation 1.3e-06 (max|diff| on raw outputs)
- End-to-end (breast cancer, 120 train / 20 test, 3-pipeline ensemble): same accuracy 0.95, 100% class agreement, proba diff 2.5e-03
- Bench (150 train / 30 test, cls): **MLX 21.8s vs MPS 45.6s → 2.09x** (shared preprocessing; forward-only gap is larger)
- Imputation: NaNs fully recovered, shape-preserving

## Usage

```python
from limix_mlx.api import LimiXMLXPredictor

clf = LimiXMLXPredictor(
    model_path="<LimiX-2.ckpt>",                       # HF download
    inference_config="<limix_src>/config/cls_default_noretrieval_v2.json",
    seed=0)
proba = clf.predict(X_train, y_train, X_test, task_type="Classification")
y_hat = clf.predict(X_train, y_train, X_test, task_type="Regression")
imp = clf.predict(X_train, y_train, X_test, task_type="Feature_imputation")
```

First run converts the ckpt once to `~/.cache/limix_mlx/LimiX-2.npz` (1.6 GB).

## Notes

- Everything runs float32 (matches the torch CPU baseline; MPS runs mixed precision, hence slightly larger diffs there).
- The per-forward random feature-positional code is drawn with `torch.randn` under the caller's seed and fed to both backends, so ensemble members match the reference exactly.
- `LimiX-2` weights are under the StableAI non-commercial license; the port code here follows the repo's existing conventions.
