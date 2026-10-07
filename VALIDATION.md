# Verification record

Verified on 2026-10-07 with Python 3.12.4, PyTorch 2.5.1, macOS ARM64, CPU.
The exact installed packages are recorded in
[validation/environment-cpu-macos.txt](validation/environment-cpu-macos.txt).

## Results

| Check | Result | Scope |
| --- | --- | --- |
| Dependency consistency (`pip check`) | Passed | No broken requirements |
| Editable package install and wheel build | Passed | Includes the shared `Jd.pt` rotation constants |
| Python syntax / shell syntax | Passed | 130 Python files and the OC20 shell wrapper |
| Entry-point `--help` | Passed | All 7 training/evaluation entry points |
| Automated CPU suite | 7 passed | Dataset loading, nonperiodic baseline/Ewald, Chignolin, NaCl, OC20 |
| Dimer training | Passed | One reduced-model epoch, 10 bundled training structures and 3 validation structures |
| Dimer checkpoint evaluation | Passed | Reloaded saved best checkpoint and evaluated the same 3 validation structures |
| Periodic NaCl training/evaluation | Passed | One reduced-model epoch on 8 synthetic structures, split 6/2 |

The automated suite checks finite energy/force outputs, output shapes, finite
nonzero parameter gradients, optimizer steps, and model-state serialization.
OC20 tests cover both 13 and 31 reciprocal vectors; nonperiodic and NaCl tests
include the `lmax=3, mmax=2` path. Chignolin tests exercise the Lightning module
with synthetic input, not its complete dataset or distributed Trainer.

Re-run the portable suite with:

```bash
python scripts/run_smoke_tests.py
```

The runner disables pytest's optional interactive readline import because the
local Anaconda/macOS readline extension crashed before collection. This does
not replace or mock any model, numerical library, graph operation, or dataset
reader. On an unaffected Python installation, `python -m pytest -q tests` also
runs the suite. Upstream deprecation warnings (17 in this environment) remain.

## Repairs made for this release

- Pin a mutually compatible Torch/PyG stack and declare missing dependencies,
  including `torch-cluster`, `torchtnt`, `numba`, and `wandb`.
- Use the included id5 Dimer files in the default configuration.
- Read standard extended-XYZ forces through ASE's calculator interface.
- Allow Dimer CPU execution and respect `system.cuda`; correct visible GPU
  numbering after `CUDA_VISIBLE_DEVICES` remapping.
- Divide each energy by its own atom count when computing batch metrics,
  avoiding a `[batch, batch]` broadcast.
- Implement Dimer checkpoint evaluation using the saved architecture.
- Derive the OC20 Ewald projection dimension from the reciprocal grid rather
  than hard-coding 31. The supplied 31-vector configuration retains the same
  parameter shapes.
- Make OC20 training importable without immediately selecting a CUDA device;
  add `--config`/`--help` and avoid a single-process distributed initialization
  that waits for nonexistent workers.
- Remove author-specific server paths, Conda environment assumptions, fixed GPU
  IDs, and stale checkpoint LFS configuration; preserve upstream attribution.
- Include reproducible installation instructions, CPU smoke configuration,
  package build metadata, and explicit data/verification limitations.

## Not established by these checks

No complete paper benchmark, pretrained-model accuracy, multi-seed comparison,
CUDA kernel execution, multi-GPU run, or full OC20/Chignolin/NaCl dataset run was
verified. The repository has no pretrained checkpoints. The small synthetic
NaCl run checks execution only and has no physical benchmark significance.
The supplied Dimer `test-id5.xyz` is used for validation/model selection by the
example configuration; it is not an independent held-out test set.

Previously supplied CSV files under `experiments_data/` were retained and were
not regenerated or used as evidence of successful reproduction. Numerical
rotation/equivariance tolerances and all manuscript claims require separate
scientific validation. This release does not claim to cover every experiment
or model variant described in a manuscript.
