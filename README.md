# hNSBI Asimov likelihood studies

This repository contains the paper-facing reproduction notebooks for the hybrid
neural density-estimation likelihood, the neural importance-sampling
construction of unbinned Asimov data, and the semi-parametric treatment of a
detector-response uncertainty.

## Notebooks

Run the notebooks in the following order:

1. [Hybrid normalizing-flow and density-ratio estimation](Hybrid_NormalizingFlow_DensityRatio.ipynb)
   trains the PRESEL classifier, reference normalizing flow, and the two
   density-ratio ensembles; validates the hybrid densities; and performs the
   profile-likelihood and toy studies.
2. [Neural importance sampling for Asimov data](NeuralImportanceSampling_Asimov.ipynb)
   reuses the frozen nominal models, trains the importance proposal, measures
   the variance reduction, and benchmarks likelihood-evaluation time at
   matched precision.
3. [Semi-parametric systematic uncertainties](SemiParametric_Systematics.ipynb)
   reuses the frozen nominal models, trains detector-response ratios, and
   profiles the resulting nuisance parameter. This is the standard
   semi-parametric construction.

Each notebook has an **Open in Colab** badge. By default, generated events,
model checkpoints, numerical arrays, and plots are kept in
`MyDrive/hnsbi_asimov/workspace`, so later notebooks and later Colab sessions
reuse the same artifacts. The initial event generation and model training are
computationally substantial; completed checkpoints are loaded automatically.

## Backend

Density-ratio training and the base inference model come from
[`nsbi-lhc-toolkit`](https://github.com/rafaellopesdesa/nsbi-lhc-toolkit).
The setup cells use a sparse checkout pinned to
[`832f3086bc7cf830f2b5d907cb2a2d693f50a391`](https://github.com/rafaellopesdesa/nsbi-lhc-toolkit/commit/832f3086bc7cf830f2b5d907cb2a2d693f50a391)
and import `src/nsbi_common_utils`. Pinning the exact revision is important
because the systematics wrapper uses a private interpolation helper from the
backend. A git submodule is intentionally not used: direct Colab launches do
not initialize submodules reliably.

The local modules contain only analysis-specific functionality:

- `utils.py`: ratio-valued inference and the density-ratio trainer adapter;
- `utils_plotting.py`: the plotting functions used by the notebooks;
- `utils_nf.py`: bounded-memory parquet partitions and normalizing-flow tools;
- `utils_systematics.py`: partition-normalized systematic morphing;
- `utils_distributions.py` and `generate_distributions.py`: the toy model and
  bounded-memory sample generation.

## Local execution

Create a Python environment, install the dependencies, and start Jupyter from
the repository root:

```bash
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

The notebook setup creates `workspace/` for generated artifacts and a sparse
backend checkout under `.dependencies/`. These paths are excluded from git.

## Reproducibility notes

- The default sample sizes match the paper studies and require substantial
  time and disk space. The complete nominal and detector-variation samples
  require tens of gigabytes, so check the available Drive or local quota
  before starting the systematics notebook.
- Trained checkpoints are not bundled. Run the nominal hNDE notebook once
  before the two downstream notebooks; subsequent runs reuse its checkpoints.
- Random seeds, train/evaluation partitions, network configurations, and
  inference settings are declared in the notebooks.
- The notebooks save ordinary PNG diagnostics and numerical arrays. They do
  not generate secondary stand-alone plotting programs.
- Wall times depend on the Colab allocation or local hardware; relative
  speedups should be interpreted together with the matched-precision columns.

## License

The software is provided under the MIT License; see [LICENSE.txt](LICENSE.txt).
