# hNDE Asimov demonstration

Reproduce the numerical studies in [the paper](https://github.com/rafaellopesdesa/hybrid_nsbi_2).

| Run order | Notebook | Paper |
| --- | --- | --- |
| 1 | [Hybrid_NormalizingFlow_DensityRatio.ipynb](Hybrid_NormalizingFlow_DensityRatio.ipynb) | Figures 1, 2, 3(a), 4, 8 and 10: $q_{\boldsymbol{\phi}}$, $r_{s,\boldsymbol{\psi}}$, $t_\mu$, $\widehat\mu$ and $q_0$. |
| 2 | [NeuralImportanceSampling_Asimov.ipynb](NeuralImportanceSampling_Asimov.ipynb) | Figures 3(b), 5 and 6; Section 5 timing: $g_{\boldsymbol{\eta}}$, $g_\epsilon$ and $\mathcal A_M$. |
| 3 | [SemiParametric_Systematics.ipynb](SemiParametric_Systematics.ipynb) | Figure 7 and Section 5.1: $g_{s,\eta}^{\pm}$ and $\alpha_{\text{scale}}$. |

```bash
python -m pip install -r requirements.txt jupyterlab
jupyter lab
```

Start Jupyter in the repository root and run each notebook from top to bottom.
The notebooks also run in Colab. Saved models are reused from `workspace/`
(Colab: `MyDrive/hnsbi_asimov/workspace`).

[MIT license](LICENSE.txt).
