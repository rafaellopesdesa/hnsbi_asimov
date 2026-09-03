"""Density-ratio compatibility helpers for the hNDE analyses."""

import numpy as np


# ---------------------------------------------------------------------------
# Analysis-local density-ratio compatibility layer
# ---------------------------------------------------------------------------


def convert_score_to_ratio(score, epsilon=1.0e-9):
    """Convert a classifier score to a finite density ratio.

    The upstream package intentionally returns classifier scores.  The hNDE
    analyses work directly with density ratios, so that conversion belongs in
    the analysis rather than in the package-wide inference API.
    """
    score = np.asarray(score, dtype=np.float64)
    score = np.clip(score, 0.0, 1.0 - float(epsilon))
    return score / (1.0 - score)


def predict_with_onnx(dataset, scaler, model, batch_size=10_000):
    """Run ONNX inference using analysis-safe providers and float32 inputs."""
    import onnx
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    if isinstance(model, onnx.ModelProto):
        available = ort.get_available_providers()
        providers = [
            provider
            for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider in available
        ]
        model = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=providers or available,
        )
    elif not isinstance(model, ort.InferenceSession):
        raise TypeError(f"Unsupported model type: {type(model)}")

    scaled = scaler.transform(dataset)
    if hasattr(scaled, "toarray"):
        scaled = scaled.toarray()
    scaled = np.ascontiguousarray(scaled, dtype=np.float32)
    if len(scaled) == 0:
        return np.empty(0, dtype=np.float32)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    predictions = []
    for start in range(0, len(scaled), int(batch_size)):
        batch = scaled[start : start + int(batch_size)]
        predictions.append(model.run([output_name], {input_name: batch})[0])
    return np.concatenate(predictions, axis=0).reshape(-1)


def predict_with_model(
    data,
    scaler,
    model,
    calibration_model=None,
    use_log_loss=False,
    batch_size=10_000,
):
    """Evaluate a classifier and return ``p_num(x) / p_den(x)``.

    This is the analysis counterpart of
    :func:`nsbi_common_utils.training.predict_with_model`, whose upstream API
    returns a score.  Keeping the ratio-returning convention here lets the
    hNDE notebooks use ratios directly without changing the shared package.
    """
    raw_prediction = predict_with_onnx(
        data,
        scaler=scaler,
        model=model,
        batch_size=batch_size,
    )
    if use_log_loss:
        # Stable sigmoid: the upstream calibrators operate in score space.
        raw_prediction = np.asarray(raw_prediction, dtype=np.float64)
        score = np.empty_like(raw_prediction)
        positive = raw_prediction >= 0.0
        score[positive] = 1.0 / (1.0 + np.exp(-raw_prediction[positive]))
        exp_prediction = np.exp(raw_prediction[~positive])
        score[~positive] = exp_prediction / (1.0 + exp_prediction)
    else:
        score = raw_prediction

    if calibration_model is not None:
        score = calibration_model.cali_pred(score)
    return convert_score_to_ratio(score)


def _capture_plotting_call(plotter, *args, **kwargs):
    """Run an upstream plotter and retain figures it normally shows/clears.

    Colab's inline backend may close a figure as soon as ``plt.show()`` is
    called.  Suppress both ``show`` and ``clf`` while the upstream plotting
    function runs so the completed figure can be collected reliably in local
    notebooks and Colab alike.
    """
    import matplotlib.pyplot as plt

    figure_numbers_before = set(plt.get_fignums())
    original_clf = plt.clf
    original_show = plt.show
    plt.clf = lambda: None
    plt.show = lambda *_, **__: None
    try:
        plotter(*args, **kwargs)
        new_numbers = [
            number
            for number in plt.get_fignums()
            if number not in figure_numbers_before
        ]
        figures = [plt.figure(number) for number in new_numbers]
    finally:
        plt.clf = original_clf
        plt.show = original_show
    for figure in figures:
        plt.close(figure)
    return figures


try:
    from nsbi_common_utils.training import density_ratio_trainer as _BaseRatioTrainer
except ImportError:  # Allow data-generation helpers to load without the backend.
    _BaseRatioTrainer = None


if _BaseRatioTrainer is not None:

    class density_ratio_trainer(_BaseRatioTrainer):
        """Analysis adapter around the upstream density-ratio trainer.

        It preserves the upstream training implementation while exposing the
        ratio-valued attributes and returned diagnostic figures used in the
        hNDE analyses.
        """

        def train(self, *args, **kwargs):
            import matplotlib.pyplot as plt
            import nsbi_common_utils.training.neural_ratio_estimation as nre

            # Upstream already selects the best validation checkpoint.
            kwargs.pop("use_best_checkpoint_model", None)
            self.loss_figure = None
            ensemble_index = kwargs.get("ensemble_index", 0)
            original_plot_loss = nre.plot_loss

            def capture_loss(loss_history, path_to_figures="", **_):
                figure, axis = plt.subplots()
                axis.plot(loss_history.train_loss, label="train")
                axis.plot(loss_history.val_loss, label="validation")
                axis.set_title("model loss", size=12)
                axis.set_ylabel("loss", size=12)
                axis.set_xlabel("epoch", size=12)
                axis.legend(loc="upper left")
                figure.savefig(
                    f"{path_to_figures}/loss_plot_{ensemble_index}.png",
                    bbox_inches="tight",
                )
                plt.close(figure)
                self.loss_figure = figure
                return figure

            nre.plot_loss = capture_loss
            try:
                result = super().train(*args, **kwargs)
            finally:
                nre.plot_loss = original_plot_loss

            self.full_data_ratio = convert_score_to_ratio(
                self.full_data_prediction
            )
            self.ratio_den_training = convert_score_to_ratio(
                self.score_den_training
            )
            self.ratio_num_training = convert_score_to_ratio(
                self.score_num_training
            )
            self.ratio_den_holdout = convert_score_to_ratio(
                self.score_den_holdout
            )
            self.ratio_num_holdout = convert_score_to_ratio(
                self.score_num_holdout
            )
            return result

        def make_calib_plots(self, observable="score", nbins=10, ensemble_index=0):
            from nsbi_common_utils.plotting import (
                plot_calibration_curve,
                plot_calibration_curve_ratio,
            )

            score_den_training = self.ratio_den_training / (
                1.0 + self.ratio_den_training
            )
            score_num_training = self.ratio_num_training / (
                1.0 + self.ratio_num_training
            )
            score_den_holdout = self.ratio_den_holdout / (
                1.0 + self.ratio_den_holdout
            )
            score_num_holdout = self.ratio_num_holdout / (
                1.0 + self.ratio_num_holdout
            )
            common = (
                score_den_training,
                self.weight_den_training,
                score_num_training,
                self.weight_num_training,
                score_den_holdout,
                self.weight_den_holdout,
                score_num_holdout,
                self.weight_num_holdout,
            )
            if observable == "score":
                plotter = plot_calibration_curve
            elif observable == "llr":
                plotter = plot_calibration_curve_ratio
            else:
                raise ValueError("observable must be 'score' or 'llr'")
            figures = _capture_plotting_call(
                plotter,
                *common,
                path_to_figures=self.path_to_figures,
                nbins=nbins,
                label="Calibration Curve - " + str(self.sample_name[0]),
                ensemble_index=ensemble_index,
            )
            return figures[-1]

        def make_reweighted_plots(
            self, variables, scale, num_bins, ensemble_index=0
        ):
            from nsbi_common_utils.plotting import plot_reweighted

            score_den_training = self.ratio_den_training / (
                1.0 + self.ratio_den_training
            )
            score_num_training = self.ratio_num_training / (
                1.0 + self.ratio_num_training
            )
            score_den_holdout = self.ratio_den_holdout / (
                1.0 + self.ratio_den_holdout
            )
            score_num_holdout = self.ratio_num_holdout / (
                1.0 + self.ratio_num_holdout
            )
            return _capture_plotting_call(
                plot_reweighted,
                self.dataset_training,
                score_den_training,
                self.weight_den_training,
                score_num_training,
                self.weight_num_training,
                self.dataset_holdout,
                score_den_holdout,
                self.weight_den_holdout,
                score_num_holdout,
                self.weight_num_holdout,
                variables=variables,
                num=num_bins,
                sample_name=self.sample_name,
                scale=scale,
                path_to_figures=self.path_to_figures,
                label_left="Training Data Diagnostic",
                label_right="Holdout Data Diagnostic",
                ensemble_index=ensemble_index,
            )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


FEATURES = ["x1", "x2", "x3", "x4", "x5"]

from utils_distributions import (  # noqa: E402,F401
    background_components,
    signal_components,
    smearing_parameters,
)
