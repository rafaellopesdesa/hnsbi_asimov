"""Reference-normalized systematic morphs for the hNDE analysis."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from nsbi_common_utils.models.sbi_parametric_model import (
    _calculate_combined_var,
    sbi_parametric_model,
)


class ReferenceNormalizedSystematicsModel:
    """Wrap a one-channel reference-sampled unbinned systematic model."""

    def __init__(self, workspace, measurement_to_fit):
        self.raw_model = sbi_parametric_model(
            workspace=workspace, measurement_to_fit=measurement_to_fit
        )
        self.list_parameters = list(self.raw_model.list_parameters)
        self.initial_parameter_values = self.raw_model.initial_parameter_values
        self.num_unconstrained_param = self.raw_model.num_unconstrained_param
        self._model_data = self.raw_model._model_data
        self._jit_nll, self._jit_value_and_grad, self._jit_many = self._build_jit_functions()

    def get_model_parameters(self):
        """Return parameter names and initial values in fit order."""
        return (self.list_parameters, self.initial_parameter_values)

    def _build_jit_functions(self):
        num_unconstrained = self.num_unconstrained_param
        batched_variation = jax.vmap(_calculate_combined_var, in_axes=(None, 0, 0))

        def nll(param_vec, data):
            nuisance_parameters = param_vec[num_unconstrained:]
            norm_modifiers = jnp.prod(
                jnp.where(data["norm_matrix"], param_vec[None, :], 1.0), axis=1
            )
            yield_variations = batched_variation(
                nuisance_parameters, data["tot_up_unbinned"], data["tot_dn_unbinned"]
            )
            raw_shape_variations = batched_variation(
                nuisance_parameters, data["var_up_unbinned"], data["var_dn_unbinned"]
            )
            shape_partition = jnp.mean(data["ratios"] * raw_shape_variations, axis=1)
            shape_variations = raw_shape_variations / shape_partition[:, None]
            expected_rate = jnp.sum(
                norm_modifiers[:, None] * data["unbinned_total"] * yield_variations, axis=0
            )
            differential_rate_over_reference = jnp.sum(
                norm_modifiers[:, None]
                * data["unbinned_total"]
                * yield_variations
                * data["ratios"]
                * shape_variations,
                axis=0,
            )
            rate_term = -2.0 * jnp.sum(
                data["expected_rate"] * jnp.log(expected_rate) - expected_rate
            )
            event_term = -2.0 * jnp.sum(
                data["weights"]
                * (jnp.log(differential_rate_over_reference) - jnp.log(expected_rate))
            )
            constraint_term = jnp.sum(nuisance_parameters**2)
            return rate_term + event_term + constraint_term

        return (
            jax.jit(nll),
            jax.jit(jax.value_and_grad(nll, argnums=0)),
            jax.jit(jax.vmap(nll, in_axes=(0, None))),
        )

    def model(self, param_array):
        """Evaluate the reference-normalized ``-2 log L``."""
        parameters = jnp.asarray(param_array)
        return self._jit_nll(parameters, self._model_data)

    def model_grad(self, param_array):
        """Evaluate the gradient of the reference-normalized likelihood."""
        parameters = jnp.asarray(param_array)
        _, gradient = self._jit_value_and_grad(parameters, self._model_data)
        return np.asarray(gradient)

    def model_many(self, parameter_points, batch_size=4):
        """Evaluate many parameter points in bounded-memory JAX batches."""
        points = np.asarray(parameter_points, dtype=float)
        batch_size = int(batch_size)
        values = []
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            valid_size = len(batch)
            if valid_size < batch_size:
                padding = np.repeat(batch[-1:, :], batch_size - valid_size, axis=0)
                batch = np.concatenate([batch, padding], axis=0)
            batch_values = np.asarray(
                self._jit_many(jnp.asarray(batch), self._model_data), dtype=float
            )
            values.append(batch_values[:valid_size])
        return np.concatenate(values)
