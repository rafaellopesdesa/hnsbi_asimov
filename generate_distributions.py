"""Generate the nominal samples and detector-scale templates."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from utils_distributions import background_components, signal_components, smearing_parameters

features = ["z1", "z2", "z3", "z4", "z5"]
reco = ["x1", "x2", "x3", "x4", "x5"]
LAM_BKG, LAM_SIG = 1_000_000.0, 1_100.0
SCALE_VARIATIONS = {"scale_up": 1.1, "scale_down": 0.9}


def sample_mixture(components, n, rng):
    """Sample ``n`` events from a Gaussian mixture."""
    fracs = np.array([c[0] for c in components], dtype=float)
    fracs = fracs / fracs.sum()
    counts = rng.multinomial(n, fracs)
    n_dimensions = len(np.asarray(components[0][1]))
    sample = np.empty((n, n_dimensions), dtype=float)
    offset = 0
    for (_, mean, cov), c in zip(components, counts):
        if c > 0:
            sample[offset : offset + c] = rng.multivariate_normal(mean, cov, size=c)
            offset += c
    rng.shuffle(sample)
    return sample


def add_reco_smearing(df, rng):
    """Add x1,...,x5 as independently smeared versions of z1,...,z5."""
    scale, resolution = smearing_parameters()
    y = df[features].to_numpy(dtype=float, copy=False)
    x = rng.normal(loc=0.0, scale=resolution[None, :], size=y.shape)
    for index, response_scale in enumerate(scale):
        x[:, index] += response_scale * y[:, index]
    df[reco] = x
    return df


def scale_variation_dataframe(nominal_df, mean_multiplier):
    """Return a detector-scale variation of an existing nominal sample."""
    mean_multiplier = float(mean_multiplier)
    scale, _ = smearing_parameters()
    scale = np.asarray(scale, dtype=float)
    z = nominal_df[features].to_numpy(dtype=float)
    x_nominal = nominal_df[reco].to_numpy(dtype=float)
    varied = nominal_df.copy()
    varied[reco] = x_nominal + (mean_multiplier - 1.0) * (z * scale[None, :])
    return varied


def nominal_batch_dataframe(
    components, n_events, rng, *, label, expected_yield=None, total_events=None
):
    """Generate one nominal batch with the established parquet schema."""
    n_events = int(n_events)
    batch = pd.DataFrame(sample_mixture(components, n_events, rng), columns=features)
    batch["fold"] = rng.integers(0, 2, size=n_events)
    batch["label"] = int(label)
    if expected_yield is not None:
        batch["weight"] = float(expected_yield) / int(total_events)
    return add_reco_smearing(batch, rng)


def write_nominal_sample(
    path, components, n_events, rng, label, expected_yield, batch_size=100_000
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for start in range(0, n_events, batch_size):
        batch = nominal_batch_dataframe(
            components,
            min(batch_size, n_events - start),
            rng,
            label=label,
            expected_yield=expected_yield,
            total_events=n_events,
        )
        table = pa.Table.from_pandas(batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, use_dictionary=False)
        writer.write_table(table)
    writer.close()


def write_scale_variations(path, batch_size=100_000):
    path = Path(path)
    writers = {}
    for record in pq.ParquetFile(path, pre_buffer=False).iter_batches(
        batch_size=batch_size, use_threads=False
    ):
        nominal = record.to_pandas()
        for variation, multiplier in SCALE_VARIATIONS.items():
            table = pa.Table.from_pandas(
                scale_variation_dataframe(nominal, multiplier), preserve_index=False
            )
            if variation not in writers:
                writers[variation] = pq.ParquetWriter(
                    path.with_name(f"{path.stem}_{variation}.parquet"),
                    table.schema,
                    use_dictionary=False,
                )
            writers[variation].write_table(table)
    for writer in writers.values():
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_bkg", type=int, default=100_000_000)
    parser.add_argument("--n_sig", type=int, default=20_000_000)
    parser.add_argument("--systematics-only", action="store_true")
    args = parser.parse_args()
    if args.systematics_only:
        for name in ["background", "signal"]:
            write_scale_variations(Path("dataframes") / f"{name}.parquet")
    else:
        rng = np.random.default_rng(42)
        write_nominal_sample(
            "dataframes/background.parquet", background_components(), args.n_bkg, rng, 0, LAM_BKG
        )
        write_nominal_sample(
            "dataframes/signal.parquet", signal_components(), args.n_sig, rng, 1, LAM_SIG
        )
