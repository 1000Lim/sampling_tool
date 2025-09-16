"""CLI App for runs the app.
"""

import asyncio
import click
from loguru import logger
from controls.dataset_sampler import run_dataset_sampling


@click.command()
@click.option("--dataset_id", prompt="Input the dataset id", type=str)
@click.option("--rawdata_key", prompt="Input the rawdata key", type=str)
@click.option("--sampling_ratio", prompt="Input the sampling ratio (N lidars per lidar)",
              type=int, default=1)
@click.option("--mv_only", prompt="Sampling MultiView only", type=bool, default=True)
def run_cli_sample_dataset(dataset_id: str, rawdata_key: str,
                           sampling_ratio: int = 1, mv_only: bool = True):
    """API for sampling dataset with keys."""
    try:
        asyncio.run(
            run_dataset_sampling(rawdata_keys=[rawdata_key], dataset_id=dataset_id,
                                 mv_only=mv_only, sampling_ratio=sampling_ratio)
        )
    except Exception as e:
        logger.error(f"Failed to get rawdata keys from dataset: {dataset_id},{e}")
        return


if __name__ == "__main__":
    run_cli_sample_dataset()
