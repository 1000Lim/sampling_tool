"""CLI App for runs the app.
"""

import asyncio
import click
from loguru import logger
from rest import data_center_rest as dc_api
from controls.dataset_sampler import run_dataset_sampling


@click.command()
@click.option("--dataset_id", prompt="Input the dataset id", type=str)
@click.option("--sampling_ratio", prompt="Input the sampling ratio (N lidars per lidar)",
              type=int, default=1)
@click.option("--mv_only", prompt="Input Sampling MultiView only", type=bool, default=True)
def run_cli_sample_dataset(dataset_id: str, sampling_ratio: int = 1, mv_only: bool = True):
    """Generates the dataset id with tagging information.

    Args:
        dataset_id (str): dataset id to generate the dataset.
    """
    click.echo(f"Sampling dataset: {dataset_id}")

    try:
        rawdata_keys = dc_api.get_rawdata_keys_from_dataset(dataset_id)
    except Exception as e:
        logger.error(f"Failed to get rawdata keys from dataset: {dataset_id},{e}")
        return

    if not rawdata_keys:
        logger.error(f"No rawdata keys found for dataset: {dataset_id}")
        return

    asyncio.run(run_dataset_sampling(rawdata_keys=rawdata_keys, dataset_id=dataset_id,
                                     sampling_ratio=sampling_ratio, mv_only=mv_only))


if __name__ == "__main__":
    run_cli_sample_dataset()
