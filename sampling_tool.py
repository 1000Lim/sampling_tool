import os
import sys
from loguru import logger

from consts import SAMPLING_DIR, DataType
from util.rawdata_util import search_rawdata_dirs_from
from pipelines.surf_pipeline import run_surf_pipeline
from pipelines.valeo_pipeline import run_valeo_pipeline

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.json import JSON
    from rich import print as rprint
    from rich.prompt import Confirm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install typer rich")
    sys.exit(1)


def main(
    stride: int,
    work_path: str,
    cam_num: int,
    export: str,
    compress: bool,
    remove: bool,
    cam_info: str,
    data_type: str,
    overlay_every: int,
    overlay_intensity: bool,
    overlay_point_radius: int,
    overlay_alpha: float,
):
    if not work_path or not os.path.isdir(work_path):
        logger.error("Error: Please input the correct path.")
        return

    rawdata_set = search_rawdata_dirs_from(work_path)
    if not rawdata_set:
        logger.error(f"No rawdata found in the given path: {work_path}")
        return

    if not export:
        export = SAMPLING_DIR

    # Normalize data type
    try:
        data_type_enum = DataType(data_type) if data_type else DataType.SURF
    except ValueError:
        logger.error(f"Unknown data_type: {data_type}. Allowed: {[dt.value for dt in DataType]}")
        return

    # Validate SURF-specific args
    if data_type_enum == DataType.SURF and cam_num is None:
        logger.error("Error: Please input the correct reference camera number.")
        return

    # Validate stride
    if stride < 1:
        logger.error("stride must be >= 1")
        return

    # Dispatch to pipelines
    if data_type_enum == DataType.SURF:
        run_surf_pipeline(rawdata_set, export, compress, remove, cam_info, cam_num, stride, overlay_every, overlay_intensity, overlay_point_radius, overlay_alpha)
    else:
        run_valeo_pipeline(rawdata_set, export, compress, remove, stride, overlay_every, overlay_intensity, overlay_point_radius, overlay_alpha)




app = typer.Typer(add_completion=False, no_args_is_help=True, help='Sampling tools for the standalone environment.')
console = Console()


@app.command(help='Run sampling pipeline for SURF or VALEO dataset')
def run(
    path: str = typer.Option(..., "--path", "-p", help='The path to work on.'),
    cam_num: int = typer.Option(None, "--cam-num", "-c", help='Reference camera number for SURF.'),
    stride: int = typer.Option(1, "--stride", "-s", help='Sample every Nth LiDAR frame (1 = all frames).'),
    compress: bool = typer.Option(False, "--compress", "-z", help='Compress to tar file.'),
    delete: bool = typer.Option(False, "--delete", "-d", help='Delete images/lidars after compression.'),
    export: str = typer.Option(None, "--export", "-e", help='The export path.'),
    cam_info: str = typer.Option(None, "--cam-info", "-i", help='Path to cam info file (SURF only).'),
    data_type: DataType = typer.Option(DataType.SURF, "--data-type", "-t", help='Data type.'),
    overlay_every: int = typer.Option(0, "--overlay-every", help='Generate WEBP overlay every N pairs (0 = disabled).'),
    overlay_intensity: bool = typer.Option(False, "--overlay-intensity", help='Color overlay by intensity instead of distance.'),
    overlay_point_radius: int = typer.Option(2, "--overlay-point-radius", help='Overlay point radius in pixels (>=1).'),
    overlay_alpha: float = typer.Option(1.0, "--overlay-alpha", help='Overlay alpha blending (1.0 = opaque, 0.0 = transparent).'),
):
    main(
        stride=stride,
        work_path=path,
        cam_num=cam_num,
        export=export,
        compress=compress,
        remove=delete,
        cam_info=cam_info,
        data_type=data_type.value,
        overlay_every=overlay_every,
        overlay_intensity=overlay_intensity,
        overlay_point_radius=overlay_point_radius,
        overlay_alpha=overlay_alpha,
    )


if __name__ == '__main__':
    app()
