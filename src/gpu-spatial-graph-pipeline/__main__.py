"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Gpu Spatial Graph Pipeline."""


if __name__ == "__main__":
    main(prog_name="gpu-spatial-graph-pipeline")  # pragma: no cover
