from datetime import datetime
from pathlib import Path


def _get_timestamp() -> str:
    return datetime.now().strftime("%F_%H_%M_%S")


def get_savepath(dir: Path, suffix: str, note: str = None):
    filename = _get_timestamp()

    if note:
        filename += f"_{note}"

    filename += suffix

    return Path(dir) / filename
