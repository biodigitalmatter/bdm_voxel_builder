import io
import zipfile
from collections import OrderedDict
from pathlib import Path

import requests

REPO_DIR = Path(__file__).parent.parent.parent
DATA_DIR = REPO_DIR / "data"
TEMP_DIR = REPO_DIR / "temp"
TEMP_DIR_CLOUD = (
    Path.home()
    / r"Lund University\MAEF, bioDigital Matter & ABM makerspace - General\data\2024_bdm_voxel_builder__TEMP"
)

if not TEMP_DIR_CLOUD.exists():
    print(f"Temporary dir on OneDrive not accesible, looking for {TEMP_DIR_CLOUD}")


def get(file_name: str, force_redownload=False):
    match Path(file_name).suffix:
        case ".vdb":
            url = f"https://artifacts.aswf.io/io/aswf/openvdb/models/{file_name}/1.0.0/{file_name}-1.0.0.zip"
        case ".pcd":
            commit = "fe1e420bf13455f9dddfe0d324350f8ff98fceee"
            url = f"https://raw.githubusercontent.com/MapIV/pypcd4/{commit}/tests/pcd/{file_name}"
        case _:
            url = None

    filepath = DATA_DIR / "examples" / file_name

    if url is None and not filepath.exists():
        raise ValueError(
            f"File {file_name} not found in the data directory,"
            + "and no url scheme is available"
        )

    if not filepath.exists() or force_redownload:
        r = requests.get(url)
        if url.endswith(".zip"):
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extract(file_name, path=DATA_DIR / "examples")
        else:
            with open(filepath, "wb") as f:
                f.write(r.content)

    return filepath


def get_teapot(force_redownload=False):
    return get("utahteapot.vdb")


def get_direction_dictionary():
    return OrderedDict(
        [
            (
                "up",
                (0, 0, 1),
            ),
            (
                "left",
                (-1, 0, 0),
            ),
            (
                "down",
                (0, 0, -1),
            ),
            (
                "right",
                (1, 0, 0),
            ),
            (
                "front",
                (0, -1, 0),
            ),
            (
                "back",
                (0, 1, 0),
            ),
        ]
    )
