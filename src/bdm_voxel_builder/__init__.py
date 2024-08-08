import io
import zipfile
from pathlib import Path

import requests

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
TEMP_DIR = REPO_DIR / "temp"


def get(file_name: str, force_redownload=False):
    url = f"https://artifacts.aswf.io/io/aswf/openvdb/models/{file_name}/1.0.0/{file_name}-1.0.0.zip"

    filepath = DATA_DIR / "examples" / file_name

    if not filepath.exists() or force_redownload:
        r = requests.get(url)

        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extract(filepath.name, path=DATA_DIR / "examples")

    return filepath


def get_teapot(force_redownload=False):
    return get("utahteapot.vdb")
