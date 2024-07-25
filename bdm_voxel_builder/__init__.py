from pathlib import Path
import requests, zipfile, io

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / "data"
TEMP_DIR = REPO_DIR / "temp"

def get_teapot(force_redownload=False):
    teapot_url = "https://artifacts.aswf.io/io/aswf/openvdb/models/utahteapot.vdb/1.0.0/utahteapot.vdb-1.0.0.zip"

    filepath = DATA_DIR / "examples" / "utahteapot.vdb"

    if not filepath.exists() or force_redownload:

        r = requests.get(teapot_url)

        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extract(filepath.name, path=DATA_DIR / "examples")

    return filepath
