from pathlib import Path

import requests
from tqdm import tqdm

def download_file(url: str, dest: Path) -> None:
    """
    Download a file from a url and save it to dest, displaying a progress bar.

    :param url: URL to download from.
    :param dest: Path to save the file to.
    :return: None or throws an error if the file cannot be downloaded.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    chunk_size = 8 * 1024  # 8 KiB

    with open(dest, "wb") as f, tqdm(
        desc=dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            progress.update(len(chunk))