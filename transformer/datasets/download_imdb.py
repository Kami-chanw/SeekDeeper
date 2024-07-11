import requests
import tarfile
import os
from tqdm import tqdm

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
download_path = "."
extract_to = "."

os.makedirs(download_path, exist_ok=True)
file_name = url.split("/")[-1]
file_path = os.path.join(download_path, file_name)

if not os.path.exists(file_path):
    print("Start downloading IMDB datasets...")
    response = requests.head(url)
    file_size = int(response.headers.get("content-length", 0))

    response = requests.get(url, stream=True)
    with open(file_path, "wb") as f, tqdm(
        desc=file_path,
        total=file_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"\nDownload complete.")
else:
    print(f"{file_path} already exists. Skipping download.")

if not os.path.exists(extract_to):
    os.makedirs(extract_to)

with tarfile.open(file_path, "r:*") as tar:
    members = tar.getmembers()
    with tqdm(total=len(members), desc="Extracting", unit="file") as bar:
        for member in members:
            tar.extract(member, path=extract_to)
            bar.update(1)
print(f"Extracted to {extract_to}")
