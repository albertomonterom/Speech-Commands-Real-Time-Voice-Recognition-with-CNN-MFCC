import os
import tarfile
import urllib.request
from config import DATA_DIR, DATASET_DIR, DATASET_URL, TARGET_LABELS

os.makedirs(DATASET_DIR, exist_ok=True)
tar_path = os.path.join(DATA_DIR, "speech_commands.tar.gz")

def download_with_resume(url, file_path):
    resume_byte_pos = 0
    if os.path.exists(file_path):
        resume_byte_pos = os.path.getsize(file_path)

    req = urllib.request.Request(url)
    
    if resume_byte_pos > 0:
        print(f"🔄 Resuming download from byte {resume_byte_pos}...")
        req.add_header("Range", f"bytes={resume_byte_pos}-")

    with urllib.request.urlopen(req) as response, open(file_path, 'ab') as out_file:
        total_size = response.length + resume_byte_pos if response.length else None
        downloaded = resume_byte_pos

        print("📥 Downloading Speech Commands...")

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded * 100 / total_size
                print(f"\rProgress: {percent:.2f}%  ({downloaded/1e6:.2f} MB)", end="")
        print("\n✔️ Download complete.")


# === DESCARGA ===
if not os.path.exists(tar_path):
    download_with_resume(DATASET_URL, tar_path)
else:
    print("📦 Dataset file already exists. Skipping download.")

# === EXTRACCIÓN ===
print("📦 Extracting selected classes...")
with tarfile.open(tar_path, "r:gz") as tar:
    members = [
        m for m in tar.getmembers()
        if any(
            m.name.startswith(label + "/") or m.name.startswith("./" + label + "/")
            for label in TARGET_LABELS
        )
    ]
    tar.extractall(path=DATASET_DIR, members=members)

print("🎉 Extraction complete.")