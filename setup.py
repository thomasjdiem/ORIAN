import os
import gdown

files = {
    "inference/without_panels/state_dict_without_panels.pth": "1GG1uvokdbM4udjRcDVe0r39rkqPMUYED",
    "inference/with_panels/state_dict_with_panels.pth.tar": "1rz4FsljZT6wRg1cVo6H4oEyKs0jKkKnu"
}

for path, file_id in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    else:
        print(f"{path} already exists, skipping")