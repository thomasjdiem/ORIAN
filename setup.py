import os
import gdown

files = {
    "without": {
        "path": "inference/without_panels/state_dict_without_panels.pth",
        "file_id": "1GG1uvokdbM4udjRcDVe0r39rkqPMUYED"
    },
    "with": {
        "path": "inference/with_panels/state_dict_with_panels.pth",
        "file_id": "1SogsRYLDEGvkhGxSTyJ9ZUVCsUblR_Ra"
    }
}

def download_file(path, file_id):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    else:
        print(f"{path} already exists, skipping.")

def main():
    print("Which pretrained model(s) would you like to download?")
    print("1 = With panels")
    print("2 = Without panels")
    print("3 = Both")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        download_file(**files["with"])
    elif choice == "2":
        download_file(**files["without"])
    elif choice == "3":
        for f in files.values():
            download_file(**f)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
