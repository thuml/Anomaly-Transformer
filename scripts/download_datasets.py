import argparse
import os
import shutil
import zipfile

import gdown
import wget

# Link
datasets = {
    "SMD": "https://drive.google.com/file/d/187cjlXCedf4v3-Xm-fK7iTQZ6k6BFaDK/view?usp=sharing",
    "SMAP": "https://drive.google.com/file/d/1DRj6A4wFGx7SNEalGkEPzRi8Td8RdEK1/view?usp=sharing",
    "PSM": "https://drive.google.com/file/d/1kohMqejb7f787XtpM4b5HR7G22nH-rEF/view?usp=sharing",
    "MSL": "https://drive.google.com/file/d/1BGeu0yiV4T_nsI1G2ayuGfLjIKw9ArZ_/view?usp=sharing",
}

dbsherlock_datasets = {
    "tpcc_16w": "https://github.com/dongyoungy/dbsherlock-reproducibility/raw/master/datasets/dbsherlock_dataset_tpcc_16w.mat",
    "tpcc_500w": "https://github.com/dongyoungy/dbsherlock-reproducibility/raw/master/datasets/dbsherlock_dataset_tpcc_500w.mat",
    "tpce_3000": "https://github.com/dongyoungy/dbsherlock-reproducibility/raw/master/datasets/dbsherlock_dataset_tpce_3000.mat",
}


def move_files_to_parent_dir(parent_dir_path: str, child_dir_name: str) -> None:
    """Move all files in child directory to parent directory."""
    child_dir_path = os.path.join(parent_dir_path, child_dir_name)
    for filename in os.listdir(child_dir_path):
        shutil.move(
            os.path.join(child_dir_path, filename),
            os.path.join(parent_dir_path, filename),
        )
    # Clean up child directory
    os.rmdir(child_dir_path)


def main(output_dir: str) -> None:
    """Download all dataset"""
    # Create directory
    os.makedirs(output_dir, exist_ok=True)

    # Start downloading
    for name, url in datasets.items():
        # Download dataset
        download_path = os.path.join(output_dir, f"{name}.zip")
        print(f"Downloading {name} dataset...")
        gdown.download(url, download_path, quiet=False, fuzzy=True)

        # Unzip dataset
        unzip_path = download_path[:-4]
        print("Extracting...")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        # Move files to parent directory
        child_dir = os.path.join(output_dir, name)
        if os.path.exists(child_dir):
            move_files_to_parent_dir(child_dir, name)
            macosx_dir = os.path.join(output_dir, name, "__MACOSX")
            if os.path.exists(macosx_dir):
                print("Clean __MACOSX directory...")
                shutil.rmtree(macosx_dir)

        # Clean up
        print("Removing zip file...")
        os.remove(download_path)

    # Download DBSherlock Dataset
    print("Downloading DBSherlock dataset...")

    # Create directory
    dir_path = os.path.join(output_dir, "dbsherlock")
    os.makedirs(dir_path, exist_ok=True)

    # Download dataset
    for name, url in dbsherlock_datasets.items():
        download_path = os.path.join(dir_path, f"{name}.mat")
        wget.download(url, out=download_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/root/Anomaly_Explanation/dataset/",
        help="path to save the dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    main(output_dir=output_dir)
