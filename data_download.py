import os
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

load_dotenv()

DATA_ROOT = Path("data")
RAW_MD_PATH = DATA_ROOT / "raw" / "medical_device"
RAW_QA_PATH = DATA_ROOT / "raw" / "medical_qna"


def ensure_dirs() -> None:
    RAW_MD_PATH.mkdir(parents=True, exist_ok=True)
    RAW_QA_PATH.mkdir(parents=True, exist_ok=True)


def download_datasets() -> None:
    """Download both datasets from Kaggle using KaggleApi."""
    ensure_dirs()

    api = KaggleApi()
    api.authenticate()

    md_dataset = "pratyushpuri/global-medical-device-manuals-dataset-2025"
    qa_dataset = "thedevastator/comprehensive-medical-q-a-dataset"

    print("Downloading medical device manuals dataset...")
    api.dataset_download_files(
        md_dataset,
        path=str(RAW_MD_PATH),
        unzip=True,
    )

    print("Downloading medical QnA dataset...")
    api.dataset_download_files(
        qa_dataset,
        path=str(RAW_QA_PATH),
        unzip=True,
    )

    print("Download completed.")


if __name__ == "__main__":
    download_datasets()