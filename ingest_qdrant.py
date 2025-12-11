import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer

from qdrant_utils import (
    get_embedder,
    get_qdrant_client,
    recreate_collection,
    upsert_points,
)

DATA_ROOT = Path("data")
PROCESSED_PATH = DATA_ROOT / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

QNA_COLLECTION = "medical_qna"
MD_COLLECTION = "medical_device_manual"


def build_qna_dataframe() -> pd.DataFrame:
    raw_path = DATA_ROOT / "raw" / "medical_qna"
    # you may need to adjust file name based on what Kaggle downloads
    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No QnA csv file found in data/raw/medical_qna")

    df = pd.read_csv(csv_files[0])

    df = df.sample(min(len(df), 500), random_state=0).reset_index(drop=True)

    df["combined_text"] = (
        "Question: "
        + df["Question"].astype(str)
        + ". "
        + "Answer: "
        + df["Answer"].astype(str)
        + ". "
        + "Type: "
        + df.get("qtype", "").astype(str)
        + ". "
    )

    out_path = PROCESSED_PATH / "medical_qna.csv"
    df.to_csv(out_path, index=False)
    return df


def build_md_dataframe() -> pd.DataFrame:
    raw_path = DATA_ROOT / "raw" / "medical_device"
    csv_files = list(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No medical device csv file found in data/raw/medical_device")

    df = pd.read_csv(csv_files[0])

    df = df.sample(min(len(df), 500), random_state=0).reset_index(drop=True)

    df["combined_text"] = (
        "Device Name: "
        + df["Device_Name"].astype(str)
        + ". "
        + "Model: "
        + df["Model_Number"].astype(str)
        + ". "
        + "Manufacturer: "
        + df["Manufacturer"].astype(str)
        + ". "
        + "Indications: "
        + df["Indications_for_Use"].astype(str)
        + ". "
        + "Contraindications: "
        + df["Contraindications"].fillna("None").astype(str)
    )

    out_path = PROCESSED_PATH / "medical_device.csv"
    df.to_csv(out_path, index=False)
    return df


def ingest_collection(
    df: pd.DataFrame,
    collection_name: str,
    embedder: SentenceTransformer,
) -> None:
    client = get_qdrant_client()

    vectors = embedder.encode(df["combined_text"].tolist(), show_progress_bar=True)
    vector_size = len(vectors[0])

    recreate_collection(client, collection_name, vector_size)

    payloads = df.to_dict(orient="records")
    ids = df.index.astype(int).tolist()

    upsert_points(
        client,
        collection_name=collection_name,
        vectors=vectors.tolist(),
        payloads=payloads,
        ids=ids,
    )


def main() -> None:
    embedder = get_embedder()

    print("Building QnA dataframe and ingesting...")
    df_qna = build_qna_dataframe()
    ingest_collection(df_qna, QNA_COLLECTION, embedder)

    print("Building medical device dataframe and ingesting...")
    df_md = build_md_dataframe()
    ingest_collection(df_md, MD_COLLECTION, embedder)

    print("Ingestion completed.")


if __name__ == "__main__":
    main()