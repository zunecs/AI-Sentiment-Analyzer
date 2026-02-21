import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install missing packages from requirements.txt"""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
    except subprocess.CalledProcessError:
        print("[ERR] Failed to install requirements")
        sys.exit(1)


try:
    import numpy, pandas, matplotlib, textblob
except ImportError:
    print("[INFO] Installing required packages...")
    install_requirements()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob


CSV_PATH = "/Users/zunecs/Documents/VS_Code/GitHub/AI-Sentiment-Analyzer/AI-Sentiment-Analyzer/data/Amazon_Unlocked_Mobile.csv"
DEFAULT_PRODUCT = "ALCATEL OneTouch Idol 3 Global Unlocked 4G LTE Smartphone, 4.7 HD IPS Display, 16GB (GSM - US Warranty)"

NEG_THRESHOLD = -0.2
POS_THRESHOLD = 0.2

EXPECTED_COLS = ["Product Name", "Brand", "Reviews", "Price", "Rating", "Votes"]


def dataset_load(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded: {csv_path}  shape={df.shape}")

        missing = [c for c in ["Product Name", "Reviews"] if c not in df.columns]
        if missing:
            print(f"[WARN] Missing expected columns: {missing}")

        return df

    except FileNotFoundError:
        print(f"[ERR] File not found: {csv_path}")
    except PermissionError:
        print(f"[ERR] No permission to read: {csv_path}")
    except pd.errors.EmptyDataError:
        print(f"[ERR] CSV file is empty: {csv_path}")
    except pd.errors.ParserError:
        print(f"[ERR] CSV parse error: {csv_path}")
    except Exception as e:
        print(f"[ERR] Unexpected error: {e}")
    return None


def datatypes_restriction(df):
    for col in ["Product Name", "Brand", "Name", "Reviews"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    if "Votes" in df.columns:
        df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce", downcast="integer")

    return df


def data_cleaning(df, min_chars=10):
    before = len(df)
    df = df.dropna(how="all").drop_duplicates()  # remove Nan

    if "Reviews" not in df.columns:
        print("[ERR] 'Reviews' column not found. Sentiment cannot proceed.")
        return df.iloc[0:0]  # empty

    reviews = df["Reviews"].astype("string")
    mask = reviews.str.len().fillna(0) >= min_chars
    df = df.loc[mask]

    after = len(df)
    print(f"[CLEAN] Rows: {before} → {after} (min review length = {min_chars})")
    return df


def subjectivity_label(s):
    if pd.isna(s):
        return "Nan"
    if s <= 0.05:
        return "Highly objective"
    if s < 0.35:
        return "Objective"
    if s <= 0.65:
        return "medium"
    if s < 0.95:
        return "Subjective"
    return "Highly subjective"


"""

def compute_sentiment_columns(df, text_col = "Reviews") :
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found.")

    texts = df[text_col].astype("string")
    mask = texts.str.len().fillna(0) > 0
    if not mask.all():
        print(f"[INFO] Skipping {(~mask).sum()} empty reviews during sentiment scoring.")

    sentiments = texts.where(mask, "").map(lambda t: TextBlob(t).sentiment if t else TextBlob("").sentiment)

    df["Polarity"] = sentiments.map(lambda s: float(s.polarity))
    df["Subjectivity"] = sentiments.map(lambda s: float(s.subjectivity))
    df["Sentiment"] = df["Polarity"].map(
        lambda p: "Negative" if p < NEG_THRESHOLD else ("Positive" if p > POS_THRESHOLD else "Neutral")
    )
    df["SubjectivityBucket"] = df["Subjectivity"].map(subjectivity_label)
    df["len"] = texts.str.len().fillna(0).astype(int)
    return df

"""


def filter_by_product(df, product_name):
    if "Product Name" not in df.columns:
        print("[WARN] No 'Product Name' column to filter on.")
        return df

    exact = df[df["Product Name"].astype("string") == str(product_name)]
    if len(exact):
        print(f"[INFO] Exact match rows for product: {len(exact)}")
        return exact.copy()  # copy for integrity

    mask = df["Product Name"].str.contains(product_name, case=False, na=False)
    partial = df.loc[mask].copy()
    print(f"[INFO] Partial match rows for product: {len(partial)}")
    return partial


def plot_sentiment_distribution(df, sentiment_col="Sentiment", out_path=None):
    if sentiment_col not in df.columns:
        print(f"[WARN] '{sentiment_col}' not found; skip plot.")
        return
    counts = (
        df[sentiment_col]
        .value_counts()
        .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    )
    ax = counts.plot(kind="bar")
    ax.set_title("Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"[OK] Saved figure → {out_path}")
    plt.close()


def plot_polarity_histogram(df, out_path=None):
    if "Polarity" not in df.columns:
        print("[WARN] 'Polarity' not found; skip plot.")
        return
    ax = df["Polarity"].plot(kind="hist", bins=30)
    ax.set_title("Polarity Histogram")
    ax.set_xlabel("Polarity (-1 to 1)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"[OK] Saved figure → {out_path}")
    plt.close()


def get_product_and_reviews(df):
    cols = [
        c
        for c in ["Product Name", "Reviews", "Brand", "Price", "Rating", "Votes"]
        if c in df.columns
    ]
    return df[cols].copy()


def export_to_csv(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Exported CSV → {out_path}")


def main(
    csv_path: str = CSV_PATH,
    product_name: str = DEFAULT_PRODUCT,
    min_chars: int = 10,
    out_dir: str = "report",
    export_csv: str = "data/predicted.csv",
):
    print("\n" + "$" * 62 + " START OF SENTIMENT ANALYZER " + "$" * 62)

    df = dataset_load(csv_path)
    if df is None or df.empty:
        print("[ERR] Unable to continue (no data).")
        sys.exit(1)

    df = datatypes_restriction(df)
    df = data_cleaning(df, min_chars=min_chars)
    if df.empty:
        print("[ERR] All rows filtered out after cleaning. Nothing to analyze.")
        sys.exit(1)

    df = get_product_and_reviews(df)

    if product_name:
        before = len(df)
        df = filter_by_product(df, product_name)
        print(f"[INFO] Product filter applied → rows {before} → {len(df)}")
        if df.empty:
            print("[WARN] No rows left after product filter.")

    if df.empty:
        print("[ERR] Nothing to score. Exiting.")
        sys.exit(1)

    # df = compute_sentiment_columns(df, text_col="Reviews")

    print("\n=== SENTIMENT SUMMARY ===")
    if "Sentiment" in df.columns:
        print(
            df["Sentiment"]
            .value_counts(dropna=False)
            .reindex(["Positive", "Neutral", "Negative"], fill_value=0)
        )
    if "Polarity" in df.columns:
        print("\nPolarity stats:")
        print(df["Polarity"].describe())
    if "Subjectivity" in df.columns:
        print("\nSubjectivity stats:")
        print(df["Subjectivity"].describe())

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    plot_sentiment_distribution(
        df, out_path=str(out_dir_path / "sentiment_distribution.png")
    )
    plot_polarity_histogram(df, out_path=str(out_dir_path / "polarity_histogram.png"))

    export_cols = [
        c
        for c in [
            "Product Name",
            "Brand",
            "Price",
            "Rating",
            "Votes",
            "Reviews",
            "Polarity",
            "Subjectivity",
            "Sentiment",
            "SubjectivityBucket",
            "len",
        ]
        if c in df.columns
    ]
    export_to_csv(df[export_cols], out_path=export_csv)

    print(
        f"[DONE] Analysis complete.\nCSV: {export_csv}\nPlots: {out_dir_path}/sentiment_distribution.png, {out_dir_path}/polarity_histogram.png\n"
    )


if __name__ == "__main__":
    pass
