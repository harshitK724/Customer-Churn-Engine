"""Download real customer churn datasets.

Default: IBM Telco Customer Churn (Kaggle: blastchar/telco-customer-churn)
Optional: Bank Customer Churn (--bank)
Optional: Kaggle CLI (--kaggle)
"""
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

TELCO_MIRROR = (
    "https://raw.githubusercontent.com/plotly/datasets/master/"
    "telco-customer-churn-by-IBM.csv"
)
BANK_MIRROR = (
    "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/"
    "Bank%20Churn%20Modelling.csv"
)

DATASETS = {
    "telco": {
        "url": TELCO_MIRROR,
        "out": Path("data/raw/telco-customer-churn.csv"),
        "kaggle": "blastchar/telco-customer-churn",
    },
    "bank": {
        "url": BANK_MIRROR,
        "out": Path("data/raw/bank-churn.csv"),
        "kaggle": "gauravduttakiit/bank-customer-churn-modeling",
    },
}


def download_from_mirror(name: str) -> Path:
    cfg = DATASETS[name]
    out = cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} dataset (Kaggle mirror) to {out}...")
    urllib.request.urlretrieve(cfg["url"], out)
    print("Done.")
    return out


def download_from_kaggle(name: str) -> bool:
    cfg = DATASETS[name]
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", cfg["kaggle"], "-p", str(cfg["out"].parent)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        for zip_path in cfg["out"].parent.glob("*.zip"):
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                for member in zf.namelist():
                    if member.endswith(".csv"):
                        with zf.open(member) as src, open(cfg["out"], "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        zip_path.unlink(missing_ok=True)
                        print(f"Downloaded from Kaggle to {cfg['out']}")
                        return True
        return False
    except FileNotFoundError:
        return False


def main() -> None:
    name = "telco"
    use_kaggle = False
    for arg in sys.argv[1:]:
        if arg == "--kaggle":
            use_kaggle = True
        elif arg == "--bank":
            name = "bank"
        elif arg == "--telco":
            name = "telco"

    if use_kaggle and download_from_kaggle(name):
        return
    if use_kaggle:
        print("Kaggle CLI failed or not configured. Falling back to mirror.")
    download_from_mirror(name)


if __name__ == "__main__":
    main()
