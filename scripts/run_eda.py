"""Run EDA and export summary JSON."""
import json
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.data.load_data import load_raw_data


def main() -> None:
    df = load_raw_data()

    summary = {
        "dataset": "telco-customer-churn",
        "kaggle_source": "blastchar/telco-customer-churn",
        "row_count": len(df),
        "churn_rate": float((df["churn"] == "Yes").mean()),
        "churn_counts": df["churn"].value_counts().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df[["tenure", "monthly_charges", "total_charges"]].describe().to_dict(),
    }

    contract_churn = pd.crosstab(df["contract"], df["churn"], normalize="index")
    summary["contract_vs_churn"] = contract_churn.to_dict()

    out_path = Path(CONFIG["paths"]["eda_summary_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"EDA summary saved to {out_path}")
    print(f"Churn rate: {summary['churn_rate']:.2%}")


if __name__ == "__main__":
    main()
