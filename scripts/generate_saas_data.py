"""Generate synthetic SaaS customer churn dataset aligned with problem statement."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/raw/saas-customer-churn.csv")
RNG = np.random.default_rng(42)
N = 6000

PLAN_TIERS = ["Basic", "Pro", "Enterprise"]
CONTRACT_TYPES = ["Monthly", "Annual"]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    tenure = RNG.integers(1, 48, size=N)
    plan_idx = RNG.choice(3, size=N, p=[0.45, 0.35, 0.20])
    plan_tier = [PLAN_TIERS[i] for i in plan_idx]
    contract_type = RNG.choice(CONTRACT_TYPES, size=N, p=[0.55, 0.45])

    base_charge = np.array([29.0, 79.0, 199.0])[plan_idx]
    monthly_charges = base_charge + RNG.normal(0, 5, size=N)
    monthly_charges = monthly_charges.clip(min=10)

    login_frequency_30d = RNG.integers(0, 45, size=N)
    days_since_last_login = RNG.integers(0, 60, size=N)
    avg_session_minutes = RNG.uniform(2, 120, size=N).round(1)
    feature_usage_count = RNG.integers(0, 25, size=N)
    support_tickets_30d = RNG.integers(0, 8, size=N)
    avg_resolution_hours = RNG.uniform(1, 48, size=N).round(1)
    escalations_90d = RNG.integers(0, 4, size=N)
    failed_payments_90d = RNG.integers(0, 3, size=N)
    payment_method_changes = RNG.integers(0, 3, size=N)

    is_monthly = (np.array(contract_type) == "Monthly").astype(float)
    is_basic = (np.array(plan_tier) == "Basic").astype(float)

    logit = (
        -2.8
        + 0.09 * days_since_last_login
        - 0.08 * login_frequency_30d
        - 0.05 * feature_usage_count
        - 0.015 * avg_session_minutes
        + 0.42 * support_tickets_30d
        + 0.65 * failed_payments_90d
        + 0.50 * escalations_90d
        + 0.20 * payment_method_changes
        + 0.85 * is_monthly
        + 0.65 * is_basic
        - 0.05 * tenure
        + RNG.normal(0, 0.22, size=N)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = RNG.binomial(1, churn_prob)

    # Inject realistic correlations for at-risk profiles
    churn = np.where(
        (days_since_last_login > 35) & (support_tickets_30d >= 3),
        1,
        churn,
    )
    churn = np.where(
        (login_frequency_30d >= 20) & (failed_payments_90d == 0) & (tenure > 12),
        0,
        churn,
    )

    df = pd.DataFrame({
        "customer_id": [f"SaaS-{i:05d}" for i in range(N)],
        "tenure_months": tenure,
        "plan_tier": plan_tier,
        "contract_type": contract_type,
        "monthly_charges": monthly_charges.round(2),
        "login_frequency_30d": login_frequency_30d,
        "days_since_last_login": days_since_last_login,
        "avg_session_minutes": avg_session_minutes,
        "feature_usage_count": feature_usage_count,
        "support_tickets_30d": support_tickets_30d,
        "avg_resolution_hours": avg_resolution_hours,
        "escalations_90d": escalations_90d,
        "failed_payments_90d": failed_payments_90d,
        "payment_method_changes": payment_method_changes,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })

    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")


if __name__ == "__main__":
    main()
