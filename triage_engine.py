#!/usr/bin/env python3
"""Production-grade chargeback / payment-dispute triage engine for an Indian BFSI PG.

This module synthesizes Indian-market dispute data, trains a cost-sensitive
Random Forest for merchant win prediction, and dispatches Generative-AI
representment payloads for high-EV cases.

Regulatory anchors (documentation only — synthetic data, not live filings):
  * RBI Master Direction on Digital Payment Security Controls (DPSS).
  * RBI guidelines on customer liability and unauthorised electronic banking
    transactions (limited-liability / 2FA / AFA evidence).
  * NPCI UPI Operating and Settlement Procedures and UPI dispute / T+N
    chargeback reason-code handling.
  * Visa / Mastercard chargeback representment (compelling evidence, AFA,
    proof of delivery / fulfilment).

Dependencies: pandas, numpy, scikit-learn, and the Python standard library.
Fully reproducible with seed=42 / random_state=42.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Operational constants (INR). RBI-aligned labour proxy for L1 dispute desks.
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
LABOR_COST_INR: float = 500.0
LABEL_NOISE_RATE: float = 0.06
TEST_SIZE: float = 0.20
MIN_TEST_SAMPLES_PER_METHOD: int = 15
CSV_PATH: str = "chargeback_data.csv"

PAYMENT_METHODS: Tuple[str, ...] = (
    "UPI",
    "Credit_Card",
    "Debit_Card",
    "NetBanking",
)
PAYMENT_PROBS: Tuple[float, ...] = (0.50, 0.25, 0.15, 0.10)

# Causal logit weights — merchant-win direction (positive => merchant favoured).
LOGIT_BASE: float = 0.0
LOGIT_STRONG_DEFENSE: float = 1.8
LOGIT_LACK_OF_DELIVERY: float = -1.6
LOGIT_UNAUTHORIZED_UPI: float = -1.3
LOGIT_TECHNICAL_FAILURE: float = -1.0
SERIAL_DISPUTER_WIN_PROB: float = 0.12
SERIAL_DISPUTER_THRESHOLD: int = 2  # override when Prior_Dispute_Count > 2

FEATURE_BOOL_COLS: Tuple[str, ...] = (
    "Two_Factor_Auth_Success",
    "Proof_Of_Delivery",
    "Technical_Failure_Flag",
)


def generate_chargeback_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Synthesize Indian BFSI dispute rows and persist ``chargeback_data.csv``.

    Amounts follow a right-skewed lognormal calibrated to a ~₹2,500 mean and
    clipped to the mass-market PG ticket band (₹300–₹12,000), with ~3%
    high-ticket outliers (₹25,000–₹1,00,000) typical of travel / electronics
    merchants. Technical-failure incidence is higher on UPI, reflecting NPCI
    switch / PSP timeout reality. Merchant-win labels are drawn from a
    documented causal logit, then corrupted with 6% flips to emulate card-
    network / bank arbitration variance.

    Args:
        n: Number of synthetic dispute records.
        seed: NumPy Generator seed for bit-exact reproducibility.

    Returns:
        DataFrame with schema documented in the module header and a boolean
        target ``Merchant_Won_Dispute``.
    """
    rng = np.random.default_rng(seed)

    # --- Ticket size: lognormal with exact mean ₹2,500 before clipping -------
    # E[LogNormal(mu, sigma)] = exp(mu + sigma^2 / 2). Invert for mu.
    sigma: float = 0.70
    mu: float = np.log(2500.0) - 0.5 * sigma**2
    amounts: np.ndarray = rng.lognormal(mean=mu, sigma=sigma, size=n)
    amounts = np.clip(amounts, 300.0, 12_000.0)

    n_outliers: int = int(round(0.03 * n))
    outlier_idx: np.ndarray = rng.choice(n, size=n_outliers, replace=False)
    amounts[outlier_idx] = rng.uniform(25_000.0, 100_000.0, size=n_outliers)
    amounts = np.round(amounts, 2)

    payment_method: np.ndarray = rng.choice(
        PAYMENT_METHODS, size=n, p=PAYMENT_PROBS
    )

    # AFA / 2FA success ~85% — RBI unauthorised-transaction liability turns
    # on whether additional factor of authentication completed.
    two_factor: np.ndarray = rng.random(n) < 0.85

    # Carrier / fulfilment POD ~70% — Visa/MC compelling-evidence pillar.
    proof_of_delivery: np.ndarray = rng.random(n) < 0.70

    # Gateway / switch timeout: ~20% UPI, ~4% other rails => ~12% overall.
    # A logged technical failure supports the customer's "debited but not
    # fulfilled / not authorised as claimed" theory (customer-favourable).
    technical_failure: np.ndarray = np.zeros(n, dtype=bool)
    upi_mask: np.ndarray = payment_method == "UPI"
    technical_failure[upi_mask] = rng.random(int(upi_mask.sum())) < 0.20
    technical_failure[~upi_mask] = rng.random(int((~upi_mask).sum())) < 0.04

    # Repeat-disputer velocity (issuer / NPCI watch-list analogue).
    prior_dispute_count: np.ndarray = rng.poisson(lam=0.45, size=n)

    # --- Causal logit for P(merchant wins arbitration) ----------------------
    logit: np.ndarray = np.full(n, LOGIT_BASE, dtype=float)

    strong_defense: np.ndarray = two_factor & proof_of_delivery
    logit[strong_defense] += LOGIT_STRONG_DEFENSE

    logit[~proof_of_delivery] += LOGIT_LACK_OF_DELIVERY

    unauthorized_upi: np.ndarray = upi_mask & (~technical_failure)
    logit[unauthorized_upi] += LOGIT_UNAUTHORIZED_UPI

    logit[technical_failure] += LOGIT_TECHNICAL_FAILURE

    win_prob: np.ndarray = 1.0 / (1.0 + np.exp(-logit))

    # Hard bank policy: serial disputers (count > 2) are not logit-capped;
    # win probability is unconditionally replaced (issuer fraud-ring controls).
    serial_mask: np.ndarray = prior_dispute_count > SERIAL_DISPUTER_THRESHOLD
    win_prob = np.where(serial_mask, SERIAL_DISPUTER_WIN_PROB, win_prob)

    merchant_won: np.ndarray = rng.binomial(1, win_prob).astype(bool)

    # 6% i.i.d. label flips — network arbitration / reason-code mapping noise.
    flip_mask: np.ndarray = rng.random(n) < LABEL_NOISE_RATE
    merchant_won = np.where(flip_mask, ~merchant_won, merchant_won).astype(bool)

    df = pd.DataFrame(
        {
            "Transaction_Amount_INR": amounts,
            "Payment_Method": payment_method,
            "Two_Factor_Auth_Success": two_factor,
            "Proof_Of_Delivery": proof_of_delivery,
            "Technical_Failure_Flag": technical_failure,
            "Prior_Dispute_Count": prior_dispute_count.astype(int),
            "Merchant_Won_Dispute": merchant_won,
        }
    )
    df.to_csv(CSV_PATH, index=False)

    _print_data_diagnostics(df)
    return df


def _prior_bucket(count: int) -> str:
    """Map prior-dispute velocity onto reporting buckets 0 / 1 / 2 / 3+."""
    if count >= 3:
        return "3+"
    return str(int(count))


def _print_data_diagnostics(df: pd.DataFrame) -> None:
    """Print class balance and win-rate crosstabs required by Task 1."""
    print("\n" + "=" * 72)
    print("TASK 1 — Synthetic Indian BFSI dispute corpus")
    print("=" * 72)
    print(f"Rows written to {CSV_PATH}: {len(df):,}")
    print(
        "Ticket mean (INR): "
        f"{df['Transaction_Amount_INR'].mean():,.2f} | "
        f"min={df['Transaction_Amount_INR'].min():,.2f} | "
        f"max={df['Transaction_Amount_INR'].max():,.2f}"
    )
    print(
        "Technical_Failure_Flag overall rate: "
        f"{df['Technical_Failure_Flag'].mean():.1%}"
    )
    print(
        "Technical_Failure_Flag | UPI: "
        f"{df.loc[df['Payment_Method'] == 'UPI', 'Technical_Failure_Flag'].mean():.1%}"
    )

    print("\nClass balance — Merchant_Won_Dispute")
    counts = df["Merchant_Won_Dispute"].value_counts().rename("count")
    rates = df["Merchant_Won_Dispute"].value_counts(normalize=True).rename("rate")
    balance = pd.concat([counts, rates], axis=1)
    print(balance.to_string())

    print("\nMerchant win rate by Payment_Method (NPCI vs card-network rails)")
    by_method = (
        df.groupby("Payment_Method")["Merchant_Won_Dispute"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "n"})
        .reindex(list(PAYMENT_METHODS))
    )
    by_method["win_rate"] = by_method["win_rate"].map(lambda x: f"{x:.1%}")
    print(by_method.to_string())

    buckets = df["Prior_Dispute_Count"].map(_prior_bucket)
    print("\nMerchant win rate by Prior_Dispute_Count bucket (serial-disputer policy)")
    by_prior = (
        df.assign(Prior_Bucket=buckets)
        .groupby("Prior_Bucket")["Merchant_Won_Dispute"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "win_rate", "count": "n"})
        .reindex(["0", "1", "2", "3+"])
    )
    by_prior["win_rate"] = by_prior["win_rate"].map(
        lambda x: f"{x:.1%}" if pd.notna(x) else "n/a"
    )
    print(by_prior.to_string())

    print("\nPayment_Method × Merchant_Won_Dispute (counts)")
    print(
        pd.crosstab(
            df["Payment_Method"],
            df["Merchant_Won_Dispute"],
            margins=True,
        ).to_string()
    )
    print("\nPrior_Bucket × Merchant_Won_Dispute (counts)")
    print(
        pd.crosstab(
            buckets.rename("Prior_Bucket"),
            df["Merchant_Won_Dispute"],
            margins=True,
        )
        .reindex(["0", "1", "2", "3+"])
        .to_string()
    )


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode rails and cast boolean evidence flags to integers.

    Payment_Method dummies are retained in full (no drop_first) so UPI vs
    card-network effects remain directly interpretable against NPCI / scheme
    representment playbooks.
    """
    X = df.copy()
    for col in FEATURE_BOOL_COLS:
        X[col] = X[col].astype(int)
    X = pd.get_dummies(X, columns=["Payment_Method"], dtype=int)
    return X


def _validate_test_method_coverage(payment_method: pd.Series) -> None:
    """Warn if a rail is too thin in hold-out for reliable per-method metrics."""
    counts = payment_method.value_counts()
    print("\nTest-set Payment_Method coverage (minimum "
          f"{MIN_TEST_SAMPLES_PER_METHOD} required):")
    for method in PAYMENT_METHODS:
        n_method = int(counts.get(method, 0))
        print(f"  {method}: {n_method}")
        if n_method < MIN_TEST_SAMPLES_PER_METHOD:
            print(
                f"WARNING: Payment_Method category '{method}' has only "
                f"{n_method} test samples (threshold="
                f"{MIN_TEST_SAMPLES_PER_METHOD}). Per-rail metrics for this "
                "category are statistically fragile — do not silently proceed "
                "to production calibration on this split."
            )


def _labeled_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return TN/FP/FN/TP with dispute-ops labels."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    labeled = {
        "TN_Auto_Conceded": int(tn),
        "FP_Wasted_Labor": int(fp),
        "FN_Lost_Revenue": int(fn),
        "TP_Successfully_Defended": int(tp),
    }
    print(
        "\nConfusion matrix (rows=actual merchant-win, cols=predicted dispute):\n"
        f"  TN Auto-Conceded        : {labeled['TN_Auto_Conceded']}\n"
        f"  FP Wasted Labor         : {labeled['FP_Wasted_Labor']}\n"
        f"  FN Lost Revenue         : {labeled['FN_Lost_Revenue']}\n"
        f"  TP Successfully Defended: {labeled['TP_Successfully_Defended']}"
    )
    return labeled


def _cost_optimal_decisions(
    win_prob: np.ndarray, amounts: np.ndarray
) -> np.ndarray:
    """Dispute iff expected recovered value exceeds desk labour (₹500).

    EV = P(merchant win) × ticket. Fighting a ₹400 UPI debit at P=0.9 still
    fails the hurdle; fighting a ₹80,000 electronics charge at P=0.4 clears it.
    """
    expected_value = win_prob * amounts
    return (expected_value > LABOR_COST_INR).astype(int)


def _print_financial_ledger(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    cm: Dict[str, int],
) -> None:
    """Print FP/FN/baseline labour and net savings vs a fight-all policy.

    Baseline is the labour cost of disputing **every** test-set case
    (``n_test × ₹500``). That is the no-triage PG operations policy.
    Selected-case labour ``(FP+TP)×₹500`` is also printed for transparency;
    using it as baseline would cancel TP/FP labour and collapse savings to
    ``−FN``, which is not an economic comparison.
    """
    fp = cm["FP_Wasted_Labor"]
    fn = cm["FN_Lost_Revenue"]
    tp = cm["TP_Successfully_Defended"]

    fp_cost = fp * LABOR_COST_INR
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_cost = float(amounts[fn_mask].sum())
    tp_labor = tp * LABOR_COST_INR
    n_test = int(len(y_true))
    baseline_cost = n_test * LABOR_COST_INR
    selected_labor = (fp + tp) * LABOR_COST_INR
    net_savings = baseline_cost - (fp_cost + fn_cost + tp_labor)

    print("\n" + "-" * 72)
    print("Financial ledger (INR) — test set")
    print("-" * 72)
    print(
        f"  False Positive Cost (FP × ₹{LABOR_COST_INR:,.0f} wasted labour): "
        f"₹{fp_cost:,.2f}"
    )
    print(
        f"  False Negative Cost (sum of tickets on conceded winnable cases, "
        f"n={fn}): ₹{fn_cost:,.2f}"
    )
    print(
        f"  TP Labour Cost (successfully defended × ₹{LABOR_COST_INR:,.0f}): "
        f"₹{tp_labor:,.2f}"
    )
    print(
        f"  Baseline Cost (dispute 100% of test set blindly, "
        f"n={n_test} × ₹{LABOR_COST_INR:,.0f}): ₹{baseline_cost:,.2f}"
    )
    print(
        f"  Selected-case labour (FP+TP) × ₹{LABOR_COST_INR:,.0f} "
        f"[informational]: ₹{selected_labor:,.2f}"
    )
    print(
        f"  Net Financial Savings = Baseline − (FP Cost + FN Cost + TP Labour): "
        f"₹{net_savings:,.2f}"
    )


def _evidence_strength_summary(row: pd.Series) -> str:
    """Compose a human-readable evidence narrative from the row's features."""
    fragments: List[str] = []

    two_fa = bool(row["Two_Factor_Auth_Success"])
    pod = bool(row["Proof_Of_Delivery"])
    tech = bool(row["Technical_Failure_Flag"])
    method = str(row["Payment_Method"])
    prior = int(row["Prior_Dispute_Count"])

    if two_fa and pod:
        fragments.append(
            "Strong merchant defence: additional factor of authentication "
            "(2FA/AFA OTP) succeeded and carrier Proof of Delivery is on file "
            "(Visa/Mastercard compelling-evidence standard; RBI 2FA liability "
            "allocation favours the merchant where AFA completed)."
        )
    elif two_fa:
        fragments.append(
            "AFA/2FA succeeded, supporting authorised-use rebuttal under RBI "
            "unauthorised electronic banking transaction guidance, but "
            "fulfilment evidence is incomplete."
        )
    else:
        fragments.append(
            "AFA/2FA did not succeed — merchant cannot rely on RBI 2FA "
            "safe-harbour; unauthorised-use claims are customer-leaning."
        )

    if not pod:
        fragments.append(
            "Lack of delivery / fulfilment confirmation: no Proof of Delivery. "
            "Card-network representment for services-not-provided / "
            "merchandise-not-received is weak without POD or usage logs."
        )
    else:
        fragments.append(
            "Proof of Delivery confirmed by carrier — supports merchandise/"
            "service rendered representment."
        )

    if tech:
        fragments.append(
            "Technical_Failure_Flag=True: gateway/switch timeout or error was "
            "logged. Under NPCI UPI failure handling and PG reconciliation "
            "practice this corroborates a customer claim of debit-without-"
            "fulfilment and is customer-favourable, not merchant-favourable."
        )
    elif method == "UPI":
        fragments.append(
            "UPI rail with no logged technical failure: a clean 'I did not "
            "authorise this' claim is hard to rebut (NPCI UPI mandate / "
            "device-binding trail exists; no timeout excuse)."
        )
    else:
        fragments.append(
            f"{method} authorisation completed without a logged switch error; "
            "scheme reason-code defence should cite auth code and AVS/OTP "
            "where available."
        )

    if prior > SERIAL_DISPUTER_THRESHOLD:
        fragments.append(
            f"Serial-disputer velocity: Prior_Dispute_Count={prior} (>2). "
            "Issuer/bank hard policy typically suppresses merchant win rate "
            "regardless of evidence strength."
        )
    elif prior > 0:
        fragments.append(
            f"Customer has {prior} prior dispute(s) — elevated velocity, "
            "below the hard serial-disputer override."
        )

    return " ".join(fragments)


def _scheme_guidelines_clause(payment_method: str) -> str:
    """Select the scheme / network corpus the letter should cite."""
    if payment_method == "UPI":
        return (
            "NPCI UPI Operating and Settlement Procedures, UPI dispute and "
            "chargeback reason-code notes, and RBI Master Direction on Digital "
            "Payment Security Controls (AFA, reconciliation, customer liability)."
        )
    if payment_method == "Credit_Card":
        return (
            "Visa Core Rules / Mastercard Chargeback Guide representment "
            "rights (compelling evidence, 3-D Secure / AFA, proof of delivery), "
            "and RBI card-present/not-present authentication requirements."
        )
    if payment_method == "Debit_Card":
        return (
            "Visa/Mastercard debit chargeback representment standards, NPCI "
            "RuPay dispute operating guidelines where the BIN is domestic, and "
            "RBI unauthorised debit-card transaction liability circulars."
        )
    return (
        "Acquiring-bank net-banking dispute SOPs, RBI electronic banking "
        "transaction liability framework, and the merchant's authenticated "
        "session / OTP evidence pack."
    )


def build_llm_payloads(
    df_test: pd.DataFrame,
    y_pred_prob: np.ndarray,
    top_k: int = 3,
) -> str:
    """Build JSON auto-responder payloads for the highest-confidence disputes.

    Filters to rows the cost-optimal policy would fight
    (``P(win) × amount > ₹500``), then keeps the ``top_k`` cases by predicted
    merchant-win probability for Generative-AI representment drafting.

    Args:
        df_test: Test-frame with original (pre-dummy) feature columns and
            ``Transaction_Amount_INR``. Index should be stable dispute keys.
        y_pred_prob: Model P(merchant wins), aligned to ``df_test`` row order.
        top_k: Number of highest-probability fight cases to dispatch.

    Returns:
        Pretty-printed JSON array string (also written to stdout).
    """
    if len(df_test) != len(y_pred_prob):
        raise ValueError(
            "df_test and y_pred_prob must be the same length "
            f"({len(df_test)} vs {len(y_pred_prob)})."
        )

    work = df_test.copy()
    work = work.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    work["_win_prob"] = np.asarray(y_pred_prob, dtype=float)
    work["_expected_value"] = work["_win_prob"] * work["Transaction_Amount_INR"]
    fight = work[work["_expected_value"] > LABOR_COST_INR].copy()
    fight = fight.sort_values("_win_prob", ascending=False).head(top_k)

    payloads: List[Dict[str, object]] = []
    for _, row in fight.iterrows():
        orig_idx = int(row["_orig_idx"])
        prob = float(row["_win_prob"])
        method = str(row["Payment_Method"])
        recommendation = (
            "AUTO_GENERATE_EVIDENCE_PACK"
            if prob > 0.80
            else "HUMAN_REVIEW_RECOMMENDED"
        )
        guidelines = _scheme_guidelines_clause(method)
        system_prompt = (
            "You are a senior representment specialist at an Indian payment "
            "gateway writing a formal Representment Letter to the acquiring "
            f"bank for dispute {orig_idx:04d} on the {method} rail. "
            "Cite the following corpus where applicable: "
            f"{guidelines} "
            "Use only the evidence in this payload; do not invent "
            "authorisation codes, ARNs, RRN/UTR, or delivery consignment "
            "numbers. Structure the letter as: (1) transaction identification, "
            "(2) scheme/NPCI reason-code rebuttal, (3) AFA/2FA and fulfilment "
            "evidence, (4) technical-failure analysis if flagged, (5) request "
            "to reverse the chargeback and re-debit the issuer. Tone: formal, "
            "evidentiary, non-accusatory toward the cardholder. Currency INR."
        )
        payloads.append(
            {
                "dispute_id": f"DSP-2026-{orig_idx:04d}",
                "transaction_details": {
                    "Amount_INR": round(float(row["Transaction_Amount_INR"]), 2),
                    "Payment_Method": method,
                    "Two_Factor_Auth_Success": bool(row["Two_Factor_Auth_Success"]),
                },
                "evidence_strength": _evidence_strength_summary(row),
                "predicted_win_probability": round(prob, 3),
                "triage_recommendation": recommendation,
                "llm_system_prompt": system_prompt,
            }
        )

    rendered = json.dumps(payloads, indent=2, ensure_ascii=False)
    print("\n" + "=" * 72)
    print("TASK 3 — Generative AI auto-responder payload dispatcher")
    print("=" * 72)
    print(
        f"Cost-optimal fight universe: {int((work['_expected_value'] > LABOR_COST_INR).sum())} "
        f"cases | dispatched top_k={len(payloads)}"
    )
    print(rendered)
    return rendered


def train_and_triage(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Encode, split, train RF, and report classification plus financial metrics.

    Returns:
        Tuple of (raw test features, true labels, predicted win probabilities)
        for the LLM payload dispatcher.
    """
    print("\n" + "=" * 72)
    print("TASK 2 — Cost-sensitive Random Forest triage")
    print("=" * 72)

    y = df["Merchant_Won_Dispute"].astype(int)
    raw_features = df.drop(columns=["Merchant_Won_Dispute"])
    X = encode_features(raw_features)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        raw_train,
        raw_test,
    ) = train_test_split(
        X,
        y,
        raw_features,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    # Silence unused-variable lint for the unused raw train split.
    del raw_train

    _validate_test_method_coverage(raw_test["Payment_Method"])

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=4,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    amounts = raw_test["Transaction_Amount_INR"].to_numpy(dtype=float)
    y_true = y_test.to_numpy(dtype=int)

    y_pred_half = (y_prob >= 0.50).astype(int)
    y_pred_cost = _cost_optimal_decisions(y_prob, amounts)

    print("\nClassification report — standard 0.50 probability threshold")
    print(
        classification_report(
            y_true,
            y_pred_half,
            target_names=["Merchant_Lost (0)", "Merchant_Won (1)"],
            digits=3,
            zero_division=0,
        )
    )
    print("Confusion matrix — 0.50 threshold")
    _labeled_confusion_matrix(y_true, y_pred_half)

    print("\nClassification report — cost-optimal rule "
          "(P(win) × Amount > ₹500 ⇒ Dispute)")
    print(
        classification_report(
            y_true,
            y_pred_cost,
            target_names=["Concede (0)", "Dispute (1)"],
            digits=3,
            zero_division=0,
        )
    )
    print("Confusion matrix — cost-optimal decision rule")
    cm_cost = _labeled_confusion_matrix(y_true, y_pred_cost)

    importances = (
        pd.Series(model.feature_importances_, index=list(X.columns))
        .sort_values(ascending=False)
        .rename("importance")
    )
    print("\nSorted feature importances (RandomForest Gini)")
    print(importances.to_string())

    _print_financial_ledger(y_true, y_pred_cost, amounts, cm_cost)

    # Align raw_test index with probability vector order (split preserves index).
    return raw_test, y_true, y_prob


def main() -> None:
    """Run data generation, model triage, and LLM payload dispatch end-to-end."""
    np.random.seed(RANDOM_SEED)
    df = generate_chargeback_data(n=1000, seed=RANDOM_SEED)
    raw_test, _y_true, y_prob = train_and_triage(df)
    build_llm_payloads(raw_test, y_pred_prob=y_prob, top_k=3)


if __name__ == "__main__":
    main()
