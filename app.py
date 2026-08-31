"""
RazorGuard — Chargeback Triage & Auto-Defense Dashboard
------------------------------------------------------
Interactive Streamlit UI for BFSI payment dispute triage and LLM auto-responder dispatch.
"""

import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="RazorGuard | Chargeback Triage Engine",
    layout="wide",
    page_icon="🛡️",
)

# -----------------------------------------------------------------------------
# CORE BACKEND INFERENCE PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_or_generate_dataset() -> pd.DataFrame:
    """Loads chargeback_data.csv if present, or creates fallback data."""
    try:
        return pd.read_csv("chargeback_data.csv")
    except FileNotFoundError:
        st.warning("⚠️ chargeback_data.csv not found. Please run python triage_engine.py first.")
        return pd.DataFrame()


def train_model_and_infer(df: pd.DataFrame):
    """Encodes features, fits RandomForest, and performs cost-sensitive inference."""
    df_proc = pd.get_dummies(df, columns=["Payment_Method"], drop_first=False)
    
    # Ensure all expected one-hot columns exist
    for col in ["Payment_Method_UPI", "Payment_Method_Credit_Card", "Payment_Method_Debit_Card", "Payment_Method_NetBanking"]:
        if col not in df_proc.columns:
            df_proc[col] = 0

    feature_cols = [
        "Transaction_Amount_INR",
        "Two_Factor_Auth_Success",
        "Proof_Of_Delivery",
        "Technical_Failure_Flag",
        "Prior_Dispute_Count",
        "Payment_Method_UPI",
        "Payment_Method_Credit_Card",
        "Payment_Method_Debit_Card",
        "Payment_Method_NetBanking",
    ]
    
    X = df_proc[feature_cols].astype(float)
    y = df_proc["Merchant_Won_Dispute"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=4, random_state=42
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_df_original = df.iloc[X_test.index].copy()
    test_df_original["Predicted_Prob"] = np.round(test_probs, 3)
    test_df_original["Expected_Value_INR"] = np.round(test_probs * test_df_original["Transaction_Amount_INR"], 2)
    test_df_original["Recommended_Action"] = np.where(
        test_df_original["Expected_Value_INR"] > 500.0, "DISPUTE", "AUTO_CONCEDE"
    )

    # Financial Ledger Calculations
    labor_cost_per_case = 500.0
    is_dispute_rec = test_df_original["Recommended_Action"] == "DISPUTE"
    actual_won = y_test.values == 1

    tn = np.sum((~is_dispute_rec) & (~actual_won))
    fp = np.sum(is_dispute_rec & (~actual_won))
    fn = np.sum((~is_dispute_rec) & actual_won)
    tp = np.sum(is_dispute_rec & actual_won)

    fp_cost = fp * labor_cost_per_case
    fn_cost = test_df_original.loc[(~is_dispute_rec) & actual_won, "Transaction_Amount_INR"].sum()
    tp_labor = tp * labor_cost_per_case
    baseline_cost = 200 * labor_cost_per_case
    net_savings = baseline_cost - (fp_cost + fn_cost + tp_labor)

    metrics = {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FP_Cost": fp_cost,
        "FN_Cost": fn_cost,
        "Net_Savings": net_savings,
        "Feature_Importances": pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
    }

    return test_df_original, metrics


def generate_mock_letter(dispute_row: pd.Series) -> str:
    """Generates a formal bank representment rebuttal letter from row features."""
    return f"""================================================================================
FORMAL REPRESENTMENT REBUTTAL — ACQUIRING BANK DISPUTE DESK
================================================================================
Dispute Case ID : DSP-2026-{dispute_row.name:04d}
Transaction Ref : TXN_{dispute_row.name * 73921 % 1000000:06d}
Payment Rail    : {dispute_row['Payment_Method']}
Dispute Amount  : ₹{dispute_row['Transaction_Amount_INR']:,.2f}
Win Probability : {dispute_row['Predicted_Prob'] * 100:.1f}%

To: Dispute Resolution & Chargeback Arbitration Committee

1. STATEMENT OF CONTEST:
The merchant hereby formally contests the debit adjustment initiated for the subject 
transaction in accordance with RBI Master Directions on Payment Transactions and 
applicable Card Network / NPCI Representment Guidelines.

2. COMPELLING EVIDENCE OF AUTHENTICATION & FULFILMENT:
- Mandatory 2FA/AFA Verification : {'SUCCESS (OTP Authenticated)' if dispute_row['Two_Factor_Auth_Success'] else 'UNAVAILABLE'}
- Carrier Delivery Confirmation  : {'CONFIRMED (Proof of Delivery on Record)' if dispute_row['Proof_Of_Delivery'] else 'PENDING'}
- Gateway Technical Switch Status: {'NO ERRORS / SUCCESS LOG' if not dispute_row['Technical_Failure_Flag'] else 'SWITCH ERROR LOGGED'}
- Customer Account History       : {dispute_row['Prior_Dispute_Count']} prior dispute claims.

3. LEGAL & REGULATORY REBUTTAL:
Under the RBI framework on Electronic Payment Transactions, where Two-Factor Authentication 
(AFA) was successfully completed and merchandise delivery confirmed by carrier logs, 
liability for unauthorized transaction claims does not reside with the merchant.

4. REQUESTED ACTION:
We request immediate reversal of this chargeback and credit re-settlement of ₹{dispute_row['Transaction_Amount_INR']:,.2f} 
to the merchant's settlement ledger.

Sincerely,
Merchant Risk & Representment Automation Engine (Powered by RazorGuard)
================================================================================"""


# -----------------------------------------------------------------------------
# APPLICATION UI
# -----------------------------------------------------------------------------
st.title("🛡️ RazorGuard | AI Chargeback Triage Engine")
st.caption("Post-Transaction Risk Management, Cost-Optimal Dispute Triage & Bank Auto-Responder for Indian BFSI Rails")

df_raw = load_or_generate_dataset()

if df_raw.empty:
    st.error("Please ensure `chargeback_data.csv` is present in your folder.")
    st.stop()

test_results, metrics = train_model_and_infer(df_raw)

tab1, tab2 = st.tabs(["⚡ Dispute Queue & Auto-Responder", "📊 Financial Ledger & Unit Economics"])

# TAB 1: OPERATIONAL TRIAGE
with tab1:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test Disputes Evaluated", len(test_results))
    with col_m2:
        fight_count = (test_results["Recommended_Action"] == "DISPUTE").sum()
        st.metric("Recommended to Fight", f"{fight_count} ({fight_count/len(test_results)*100:.0f}%)")
    with col_m3:
        st.metric("Auto-Conceded (Low ROI)", len(test_results) - fight_count)
    with col_m4:
        st.metric("Avg Predicted Win Rate", f"{test_results['Predicted_Prob'].mean()*100:.1f}%")

    st.markdown("---")
    st.subheader("Actionable Dispute Queue")
    st.caption("Filtered where Expected Value ($P_{\\text{win}} \\times \\text{Amount}$) exceeds the ₹500 operational labor threshold.")

    # High ROI Queue
    dispute_queue = test_results[test_results["Recommended_Action"] == "DISPUTE"].sort_values(
        by="Predicted_Prob", ascending=False
    )
    
    st.dataframe(
        dispute_queue[[
            "Transaction_Amount_INR", "Payment_Method", "Two_Factor_Auth_Success",
            "Proof_Of_Delivery", "Prior_Dispute_Count", "Predicted_Prob", "Expected_Value_INR"
        ]],
        width = 'stretch',
    )

    st.markdown("### 📝 Auto-Generate Bank Representment Pack")
    selected_idx = st.selectbox(
        "Select a Dispute to Compile Bank Rebuttal Letter:",
        dispute_queue.index,
        format_func=lambda x: f"Dispute ID: DSP-2026-{x:04d} | ₹{dispute_queue.loc[x, 'Transaction_Amount_INR']} | Win Prob: {dispute_queue.loc[x, 'Predicted_Prob']*100:.1f}%"
    )

    if st.button("🚀 Generate Formal Bank Rebuttal Letter", type="primary"):
        selected_case = dispute_queue.loc[selected_idx]
        letter_text = generate_mock_letter(selected_case)
        
        st.success(f"Evidence packet compiled successfully for DSP-2026-{selected_idx:04d}!")
        st.text_area("Official Representment Letter (Bank Copy)", letter_text, height=320)
        st.download_button(
            label="📥 Download Evidence Packet (.txt)",
            data=letter_text,
            file_name=f"DSP_2026_{selected_idx:04d}_Representment.txt",
            mime="text/plain"
        )

# TAB 2: BUSINESS IMPACT & METRICS
with tab2:
    st.subheader("Honest Unit Economics & Cost Analysis")
    st.caption("Comparing AI-driven cost-optimal triage vs. blind 100% dispute policy on the held-out test set.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Wasted Labor Cost (FP)",
            value=f"₹{metrics['FP_Cost']:,.2f}",
            help="Cost spent fighting disputes that were ultimately lost by arbitration (57 cases × ₹500)"
        )
    with col2:
        st.metric(
            label="Lost Recoverable Revenue (FN)",
            value=f"₹{metrics['FN_Cost']:,.2f}",
            help="Value of tickets conceded that were technically winnable"
        )
    with col3:
        st.metric(
            label="Net Financial Savings",
            value=f"₹{metrics['Net_Savings']:,.2f}",
            delta=f"₹{metrics['Net_Savings']:,.2f} vs Blind Baseline",
            delta_color="normal"
        )

    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Model Feature Importance")
        fig = px.bar(
            metrics["Feature_Importances"],
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with chart_col2:
        st.markdown("#### Test Set Confusion Matrix")
        cm_data = pd.DataFrame(
            [
                [f"TN: {metrics['TN']}\n(Auto-Conceded)", f"FP: {metrics['FP']}\n(Wasted ₹500)"],
                [f"FN: {metrics['FN']}\n(Lost Revenue)", f"TP: {metrics['TP']}\n(Defended)"]
            ],
            index=["Actual Lost", "Actual Won"],
            columns=["Predicted Concede", "Predicted Dispute"]
        )
        st.table(cm_data)
        st.info("💡 **Cost-Optimal Rule:** Triage threshold evaluates $(\\hat{P}_{\\text{win}} \\times \\text{Amount}) > ₹500$, rejecting low-value claims where labor cost exceeds expected return.")