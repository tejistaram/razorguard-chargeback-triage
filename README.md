# 🛡️ RazorGuard | AI Chargeback Triage Engine

**Built for Razorpay Buildathon | Track 02 — AI Risk Manager**

RazorGuard is a cost-optimal chargeback triage and automated representment engine for Indian BFSI payment rails. It protects merchant margins from "friendly fraud" by calculating the exact operational ROI of disputing a chargeback, and automatically drafts RBI/NPCI-compliant rebuttal letters for high-yield cases.

## 🚨 The Problem

Merchants lose substantial margins to chargebacks because disputing claims manually costs ~₹500 in operational labor per case. If a merchant fights a ₹350 dispute and wins, they still lose ₹150. Most ML fraud models evaluate pure "accuracy" or use a naive 0.5 probability threshold, completely ignoring unit economics. 

## 💡 The Solution

RazorGuard replaces standard binary classification with a **Dynamic Cost-Optimal Threshold**. 

Instead of fighting every dispute, RazorGuard recommends fighting a case **IF AND ONLY IF**:

`Expected Value = (Predicted Win Probability × Transaction Amount) > ₹500 Labor Cost`

### Core Features

* **Indian BFSI Causal Logic:** The data engine is modeled strictly on domestic realities, incorporating UPI technical failure logs, RBI 2FA/AFA liability mandates, and serial-disputer bank policies.

* **Cost-Sensitive ML:** Uses a Random Forest classifier to predict win probability, optimizing strictly for False-Positive labor waste vs. False-Negative lost revenue.

* **LLM Auto-Responder Pipeline:** Generates structured JSON payloads for high-ROI cases, providing a strict extraction template for LLMs to generate formal Bank Representment Letters without hallucinating ARNs or tracking numbers.

## 📊 Honest Business Metrics (Test Set)

Compared to a baseline policy of disputing all chargebacks blindly, RazorGuard's cost-optimal threshold delivered the following on a 200-case test set:

* **False Positive Labor Waste:** ₹28,500

* **False Negative Lost Revenue:** ₹23,159

* **Net Financial Savings:** ₹6,340.93

## 🚀 How to Run Locally

1. **Clone the repository:**

   ```bash

   git clone [[https://github.com/tejistaram/razorguard-chargeback-triage.git](https://github.com/tejistaram/razorguard-chargeback-triage.git)](https://github.com/tejistaram/razorguard-chargeback-triage.git](https://github.com/tejistaram/razorguard-chargeback-triage.git))

   cd razorguard-chargeback-triage