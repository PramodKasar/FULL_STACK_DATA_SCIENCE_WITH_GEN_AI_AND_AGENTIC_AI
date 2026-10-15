"""
Customer Churn Prediction – Streamlit UI
Run with:  streamlit run app.py
Requires:  final_gb_classifier.pkl in the same folder
"""

import joblib
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# STYLING (works in both light and dark themes)
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}

.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    padding: 2rem 2.2rem; border-radius: 20px; color: #fff; margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(79,70,229,.25);
}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; color: #fff;}
.hero p  {margin: .4rem 0 0; opacity: .92; font-size: 1.02rem; color: #fff;}

.card {
    background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.22);
    border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.card h4 {margin: 0 0 .6rem; font-size: 1.05rem;}

.kpi-row {display: flex; gap: .8rem; flex-wrap: wrap; margin-bottom: 1rem;}
.kpi {
    flex: 1; min-width: 130px; background: rgba(128,128,128,.07);
    border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: .8rem 1rem;
}
.kpi .l {font-size: .75rem; opacity: .65; text-transform: uppercase; letter-spacing: .05em;}
.kpi .v {font-size: 1.25rem; font-weight: 700; margin-top: .15rem;}

.badge {
    display: inline-block; padding: .35rem .9rem; border-radius: 999px;
    font-weight: 700; font-size: .9rem; color: #fff;
}
.chip {
    display: inline-block; padding: .25rem .7rem; margin: .15rem .25rem .15rem 0;
    border-radius: 999px; font-size: .82rem; font-weight: 600;
}
.chip.bad  {background: rgba(239,68,68,.15);  color: #ef4444; border: 1px solid rgba(239,68,68,.4);}
.chip.good {background: rgba(16,185,129,.15); color: #10b981; border: 1px solid rgba(16,185,129,.4);}

.stTabs [data-baseweb="tab-list"] {gap: .5rem;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: .6rem 1.1rem; font-weight: 600;}
div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #4f46e5, #7c3aed); border: none; color: #fff;
    font-weight: 700; padding: .7rem 1rem; border-radius: 12px; font-size: 1.05rem;
}
div.stButton > button[kind="primary"]:hover {filter: brightness(1.1);}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load("final_gb_classifier.pkl")


try:
    model = load_model()
except Exception as e:  # noqa: BLE001
    st.error(
        "Could not load **final_gb_classifier.pkl**. Make sure it is in the same "
        f"folder as app.py.\n\n`{e}`"
    )
    st.stop()

# ----------------------------------------------------------------------------
# ENCODINGS – identical to sklearn LabelEncoder (alphabetical) used in training
# ----------------------------------------------------------------------------
GENDER = {"Female": 0, "Male": 1}
YES_NO = {"No": 0, "Yes": 1}
MULTI_LINES = {"No": 0, "No phone service": 1, "Yes": 2}
INTERNET = {"DSL": 0, "Fiber optic": 1, "No": 2}
ADDON = {"No": 0, "No internet service": 1, "Yes": 2}
CONTRACT = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}
ADDON_FIELDS = {
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
}

# Exact column order used when the model was trained
FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges", "tenure_group",
]


def tenure_to_group(months: int) -> int:
    """Same bins as training: 1-12, 13-24, 25-36, 37-48, 49-60, 61-72 -> 0..5"""
    return min(max(months - 1, 0) // 12, 5)


def tenure_group_label(code: int) -> str:
    return ["1–12", "13–24", "25–36", "37–48", "49–60", "61–72"][code] + " months"


# ----------------------------------------------------------------------------
# SESSION DEFAULTS + PRESETS
# ----------------------------------------------------------------------------
DEFAULTS = {
    "gender": "Female", "senior": "No", "partner": "No", "dependents": "No",
    "tenure": 12, "phone": "Yes", "multi": "No", "internet": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "contract": "Month-to-month", "paperless": "Yes",
    "payment": "Electronic check", "monthly": 70.0,
    "manual_total": False, "total": 0.0,
}

PRESETS = {
    "🔴 High-risk customer": {
        **DEFAULTS, "senior": "Yes", "tenure": 3, "monthly": 95.0,
        "internet": "Fiber optic", "contract": "Month-to-month",
        "payment": "Electronic check", "paperless": "Yes",
    },
    "🟢 Loyal customer": {
        **DEFAULTS, "partner": "Yes", "dependents": "Yes", "tenure": 60,
        "monthly": 65.0, "internet": "DSL", "contract": "Two year",
        "payment": "Credit card (automatic)", "paperless": "No", "multi": "Yes",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes",
    },
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def reset_form():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Quick start")
    st.caption("Load an example customer, then tweak any field.")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    st.button("↺ Reset form", on_click=reset_form, use_container_width=True)

    st.divider()
    st.markdown("### 🧠 About the model")
    st.markdown(
        "**Algorithm:** Gradient Boosting Classifier  \n"
        "**Trees:** 100 · **Depth:** 7  \n"
        "**Learning rate:** 0.1  \n"
        "**Data:** Telco Customer Churn"
    )
    st.caption(
        "Predictions are probabilistic estimates to support retention decisions — "
        "not guarantees."
    )

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>📡 Customer Churn Predictor</h1>
  <p>Enter a customer's profile, services and billing details to estimate the
  likelihood they will leave — and get tailored retention ideas.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# INPUT FORM (tabs)
# ----------------------------------------------------------------------------
tab_profile, tab_services, tab_billing = st.tabs(
    ["👤 Customer Profile", "🛠️ Services", "💳 Contract & Billing"]
)

with tab_profile:
    c1, c2 = st.columns(2)
    with c1:
        st.radio("Gender", list(GENDER), key="gender", horizontal=True)
        st.radio("Senior citizen", list(YES_NO), key="senior", horizontal=True)
    with c2:
        st.radio("Has a partner", list(YES_NO), key="partner", horizontal=True)
        st.radio("Has dependents", list(YES_NO), key="dependents", horizontal=True)
    st.slider(
        "Tenure (months with the company)", 1, 72, key="tenure",
        help="Converted internally to the tenure group the model was trained on.",
    )
    st.caption(f"Tenure group → **{tenure_group_label(tenure_to_group(st.session_state.tenure))}**")

with tab_services:
    c1, c2 = st.columns(2)
    with c1:
        st.radio("Phone service", list(YES_NO), key="phone", horizontal=True)
        if st.session_state.phone == "Yes":
            st.radio("Multiple lines", ["No", "Yes"], key="multi", horizontal=True)
        else:
            st.info("No phone service → *Multiple lines* is set automatically.")
    with c2:
        st.selectbox("Internet service", list(INTERNET), key="internet")

    if st.session_state.internet == "No":
        st.info("No internet service → all internet add-ons are set automatically.")
    else:
        st.markdown("**Internet add-ons**")
        cols = st.columns(3)
        for i, (key, label) in enumerate(ADDON_FIELDS.items()):
            with cols[i % 3]:
                st.radio(label, ["No", "Yes"], key=key, horizontal=True)

with tab_billing:
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("Contract type", list(CONTRACT), key="contract")
        st.selectbox("Payment method", list(PAYMENT), key="payment")
        st.radio("Paperless billing", list(YES_NO), key="paperless", horizontal=True)
    with c2:
        st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=500.0, step=1.0,
            key="monthly",
        )
        st.checkbox("Enter total charges manually", key="manual_total")
        if st.session_state.manual_total:
            st.number_input(
                "Total charges ($)", min_value=0.0, step=10.0, key="total"
            )
            total_charges = st.session_state.total
        else:
            total_charges = st.session_state.monthly * st.session_state.tenure
            st.caption(
                f"Estimated total charges: **${total_charges:,.2f}** "
                "(monthly × tenure)"
            )

st.write("")
predict_clicked = st.button(
    "🔮 Predict churn risk", type="primary", use_container_width=True
)


# ----------------------------------------------------------------------------
# BUILD MODEL INPUT
# ----------------------------------------------------------------------------
def build_features() -> pd.DataFrame:
    s = st.session_state
    phone_yes = s.phone == "Yes"
    no_internet = s.internet == "No"

    row = {
        "gender": GENDER[s.gender],
        "SeniorCitizen": YES_NO[s.senior],
        "Partner": YES_NO[s.partner],
        "Dependents": YES_NO[s.dependents],
        "PhoneService": YES_NO[s.phone],
        "MultipleLines": MULTI_LINES["Yes" if (phone_yes and s.multi == "Yes")
                                     else ("No" if phone_yes else "No phone service")],
        "InternetService": INTERNET[s.internet],
        "Contract": CONTRACT[s.contract],
        "PaperlessBilling": YES_NO[s.paperless],
        "PaymentMethod": PAYMENT[s.payment],
        "MonthlyCharges": float(s.monthly),
        "TotalCharges": float(total_charges),
        "tenure_group": tenure_to_group(s.tenure),
    }
    for key in ADDON_FIELDS:
        row[key] = ADDON["No internet service"] if no_internet else ADDON[s[key]]
    return pd.DataFrame([row])[FEATURE_ORDER]


def risk_factors():
    """Simple, explainable business rules based on common churn patterns."""
    s = st.session_state
    bad, good = [], []
    if s.contract == "Month-to-month":
        bad.append("Month-to-month contract")
    else:
        good.append(f"{s.contract} contract")
    if s.tenure <= 12:
        bad.append("New customer (≤ 12 months)")
    elif s.tenure >= 48:
        good.append("Long-term customer (4+ years)")
    if s.internet == "Fiber optic":
        bad.append("Fiber optic plan")
    if s.payment == "Electronic check":
        bad.append("Pays by electronic check")
    elif "automatic" in s.payment:
        good.append("Automatic payments")
    if s.internet != "No":
        if s.TechSupport == "No":
            bad.append("No tech support")
        else:
            good.append("Has tech support")
        if s.OnlineSecurity == "No":
            bad.append("No online security")
        else:
            good.append("Has online security")
    if s.senior == "Yes":
        bad.append("Senior citizen")
    if s.monthly >= 85:
        bad.append("High monthly bill")
    if s.partner == "Yes" or s.dependents == "Yes":
        good.append("Household ties (partner/dependents)")
    return bad, good


def recommendations(bad):
    tips = []
    if "Month-to-month contract" in bad:
        tips.append("Offer a discount to move to a 1- or 2-year contract.")
    if "Pays by electronic check" in bad:
        tips.append("Incentivise automatic payment (bank transfer / credit card).")
    if "No tech support" in bad or "No online security" in bad:
        tips.append("Bundle free tech support / online security for a trial period.")
    if "High monthly bill" in bad:
        tips.append("Review the plan — offer a loyalty price or a right-sized package.")
    if "New customer (≤ 12 months)" in bad:
        tips.append("Trigger an onboarding check-in call within the first 90 days.")
    if "Fiber optic plan" in bad:
        tips.append("Proactively check service quality for this customer.")
    if not tips:
        tips.append("Maintain engagement with periodic loyalty rewards.")
    return tips


def gauge_svg(pct: float, color: str) -> str:
    arc = 251.3  # length of a radius-80 half circle
    filled = arc * pct / 100
    return f"""
<svg viewBox="0 0 200 120" width="100%" style="max-width:360px;display:block;margin:auto">
  <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="rgba(128,128,128,.25)"
        stroke-width="16" stroke-linecap="round"/>
  <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="{color}"
        stroke-width="16" stroke-linecap="round"
        stroke-dasharray="{filled:.1f} {arc}"/>
  <text x="100" y="92" text-anchor="middle" font-size="30" font-weight="800"
        fill="{color}">{pct:.0f}%</text>
  <text x="100" y="112" text-anchor="middle" font-size="9" fill="gray">churn probability</text>
</svg>"""


# ----------------------------------------------------------------------------
# RESULT
# ----------------------------------------------------------------------------
if predict_clicked:
    features = build_features()
    with st.spinner("Analysing customer…"):
        prob = float(model.predict_proba(features)[0][1])
        will_churn = prob >= 0.5

    pct = prob * 100
    if pct < 30:
        level, color = "LOW RISK", "#10b981"
    elif pct < 60:
        level, color = "MEDIUM RISK", "#f59e0b"
    else:
        level, color = "HIGH RISK", "#ef4444"

    st.divider()
    st.subheader("📊 Prediction result")

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown(
            f"""
<div class="card" style="text-align:center">
  {gauge_svg(pct, color)}
  <div style="margin-top:.6rem"><span class="badge" style="background:{color}">{level}</span></div>
  <p style="margin:.9rem 0 0;font-size:1.05rem">
    {"⚠️ This customer is <b>likely to churn</b>." if will_churn
     else "✅ This customer is <b>likely to stay</b>."}
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        s = st.session_state
        st.markdown(
            f"""
<div class="kpi-row">
  <div class="kpi"><div class="l">Tenure</div><div class="v">{s.tenure} mo</div></div>
  <div class="kpi"><div class="l">Monthly</div><div class="v">${s.monthly:,.0f}</div></div>
  <div class="kpi"><div class="l">Contract</div><div class="v">{s.contract}</div></div>
  <div class="kpi"><div class="l">Internet</div><div class="v">{s.internet}</div></div>
</div>
""",
            unsafe_allow_html=True,
        )

        bad, good = risk_factors()
        chips = "".join(f'<span class="chip bad">⚠ {b}</span>' for b in bad) or "<i>None detected</i>"
        chips_g = "".join(f'<span class="chip good">✓ {g}</span>' for g in good) or "<i>None detected</i>"
        st.markdown(
            f"""
<div class="card"><h4>🚩 Risk indicators</h4>{chips}</div>
<div class="card"><h4>💚 Protective factors</h4>{chips_g}</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 💡 Suggested retention actions")
    for tip in recommendations(bad):
        st.markdown(f"- {tip}")
    st.caption(
        "Risk indicators and actions are rule-of-thumb insights from common churn "
        "patterns; the probability itself comes from the trained model."
    )

    with st.expander("🔍 View model input (encoded features)"):
        st.dataframe(features, use_container_width=True, hide_index=True)

    report = features.copy()
    report.insert(0, "churn_probability_pct", round(pct, 2))
    report.insert(1, "prediction", "Churn" if will_churn else "Stay")
    st.download_button(
        "⬇️ Download result (CSV)",
        report.to_csv(index=False).encode("utf-8"),
        file_name="churn_prediction.csv",
        mime="text/csv",
    )
else:
    st.info("Fill in the tabs above and click **Predict churn risk** to see the result.")