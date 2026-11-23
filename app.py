
%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

st.set_page_config(
    page_title="ShopperAI | Purchase Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0B1020, #141B33);
    color: #F8FAFC;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
[data-testid="stSidebar"] {
    background: #10182D;
}
.hero {
    padding: 34px;
    border-radius: 22px;
    background: linear-gradient(110deg, #4931A8, #2563EB);
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 2.5rem;
    margin-bottom: 10px;
}
.hero p {
    color: #E0E7FF;
    font-size: 1.05rem;
}
.section-title {
    color: #C4B5FD;
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 10px;
}
.result-card {
    background: #1E293B;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #475569;
    margin-top: 15px;
}
.result-card h2 {
    color: #FFFFFF;
}
.footer {
    text-align: center;
    color: #94A3B8;
    padding: 25px;
    font-size: 13px;
}
div.stButton > button {
    background: linear-gradient(90deg, #7C3AED, #2563EB);
    color: white;
    border-radius: 12px;
    font-weight: 700;
    padding: 12px 28px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    preprocessor = joblib.load("preprocessor.pkl")
    classifier = XGBClassifier()
    classifier.load_model("xgboost_model.json")
    return preprocessor, classifier

preprocessor, classifier = load_model()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("🛍️ ShopperAI")
    st.caption("Machine Learning Prediction Dashboard")
    st.divider()

    st.subheader("Navigation")
    st.markdown("🏠 Dashboard")
    st.markdown("📊 Purchase Prediction")
    st.markdown("🤖 XGBoost Model")

    st.divider()

    st.subheader("About This Project")
    st.write(
        "This application predicts online shopper "
        "purchase intention using a trained XGBoost model."
    )

    st.info(
        "Enter customer browsing information and "
        "click Predict to generate a result."
    )

    st.divider()
    st.caption("Final Year Data Science Project")

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>🛒 Online Shopper Purchase Prediction</h1>
    <p>
        An intelligent machine learning dashboard that
        predicts customer purchase intention based on
        online browsing behaviour.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- OVERVIEW ----------
a, b, c = st.columns(3)

with a:
    st.metric("Machine Learning Model", "XGBoost")

with b:
    st.metric("Training Dataset", "12,330 Sessions")

with c:
    st.metric("Input Features", "17")

st.divider()

st.title("📊 Customer Behaviour Analysis")
st.write(
    "Complete the following information to predict "
    "whether a shopping session will result in a purchase."
)

# ---------- INPUT FORM ----------
with st.form("prediction_form"):

    st.markdown(
        '<div class="section-title">🧭 Browsing Activity</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        administrative = st.number_input(
            "Administrative Pages", min_value=0, value=2
        )
        administrative_duration = st.number_input(
            "Administrative Duration (seconds)",
            min_value=0.0, value=50.0
        )
        informational = st.number_input(
            "Informational Pages", min_value=0, value=0
        )
        informational_duration = st.number_input(
            "Informational Duration (seconds)",
            min_value=0.0, value=0.0
        )

    with col2:
        product_related = st.number_input(
            "Product-Related Pages", min_value=0, value=20
        )
        product_duration = st.number_input(
            "Product-Related Duration (seconds)",
            min_value=0.0, value=500.0
        )
        page_values = st.number_input(
            "Page Values", min_value=0.0, value=0.0
        )

    st.markdown(
        '<div class="section-title">📈 Engagement Metrics</div>',
        unsafe_allow_html=True
    )

    col3, col4 = st.columns(2)

    with col3:
        bounce_rates = st.number_input(
            "Bounce Rates",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            format="%.4f"
        )

    with col4:
        exit_rates = st.number_input(
            "Exit Rates",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            format="%.4f"
        )

    st.markdown(
        '<div class="section-title">📅 Visit Information</div>',
        unsafe_allow_html=True
    )

    col5, col6 = st.columns(2)

    with col5:
        month = st.selectbox(
            "Month",
            ["Feb", "Mar", "May", "June", "Jul",
             "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
        visitor_type = st.selectbox(
            "Visitor Type",
            ["Returning_Visitor", "New_Visitor", "Other"]
        )

    with col6:
        special_day = st.number_input(
            "Special Day",
            min_value=0.0,
            max_value=1.0,
            value=0.0
        )
        weekend = st.selectbox(
            "Weekend", [False, True]
        )

    st.markdown(
        '<div class="section-title">💻 Technical Information</div>',
        unsafe_allow_html=True
    )

    col7, col8 = st.columns(2)

    with col7:
        operating_system = st.number_input(
            "Operating System", min_value=1, value=2
        )
        browser = st.number_input(
            "Browser", min_value=1, value=2
        )

    with col8:
        region = st.number_input(
            "Region", min_value=1, value=1
        )
        traffic_type = st.number_input(
            "Traffic Type", min_value=1, value=2
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔮 Predict Purchase Intention",
        use_container_width=True
    )

# ---------- PREDICTION ----------
if submitted:

    input_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_system,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend
    }])

    try:
        processed = preprocessor.transform(input_data)

        prediction = int(classifier.predict(processed)[0])

        probability = float(
            classifier.predict_proba(processed)[0][1]
        )

        st.divider()
        st.title("🎯 Prediction Results")

        if prediction == 1:
            st.success("🛍️ Prediction: Purchase")
        else:
            st.warning("🛒 Prediction: No Purchase")

        result1, result2 = st.columns(2)

        with result1:
            st.metric(
                "Purchase Probability",
                f"{probability * 100:.2f}%"
            )

        with result2:
            st.metric(
                "No Purchase Probability",
                f"{(1 - probability) * 100:.2f}%"
            )

        st.progress(probability)

        st.caption(
            "This prediction is generated using "
            "the trained XGBoost machine learning model."
        )

        st.info(
            "Predictions are estimates and do not "
            "guarantee actual customer behaviour."
        )

    except Exception as error:
        st.error(f"Prediction error: {error}")

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    🛍️ ShopperAI — Online Shopper Purchase Prediction<br>
    Final Year Project | BSc (Hons) Data Science<br>
    University of Hertfordshire
</div>
""", unsafe_allow_html=True)
