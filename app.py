
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Online Shopper Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Online Shopper Purchase Prediction")

st.write(
    "Enter customer browsing information to predict "
    "whether a shopping session will result in a purchase."
)

from xgboost import XGBClassifier

@st.cache_resource
def load_model():
    preprocessor = joblib.load("preprocessor.pkl")

    classifier = XGBClassifier()
    classifier.load_model("xgboost_model.json")

    return preprocessor, classifier

preprocessor, classifier = load_model()

st.subheader("Customer Browsing Information")

administrative = st.number_input(
    "Administrative Pages",
    min_value=0,
    value=0
)

administrative_duration = st.number_input(
    "Administrative Duration (seconds)",
    min_value=0.0,
    value=0.0
)

informational = st.number_input(
    "Informational Pages",
    min_value=0,
    value=0
)

informational_duration = st.number_input(
    "Informational Duration (seconds)",
    min_value=0.0,
    value=0.0
)

product_related = st.number_input(
    "Product-Related Pages",
    min_value=0,
    value=5
)

product_related_duration = st.number_input(
    "Product-Related Duration (seconds)",
    min_value=0.0,
    value=300.0
)

bounce_rates = st.number_input(
    "Bounce Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.02
)

exit_rates = st.number_input(
    "Exit Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.05
)

page_values = st.number_input(
    "Page Values",
    min_value=0.0,
    value=0.0
)

special_day = st.slider(
    "Special Day",
    min_value=0.0,
    max_value=1.0,
    value=0.0
)

month = st.selectbox(
    "Month",
    ["Feb", "Mar", "May", "June", "Jul",
     "Aug", "Sep", "Oct", "Nov", "Dec"]
)

operating_systems = st.number_input(
    "Operating System Code",
    min_value=1,
    value=2
)

browser = st.number_input(
    "Browser Code",
    min_value=1,
    value=2
)

region = st.number_input(
    "Region Code",
    min_value=1,
    value=1
)

traffic_type = st.number_input(
    "Traffic Type Code",
    min_value=1,
    value=2
)

visitor_type = st.selectbox(
    "Visitor Type",
    ["Returning_Visitor", "New_Visitor", "Other"]
)

weekend = st.checkbox("Weekend")

if st.button("Predict Purchase Intention"):

    input_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_systems,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend
    }])

    processed_input = preprocessor.transform(input_data)

prediction = classifier.predict(processed_input)[0]

probability = classifier.predict_proba(
    processed_input
)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Purchase")
    else:
        st.warning("Prediction: No Purchase")

    st.metric(
        "Estimated Purchase Probability",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This is a machine learning estimate, "
        "not a guaranteed purchasing outcome."
    )
