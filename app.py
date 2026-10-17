import streamlit as st
import pandas as pd

from model_demo import predict_case, train_kyphosis_model

st.set_page_config(page_title="Kyphosis Classification Study", page_icon="🩺")
st.title("Kyphosis Classification Study")
st.caption("A Random Forest workflow for an appropriately authorized CSV dataset.")
st.error(
    "Educational software only. This model is not clinically validated, is not "
    "a diagnostic tool, and must not be used to make patient-care decisions."
)
st.info(
    "Upload your own authorized CSV with a categorical `Kyphosis` target and "
    "numeric predictor fields such as `Age`, `Number`, and `Start`. No patient "
    "dataset is bundled or sent to a remote service."
)
uploaded = st.file_uploader("Upload authorized CSV", type=["csv"])
if uploaded is None:
    st.stop()
if uploaded.size > 5 * 1024 * 1024:
    st.error("CSV exceeds the 5 MB upload limit.")
    st.stop()
try:
    frame = pd.read_csv(uploaded)
except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
    st.error(f"Could not read CSV: {error}")
    st.stop()
if "Kyphosis" not in frame.columns:
    st.error("CSV must contain a `Kyphosis` target column.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(frame.head(100), use_container_width=True)
st.caption(f"{len(frame):,} rows · {len(frame.columns)} columns")
try:
    result = train_kyphosis_model(frame)
except ValueError as error:
    st.warning(str(error))
    st.stop()

left, right = st.columns(2)
left.metric("Hold-out accuracy", f"{result['accuracy']:.1%}")
right.metric("Test rows", result["test_count"])
st.subheader("Confusion matrix")
st.caption("Rows: actual labels · columns: predicted labels · " + ", ".join(result["labels"]))
st.dataframe(result["confusion_matrix"], use_container_width=True)
st.subheader("Feature importance")
st.bar_chart(
    pd.Series(result["importance"], index=result["feature_columns"]).sort_values()
)

st.subheader("Explore one example row")
inputs = {}
for column in result["feature_columns"]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        inputs[column] = 0.0
    else:
        minimum, maximum = float(values.min()), float(values.max())
        default = float(values.median())
        inputs[column] = st.number_input(
            column,
            min_value=minimum if minimum < default else default - 1.0,
            max_value=maximum if maximum > default else default + 1.0,
            value=default,
        )
if st.button("Run educational example"):
    prediction, probabilities = predict_case(
        result["model"], result["feature_columns"], inputs
    )
    st.write(f"Model class output: **{prediction}** (not a medical conclusion)")
    st.json(probabilities)
