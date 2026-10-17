import streamlit as st

from model_demo import evaluate_demo

st.set_page_config(page_title="Classifier Learning Demo", page_icon="🧪")
st.title("Classifier Learning Demo")
st.caption("A reproducible machine-learning exercise on generated synthetic data.")
test_size = st.slider("Synthetic test-set fraction", 0.1, 0.5, 0.25, 0.05)
seed = st.number_input("Random seed", min_value=0, max_value=10000, value=7)
metrics = evaluate_demo(test_size, int(seed))

left, right = st.columns(2)
left.metric("Synthetic test accuracy", f"{metrics['accuracy']:.1%}")
right.metric("Test examples", metrics["test_size"])
st.subheader("Synthetic confusion matrix")
st.dataframe(metrics["confusion_matrix"])
st.warning(
    "No clinical data or patient-level inputs are used. The generated labels and "
    "model scores are arbitrary demonstrations, not kyphosis predictions or medical advice."
)
