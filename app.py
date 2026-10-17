
import json
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Brain Tumor AI",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection AI")
st.write("Upload an MRI image to see the model's prediction.")

st.warning(
    "Educational project only. This AI is not a medical "
    "diagnostic tool. Consult a qualified medical professional."
)

@st.cache_resource
def load_ai_model():
    return load_model("brain_tumor_model.keras")

@st.cache_data
def load_labels():
    with open("class_names.json", "r") as file:
        return json.load(file)

try:
    model = load_ai_model()
    class_names = load_labels()
except Exception as error:
    st.error(f"Model could not be loaded: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", width=350)

    if st.button("Analyze Image", type="primary"):
        with st.spinner("Analyzing image..."):
            resized_image = image.resize((224, 224))

            image_array = np.array(
                resized_image, dtype=np.float32
            )
            image_array = np.expand_dims(image_array, axis=0)

            # Model ke andar Rescaling layer pehle se hai.
            predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.subheader("Model Prediction")
        st.write(f"**Predicted class:** {predicted_class}")
        st.metric("Model confidence score", f"{confidence:.2f}%")

        st.subheader("Scores for Each Class")
        st.bar_chart({
            "Score (%)": {
                name: float(score) * 100
                for name, score in zip(class_names, predictions)
            }
        })

        st.caption(
            "The score is the model's confidence, not the "
            "probability that a medical diagnosis is correct."
        )
