import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="EcoVision | Smart Waste Classifier", page_icon="♻️", layout="centered")

MODEL_PATH = os.path.join("models", "ecovision.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.txt")
IMG_SIZE = (224, 224)

st.title("♻️ EcoVision")
st.subheader("AI-powered waste classification")
st.write("Upload a photo of a single waste item. EcoVision predicts its waste category and gives a simple disposal suggestion.")

GUIDANCE = {
    "cardboard": ("Recycle", "Keep cardboard clean and dry. Flatten boxes before placing them in paper/cardboard recycling."),
    "glass": ("Recycle carefully", "Separate glass where local facilities accept it. Do not mix broken glass loosely with other recyclables."),
    "metal": ("Recycle", "Empty and rinse cans or containers. Check local rules for foil and other metal items."),
    "paper": ("Recycle", "Keep paper clean and dry. Greasy or food-soiled paper may not be accepted by local recycling programs."),
    "plastic": ("Check local recycling rules", "Empty and rinse the item. Plastic recycling rules vary by resin type and location."),
    "trash": ("General waste", "This model labels the item as general waste. Check local guidance before disposal; some items may have specialist collection."),
}

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    return model, names

model, class_names = load_model()

with st.expander("About this project"):
    st.write(
        "EcoVision is a student machine-learning project that uses transfer learning "
        "(MobileNetV2) to classify common waste categories. It is an educational prototype, "
        "not a certified waste-management or hazardous-material identification system."
    )

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if model is None:
    st.warning("The trained model is not included yet. Train it using the instructions in README.md, then place the resulting file at models/ecovision.keras.")
    st.code("python train.py --data_dir data/waste --epochs 10")
elif uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        resized = image.resize(IMG_SIZE)
        arr = tf.keras.utils.img_to_array(resized)
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        probabilities = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        best_idx = int(np.argmax(probabilities))
        label = class_names[best_idx].lower()
        confidence = float(probabilities[best_idx])

        st.metric("Predicted category", label.title(), f"{confidence:.1%} confidence")
        action, advice = GUIDANCE.get(label, ("Review manually", "Please verify this prediction before disposal."))
        st.info(f"**Suggested action: {action}.** {advice}")
        st.caption("Model predictions can be wrong, especially for mixed, dirty, unusual, or partially visible objects.")
        st.write("### Model confidence")
        st.bar_chart({name.title(): float(prob) for name, prob in zip(class_names, probabilities)})
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.info("Upload an image to get a prediction.")

st.divider()
st.caption("EcoVision • Built with Python, TensorFlow, MobileNetV2 and Streamlit")
