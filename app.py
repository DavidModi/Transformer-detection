import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
import pickle
import base64

# Clear cache to avoid potential corrupted data
st.cache_data.clear()

# Function to load image and convert it to base64
def load_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Paths to your images
background_image_path = "C:\\Users\\Owner\\Downloads\\1000_F_433027896_PoQOnWcXtmgmuvIfA3ye5zrSzDyxDzHS.jpg"

# Encode images to base64
background_image_base64 = load_image(background_image_path)

# Load your model
@st.cache_data(show_spinner=False)
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model from {model_path}. Error: {e}")
        return None

# Load the predicted labels
@st.cache_data(show_spinner=False)
def load_labels(label_path):
    try:
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)
        return labels
    except Exception as e:
        st.error(f"Failed to load labels from {label_path}. Error: {e}")
        return None

# Custom CSS to add background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url(data:image/jpg;base64,{background_image_base64});
        background-size: cover;
    }}
    .sidebar .sidebar-content {{
        background: url(data:image/jpg;base64,{background_image_base64});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Create a Streamlit app
st.title("Transformer Fault Detection and Analysis")

# Add a login page
st.sidebar.title("Login")
username = st.sidebar.text_input("Username:")
password = st.sidebar.text_input("Password:", type="password")
if st.sidebar.button("Login"):
    if username == "user" and password == "pass":
        st.sidebar.success("Logged in successfully!")
    else:
        st.sidebar.error("Invalid username or password")
        st.stop()

# Add a dropdown menu to select the model
st.sidebar.title("Model Selection")
model_name = st.sidebar.selectbox("Select model:", ["Model 1", "Model 2", "Model 3"])

# Map model names to paths
model_path = "C:\\Users\\Owner\\Desktop\\Final Year Project\\Notebooks\\my_model.keras"
label_path = "C:\\Users\\Owner\\Desktop\\Final Year Project\\Notebooks\\predicted_labels.pkl"

# Load the selected model and labels
if os.path.exists(model_path):
    model = load_model(model_path)
    if model is None:
        st.stop()
else:
    st.error(f"Model {model_name} not found at {model_path}")
    st.stop()

if os.path.exists(label_path):
    labels = load_labels(label_path)
    if labels is None:
        st.stop()
else:
    st.error(f"Labels not found at {label_path}")
    st.stop()

# Add file uploaders for scalogram images
st.header("Upload Scalogram Images")
scalogram_files = [st.file_uploader(f"Upload scalogram {i+1}:", type=["png", "jpg", "jpeg"]) for i in range(6)]

# Add a button to run the analysis
if st.button("Run Analysis"):
    scalogram_data = []
    for scalogram_file in scalogram_files:
        if scalogram_file is not None:
            try:
                scalogram = np.frombuffer(scalogram_file.read(), np.uint8)
                scalogram = cv2.imdecode(scalogram, cv2.IMREAD_COLOR)
                scalogram_data.append(scalogram)
                # Display each uploaded scalogram
                st.image(scalogram, caption=f"Scalogram {scalogram_files.index(scalogram_file) + 1}", use_column_width=True)
            except Exception as e:
                st.error(f"Error reading scalogram file: {e}")
                st.stop()
        else:
            st.error("Please upload all scalogram files")
            st.stop()

    # Ensure all scalograms have 3 channels (convert RGBA to RGB)
    def ensure_three_channels(image):
        if image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    scalogram_data = [ensure_three_channels(img) for img in scalogram_data]

    # Combine the scalograms
    combined_image = np.concatenate(scalogram_data, axis=1)

    # Resize to 128x128 for the model's input size
    combined_image = cv2.resize(combined_image, (128, 128))

    # Ensure the combined image has the right shape for the model
    st.write(f"Combined image shape before reshaping: {combined_image.shape}")
    try:
        combined_image = combined_image.reshape(1, 128, 128, 3)  # Reshape for the model
    except ValueError as e:
        st.error(f"Reshape error: {e}. Combined image shape: {combined_image.shape}")
        st.stop()

    # Run the analysis
    prediction = model.predict(combined_image)
    predicted_label = labels[np.argmax(prediction)]

    # Display the results
    st.write(f"Transformer status: {predicted_label}")
    st.image(combined_image.reshape(128, 128, 3), caption="Combined Scalogram", use_column_width=True)
else:
    st.error("No valid scalograms to process")

# Add documentation links
st.sidebar.title("Help")
st.sidebar.info("""
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [TensorFlow Documentation](https://www.tensorflow.org/guide)
    - [OpenCV Documentation](https://docs.opencv.org/)
""")
