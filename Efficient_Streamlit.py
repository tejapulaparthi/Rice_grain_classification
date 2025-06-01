import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Rice Classifier 🌾", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_efficientnet_with_conv_layers.keras')

model = load_model()
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Preprocess image
def preprocess_image(image):
    image = image.resize((100, 100))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Initialize session state
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

# Navbar
def navbar():
    st.markdown("""
        <style>
            .nav-container {
                display: flex;
                justify-content: space-evenly;
                background-color: #f1f8e9;
                padding: 0.7rem;
                border-radius: 10px;
                margin-bottom: 2rem;
            }
            .nav-button button {
                background-color: #c5e1a5;
                border: none;
                padding: 0.5rem 1.2rem;
                font-size: 1rem;
                font-weight: 600;
                color: #2e7d32;
                border-radius: 8px;
                cursor: pointer;
            }
            .nav-button button:hover {
                background-color: #aed581;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🏠 Home"):
            st.session_state.active_page = "Home"
    with col2:
        if st.button("🛠️ Services"):
            st.session_state.active_page = "Services"
    with col3:
        if st.button("🔍 Predict"):
            st.session_state.active_page = "Predict"
    with col4:
        if st.button("📬 Contact"):
            st.session_state.active_page = "Contact"

# Page sections
def show_home():
    st.markdown("<h2>Welcome to AI-Powered Rice Grain Classification</h2>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1614691811874-09c7c1a13885", use_column_width=True)
    st.write("This system uses machine learning to classify and grade rice grains with high accuracy.")

def show_services():
    st.header("🛠️ Services Offered")
    st.markdown("""
        - 📷 **Rice Grain Classification**  
        - 📊 **Grain Quality Grading** *(Good / Average / Poor)*  
        - 🧠 Powered by **EfficientNet + Transfer Learning**
        - ⏱️ Fast and Accurate Predictions
    """)

def show_predict():
    st.header("🔍 Upload & Predict Rice Grain Type")
    uploaded_file = st.file_uploader("Choose a rice grain image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)

        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                processed = preprocess_image(image)
                prediction = model.predict(processed)
                pred_class = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

            st.success(f"✅ Predicted: **{pred_class}** with {confidence:.2f}% confidence")
            st.progress(int(confidence))

def show_contact():
    st.header("📬 Contact Us")
    st.markdown("""
        - 📧 Email: riceai@agrotech.com  
        - 🌐 Website: [www.riceai-app.com](http://www.riceai-app.com)  
        - 📍 Location: Andhra Pradesh, India
    """)
    st.info("For queries, collaborations or demo access, feel free to reach out!")

# Render
navbar()

if st.session_state.active_page == "Home":
    show_home()
elif st.session_state.active_page == "Services":
    show_services()
elif st.session_state.active_page == "Predict":
    show_predict()
elif st.session_state.active_page == "Contact":
    show_contact()
