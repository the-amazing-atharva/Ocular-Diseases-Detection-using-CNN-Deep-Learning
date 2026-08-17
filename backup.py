import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os  # Added for path handling

# Load the custom CNN model
# Ensure 'best_model.h5' is in the same directory or provide the full path
try:
    model = keras.models.load_model('best_model.h5')
    print("Custom CNN model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

print("Model input shape:", model.input_shape)

# Class names should match the order during training
Class_Names_Dict = {
    'glaucoma': 0,
    'normal': 1,
    'cataract': 2,
    'diabetic_retinopathy': 3
}
class_names = list(Class_Names_Dict.keys())

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Build Streamlit Interface
st.cache_data.clear()  # Clear cache to ensure fresh run
st.set_page_config(page_title="Eye Disease Classification",
                   page_icon='👁️', layout="wide")  # Added layout="wide"

st.title("Eye Diseases Classification 👁️")
st.markdown("**Welcome to the Eye Disease Classification App! This tool uses a deep learning model to help identify potential eye conditions from retinal images. Please note: This is for informational purposes only and not a substitute for professional medical advice.**")
st.image('diabetic-eye-issues-5-ways-diabetes-impacts-vision.jpg')
st.divider()

# Sidebar for disease descriptions (existing)
st.sidebar.markdown("## 🩺 Eye Disease Descriptions")
st.sidebar.markdown("""
### 👁️ Glaucoma  
Damage to the optic nerve, often caused by high intraocular pressure.  
It can lead to gradual, irreversible vision loss if untreated.

---

### 👁️ Diabetic Retinopathy  
Caused by diabetes damaging the retina’s blood vessels.  
May lead to blurred vision and blindness without early treatment.

---

### 👁️ Cataract  
Clouding of the eye's lens, usually due to aging.  
It causes blurry vision and glare, treatable with surgery.

---

### 👁️ Normal  
Healthy eye with no signs of disease or retinal abnormalities.  
Vision remains clear and unaffected.
""")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Prediction", "About the Project", "How it Works", "Visualizations", "Disclaimer"])  # Added 'Visualizations' tab

with tab1:
    st.header("Upload Retinal Images for Prediction")
    st.markdown(
        "Upload images of the left and right eye (JPG, JPEG, PNG) to get a classification.")

    col_upload_left, col_upload_right = st.columns(2)
    with col_upload_left:
        Left_Eye = st.file_uploader(
            "**Upload Left Eye Image**", type=["jpg", "jpeg", "png"])
    with col_upload_right:
        Right_Eye = st.file_uploader(
            "**Upload Right Eye Image**", type=["jpg", "jpeg", "png"])

    col_display_left, col_display_right = st.columns(2)
    with col_display_left:
        if Left_Eye is not None:
            st.image(Left_Eye, caption="Left Eye - Uploaded Image",
                     width=300)  # Fixed width for better control
    with col_display_right:
        if Right_Eye is not None:
            st.image(Right_Eye, caption="Right Eye - Uploaded Image",
                     width=300)  # Fixed width

    if Right_Eye is not None and Left_Eye is not None:
        def predict_image(image_file):
            image = Image.open(image_file).convert('RGB')
            image = image.resize((224, 224))
            image_np = np.array(image, dtype='float32')
            image_np = image_np / 255.0
            image_batch = np.expand_dims(image_np, axis=0)

            probs = model.predict(image_batch)[0]
            predicted_class_index = int(np.argmax(probs))
            return predicted_class_index, probs

        predicted_class_index_left, prop_left = predict_image(Left_Eye)
        predicted_class_index_right, prop_right = predict_image(Right_Eye)

        st.subheader("Prediction Results")
        col_results_left, col_results_right = st.columns(2)

        with col_results_left:
            st.markdown("### Left Eye Prediction")
            st.success(
                f"**Predicted Class: {class_names[predicted_class_index_left]}**")
            col_results_left.metric(
                f"Confidence", f"{prop_left[predicted_class_index_left]*100:.2f} %")  # Re-added metric
            st.write("---")
            st.markdown("##### All Class Confidences:")
            for i, prob in enumerate(prop_left):
                if i == predicted_class_index_left:
                    # Highlight predicted class
                    st.markdown(f"**{class_names[i]}: {prob:.2%}** :star2:")
                else:
                    st.write(f"{class_names[i]}: {prob:.2%}")

        with col_results_right:
            st.markdown("### Right Eye Prediction")
            st.success(
                f"**Predicted Class: {class_names[predicted_class_index_right]}**")
            col_results_right.metric(
                f"Confidence", f"{prop_right[predicted_class_index_right]*100:.2f} %")  # Re-added metric
            st.write("---")
            st.markdown("##### All Class Confidences:")
            for i, prob in enumerate(prop_right):
                if i == predicted_class_index_right:
                    st.markdown(f"**{class_names[i]}: {prob:.2%}** :star2:")
                else:
                    st.write(f"{class_names[i]}: {prob:.2%}")

with tab2:  # About the Project
    st.header("About the Eye Disease Classification Project")
    st.markdown("""
    This project aims to develop an automated system for classifying common eye diseases from retinal images.
    Early detection of eye conditions like glaucoma, diabetic retinopathy, and cataracts is crucial for effective treatment and preventing permanent vision loss.

    **Project Goals:**
    *   To build a robust deep learning model capable of accurately classifying retinal images into specific disease categories.
    *   To provide an accessible and easy-to-use web application for demonstrating the model's capabilities.
    *   To aid in preliminary screening and educational purposes, highlighting the potential of AI in healthcare.

    **Diseases Classified:**
    *   **Glaucoma**: A group of eye conditions that damage the optic nerve, often caused by high intraocular pressure.
    *   **Diabetic Retinopathy**: A complication of diabetes that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
    *   **Cataract**: A clouding of the normally clear lens of your eye. For people who have cataracts, seeing through cloudy lenses is a bit like looking through a frosty or fogged-up window.
    *   **Normal**: Images indicating a healthy retina without signs of the classified diseases.

    **Model Used:**
    The core of this application is a custom-built Convolutional Neural Network (CNN) trained on a diverse dataset of retinal images. The model has been optimized for accuracy and performance to provide reliable classifications.
    """)

with tab3:  # How it Works
    st.header("How the Eye Disease Classification App Works")
    st.markdown("""
    This application utilizes a custom-trained Convolutional Neural Network (CNN) to analyze uploaded retinal images. Here's a brief overview of the process:

    ### 1. Image Upload and Preprocessing
    *   **Input**: You upload images of the left and/or right eye.
    *   **Resizing**: Each image is resized to `224x224` pixels to match the input dimensions expected by the trained model.
    *   **Color Conversion**: Images are ensured to be in RGB format.
    *   **Normalization**: Pixel values, originally ranging from 0-255, are scaled down to 0-1. This normalization step is crucial for optimal model performance.

    ### 2. Deep Learning Model (Custom CNN)
    *   **Architecture**: The application uses a custom CNN architecture comprising multiple convolutional layers, pooling layers, and fully connected (dense) layers.
        *   **Convolutional Layers**: Extract features from the images (e.g., edges, textures, patterns specific to eye diseases).
        *   **Pooling Layers**: Reduce the spatial dimensions of the feature maps, helping to make the model more robust to variations in image position.
        *   **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
        *   **Dense Layers**: Learn complex patterns from the flattened features for classification.
        *   **Dropout**: A regularization technique used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
        *   **Softmax Activation**: The final layer uses a softmax activation function to output probabilities for each of the four disease classes.
    *   **Training**: The model was trained on a large dataset of labeled retinal images, learning to identify the visual characteristics associated with each eye condition.

    ### 3. Prediction
    *   Once an image is preprocessed, it's fed into the loaded CNN model.
    *   The model outputs a probability distribution across the four classes (Glaucoma, Normal, Cataract, Diabetic Retinopathy).
    *   The class with the highest probability is selected as the predicted disease.
    *   The application then displays the predicted class and the confidence score for each class.

    This entire process happens rapidly, providing you with an instant classification of the uploaded retinal images.
    """)

with tab4:  # Visualizations tab

    st.header("📊 Key Model Visualizations")

    st.markdown(
        """
        Explore the dataset distribution, training process, model evaluation,
        and classification performance through the visualizations generated
        during the project.
        """
    )

    st.divider()

    # -------------------------------------------------------------------------
    # Get the directory where this Streamlit Python file is located
    # -------------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Visualization folder
    VISUALIZATION_DIR = os.path.join(BASE_DIR, "Visualizations")

    # =========================================================================
    # 1. DATASET DISTRIBUTION
    # =========================================================================

    distribution_path = os.path.join(
        VISUALIZATION_DIR,
        "01_distribution.png"
    )

    st.subheader("1. 📈 Dataset Class Distribution")

    st.markdown(
        """
        This visualization shows the distribution of retinal images across
        the different classes in the training, validation, and testing sets.
        It helps us understand whether the dataset is balanced across the
        four eye disease categories.
        """
    )

    if os.path.isfile(distribution_path):

        st.image(
            distribution_path,
            caption="Distribution of Classes in Train, Validation, and Test Sets",
            use_container_width=True
        )

    else:

        st.error(
            f"❌ Image not found:\n\n`{distribution_path}`"
        )

    st.divider()

    # =========================================================================
    # 2. TRAIN / VALIDATION / TEST PIE CHART
    # =========================================================================

    train_test_path = os.path.join(
        VISUALIZATION_DIR,
        "02_train_test.png"
    )

    st.subheader("2. 🥧 Dataset Distribution")

    st.markdown(
        """
        These pie charts show how the dataset was divided into training,
        validation, and testing subsets.
        """
    )

    if os.path.isfile(train_test_path):

        st.image(
            train_test_path,
            caption="Training, Validation, and Testing Dataset Distribution",
            use_container_width=True
        )

    else:

        st.error(
            f"❌ Image not found:\n\n`{train_test_path}`"
        )

    st.divider()

    # =========================================================================
    # 3. TRAINING SAMPLE IMAGES
    # =========================================================================

    model_building_path = os.path.join(
        VISUALIZATION_DIR,
        "03_model_building.png"
    )

    st.subheader("3. 🖼️ Sample Training Images")

    st.markdown(
        """
        This visualization shows representative retinal images used during
        the model development and training process, including examples of
        augmented training images.
        """
    )

    if os.path.isfile(model_building_path):

        st.image(
            model_building_path,
            caption="Sample Images Used During Model Training",
            use_container_width=True
        )

    else:

        st.error(
            f"❌ Image not found:\n\n`{model_building_path}`"
        )

    st.divider()

    # =========================================================================
    # 4. EVALUATION CURVES
    # =========================================================================

    evaluation_path = os.path.join(
        VISUALIZATION_DIR,
        "04_evaluate.png"
    )

    st.subheader("4. 📉 Model Evaluation Curves")

    st.markdown(
        """
        These curves show the model's training and validation performance
        across epochs. Accuracy and loss curves can be used to understand
        how well the model learned and whether overfitting occurred.
        """
    )

    if os.path.isfile(evaluation_path):

        st.image(
            evaluation_path,
            caption="Training and Validation Accuracy/Loss over Epochs",
            use_container_width=True
        )

    else:

        st.error(
            f"❌ Image not found:\n\n`{evaluation_path}`"
        )

    st.divider()

    # =========================================================================
    # 5. CONFUSION MATRIX
    # =========================================================================

    confusion_matrix_path = os.path.join(
        VISUALIZATION_DIR,
        "05_confusion_matrix.png"
    )

    st.subheader("5. 🔲 Confusion Matrix")

    st.markdown(
        """
        The confusion matrix shows the classification performance of the
        model on the test dataset. It indicates which eye disease classes
        were correctly classified and which classes were confused with
        one another.
        """
    )

    if os.path.isfile(confusion_matrix_path):

        st.image(
            confusion_matrix_path,
            caption="Confusion Matrix for Test Set Predictions",
            use_container_width=True
        )

    else:

        st.error(
            f"❌ Image not found:\n\n`{confusion_matrix_path}`"
        )


with tab5:  # Disclaimer
    st.header("Important Medical Disclaimer")
    st.warning("""
    **This Eye Disease Classification application is developed for educational and demonstration purposes only. It is NOT intended to be a substitute for professional medical advice, diagnosis, or treatment.**

    *   **Do Not Use for Self-Diagnosis**: The predictions provided by this tool are based on a trained machine learning model and should not be used to diagnose any medical condition.
    *   **Consult a Healthcare Professional**: Always seek the advice of a qualified healthcare professional (e.g., an ophthalmologist or optometrist) for any medical questions or concerns you may have regarding your eye health.
    *   **Accuracy Limitations**: While deep learning models can achieve high accuracy, they are not infallible. Factors like image quality, rare conditions, or variations in disease presentation can affect the model's performance.
    *   **No Doctor-Patient Relationship**: The use of this application does not create a doctor-patient relationship between you and the developers or the model.

    **By using this application, you acknowledge and agree to this disclaimer.**
    """)
