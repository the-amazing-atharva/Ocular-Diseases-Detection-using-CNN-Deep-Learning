import os
import io
import base64

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁️",
    layout="wide"
)


# =============================================================================
# BASE DIRECTORY
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# FILE PATHS
# =============================================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_model.h5"
)

HERO_IMAGE_PATH = os.path.join(
    BASE_DIR,
    "diabetic-eye-issues-5-ways-diabetes-impacts-vision.jpg"
)

VISUALIZATION_DIR = os.path.join(
    BASE_DIR,
    "Visualizations"
)


# =============================================================================
# CLASS NAMES
# =============================================================================

Class_Names_Dict = {
    "glaucoma": 0,
    "normal": 1,
    "cataract": 2,
    "diabetic_retinopathy": 3
}

class_names = list(Class_Names_Dict.keys())


# =============================================================================
# CLASS DISPLAY NAMES
# =============================================================================

DISPLAY_NAMES = {
    "glaucoma": "Glaucoma",
    "normal": "Normal",
    "cataract": "Cataract",
    "diabetic_retinopathy": "Diabetic Retinopathy"
}


# =============================================================================
# DISEASE INFORMATION
# =============================================================================

DISEASE_INFO = {

    "glaucoma": {
        "title": "👁️ Glaucoma",
        "description": (
            "Glaucoma is a group of eye conditions that can damage the optic "
            "nerve, which is important for vision. It is often associated "
            "with increased intraocular pressure."
        ),
        "symptoms": [
            "Gradual loss of peripheral vision",
            "Blurred vision",
            "Eye pain in some forms",
            "Halos around lights",
            "Redness of the eye"
        ],
        "risk_factors": [
            "Increasing age",
            "Family history",
            "Elevated intraocular pressure",
            "Certain medical conditions",
            "Previous eye injury"
        ],
        "management": (
            "Management may include prescription eye drops, laser treatment, "
            "or surgery depending on the type and severity."
        )
    },

    "normal": {
        "title": "✅ Normal",
        "description": (
            "The image was classified as normal by the trained model, "
            "meaning the model did not identify strong visual patterns "
            "associated with the four disease categories."
        ),
        "symptoms": [
            "No disease-specific pattern detected by this model",
            "Normal classification does not guarantee perfect eye health"
        ],
        "risk_factors": [
            "Eye health can change over time",
            "Some eye conditions may not be visible in a single retinal image"
        ],
        "management": (
            "Continue routine eye examinations and follow the advice of "
            "a qualified eye-care professional."
        )
    },

    "cataract": {
        "title": "👁️ Cataract",
        "description": (
            "A cataract is clouding of the normally clear lens of the eye. "
            "It can cause blurry vision and increased sensitivity to glare."
        ),
        "symptoms": [
            "Cloudy or blurry vision",
            "Glare sensitivity",
            "Difficulty seeing at night",
            "Faded colors",
            "Frequent prescription changes"
        ],
        "risk_factors": [
            "Increasing age",
            "Diabetes",
            "Previous eye injury",
            "Certain medications",
            "Long-term exposure to ultraviolet light"
        ],
        "management": (
            "Early cataracts may be managed with vision correction. "
            "When cataracts significantly affect vision, surgery may be "
            "recommended by an ophthalmologist."
        )
    },

    "diabetic_retinopathy": {
        "title": "🩸 Diabetic Retinopathy",
        "description": (
            "Diabetic retinopathy is an eye complication of diabetes that "
            "results from damage to retinal blood vessels."
        ),
        "symptoms": [
            "Blurred or fluctuating vision",
            "Dark spots or floaters",
            "Difficulty seeing colors",
            "Vision loss in advanced cases"
        ],
        "risk_factors": [
            "Diabetes duration",
            "Poor blood glucose control",
            "High blood pressure",
            "Abnormal cholesterol levels",
            "Pregnancy in people with diabetes"
        ],
        "management": (
            "Management may involve better control of blood glucose and "
            "blood pressure, regular retinal examinations, injections, "
            "laser treatment, or surgery depending on severity."
        )
    }
}


# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource
def load_cnn_model():

    try:

        loaded_model = keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        # IMPORTANT:
        # Explicitly call the model once so that its weights and
        # computation are initialized before Grad-CAM.
        dummy_input = tf.zeros(
            (1, 224, 224, 3),
            dtype=tf.float32
        )

        _ = loaded_model(dummy_input, training=False)

        return loaded_model, None

    except Exception as e:

        return None, str(e)


model, model_error = load_cnn_model()


# =============================================================================
# MODEL ERROR HANDLING
# =============================================================================

if model is None:

    st.error(
        f"❌ Error loading model:\n\n{model_error}"
    )

    st.info(
        "Make sure 'best_model.h5' is located in the same directory "
        "as this Streamlit Python file."
    )

    st.stop()


# =============================================================================
# MODEL INFORMATION
# =============================================================================

print("Custom CNN model loaded successfully.")
print("Model input shape:", model.input_shape)


# =============================================================================
# PREPROCESS IMAGE
# =============================================================================

def preprocess_image(image_file):

    image = Image.open(image_file).convert("RGB")

    original_image = image.copy()

    image = image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_np = np.array(
        image,
        dtype=np.float32
    )

    image_np = image_np / 255.0

    image_batch = np.expand_dims(
        image_np,
        axis=0
    )

    return image_batch, original_image, image


# =============================================================================
# PREDICT IMAGE
# =============================================================================

def predict_image(image_file):

    image_batch, original_image, processed_image = preprocess_image(
        image_file
    )

    predictions = model(
        image_batch,
        training=False
    )

    probs = predictions.numpy()[0]

    predicted_class_index = int(
        np.argmax(probs)
    )

    return (
        predicted_class_index,
        probs,
        image_batch,
        original_image,
        processed_image
    )


# =============================================================================
# GRAD-CAM
# =============================================================================
#
# IMPORTANT:
#
# We DO NOT use:
#
# model.input
# model.output
#
# because the loaded Sequential model may not expose a compatible
# symbolic computation graph under the current Keras version.
#
# Instead, we rebuild a small Functional graph using the existing
# trained layers and capture conv2d_6 directly during the forward pass.
#
# This is the key fix for:
#
# "The layer sequential has never been called and thus has no defined output."
#
# =============================================================================

GRADCAM_LAYER_NAME = "conv2d_6"


def build_gradcam_model(target_layer_name):
    """
    Build a fresh Functional graph from the already-loaded Sequential
    model layers.

    This avoids relying on model.input/model.output from the loaded
    H5 Sequential object.
    """

    target_layer = None

    for layer in model.layers:

        if layer.name == target_layer_name:

            target_layer = layer
            break

    if target_layer is None:

        raise ValueError(
            f"Layer '{target_layer_name}' was not found in the model."
        )

    # Create a new symbolic input.
    inputs = keras.Input(
        shape=(224, 224, 3),
        name="gradcam_input"
    )

    x = inputs
    conv_outputs = None

    # Replay the ORIGINAL trained layers.
    for layer in model.layers:

        x = layer(x)

        if layer.name == target_layer_name:

            conv_outputs = x

    if conv_outputs is None:

        raise ValueError(
            f"Could not obtain output from layer '{target_layer_name}'."
        )

    predictions = x

    grad_model = keras.Model(
        inputs=inputs,
        outputs=[
            conv_outputs,
            predictions
        ],
        name="gradcam_model"
    )

    return grad_model


@st.cache_resource
def get_gradcam_model():

    try:

        grad_model = build_gradcam_model(
            GRADCAM_LAYER_NAME
        )

        return grad_model, None

    except Exception as e:

        return None, str(e)


# =============================================================================
# GENERATE GRAD-CAM
# =============================================================================

def generate_gradcam(
    image_batch,
    predicted_class_index
):
    """
    Generate Grad-CAM heatmap for the predicted class.
    """

    grad_model, error = get_gradcam_model()

    if grad_model is None:

        raise RuntimeError(
            f"Could not construct the Grad-CAM model. "
            f"Reason: {error}"
        )

    image_tensor = tf.cast(
        image_batch,
        tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False
        )

        class_score = predictions[
            :, predicted_class_index
        ]

    # Gradient of target class score with respect to
    # convolutional feature maps.
    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    if grads is None:

        raise RuntimeError(
            "Gradients could not be calculated for the selected "
            "convolutional layer."
        )

    # Global average pooling of gradients.
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Remove batch dimension.
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    # Weighted combination of feature maps.
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU: only retain positive influence.
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize.
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    return heatmap


# =============================================================================
# CREATE GRAD-CAM OVERLAY
# =============================================================================

def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=0.45
):
    """
    Create a heatmap + original-image overlay.

    OpenCV is NOT required.
    """

    # Convert original image to RGB.
    original_image = original_image.convert(
        "RGB"
    )

    # Resize heatmap to original image dimensions.
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255),
        mode="L"
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_np = np.array(
        heatmap_image
    ) / 255.0

    # Use matplotlib's jet colormap.
    cmap = plt.get_cmap("jet")

    colored_heatmap = cmap(
        heatmap_np
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    colored_heatmap_image = Image.fromarray(
        colored_heatmap
    ).convert("RGB")

    # Blend.
    overlay = Image.blend(
        original_image,
        colored_heatmap_image,
        alpha=alpha
    )

    return (
        heatmap_image,
        colored_heatmap_image,
        overlay
    )


# =============================================================================
# GENERATE GRAD-CAM FOR ONE EYE
# =============================================================================

def generate_eye_gradcam(
    image_batch,
    original_image,
    predicted_class_index
):

    heatmap = generate_gradcam(
        image_batch,
        predicted_class_index
    )

    heatmap_image, colored_heatmap, overlay = create_gradcam_overlay(
        original_image,
        heatmap
    )

    return (
        heatmap,
        heatmap_image,
        colored_heatmap,
        overlay
    )


# =============================================================================
# CREATE DOWNLOADABLE REPORT
# =============================================================================

def create_prediction_report(
    left_class_index,
    left_probs,
    right_class_index,
    right_probs
):

    left_class = class_names[
        left_class_index
    ]

    right_class = class_names[
        right_class_index
    ]

    left_confidence = (
        left_probs[left_class_index] * 100
    )

    right_confidence = (
        right_probs[right_class_index] * 100
    )

    left_second = np.argsort(
        left_probs
    )[-2]

    right_second = np.argsort(
        right_probs
    )[-2]

    left_second_confidence = (
        left_probs[left_second] * 100
    )

    right_second_confidence = (
        right_probs[right_second] * 100
    )

    if left_class == right_class:

        overall_assessment = (
            f"Both uploaded eyes were classified as "
            f"{DISPLAY_NAMES[left_class]}."
        )

    else:

        overall_assessment = (
            "The model produced different classifications "
            "for the left and right eyes."
        )

    report = f"""
============================================================
EYE DISEASE CLASSIFICATION REPORT
============================================================

IMPORTANT:
This report is generated by an artificial intelligence model
for educational and demonstration purposes only.

It is NOT a medical diagnosis.

------------------------------------------------------------
LEFT EYE
------------------------------------------------------------

Predicted Class:
{DISPLAY_NAMES[left_class]}

Confidence:
{left_confidence:.2f}%

Second Highest Class:
{DISPLAY_NAMES[class_names[left_second]]}

Second Highest Probability:
{left_second_confidence:.2f}%

All Class Probabilities:
"""

    for i, probability in enumerate(left_probs):

        report += (
            f"\n{DISPLAY_NAMES[class_names[i]]}: "
            f"{probability * 100:.2f}%"
        )

    report += f"""

------------------------------------------------------------
RIGHT EYE
------------------------------------------------------------

Predicted Class:
{DISPLAY_NAMES[right_class]}

Confidence:
{right_confidence:.2f}%

Second Highest Class:
{DISPLAY_NAMES[class_names[right_second]]}

Second Highest Probability:
{right_second_confidence:.2f}%

All Class Probabilities:
"""

    for i, probability in enumerate(right_probs):

        report += (
            f"\n{DISPLAY_NAMES[class_names[i]]}: "
            f"{probability * 100:.2f}%"
        )

    report += f"""

------------------------------------------------------------
OVERALL ASSESSMENT
------------------------------------------------------------

{overall_assessment}

------------------------------------------------------------
MODEL INFORMATION
------------------------------------------------------------

Model:
Custom CNN

Input Size:
224 x 224 x 3

Number of Classes:
4

Classes:
Glaucoma
Normal
Cataract
Diabetic Retinopathy

Grad-CAM Layer:
{GRADCAM_LAYER_NAME}

------------------------------------------------------------
MEDICAL DISCLAIMER
------------------------------------------------------------

This application is intended only for educational,
research, and demonstration purposes.

The predictions should NOT be used for self-diagnosis,
medical decision-making, or treatment.

Please consult a qualified ophthalmologist or optometrist
for professional evaluation.

============================================================
END OF REPORT
============================================================
"""

    return report


# =============================================================================
# APPLICATION HEADER
# =============================================================================

st.title(
    "Eye Diseases Classification 👁️"
)

st.markdown(
    """
    **Welcome to the Eye Disease Classification App!**

    This tool uses a deep learning model to help identify
    potential eye conditions from retinal images.

    **Please note: This is for informational purposes only
    and is not a substitute for professional medical advice.**
    """
)


# =============================================================================
# HERO IMAGE
# =============================================================================

if os.path.isfile(HERO_IMAGE_PATH):

    st.image(
        HERO_IMAGE_PATH,
        width="stretch"
    )

else:

    st.warning(
        f"Hero image not found: `{HERO_IMAGE_PATH}`"
    )


st.divider()


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown(
    "## 🩺 Eye Disease Descriptions"
)

st.sidebar.markdown(
    """
### 👁️ Glaucoma

Damage to the optic nerve, often caused by high
intraocular pressure.

It can lead to gradual, irreversible vision loss
if untreated.

---

### 👁️ Diabetic Retinopathy

Caused by diabetes damaging the retina's blood
vessels.

May lead to blurred vision and blindness without
early treatment.

---

### 👁️ Cataract

Clouding of the eye's lens, usually associated
with aging.

It can cause blurry vision and glare and may be
treated with surgery.

---

### 👁️ Normal

Healthy eye with no signs of disease or retinal
abnormalities detected by this model.

Vision remains clear and unaffected.
"""
)


# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "🔍 Prediction",
        "📖 About the Project",
        "⚙️ How it Works",
        "📊 Visualizations",
        "🔬 Explainable AI",
        "🩺 Eye Comparison",
        "📋 Prediction Report",
        "📚 Disease Information",
    ]
)


# =============================================================================
# TAB 1 - PREDICTION
# =============================================================================

with tab1:

    st.header(
        "🔍 Upload Retinal Images for Prediction"
    )

    st.markdown(
        "Upload images of the left and right eye "
        "(JPG, JPEG, PNG) to get a classification."
    )

    col_upload_left, col_upload_right = st.columns(2)

    with col_upload_left:

        Left_Eye = st.file_uploader(
            "**Upload Left Eye Image**",
            type=["jpg", "jpeg", "png"],
            key="left_eye_uploader"
        )

    with col_upload_right:

        Right_Eye = st.file_uploader(
            "**Upload Right Eye Image**",
            type=["jpg", "jpeg", "png"],
            key="right_eye_uploader"
        )

    # -------------------------------------------------------------------------
    # DISPLAY UPLOADED IMAGES
    # -------------------------------------------------------------------------

    col_display_left, col_display_right = st.columns(2)

    with col_display_left:

        if Left_Eye is not None:

            st.image(
                Left_Eye,
                caption="Left Eye - Uploaded Image",
                width=300
            )

    with col_display_right:

        if Right_Eye is not None:

            st.image(
                Right_Eye,
                caption="Right Eye - Uploaded Image",
                width=300
            )

    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    if (
        Right_Eye is not None
        and
        Left_Eye is not None
    ):

        try:

            (
                predicted_class_index_left,
                prop_left,
                image_batch_left,
                original_image_left,
                processed_image_left
            ) = predict_image(
                Left_Eye
            )

            (
                predicted_class_index_right,
                prop_right,
                image_batch_right,
                original_image_right,
                processed_image_right
            ) = predict_image(
                Right_Eye
            )

            # Save results into session state.
            st.session_state[
                "left_prediction"
            ] = (
                predicted_class_index_left,
                prop_left,
                image_batch_left,
                original_image_left,
                processed_image_left
            )

            st.session_state[
                "right_prediction"
            ] = (
                predicted_class_index_right,
                prop_right,
                image_batch_right,
                original_image_right,
                processed_image_right
            )

            # -----------------------------------------------------------------
            # RESULTS
            # -----------------------------------------------------------------

            st.subheader(
                "🎯 Prediction Results"
            )

            col_results_left, col_results_right = st.columns(2)

            # =================================================================
            # LEFT EYE
            # =================================================================

            with col_results_left:

                st.markdown(
                    "### 👁️ Left Eye Prediction"
                )

                left_class_name = class_names[
                    predicted_class_index_left
                ]

                left_confidence = (
                    prop_left[
                        predicted_class_index_left
                    ] * 100
                )

                st.success(
                    f"**Predicted Class: "
                    f"{DISPLAY_NAMES[left_class_name]}**"
                )

                st.metric(
                    "Confidence",
                    f"{left_confidence:.2f}%"
                )

                st.write("---")

                st.markdown(
                    "##### 📊 All Class Confidences:"
                )

                for i, prob in enumerate(prop_left):

                    if i == predicted_class_index_left:

                        st.markdown(
                            f"**{DISPLAY_NAMES[class_names[i]]}: "
                            f"{prob:.2%}** ⭐"
                        )

                    else:

                        st.write(
                            f"{DISPLAY_NAMES[class_names[i]]}: "
                            f"{prob:.2%}"
                        )

            # =================================================================
            # RIGHT EYE
            # =================================================================

            with col_results_right:

                st.markdown(
                    "### 👁️ Right Eye Prediction"
                )

                right_class_name = class_names[
                    predicted_class_index_right
                ]

                right_confidence = (
                    prop_right[
                        predicted_class_index_right
                    ] * 100
                )

                st.success(
                    f"**Predicted Class: "
                    f"{DISPLAY_NAMES[right_class_name]}**"
                )

                st.metric(
                    "Confidence",
                    f"{right_confidence:.2f}%"
                )

                st.write("---")

                st.markdown(
                    "##### 📊 All Class Confidences:"
                )

                for i, prob in enumerate(prop_right):

                    if i == predicted_class_index_right:

                        st.markdown(
                            f"**{DISPLAY_NAMES[class_names[i]]}: "
                            f"{prob:.2%}** ⭐"
                        )

                    else:

                        st.write(
                            f"{DISPLAY_NAMES[class_names[i]]}: "
                            f"{prob:.2%}"
                        )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# =============================================================================
# TAB 2 - ABOUT THE PROJECT
# =============================================================================

with tab2:

    st.header(
        "📖 About the Eye Disease Classification Project"
    )

    st.markdown(
        """
        This project aims to develop an automated system for
        classifying common eye diseases from retinal images.

        Early detection of eye conditions like glaucoma,
        diabetic retinopathy, and cataracts is important for
        timely professional evaluation.

        **Project Goals:**

        * Build a deep learning model capable of classifying
          retinal images into specific disease categories.
        * Provide an accessible web application for demonstrating
          the model's capabilities.
        * Demonstrate the potential of AI in healthcare.
        * Provide explainability using Grad-CAM visualizations.

        **Diseases Classified:**

        * **Glaucoma**
        * **Diabetic Retinopathy**
        * **Cataract**
        * **Normal**

        **Model Used:**

        The core of this application is a custom-built
        Convolutional Neural Network (CNN).

        The CNN contains multiple convolutional layers,
        pooling layers, a Flatten layer, Dense layers,
        Dropout, and a final Softmax classification layer.

        The final convolutional layer used for Grad-CAM
        analysis is:

        `conv2d_6`
        """
    )


# =============================================================================
# TAB 3 - HOW IT WORKS
# =============================================================================

with tab3:

    st.header(
        "⚙️ How the Eye Disease Classification App Works"
    )

    st.markdown(
        """
        This application utilizes a custom-trained
        Convolutional Neural Network (CNN) to analyze
        uploaded retinal images.

        ### 1. 🖼️ Image Upload and Preprocessing

        * Input images are uploaded by the user.
        * Images are converted to RGB.
        * Images are resized to `224x224`.
        * Pixel values are normalized from `0-255` to `0-1`.

        ### 2. 🧠 Deep Learning Model

        The custom CNN contains:

        * Convolutional layers
        * Max pooling layers
        * Flatten layer
        * Dense layer
        * Dropout
        * Softmax output layer

        ### 3. 🎯 Prediction

        The model produces probabilities for:

        * Glaucoma
        * Normal
        * Cataract
        * Diabetic Retinopathy

        The class with the highest probability is selected
        as the model prediction.

        ### 4. 🔬 Explainable AI

        Grad-CAM is used to visualize image regions that
        contributed to the model's classification.

        ### 5. 🩺 Eye Comparison

        When both eyes are uploaded, the application compares
        their predicted classes and confidence values.

        ### 6. 📋 Report Generation

        A detailed text report can be generated containing
        predictions, confidence values, and model information.
        """
    )


# =============================================================================
# TAB 4 - VISUALIZATIONS
# =============================================================================

with tab4:

    st.header(
        "📊 Key Model Visualizations"
    )

    st.markdown(
        """
        Explore the dataset distribution, training process,
        model evaluation, and classification performance
        through the visualizations generated during the project.
        """
    )

    st.divider()

    # =========================================================================
    # 1. DATASET DISTRIBUTION
    # =========================================================================

    distribution_path = os.path.join(
        VISUALIZATION_DIR,
        "01_distribution.png"
    )

    st.subheader(
        "1. 📈 Dataset Class Distribution"
    )

    st.markdown(
        """
        This visualization shows the distribution of retinal
        images across the different classes in the training,
        validation, and testing sets.
        """
    )

    if os.path.isfile(distribution_path):

        st.image(
            distribution_path,
            caption=(
                "Distribution of Classes in Train, "
                "Validation, and Test Sets"
            ),
            width="stretch"
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

    st.subheader(
        "2. 🥧 Dataset Distribution"
    )

    st.markdown(
        """
        These pie charts show how the dataset was divided
        into training, validation, and testing subsets.
        """
    )

    if os.path.isfile(train_test_path):

        st.image(
            train_test_path,
            caption=(
                "Training, Validation, and Testing "
                "Dataset Distribution"
            ),
            width="stretch"
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

    st.subheader(
        "3. 🖼️ Sample Training Images"
    )

    st.markdown(
        """
        This visualization shows representative retinal
        images used during the model development and training
        process, including examples of augmented training images.
        """
    )

    if os.path.isfile(model_building_path):

        st.image(
            model_building_path,
            caption="Sample Images Used During Model Training",
            width="stretch"
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

    st.subheader(
        "4. 📉 Model Evaluation Curves"
    )

    st.markdown(
        """
        These curves show the model's training and validation
        performance across epochs. Accuracy and loss curves
        can be used to understand model learning and possible
        overfitting.
        """
    )

    if os.path.isfile(evaluation_path):

        st.image(
            evaluation_path,
            caption="Training and Validation Accuracy/Loss over Epochs",
            width="stretch"
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

    st.subheader(
        "5. 🔲 Confusion Matrix"
    )

    st.markdown(
        """
        The confusion matrix shows the classification
        performance of the model on the test dataset.
        """
    )

    if os.path.isfile(confusion_matrix_path):

        st.image(
            confusion_matrix_path,
            caption="Confusion Matrix for Test Set Predictions",
            width="stretch"
        )

    else:

        st.error(
            f"❌ Image not found:\n\n`{confusion_matrix_path}`"
        )

    st.divider()

    # =========================================================================
    # 6. ROC-AUC CURVE
    # =========================================================================

    roc_auc_path = os.path.join(
        VISUALIZATION_DIR,
        "06_roc_curve.png"
    )

    st.subheader(
        "6. 📈 ROC-AUC Curve"
    )

    st.markdown(
        """
        The ROC-AUC curve evaluates the model's ability to
        distinguish between the four eye-disease classes across
        different classification thresholds. A higher AUC
        generally indicates better class discrimination.
        """
    )

    if os.path.isfile(roc_auc_path):

        st.image(
            roc_auc_path,
            caption="Multi-Class ROC-AUC Curve",
            width="stretch"
        )

    else:

        st.error(
            f"❌ Image not found:\n\n`{roc_auc_path}`"
        )

    st.divider()

    # =========================================================================
    # 7. PRECISION, RECALL AND F1-SCORE COMPARISON
    # =========================================================================

    metrics_comparison_path = os.path.join(
        VISUALIZATION_DIR,
        "07_comparison.png"
    )

    st.subheader(
        "7. 📊 Precision, Recall and F1-Score Comparison"
    )

    st.markdown(
        """
        This visualization compares precision, recall, and
        F1-score across the four eye-disease classes. It provides
        a class-wise view of the model's classification performance.
        """
    )

    if os.path.isfile(metrics_comparison_path):

        st.image(
            metrics_comparison_path,
            caption=(
                "Precision, Recall and F1-Score "
                "Comparison Across Classes"
            ),
            width="stretch"
        )

    else:

        st.error(
            f"❌ Image not found:\n\n`{metrics_comparison_path}`"
        )


# =============================================================================
# TAB 5 - EXPLAINABLE AI / GRAD-CAM
# =============================================================================

with tab5:

    st.header(
        "🔬 Explainable AI - Grad-CAM"
    )

    st.markdown(
        """
        **Grad-CAM (Gradient-weighted Class Activation Mapping)**
        provides a visual representation of image regions that
        contributed to the CNN's prediction.

        The Grad-CAM implementation in this application uses
        **`conv2d_6`**, the final convolutional layer of the
        trained custom CNN.
        """
    )

    st.warning(
        """
        ⚠️ **Important**

        The Grad-CAM heatmap represents model attention.

        It should **not** be interpreted as proof that a particular
        anatomical region is diseased or as a clinical explanation.

        The heatmap is an AI interpretability visualization,
        not a medical diagnostic tool.
        """
    )

    st.markdown(
        """
        ### 🔥 Heatmap interpretation

        🔴 **Red / Orange** → stronger model attention

        🟡 **Yellow** → moderate model attention

        🔵 **Blue** → lower model attention
        """
    )

    st.divider()

    st.subheader(
        "🔍 Generate Grad-CAM Visualizations"
    )

    if (
        "left_prediction" not in st.session_state
        or
        "right_prediction" not in st.session_state
    ):

        st.info(
            "👈 Please upload both left-eye and right-eye images "
            "in the 🔍 Prediction tab first."
        )

    else:

        (
            left_class_index,
            left_probs,
            left_image_batch,
            left_original_image,
            left_processed_image
        ) = st.session_state["left_prediction"]

        (
            right_class_index,
            right_probs,
            right_image_batch,
            right_original_image,
            right_processed_image
        ) = st.session_state["right_prediction"]

        # ---------------------------------------------------------------------
        # LEFT EYE
        # ---------------------------------------------------------------------

        st.markdown(
            "### 👁️ Left Eye - Grad-CAM"
        )

        try:

            (
                left_heatmap,
                left_gray_heatmap,
                left_colored_heatmap,
                left_overlay
            ) = generate_eye_gradcam(
                left_image_batch,
                left_original_image,
                left_class_index
            )

            left_class = class_names[
                left_class_index
            ]

            st.success(
                f"Model Prediction: "
                f"**{DISPLAY_NAMES[left_class]}** "
                f"({left_probs[left_class_index] * 100:.2f}%)"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.image(
                    left_original_image,
                    caption="Original Left Eye",
                    width="stretch"
                )

            with col2:

                st.image(
                    left_colored_heatmap,
                    caption="Grad-CAM Heatmap",
                    width="stretch"
                )

            with col3:

                st.image(
                    left_overlay,
                    caption="Grad-CAM Overlay",
                    width="stretch"
                )

        except Exception as e:

            st.error(
                "❌ Grad-CAM could not be generated "
                "for the left eye."
            )

            st.code(
                str(e)
            )

        st.divider()

        # ---------------------------------------------------------------------
        # RIGHT EYE
        # ---------------------------------------------------------------------

        st.markdown(
            "### 👁️ Right Eye - Grad-CAM"
        )

        try:

            (
                right_heatmap,
                right_gray_heatmap,
                right_colored_heatmap,
                right_overlay
            ) = generate_eye_gradcam(
                right_image_batch,
                right_original_image,
                right_class_index
            )

            right_class = class_names[
                right_class_index
            ]

            st.success(
                f"Model Prediction: "
                f"**{DISPLAY_NAMES[right_class]}** "
                f"({right_probs[right_class_index] * 100:.2f}%)"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.image(
                    right_original_image,
                    caption="Original Right Eye",
                    width="stretch"
                )

            with col2:

                st.image(
                    right_colored_heatmap,
                    caption="Grad-CAM Heatmap",
                    width="stretch"
                )

            with col3:

                st.image(
                    right_overlay,
                    caption="Grad-CAM Overlay",
                    width="stretch"
                )

        except Exception as e:

            st.error(
                "❌ Grad-CAM could not be generated "
                "for the right eye."
            )

            st.code(
                str(e)
            )


# =============================================================================
# TAB 6 - EYE-TO-EYE COMPARISON
# =============================================================================

with tab6:

    st.header(
        "🩺 Eye-to-Eye Comparison / Overall Assessment"
    )

    st.markdown(
        """
        This section compares the predictions of the left and
        right retinal images.

        **Important:** The comparison is based only on the
        classification outputs of the AI model and should not
        be interpreted as a clinical assessment.
        """
    )

    if (
        "left_prediction" not in st.session_state
        or
        "right_prediction" not in st.session_state
    ):

        st.info(
            "Please upload both eye images in the "
            "🔍 Prediction tab first."
        )

    else:

        (
            left_class_index,
            left_probs,
            left_image_batch,
            left_original_image,
            left_processed_image
        ) = st.session_state["left_prediction"]

        (
            right_class_index,
            right_probs,
            right_image_batch,
            right_original_image,
            right_processed_image
        ) = st.session_state["right_prediction"]

        left_class = class_names[
            left_class_index
        ]

        right_class = class_names[
            right_class_index
        ]

        left_confidence = (
            left_probs[left_class_index] * 100
        )

        right_confidence = (
            right_probs[right_class_index] * 100
        )

        # ---------------------------------------------------------------------
        # SIDE-BY-SIDE IMAGE COMPARISON
        # ---------------------------------------------------------------------

        col_left, col_right = st.columns(2)

        with col_left:

            st.subheader(
                "👁️ Left Eye"
            )

            st.image(
                left_original_image,
                caption="Left Eye",
                width="stretch"
            )

            st.metric(
                "Prediction",
                DISPLAY_NAMES[left_class]
            )

            st.metric(
                "Confidence",
                f"{left_confidence:.2f}%"
            )

        with col_right:

            st.subheader(
                "👁️ Right Eye"
            )

            st.image(
                right_original_image,
                caption="Right Eye",
                width="stretch"
            )

            st.metric(
                "Prediction",
                DISPLAY_NAMES[right_class]
            )

            st.metric(
                "Confidence",
                f"{right_confidence:.2f}%"
            )

        st.divider()

        # ---------------------------------------------------------------------
        # OVERALL ASSESSMENT
        # ---------------------------------------------------------------------

        st.subheader(
            "🩺 Overall Model Assessment"
        )

        if left_class == right_class:

            if left_class == "normal":

                st.success(
                    """
                    ✅ Both eyes received a **Normal** classification
                    from the model.

                    This indicates that the model produced consistent
                    predictions for both uploaded images.

                    This does NOT confirm that the eyes are medically
                    healthy.
                    """
                )

            else:

                st.warning(
                    f"""
                    ⚠️ Both eyes received the same model classification:

                    **{DISPLAY_NAMES[left_class]}**

                    The model produced consistent predictions across
                    both uploaded images.

                    Professional clinical evaluation is required
                    to determine whether disease is actually present.
                    """
                )

        else:

            st.warning(
                f"""
                ⚠️ The model produced different predictions:

                **Left Eye:** {DISPLAY_NAMES[left_class]}

                **Right Eye:** {DISPLAY_NAMES[right_class]}

                Differences between the two predictions may reflect
                image characteristics, model uncertainty, or actual
                differences between the images.

                This result should not be interpreted as a clinical
                diagnosis.
                """
            )

        # ---------------------------------------------------------------------
        # CONFIDENCE DIFFERENCE
        # ---------------------------------------------------------------------

        confidence_difference = abs(
            left_confidence - right_confidence
        )

        st.subheader(
            "📊 Confidence Comparison"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Left Eye Confidence",
                f"{left_confidence:.2f}%"
            )

        with col2:

            st.metric(
                "Right Eye Confidence",
                f"{right_confidence:.2f}%"
            )

        with col3:

            st.metric(
                "Confidence Difference",
                f"{confidence_difference:.2f}%"
            )


# =============================================================================
# TAB 7 - DETAILED PREDICTION REPORT
# =============================================================================

with tab7:

    st.header(
        "📋 Detailed Prediction Report"
    )

    st.markdown(
        """
        Generate a detailed report containing the predictions,
        confidence scores, class probabilities, model information,
        and overall comparison.
        """
    )

    if (
        "left_prediction" not in st.session_state
        or
        "right_prediction" not in st.session_state
    ):

        st.info(
            "Please upload both eye images in the "
            "🔍 Prediction tab first."
        )

    else:

        (
            left_class_index,
            left_probs,
            left_image_batch,
            left_original_image,
            left_processed_image
        ) = st.session_state["left_prediction"]

        (
            right_class_index,
            right_probs,
            right_image_batch,
            right_original_image,
            right_processed_image
        ) = st.session_state["right_prediction"]

        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------

        st.subheader(
            "📌 Prediction Summary"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 👁️ Left Eye"
            )

            st.write(
                f"**Prediction:** "
                f"{DISPLAY_NAMES[class_names[left_class_index]]}"
            )

            st.write(
                f"**Confidence:** "
                f"{left_probs[left_class_index] * 100:.2f}%"
            )

        with col2:

            st.markdown(
                "### 👁️ Right Eye"
            )

            st.write(
                f"**Prediction:** "
                f"{DISPLAY_NAMES[class_names[right_class_index]]}"
            )

            st.write(
                f"**Confidence:** "
                f"{right_probs[right_class_index] * 100:.2f}%"
            )

        st.divider()

        # ---------------------------------------------------------------------
        # DETAILED PROBABILITIES
        # ---------------------------------------------------------------------

        st.subheader(
            "📊 Detailed Class Probabilities"
        )

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:

            st.markdown(
                "### 👁️ Left Eye"
            )

            for i, probability in enumerate(left_probs):

                st.progress(
                    float(probability),
                    text=(
                        f"{DISPLAY_NAMES[class_names[i]]}: "
                        f"{probability * 100:.2f}%"
                    )
                )

        with probability_col2:

            st.markdown(
                "### 👁️ Right Eye"
            )

            for i, probability in enumerate(right_probs):

                st.progress(
                    float(probability),
                    text=(
                        f"{DISPLAY_NAMES[class_names[i]]}: "
                        f"{probability * 100:.2f}%"
                    )
                )

        st.divider()

        # ---------------------------------------------------------------------
        # GENERATE REPORT
        # ---------------------------------------------------------------------

        st.subheader(
            "📄 Download Report"
        )

        report = create_prediction_report(
            left_class_index,
            left_probs,
            right_class_index,
            right_probs
        )

        with st.expander(
            "👀 Preview Detailed Report"
        ):

            st.text(
                report
            )

        st.download_button(
            label="📥 Download Prediction Report",
            data=report,
            file_name="eye_disease_prediction_report.txt",
            mime="text/plain"
        )


# =============================================================================
# TAB 8 - DISEASE INFORMATION / EDUCATIONAL CENTER
# =============================================================================

with tab8:

    st.header(
        "📚 Disease Information / Educational Center"
    )

    st.markdown(
        """
        Learn more about the four categories included in this
        AI classification project.

        **This information is educational and should not replace
        advice from a qualified eye-care professional.**
        """
    )

    disease_selection = st.selectbox(
        "🔎 Select a condition to learn more:",
        [
            "glaucoma",
            "normal",
            "cataract",
            "diabetic_retinopathy"
        ],
        format_func=lambda x: DISPLAY_NAMES[x]
    )

    selected_info = DISEASE_INFO[
        disease_selection
    ]

    st.subheader(
        selected_info["title"]
    )

    st.markdown(
        selected_info["description"]
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### ⚠️ Common Symptoms"
        )

        for symptom in selected_info["symptoms"]:

            st.markdown(
                f"- {symptom}"
            )

    with col2:

        st.markdown(
            "### 📌 Risk Factors"
        )

        for risk_factor in selected_info["risk_factors"]:

            st.markdown(
                f"- {risk_factor}"
            )

    st.divider()

    st.markdown(
        "### 🩺 General Management Information"
    )

    st.info(
        selected_info["management"]
    )

    st.warning(
        """
        **When should you see an eye-care professional?**

        If you experience sudden vision loss, significant eye pain,
        sudden flashes/floaters, severe redness, or any other
        concerning change in vision, seek professional medical
        attention promptly.

        The AI classification provided by this application should
        not be used to decide whether medical care is necessary.
        """
    )


# =============================================================================
# FOOTER / DISCLAIMER
# =============================================================================

st.divider()

st.markdown(
    """
    ### ⚠️ Medical Disclaimer

    **This Eye Disease Classification application is developed
    for educational and demonstration purposes only.**

    It is **NOT intended to be a substitute for professional
    medical advice, diagnosis, or treatment.**

    * **Do Not Use for Self-Diagnosis:** Predictions are generated
      by a machine learning model and should not be treated as
      a medical diagnosis.

    * **Consult a Healthcare Professional:** Always seek advice
      from a qualified ophthalmologist or optometrist for concerns
      regarding eye health.

    * **Accuracy Limitations:** Deep learning models are not
      infallible. Image quality, dataset limitations, rare
      conditions, and variations in disease presentation may
      affect performance.

    * **Grad-CAM Limitations:** Grad-CAM shows regions associated
      with model attention. It does not prove the presence of
      disease in a particular anatomical region.

    * **No Doctor-Patient Relationship:** Using this application
      does not create a doctor-patient relationship between the
      user, developers, or AI model.

    **By using this application, you acknowledge and agree to
    this disclaimer.**
    """
)
