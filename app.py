import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import streamlit as st
import tensorflow as tf
import keras 
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="AI Radiologist Pro", page_icon="🩺", layout="centered")
st.title("🩺 AI Radiologist: Advanced Multi-Lesion Detector")
st.write("Upload an X-Ray. The AI will scan the entire bone structure and render isolated, discrete heatmaps at every exact fracture/damage location.")

@st.cache_resource
def load_model():
    class PatchedDense(keras.layers.Dense):
        @classmethod
        def from_config(cls, config):
            config.pop("quantization_config", None)
            return super().from_config(config)

    return keras.models.load_model(
        'mura.h5', 
        custom_objects={'Dense': PatchedDense},
        compile=False
    )

model = None
try:
    model = load_model()
except Exception as e:
    st.error(f"Model load hone mein error aaya: {e}")

def compute_raw_cam(img_array, model):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        try:
            shape = getattr(layer, 'output_shape', None)
            if shape is None and hasattr(layer, 'output'):
                shape = layer.output.shape
            if shape is not None and len(shape) == 4:
                last_conv_layer_name = layer.name
                break
        except Exception:
            continue
            
    if last_conv_layer_name is None:
        return np.zeros((224, 224), dtype=np.float32)

    grad_model = tf.keras.models.Model(
        model.inputs, 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    guided_grads = tf.cast(grads > 0, "float32") * grads
    pooled_grads = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    
    conv_outputs = last_conv_layer_output[0]
    cam = conv_outputs @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0)
    
    cam_np = cam.numpy()
    max_val = np.max(cam_np)
    if max_val > 0:
        cam_np = cam_np / max_val
    return cam_np

def generate_discrete_lesion_heatmaps(original_img, cam_map, is_abnormal=True):
    """Renders separate, pinpoint heatmaps strictly at isolated fracture points."""
    orig_w, orig_h = original_img.size
    img_np = np.array(original_img)
    
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
    output_np = img_np.copy()
    
    if not is_abnormal:
        # Clean Verification for normal X-rays
        cv2.putText(
            output_np, 
            "STATUS: NORMAL BONE ALIGNMENT", 
            (25, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.65, 
            (40, 200, 40), 
            2, 
            cv2.LINE_AA
        )
        return Image.fromarray(output_np)

    # 1. Resize CAM to original radiograph resolution
    cam_resized = cv2.resize(cam_map, (orig_w, orig_h))
    
    # 2. Local Peak Extraction (Bridge Breaking to separate multiple fractures)
    cam_8u = np.uint8(255 * cam_resized)
    
    # Adaptive thresholding to identify regional high-energy centers
    _, binary_seeds = cv2.threshold(cam_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological erosion to break any soft continuous gradient bridge between two bones
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    separated_seeds = cv2.morphologyEx(binary_seeds, cv2.MORPH_OPEN, kernel_small)
    
    contours, _ = cv2.findContours(separated_seeds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    combined_discrete_cam = np.zeros_like(cam_resized, dtype=np.float32)
    valid_sites_found = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  
            valid_sites_found += 1
            site_mask = np.zeros_like(separated_seeds)
            cv2.drawContours(site_mask, [cnt], -1, 255, -1)
            pocket = cam_resized * (site_mask > 0)
            pocket_max = np.max(pocket)
            if pocket_max > 0:
                pocket_norm = (pocket / pocket_max) ** 1.8
                pocket_smooth = cv2.GaussianBlur(pocket_norm, (19, 19), 0)
                combined_discrete_cam = np.maximum(combined_discrete_cam, pocket_smooth)
            
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            center = (int(cx), int(cy))
            rad = int(radius * 1.15)
            
            cv2.circle(output_np, center, rad, (255, 60, 60), 2, cv2.LINE_AA)
            cv2.circle(output_np, center, max(rad - 4, 1), (255, 180, 50), 1, cv2.LINE_AA)

    if valid_sites_found == 0:
        combined_discrete_cam = np.where(cam_resized > 0.55, (cam_resized - 0.55) / 0.45, 0)
    
    heatmap_uint8 = np.uint8(255 * np.clip(combined_discrete_cam, 0, 1))
    colored_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
    active_spots = (combined_discrete_cam > 0.12)[:, :, np.newaxis]
    output_np = np.where(
        active_spots,
        cv2.addWeighted(output_np, 0.60, colored_map, 0.40, 0),
        output_np
    )
    
    return Image.fromarray(output_np.astype(np.uint8))

uploaded_file = st.file_uploader("Upload Musculoskeletal X-Ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("Model initialization failed. Please check runtime.")
    else:
        original_image = Image.open(uploaded_file).convert('RGB')
        img_array = original_image.resize((224, 224))
        img_array_scaled = np.array(img_array) / 255.0
        img_array_batch = np.expand_dims(img_array_scaled, axis=0)
        
        with st.spinner("Analyzing cortical fractures & multi-site osseous trauma..."):
            prediction = float(model.predict(img_array_batch)[0][0])
            cam_map = compute_raw_cam(img_array_batch, model)
            
        st.markdown("---")
        
        is_abnormal = prediction > 0.5
        diagnostic_overlay = generate_discrete_lesion_heatmaps(original_image, cam_map, is_abnormal=is_abnormal)
        
        if is_abnormal:
            st.error(f"🚨 **ABNORMALITY DETECTED** (Confidence: {prediction*100:.2f}%)")
            st.caption("Discrete thermal target zones pinpoint each isolated fracture / damage point across the bone structure.")
        else:
            st.success(f"✅ **NORMAL BONE STRUCTURE** (Confidence: {(1-prediction)*100:.2f}%)")
            st.caption("No focal cortical break, dislocation, or joint abnormality detected.")

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption='Original Radiograph', width='stretch')
        with col2:
            st.image(
                diagnostic_overlay, 
                caption='AI Isolated Pathology Localization' if is_abnormal else 'AI Verification (Healthy)', 
                width='stretch'
            )