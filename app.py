import cv2
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow import keras

MODEL_VALIDATION_ACCURACY = 99.9  # Your model's reported accuracy

print("Loading model...")
base_model = keras.models.load_model('base_network.h5')
print("✓ Model loaded!")

def is_likely_signature(image, threshold=0.03):
    """
    Returns True if the image likely contains a signature, False if it is mostly blank or not enough ink.
    threshold: Percentage of black ('ink') pixels required.
    """
    if image is None:
        return False
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    ink_pixels = np.count_nonzero(binary)
    total_pixels = binary.shape[0] * binary.shape[1]
    ink_ratio = ink_pixels / total_pixels
    return ink_ratio > threshold  # E.g., 3% of pixels must be black (ink)

def preprocess_signature(image, target_size=(155, 220)):
    if image is None:
        return None
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    resized = cv2.resize(gray, (target_size[1], target_size[0]))
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)
    normalized = binary.astype(np.float32) / 255.0
    normalized = np.expand_dims(normalized, axis=-1)
    normalized = np.expand_dims(normalized, axis=0)
    return normalized

def calculate_similarity(img1, img2):
    processed1 = preprocess_signature(img1)
    processed2 = preprocess_signature(img2)
    embedding1 = base_model.predict(processed1, verbose=0)
    embedding2 = base_model.predict(processed2, verbose=0)
    embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
    embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
    cosine_sim = np.dot(embedding1_norm.flatten(), embedding2_norm.flatten())
    similarity = (cosine_sim + 1) / 2
    return similarity

def style_result(result):
    if "GENUINE" in result:
        return (
            '<span style="color:#237804;font-weight:700;font-size:1.7em;'
            'background:#e6ffed;padding:12px 30px;border-radius:12px;'
            'border:2.5px solid #237804;box-shadow:0 0 12px #d8f3dc;">'
            + result + '</span>'
        )
    elif "FORGERY" in result:
        return (
            '<span style="color:#d4380d;font-weight:700;font-size:2em;'
            'background:#fff1f0;padding:16px 40px;border-radius:12px;'
            'border:3px solid #d4380d;box-shadow:0 0 18px #ffd5d5;">'
            + result + '</span>'
        )
    else:
        return (
            '<span style="font-weight:700;font-size:1.3em;">'
            + result + '</span>'
        )

def verify_signature(reference_img, test_img):
    if reference_img is None or test_img is None:
        return style_result("❌ Please upload both signatures!"), "ERROR", ""
    if not is_likely_signature(reference_img):
        return style_result("❌ Reference image is not a valid signature!"), "ERROR", ""
    if not is_likely_signature(test_img):
        return style_result("❌ Test image is not a valid signature!"), "ERROR", ""
    similarity = calculate_similarity(reference_img, test_img)
    similarity_percent = round(similarity * 100, 2)
    if similarity >= 0.95:
        result = "✅ GENUINE SIGNATURE"
        confidence = "HIGH 🟢"
    else:
        result = "❌ FORGERY DETECTED"
        confidence = "HIGH 🔴"
    return style_result(result), confidence, f"{similarity_percent}%"

def verify_multi_reference(ref1, ref2, ref3, test_img):
    refs = [img for img in [ref1, ref2, ref3] if img is not None]
    if len(refs) < 2:
        return style_result("❌ Upload at least 2 reference signatures!"), "ERROR", ""
    # Reference signature validity check
    for idx, ref in enumerate(refs):
        if not is_likely_signature(ref):
            return style_result(f"❌ Reference image {idx+1} is not a valid signature!"), "ERROR", ""
    if test_img is None:
        return style_result("❌ Upload test signature!"), "ERROR", ""
    if not is_likely_signature(test_img):
        return style_result("❌ Test image is not a valid signature!"), "ERROR", ""
    similarities = [calculate_similarity(ref, test_img) for ref in refs]
    avg_similarity = np.mean(similarities)
    avg_similarity_percent = round(avg_similarity * 100, 2)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    variation = max_sim - min_sim
    scores_text = "\n".join([f"Reference {i+1}: {round(s*100,2)}%" for i, s in enumerate(similarities)])
    if avg_similarity >= 0.95 and variation < 0.10:
        result = "✅ GENUINE SIGNATURE"
        confidence = "HIGH 🟢"
    else:
        result = "❌ FORGERY DETECTED"
        confidence = "HIGH 🔴"
    return style_result(result), confidence, f"Average: {avg_similarity_percent}%\n" + scores_text

with gr.Blocks(
    title="Signature Verification System",
    theme=gr.themes.Base(),
    css="""
    .output-html {font-size: 1.22em;}
    .gradio-row {margin-bottom: 17px;}
    .gradio-column {margin-bottom: 17px;}
    .highlighted-result {font-size:2em;font-weight:700;padding:14px 32px;border-radius:12px;}
    """
) as demo:
    gr.Markdown("""
        <div style="text-align:center; margin-bottom:20px;">
            <h1 style="color:#0066cc;letter-spacing:1px;">🔏 Signature Verification System</h1>
            <h3 style="color:#222;font-weight:600;">AI-Powered Secure Authentication</h3>
        </div>
        <div style="text-align:center; font-size:1.16em; margin-bottom:18px;">
            <span style="padding:10px 28px;background:#d8f3dc;color:#237804;border-radius:8px;font-size:1.17em;">
            Validation Accuracy: <b>99.9%</b>
            </span>
        </div>
        <div style="background:#f8f9fa;border-radius:16px;padding:12px 22px;">
        <b>How it works:</b> Upload genuine reference signature(s) and a test signature.<br>
        You'll get a clear decision—highlighted for instant recognition.<br>
        In multi-reference mode, individual similarity scores are shown for each reference.
        </div>
        <br>
    """)
    with gr.Tabs():
        with gr.Tab("Simple Verification"):
            gr.Markdown("""
                <div style="color:#444;font-size:1.09em;background:#f1f8fd;padding:10px;border-radius:12px;">
                    <b>Single Reference:</b> Quickly check one test signature against a genuine sample.
                </div>
            """)
            with gr.Row():
                with gr.Column():
                    simple_ref = gr.Image(label="Reference Signature", type="numpy", height=230)
                with gr.Column():
                    simple_test = gr.Image(label="Test Signature", type="numpy", height=230)
            simple_btn = gr.Button("🔍 Verify Signature", variant="primary", size="lg")
            with gr.Row():
                simple_result = gr.HTML(label="Result")
                simple_confidence = gr.Textbox(label="Confidence Level", interactive=False)
                simple_similarity = gr.Textbox(label="Similarity Score (%)", interactive=False)
            simple_btn.click(
                fn=verify_signature,
                inputs=[simple_ref, simple_test],
                outputs=[simple_result, simple_confidence, simple_similarity]
            )
        with gr.Tab("Multi-Reference Verification"):
            gr.Markdown("""
                <div style="color:#444;font-size:1.09em;background:#f1f8fd;padding:10px;border-radius:12px;">
                    <b>Multi-Reference:</b> Advanced check—use 2 or 3 genuine signatures for robust detection.
                </div>
            """)
            with gr.Row():
                adv_ref1 = gr.Image(label="Reference 1", type="numpy", height=180)
                adv_ref2 = gr.Image(label="Reference 2", type="numpy", height=180)
                adv_ref3 = gr.Image(label="Reference 3 (Optional)", type="numpy", height=180)
            adv_test = gr.Image(label="Test Signature", type="numpy", height=230)
            adv_btn = gr.Button("🔍 Verify", variant="primary", size="lg")
            with gr.Row():
                adv_result = gr.HTML(label="Result")
                adv_confidence = gr.Textbox(label="Confidence Level", interactive=False)
                adv_similarity = gr.Textbox(label="Similarity Scores", interactive=False)
            adv_btn.click(
                fn=verify_multi_reference,
                inputs=[adv_ref1, adv_ref2, adv_ref3, adv_test],
                outputs=[adv_result, adv_confidence, adv_similarity]
            )
    gr.Markdown("""
        <br>
        <div style="background:#faf2fa;padding:10px 28px;border-radius:16px;font-size:1.03em;">
        <b>Instructions:</b>
        <ul>
            <li>Upload clear, original signature images (white paper, strong dark pen recommended)</li>
            <li>Multi-reference mode is best for high-security needs</li>
            <li>No cropping needed: System auto-optimizes the images</li>
        </ul>
        <b>Note:</b> No data is stored—your privacy is protected.
        </div>x
        <br>
        <div style="text-align:center;">
            <i style="color:#c778fe;">Powered by Deep Learning | Built with Gradio</i>
        </div>
    """)

if __name__ == "__main__":
    demo.launch()