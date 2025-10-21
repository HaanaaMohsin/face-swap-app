import os
from tempfile import NamedTemporaryFile
from io import BytesIO
import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import requests
import torch
from PIL import Image

# Swin eye disease detection
from eye_disease.data import get_transforms
from eye_disease.model import build_model, load_checkpoint

app = ''
swapper = ''
st.set_page_config(page_title="FaceSwap App by Adil Khan")

def download_model():
    url = "https://cdn.adikhanofficial.com/python/insightface/models/inswapper_128.onnx"
    filename = url.split('/')[-1]
    filepath = os.path.join(os.path.dirname(__file__),filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"{filename} already exists in the directory.")

def swap_faces(target_image, target_face, source_face):
    try:
        return swapper.get(target_image, target_face, source_face, paste_back=True)
    except Exception as e:
        st.error(f"Error during swaping: {e}")


def initialize_faceswap():
    """Lazily initialize FaceAnalysis and swapper to avoid heavy startup when unused."""
    global app, swapper
    if isinstance(app, FaceAnalysis) and hasattr(swapper, 'get'):
        return
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    download_model()
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', root=os.path.dirname(__file__))


def image_faceswap_app():
    st.title("Face Swapper for Image")
    initialize_faceswap()
    source_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
    target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
    if source_image and target_image:
        with st.spinner("Swapping... Please wait."):
            try:
                source_image = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), -1)
                target_image = cv2.imdecode(np.frombuffer(target_image.read(), np.uint8), -1)
                source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)                
                source_faces = app.get(source_image)
                source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
                if len(source_faces) == 0:
                    raise ValueError("No faces found in the source image.")
                source_face = source_faces[0]
                target_faces = app.get(target_image)
                target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
                if len(target_faces) == 0:
                    raise ValueError("No faces found in the target image.")
                target_face = target_faces[0]
                swapped_image = swap_faces(target_image, target_face, source_face)                
                message_placeholder = st.empty()
                message_placeholder.success("Swapped Successfully!")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.image(source_image, caption="Source Image", use_column_width=True)
                with col2:
                    st.image(target_image, caption="Target Image", use_column_width=True)
                with col3:
                    st.image(swapped_image, caption="Swapped Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error during image processing: {e}")


def process_video(source_img, video_path, output_video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        source_faces = app.get(source_img)
        source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
        if len(source_faces) == 0:
            raise ValueError("No faces found in the source image.")
        source_face = source_faces[0]
        progress_placeholder = st.empty()
        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            target_faces = app.get(frame)
            target_faces = sorted(target_faces, key=lambda x: x.bbox[0])
            if len(target_faces) > 0:
                frame = swap_faces(frame, target_faces[0], source_face)
            out.write(frame)
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time if elapsed_time > 0 else 0
            remaining_time_seconds = max(0, (total_frames - frame_count) / frames_per_second) if frames_per_second > 0 else 0
            remaining_minutes, remaining_seconds = divmod(remaining_time_seconds, 60)
            elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)
            progress_placeholder.text(
                f"Processed Frames: {frame_count}/{total_frames} | Elapsed Time: {int(elapsed_minutes)}m {int(elapsed_seconds)}s | Remaining Time: {int(remaining_minutes)}m {int(remaining_seconds)}s")
            frame_count += 1
        cap.release()
        out.release()
    except Exception as e:
        st.error(f"Error during video processing: {e}")


def video_faceswap_app():
    st.title("Face Swapper for Video")
    initialize_faceswap()
    source_image = st.file_uploader("Upload Source Face Image", type=["jpg", "jpeg", "png"])
    if source_image is not None:
        source_image = cv2.imdecode(np.frombuffer(source_image.read(), np.uint8), -1)
    target_video = st.file_uploader("Upload Target Video", type=["mp4"])
    if target_video is not None:
        temp_video = NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(target_video.read())
        output_video_path = os.path.splitext(temp_video.name)[0] + '_output.mp4'
        status_placeholder = st.empty()
        try:
            with st.spinner("Processing... This may take a while."):
                process_video(source_image, temp_video.name, output_video_path)
            status_placeholder.success("Processing complete!")
            st.subheader("Your video is ready:")
            st.video(output_video_path)
        except Exception as e:
            st.error(f"Error during video processing: {e}")


def eye_disease_detection_app():
    st.title("Eye Disease Detection (Swin Transformer)")

    device_choice = st.selectbox(
        "Select device",
        options=["auto", "cpu", "cuda"],
        index=0,
        help="'auto' uses CUDA if available else CPU",
    )
    device = (
        torch.device("cuda")
        if (device_choice == "cuda" or (device_choice == "auto" and torch.cuda.is_available()))
        else torch.device("cpu")
    )

    ckpt_file = st.file_uploader("Upload trained checkpoint (.pth)", type=["pth"])
    if ckpt_file is None:
        st.info("Upload a trained checkpoint to proceed.")
        return

    # Persist checkpoint to a temporary file for loading
    with NamedTemporaryFile(delete=False, suffix=".pth") as tmp_ckpt:
        tmp_ckpt.write(ckpt_file.read())
        ckpt_path = tmp_ckpt.name

    # Inspect checkpoint metadata
    ckpt = torch.load(ckpt_path, map_location=device)
    idx_to_class = ckpt.get("idx_to_class")
    image_size = int(ckpt.get("image_size", 224))
    model_name = ckpt.get("model_name", "swin_tiny_patch4_window7_224")
    num_classes = int(ckpt.get("num_classes", len(idx_to_class) if idx_to_class else 2))

    # Build and load model
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.to(device)
    model.eval()
    _ = load_checkpoint(model, ckpt_path, map_location=str(device))

    st.caption(f"Model: {model_name} | Image size: {image_size} | Classes: {num_classes}")

    uploaded_images = st.file_uploader(
        "Upload fundus image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_images:
        return

    eval_transform = get_transforms(image_size=image_size, augment=False)

    for img_file in uploaded_images:
        try:
            image_bytes = img_file.read()
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            tensor = eval_transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                top_prob, top_idx = probs.max(dim=0)

            classes = (
                [idx_to_class[i] for i in range(num_classes)] if idx_to_class else [str(i) for i in range(num_classes)]
            )

            st.subheader(img_file.name)
            cols = st.columns([1, 1])
            with cols[0]:
                st.image(pil_image, caption="Input", use_column_width=True)
            with cols[1]:
                st.metric("Predicted Class", classes[top_idx.item()], f"{top_prob.item():.2%}")
                st.write("Probabilities:")
                prob_table = {"class": classes, "prob": [float(p.item()) for p in probs]}
                st.dataframe(prob_table, hide_index=True)
        except Exception as e:
            st.error(f"Failed to process {img_file.name}: {e}")

def main():
    app_selection = st.sidebar.radio(
        "Select App",
        ("Image Face Swapping", "Video Face Swapping", "Eye Disease Detection"),
    )
    if app_selection == "Image Face Swapping":
        image_faceswap_app()
    elif app_selection == "Video Face Swapping":
        video_faceswap_app()
    elif app_selection == "Eye Disease Detection":
        eye_disease_detection_app()


if __name__ == "__main__":
    main()
