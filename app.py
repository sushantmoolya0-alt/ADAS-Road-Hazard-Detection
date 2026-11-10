import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import subprocess
from PIL import Image
from collections import Counter
from ultralytics import YOLO

st.set_page_config(
    page_title="ADAS Road Hazard Detection",
    page_icon="🚙",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("/content/best.pt")

model = load_model()

def get_severity(box, image_shape):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    height, width = image_shape[:2]
    area_ratio = ((x2-x1) * (y2-y1)) / (height * width)

    if area_ratio < 0.01:
        return "LOW"
    elif area_ratio < 0.04:
        return "MEDIUM"
    else:
        return "HIGH"

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose Module",
    ["Home", "Image Detection", "Video Detection"]
)

st.title("🚙 ADAS - Intelligent Road Hazard Detection")
st.caption("AI-powered pothole and road damage detection system")


if page == "Home":

    st.markdown("""
    <div style="background:linear-gradient(120deg,#10243A,#176B87);
                padding:30px;border-radius:15px;color:white;">
        <h1>🚗 Intelligent Road Hazard Detection</h1>
        <h3>AI-powered Advanced Driver Assistance System</h3>
        <p>Detect road damage, analyze hazards and generate safety warnings
        using deep learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.success("🟢 AI Detection Model: Loaded Successfully")

    st.subheader("📊 System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Detection Engine", "YOLOv8")
    col2.metric("Input Modes", "Image + Video")
    col3.metric("Alert System", "Active")

    st.divider()

    st.subheader("🚧 Core Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Image Detection")
        st.write("Upload road images and identify road damage.")
        st.markdown("### 🎯 Confidence Analysis")
        st.write("View the confidence score for each detection.")

    with col2:
        st.markdown("### 🎥 Video Detection")
        st.write("Analyze road videos frame by frame.")
        st.markdown("### ⚠️ Hazard Warnings")
        st.write("Receive hazard alerts and approximate severity estimates.")

    st.divider()

    st.subheader("🧠 Technology Stack")

    st.write(
        "**Deep Learning:** YOLOv8  |  "
        "**Programming:** Python  |  "
        "**Interface:** Streamlit  |  "
        "**Computer Vision:** OpenCV"
    )

    st.info(
        "Select Image Detection or Video Detection from the sidebar "
        "to start analyzing road hazards."
    )

    st.caption("ADAS | Intelligent Road Hazard Detection System")


elif page == "Image Detection":
    st.header("Image Detection")

    uploaded_image = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"]
    )

    confidence = st.slider(
        "Confidence Threshold", 0.1, 1.0, 0.25, 0.05
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Road Image",
                 use_container_width=True)

        if st.button("Detect Road Hazards"):
            with st.spinner("Analyzing image..."):
                results = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]
                annotated = result.plot()
                annotated = cv2.cvtColor(
                    annotated, cv2.COLOR_BGR2RGB
                )

                st.subheader("Detection Result")
                st.image(annotated, use_container_width=True)

                if len(result.boxes) == 0:
                    st.success("No road hazards detected.")
                else:
                    st.error(
                        f"⚠️ WARNING: {len(result.boxes)} "
                        "road hazard(s) detected!"
                    )

                    for box in result.boxes:
                        hazard = model.names[int(box.cls[0])]
                        score = float(box.conf[0])
                        severity = get_severity(
                            box, result.orig_shape
                        )

                        st.write(f"**Hazard:** {hazard}")
                        st.write(f"**Confidence:** {score:.2%}")
                        st.write(
                            f"**Estimated severity:** {severity}"
                        )

                        if severity == "HIGH":
                            st.error("🚨 HIGH SEVERITY WARNING")
                        elif severity == "MEDIUM":
                            st.warning("⚠️ MEDIUM SEVERITY WARNING")
                        else:
                            st.info("LOW SEVERITY")

                        st.divider()

elif page == "Video Detection":
    st.header("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    confidence = st.slider(
        "Video Confidence Threshold",
        0.1, 1.0, 0.25, 0.05
    )

    if uploaded_video:
        if st.button("Analyze Road Video"):
            input_path = None
            output_path = None

            try:
                suffix = os.path.splitext(
                    uploaded_video.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_input:
                    temp_input.write(uploaded_video.read())
                    input_path = temp_input.name

                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if not fps or total_frames == 0:
                    st.error("Could not read this video.")
                else:
                    output_path = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp4"
                    ).name

                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height)
                    )

                    hazard_counts = Counter()
                    highest_severity = "LOW"
                    severity_rank = {
                        "LOW": 1, "MEDIUM": 2, "HIGH": 3
                    }

                    progress = st.progress(0)
                    status = st.empty()

                    results = model.predict(
                        source=input_path,
                        conf=confidence,
                        stream=True,
                        verbose=False
                    )

                    for frame_number, result in enumerate(results, 1):
                        frame = result.plot()

                        for box in result.boxes:
                            hazard = model.names[int(box.cls[0])]
                            severity = get_severity(
                                box, result.orig_shape
                            )

                            hazard_counts[hazard] += 1

                            if severity_rank[severity] > severity_rank[
                                highest_severity
                            ]:
                                highest_severity = severity

                            if severity == "HIGH":
                                warning = "HIGH SEVERITY WARNING"
                            elif severity == "MEDIUM":
                                warning = "MEDIUM SEVERITY WARNING"
                            else:
                                warning = "LOW SEVERITY"

                            cv2.putText(
                                frame,
                                warning,
                                (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                        writer.write(frame)

                        progress.progress(
                            min(frame_number / total_frames, 1.0)
                        )
                        status.write(
                            f"Processing frame {frame_number} "
                            f"of {total_frames}"
                        )

                    writer.release()

                    st.success("Video analysis completed!")

                    h264_path = output_path.replace(
                        ".mp4", "_h264.mp4"
                    )

                    conversion = subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", output_path,
                            "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p",
                            h264_path
                        ],
                        capture_output=True
                    )

                    final_video = (
                        h264_path if conversion.returncode == 0
                        else output_path
                    )

                    st.subheader("Processed Video")
                    st.video(final_video)

                    with open(final_video, "rb") as video_file:
                        st.download_button(
                            "Download Processed Video",
                            data=video_file,
                            file_name="ADAS_Detection_Result.mp4",
                            mime="video/mp4"
                        )

                    st.subheader("Detection Summary")

                    total_hazards = sum(hazard_counts.values())

                    st.metric("Total detections across frames",
                              total_hazards)

                    if total_hazards == 0:
                        st.success("No road hazards detected.")
                    else:
                        st.error(
                            f"⚠️ WARNING: {total_hazards} "
                            "hazard detections across video frames."
                        )

                        st.write(
                            f"**Highest estimated severity:** "
                            f"{highest_severity}"
                        )

                        for hazard, count in hazard_counts.items():
                            st.write(f"**{hazard}:** {count} detections")

            except Exception as e:
                st.error(f"Video processing error: {e}")

            finally:
                if input_path and os.path.exists(input_path):
                    os.remove(input_path)
