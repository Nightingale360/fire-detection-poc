import torch


import cv2
from pathlib import Path
import sys
from ultralytics import YOLO
from PIL import Image
import pydeck as pdk
import pandas as pd
import numpy as np
import time
import datetime
import streamlit as st

# 1️⃣ PAGE CONFIG & GLOBAL STYLES
st.set_page_config(
    page_title="Alpha Firewatch",
    page_icon="🔥👁️⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* ---- Hide Streamlit header/footer ---- */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Body background ---- */
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* ---- Title styling ---- */
    .title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #E03E2D;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    /* ---- Sidebar styling ---- */
    .sidebar .sidebar-content {
        background-color: #F7F7F7;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .sidebar .sidebar-content h6 {
        color: #E03E2D;
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── LOGIN STATE ────────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

login_slot = st.empty()

# ─── LOGIN FORM ─────────────────────────────────────────────────────────────────
with login_slot.container():
    if not st.session_state.logged_in:
        st.title("🔐 Login to Alpha Firewatch")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        login_clicked = st.button("Log in", key="login_btn")

        if login_clicked:
            if password == "password":
                st.session_state.logged_in = True
                st.success(f"Welcome, {username}! Redirecting…")
            else:
                st.error("❌ Invalid password.")

# If still not logged in, stop here (login form remains visible)
if not st.session_state.logged_in:
    st.stop()

# Once logged in, clear the login form
login_slot.empty()

# Pick GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FILE = Path(__file__).resolve()
ROOT = FILE.parent

if ROOT not in sys.path:
    sys.path.append(str(ROOT))

ROOT = ROOT.relative_to(Path.cwd())

# Sources
UPLOAD = "Upload"
FEED = "Feed"

SOURCES_LIST = [UPLOAD, FEED]
UPLOAD_DIR = ROOT/'Upload'

UPLOAD_DICT = {
    'Video 1': UPLOAD_DIR/'fire_video_1_low.mp4',
    'Video 2': UPLOAD_DIR/'fire_video_1_mid.mp4',
    'Video 3': UPLOAD_DIR/'fire_video_1_hd.mp4'
}

#Model Config
MODEL_DIR = ROOT/'weights'
DETECTION_MODEL = MODEL_DIR/'yolo11l.pt'

try:
    model = YOLO(DETECTION_MODEL)
except Exception as e:
    st.error(f"Unable to load model. Check the specified path:\n`{DETECTION_MODEL}`")
    st.error(str(e))
    st.stop()  # stop execution if the model can't be loaded




# Custom CSS


# 2️⃣ SIDEBAR NAV & SETTINGS
# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Dashboard", unsafe_allow_html=True)
    # give the radio a real label (you can hide it if you don't want to show it)
    page = st.radio(
        label="Navigate to",
        options=("Deployed Drones", "Surveillance", "Alerts"),
        index=1,
        label_visibility="collapsed"  # hides the label visually but keeps it for accessibility
    )
    st.markdown("---")
    st.markdown("### 🔍 Confidence Threshold")
    confidence_pct = st.slider(
        "Select model confidence (%)",
        25, 100, 40, 1
    )
    confidence = confidence_pct / 100.0

# 3️⃣ HEADER
st.markdown('<div class="title">Alpha Firewatch</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time fire detection & drone monitoring</div>', unsafe_allow_html=True)

# 4️⃣ PAGE CONTENT
if page == "Deployed Drones":
    st.subheader("🚁 Deployed Drones")
    st.write("**Active drone fleet — live positions & status**")
    # placeholder for drone table or metrics
    st.dataframe(
        pd.DataFrame({
            "Drone ID": [f"D-{i}" for i in range(1,6)],
            "Battery (%)": np.random.randint(50,100,5),
            "Status": ["OK"]*5
        }).set_index("Drone ID")
    )

elif page == "Surveillance":
    st.subheader("📡 Surveillance")
    input_type = st.radio(
        "Input source:",
        ("Live Camera Feed", "Upload Video File")
    )
    video_path = None
    if input_type == "Live Camera Feed":
        st.info("🔴 Streaming from camera…")
        # your video loop here
    elif input_type == "Upload Video File":
        source_video = st.sidebar.selectbox(
            "Choose a Video…", list(UPLOAD_DICT.keys())
        )

        video_path = UPLOAD_DICT[source_video]
        if video_path is None:
            DEFAULT_VIDEO = UPLOAD_DICT["Video 1"]

        st.video(str(video_path))

        if st.sidebar.button("Detect Video Objects"):

            video_cap = cv2.VideoCapture(str(video_path))
            st_frame = st.empty()

            fps = video_cap.get(cv2.CAP_PROP_FPS) or 30  # fallback to 30 if unavailable
            skip_seconds = 3 * 60  # 5 minutes
            skip_frames = int(fps * skip_seconds)

            frame_idx = 0
            notice_slot = st.empty()
            try:
                while video_cap.isOpened():
                    success, frame = video_cap.read()

                    frame_idx += 1
                    if frame_idx <= skip_frames:
                        continue

                    if not success:
                        break

                    # YOLO inference
                    results = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=640,  # or whatever image size your model expects
                        device=device  # if you need to specify GPU/CPU
                    )
                    targets = results[0].boxes

                    if len(targets) > 0:
                        # we got a detection—grab its info
                        box = targets.xyxy[0].cpu().numpy().tolist()
                        score = float(targets.conf[0].cpu().numpy())
                        cls_idx = int(targets.cls[0].cpu().numpy())
                        cls_name = model.names[cls_idx]

                        # show it on the frame
                        out = results[0].plot()
                        st_frame.image(out, channels="BGR", use_container_width=True)

                        # LOG & NOTIFY ONCE
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        msg = f"🚨 Detected **{cls_name}** at **{timestamp}** (conf={score:.2f})"
                        notice_slot.success(msg)
                        detected = True
                    else:
                        notice_slot.info("No detection on this frame")

                    result_plotted = results[0].plot()  # BGR image with boxes drawn

                    # Display at full column width; Streamlit will scale it
                    st_frame.image(
                        result_plotted,
                        caption="Detected Video",
                        channels="BGR",
                        use_container_width=True
                    )
            except Exception as e:
                st.sidebar.error("Error running detection on video:")
                st.sidebar.error(str(e))
            finally:
                video_cap.release()

    st.markdown(f"**Confidence threshold:** {confidence:.2f}")

elif page == "Alerts":
    st.subheader("🚨 My Alerts")
    st.write("**Past fire detections & locations**")
    num_rows = 5
    statuses = ["ACTIVE" if i % 2 == 0 else "RESOLVED" for i in range(num_rows)]
    st.table(pd.DataFrame({
        "Timestamp": pd.date_range("2025-05-20", periods=num_rows, freq="6h"),  # lowercase 'h'
        "Location": ["NSW, Australia"]*num_rows,
        "Status": statuses,
        "Confidence": np.round(np.random.uniform(0.4, 0.9, 5), 2)
    }))

# 5️⃣ MAP SIMULATION (only on Deployed Drones or always show)
if page in ("Deployed Drones", "Surveillance"):
    st.markdown("### 🌏 Drone Patrol Map")
    n_drones = 5
    lat0, lon0 = -32.5, 149.5
    lats = lat0 + (np.random.rand(n_drones)-0.5)*2.0
    lons = lon0 + (np.random.rand(n_drones)-0.5)*2.0
    df = pd.DataFrame({"lat": lats, "lon": lons})

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["lon","lat"],
        get_radius=5000,
        get_fill_color=[224, 62, 45, 200],
        pickable=True
    )
    view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=6)
    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9")
    map_slot = st.empty()
    map_slot.pydeck_chart(deck)

    # animate
    for _ in range(30):
        df["lat"] += (np.random.rand(n_drones)-0.5)*0.01
        df["lon"] += (np.random.rand(n_drones)-0.5)*0.01
        deck.layers[0].data = df
        map_slot.pydeck_chart(deck)
        time.sleep(0.1)


st.sidebar.header("Video Upload")
