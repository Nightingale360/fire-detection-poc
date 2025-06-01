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
import random

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
    /* Full-page background image */
    .reportview-container .main .block-container {
        background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT-fQB5IIJEzGvMBfgU2Gyo6R7zaRhYQMwMng&s');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Optional: a semi-transparent overlay to improve contrast */
    .reportview-container .main .block-container::before {
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        background-color: rgba(0, 0, 0, 0.4);
        z-index: -1;
    }
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

LOG_CSV = ROOT / "detections.csv"
if not LOG_CSV.exists():
    pd.DataFrame(columns=["timestamp","drone","confidence","lat","lon","status"]).to_csv(LOG_CSV, index=False)

#Model Config
MODEL_DIR = ROOT/'weights'
# DETECTION_MODEL = MODEL_DIR/'yolo11l.pt'
DETECTION_MODEL = MODEL_DIR/'best.pt'

LOG_INTERVAL = datetime.timedelta(seconds=30)
LAST_LOG = datetime.datetime.min
TARGET_CLASS = "fire"

# base lat/lon for mocking
BASE_LAT, BASE_LON = -33.5, 151.2
MAX_JITTER = 0.05
input_type = ""


def list_cameras(max_idx=5, backend=cv2.CAP_ANY):
    valid = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            valid.append(i)
            cap.release()
    return valid


def open_first_working_camera(indices):
    for i in indices:
        cap = cv2.VideoCapture(i, cv2.CAP_ANY)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
    return None


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
        index=2,
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

        if not st.button("▶️ Connect & Start Camera Feed"):
            st.info("Click ▶️ to connect to your webcam")
            st.stop()

        st.info("🔴 Streaming from camera…")
        available = list_cameras()
        if not available:
            st.error("No webcams detected! Please connect a camera and refresh.")
            st.stop()

        cam_index = st.sidebar.selectbox("Camera index", available, index=0)

        video_cap = open_first_working_camera(available)
        st.write(f"Using camera index {cam_index}")

        st_frame = st.empty()
        notice_slot = st.empty()
        last_log = datetime.datetime.now() - LOG_INTERVAL

        # 4️⃣ Main streaming & logging loop
        try:
            while video_cap.isOpened():
                success, frame = video_cap.read()
                if not success:
                    break

                results = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=640,
                    device=device
                )
                targets = results[0].boxes
                now = datetime.datetime.now()

                if len(targets) and (now - last_log) >= LOG_INTERVAL:
                    for conf, cls in zip(targets.conf, targets.cls):
                        cls_name = model.names[int(cls.cpu().numpy())].lower()
                        if cls_name == TARGET_CLASS:
                            score = float(conf.cpu().numpy())

                            # render annotated frame once
                            out = results[0].plot()
                            st_frame.image(out, channels="BGR", use_container_width=True)

                            # mock location string
                            lat = BASE_LAT + random.uniform(-MAX_JITTER, MAX_JITTER)
                            lon = BASE_LON + random.uniform(-MAX_JITTER, MAX_JITTER)

                            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                            notice_slot.success(
                                f"🚨 Detected **{TARGET_CLASS}** by Drone D1 at {timestamp} (conf={score:.2f})")
                            drone = "D-1"
                            status = "ACTIVE"
                            # append only timestamp, class, confidence, location, status
                            with open(LOG_CSV, "a") as f:
                                f.write(f"{timestamp},{drone},{score:.2f},{lat:.5f}, {lon:.5f},{status}\n")

                            last_log = now
                            break
                annotated = results[0].plot()
                st_frame.image(annotated, channels="BGR", use_container_width=True)

        finally:
            video_cap.release()
            st.info("🔴 Streaming stopped.")

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
            skip_seconds = (5 * 60) + 5
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
                    now = datetime.datetime.now()

                    if len(targets) and (now - LAST_LOG) >= LOG_INTERVAL:
                        logged = False
                        for conf, cls in zip(targets.conf, targets.cls):
                            cls_name = model.names[int(cls.cpu().numpy())].lower()
                            if cls_name == TARGET_CLASS:
                                score = float(conf.cpu().numpy())

                                # render annotated frame once
                                out = results[0].plot()
                                st_frame.image(out, channels="BGR", use_container_width=True)

                                # mock location string
                                lat = BASE_LAT + random.uniform(-MAX_JITTER, MAX_JITTER)
                                lon = BASE_LON + random.uniform(-MAX_JITTER, MAX_JITTER)

                                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                                notice_slot.success(f"🚨 Detected **{TARGET_CLASS}** by Drone D1 at {timestamp} (conf={score:.2f})")
                                drone = "D-1"
                                status = "ACTIVE"
                                # append only timestamp, class, confidence, location, status
                                with open(LOG_CSV, "a") as f:
                                    f.write(f"{timestamp},{drone},{score:.2f},{lat:.5f}, {lon:.5f},{status}\n")

                                LAST_LOG = now
                                break

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
    df_log = pd.read_csv(LOG_CSV)
    # We log a single “location” column
    if "location" in df_log.columns:
        df_log["Location"] = df_log["location"]
    else:
        df_log["Location"] = "Unknown"

    recent = df_log.sort_values("timestamp", ascending=False).head(20)

    st.table(
        recent[["timestamp", "drone", "confidence", "lat", "lon", "status"]]
        .rename(columns={
            "timestamp": "Time",
            "drone": "Drone",
            "confidence": "Confidence",
            "lat": "Latitude",
            "lon": "Longitude",
            "status": "Status"
        })
    )

# 5️⃣ MAP SIMULATION (only on Deployed Drones or always show)
if page in ("Deployed Drones", "Surveillance", "Alerts") or input_type in ("Live Camera Feed"):
    st.markdown("### 🌏 Drone Patrol Map")
    n_drones = 5
    lat0, lon0 = -33.5, 150.2
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
