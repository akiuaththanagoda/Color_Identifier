import base64
from datetime import datetime
import io
import math
import time
import cv2
import mysql.connector
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(
    page_title="Ocean Lanka - AI Color Precision Suite",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
<style>
    .main { background-color: #0E1117; }
    .stApp { max-width: 100%; }
    .metric-card {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
    }
    .status-badge-a { background-color: #059669; color: white; padding: 6px 12px; border-radius: 20px; font-weight: bold; }
    .status-badge-b { background-color: #D97706; color: white; padding: 6px 12px; border-radius: 20px; font-weight: bold; }
    .status-badge-c { background-color: #DC2626; color: white; padding: 6px 12px; border-radius: 20px; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)


# --- MySQL Database Helpers ---
def get_db_connection():
  """Establish Connection to MySQL Database"""
  try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ocean_quality_db",
        port=3306,
    )
    return conn
  except Exception as e:
    st.error(f"❌ Database Connection Error: {e}")
    return None


def save_scan_to_mysql(inspector, batch, roll, pos, lab, delta_e):
  """Inserts Inspection Data directly into MySQL Table"""
  conn = get_db_connection()
  if conn:
    try:
      cursor = conn.cursor()
      query = """
            INSERT INTO color_inspections 
            (inspector_name, batch_no, roll_no, scan_position, lab_L, lab_a, lab_b, ciede2000_delta_e) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
      values = (inspector, batch, roll, pos, lab[0], lab[1], lab[2], delta_e)
      cursor.execute(query, values)
      conn.commit()
      cursor.close()
      conn.close()
      return True
    except Exception as e:
      st.error(f"❌ Database Insertion Failed: {e}")
      return False
  return False


# --- Voice Output Helper ---
def speak_voice(text):
  """Synthesizes Speech via Browser Speech API"""
  js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 0.95;
        msg.pitch = 1.0;
        window.speechSynthesis.speak(msg);
    </script>
    """
  components.html(js_code, height=0, width=0)


# --- Session State & Authentication Gate ---
if "authenticated" not in st.session_state:
  st.session_state.authenticated = False
if "user_name" not in st.session_state:
  st.session_state.user_name = "Inspector"


def login_gate():
  if not st.session_state.authenticated:
    st.markdown(
        "<h2 style='text-align: center; color: #38BDF8;'>🏭 Ocean Lanka AI"
        " Vision System</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h5 style='text-align: center; color: #94A3B8;'>Enterprise"
        " Zero-Hardware Color Identification Suite</h5>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
      with st.form("login_form"):
        user_name_input = st.text_input("👤 Operator Name / User ID:")
        password_input = st.text_input("🔒 System Password:", type="password")
        submit_btn = st.form_submit_button(
            "🔐 Authenticate & Launch System", use_container_width=True
        )

        if submit_btn:
          if (
              password_input == "Ocean2026"
              and user_name_input.strip() != ""
          ):
            st.session_state.authenticated = True
            st.session_state.user_name = user_name_input.strip()

            current_hour = datetime.now().hour
            greeting = (
                "Good Morning"
                if current_hour < 12
                else ("Good Afternoon" if current_hour < 17 else "Good Evening")
            )

            welcome_msg = (
                f"{greeting} {st.session_state.user_name}. Welcome to Ocean"
                " Lanka Color Identification System. All quality modules are"
                " initialized."
            )
            speak_voice(welcome_msg)
            st.success(f"Welcome, {st.session_state.user_name}!")
            st.rerun()
          else:
            st.error("❌ Invalid Password or User Name missing!")
      return False
  return True


if not login_gate():
  st.stop()

# Initialize Batch Session State Data Structure
if "batch_data" not in st.session_state:
  st.session_state.batch_data = {}


# --- Color Precision Math ---
def rgb_to_lab(rgb):
  """Precision RGB to CIE L*a*b* Standard Transformation"""
  r, g, b = [x / 255.0 for x in rgb]
  r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
  g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
  b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

  x = (r * 0.4124 + g * 0.3576 + b * 0.1805) * 100 / 95.047
  y = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 100 / 100.000
  z = (r * 0.0193 + g * 0.1192 + b * 0.9505) * 100 / 108.883

  fx = x ** (1 / 3) if x > 0.008856 else (7.787 * x) + (16 / 116)
  fy = y ** (1 / 3) if y > 0.008856 else (7.787 * y) + (16 / 116)
  fz = z ** (1 / 3) if z > 0.008856 else (7.787 * z) + (16 / 116)

  L = (116 * fy) - 16
  a = 500 * (fx - fy)
  b_val = 200 * (fy - fz)
  return round(L, 3), round(a, 3), round(b_val, 3)


def calculate_ciede2000(lab1, lab2):
  """High Accuracy CIEDE2000 Delta E Calculation"""
  L1, a1, b1 = lab1
  L2, a2, b2 = lab2

  avg_L = (L1 + L2) / 2.0
  C1 = math.sqrt(a1**2 + b1**2)
  C2 = math.sqrt(a2**2 + b2**2)
  avg_C = (C1 + C2) / 2.0

  G = 0.5 * (1 - math.sqrt(avg_C**7 / (avg_C**7 + 25**7))) if avg_C > 0 else 0
  a1_prime = (1 + G) * a1
  a2_prime = (1 + G) * a2

  C1_prime = math.sqrt(a1_prime**2 + b1**2)
  C2_prime = math.sqrt(a2_prime**2 + b2**2)
  avg_C_prime = (C1_prime + C2_prime) / 2.0

  h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
  h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360

  if abs(h1_prime - h2_prime) <= 180:
    avg_h_prime = (h1_prime + h2_prime) / 2.0
  else:
    avg_h_prime = (
        (h1_prime + h2_prime + 360) / 2.0
        if (h1_prime + h2_prime) < 360
        else (h1_prime + h2_prime - 360) / 2.0
    )

  T = (
      1
      - 0.17 * math.cos(math.radians(avg_h_prime - 30))
      + 0.24 * math.cos(math.radians(2 * avg_h_prime))
      + 0.32 * math.cos(math.radians(3 * avg_h_prime + 6))
      - 0.20 * math.cos(math.radians(4 * avg_h_prime - 63))
  )

  delta_h_prime = h2_prime - h1_prime
  if abs(delta_h_prime) > 180:
    delta_h_prime -= 360 if h2_prime > h1_prime else -360

  delta_L_prime = L2 - L1
  delta_C_prime = C2_prime - C1_prime
  delta_H_prime = (
      2
      * math.sqrt(C1_prime * C2_prime)
      * math.sin(math.radians(delta_h_prime / 2.0))
  )

  S_L = 1 + (
      (0.015 * ((avg_L - 50) ** 2)) / math.sqrt(20 + ((avg_L - 50) ** 2))
  )
  S_C = 1 + 0.045 * avg_C_prime
  S_H = 1 + 0.015 * avg_C_prime * T

  delta_ro = 30 * math.exp(-(((avg_h_prime - 275) / 25) ** 2))
  R_C = 2 * math.sqrt(avg_C_prime**7 / (avg_C_prime**7 + 25**7))
  R_T = -math.sin(math.radians(2 * delta_ro)) * R_C

  de2000 = math.sqrt(
      (delta_L_prime / S_L) ** 2
      + (delta_C_prime / S_C) ** 2
      + (delta_H_prime / S_H) ** 2
      + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
  )
  return round(de2000, 3)


def get_quality_grade(avg_de, max_de):
  """Enterprise Grade Categorization"""
  if avg_de <= 0.75 and max_de <= 1.0:
    return "Grade A (Premium Pass)", "status-badge-a", "🟢"
  elif avg_de <= 1.20 and max_de <= 1.5:
    return "Grade B (Standard Pass)", "status-badge-b", "🟡"
  else:
    return "Grade C (Reject / Out of Spec)", "status-badge-c", "🔴"


# --- Header & Live Clock ---
h_col1, h_col2 = st.columns([2.5, 1.5])

with h_col1:
  st.title("🎨 Ocean Lanka - AI Precision Color Identifier")
  st.caption(
      f"👤 Authenticated Inspector: **{st.session_state.user_name}** |"
      " Autonomous Vision System"
  )

with h_col2:
  clock_html = """
    <div style="background: #1E293B; padding: 10px 15px; border-radius: 10px; border: 1px solid #0EA5E9; text-align: center;">
        <span style="font-size: 12px; color: #94A3B8; font-weight: bold;">🇱🇰 SRI LANKA REAL-TIME</span><br>
        <span id="live_clock" style="font-size: 16px; font-weight: bold; color: #38BDF8;">--:--:--</span>
    </div>
    <script>
        function updateClock() {
            var now = new Date();
            var options = { timeZone: 'Asia/Colombo', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true };
            var formatter = new Intl.DateTimeFormat('en-US', options);
            document.getElementById('live_clock').innerHTML = formatter.format(now);
        }
        setInterval(updateClock, 1000);
        updateClock();
    </script>
    """
  components.html(clock_html, height=80)

st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.markdown("### ⚙️ Control Panel")
nav_choice = st.sidebar.radio(
    "Navigation Module:",
    [
        "🔍 Real-time Inspection",
        "📊 Batch Quality Analytics",
        "📜 Live MySQL Database Records",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Active Production Context")
batch_no = st.sidebar.text_input("Batch Number:", value="BATCH-9001").upper()
roll_no = st.sidebar.text_input("Roll Number:", value="ROLL-01").upper()

if batch_no not in st.session_state.batch_data:
  st.session_state.batch_data[batch_no] = {"rolls": {}}
if roll_no not in st.session_state.batch_data[batch_no]["rolls"]:
  st.session_state.batch_data[batch_no]["rolls"][roll_no] = []

if st.sidebar.button("🚪 Logout User"):
  st.session_state.authenticated = False
  st.rerun()

# --- Module 1: Real-Time Inspection ---
if nav_choice == "🔍 Real-time Inspection":
  st.subheader(f"📍 Active Scan Point: Batch [{batch_no}] ➔ Roll [{roll_no}]")

  col_cam, col_controls = st.columns([2, 1])

  with col_controls:
    st.markdown("#### 📷 Input Method")
    cam_source = st.radio(
        "Select Source:",
        [
            "Direct Mobile Camera (Flash/HD)",
            "Standard Camera Input",
            "Upload Image File",
        ],
    )
    position = st.selectbox(
        "Scan Position:",
        [
            "Start (0m)",
            "Middle (50m)",
            "End (100m)",
            "Edge-Left",
            "Center-Point",
            "Edge-Right",
            "Custom Meter Point",
        ],
    )

    if position == "Custom Meter Point":
      position = st.text_input("Type Distance (e.g., 25m):", value="25m")

    st.markdown("---")
    st.markdown("#### 🛠️ Image Processing Filters")
    enable_denoise = st.checkbox(
        "Denoise Image (Remove Grain)", value=True
    )
    enable_sharpen = st.checkbox(
        "Sharpen Image (Enhance Texture)", value=True
    )

  with col_cam:
    captured_img = None

    if cam_source == "Direct Mobile Camera (Flash/HD)":
      # HTML5 / JS Camera Component with High Res & Flash Control
      camera_html = """
            <div style="text-align: center; font-family: Arial, sans-serif;">
                <video id="video" autoplay playsinline style="width: 100%; max-width: 500px; border-radius: 10px; border: 2px solid #0EA5E9;"></video>
                <br><br>
                <button id="flashBtn" onclick="toggleFlash()" style="padding: 10px 18px; font-size: 15px; margin-right: 10px; border-radius: 8px; background-color: #EAB308; color: black; font-weight: bold; border: none; cursor: pointer;">💡 Toggle Flash</button>
                <button id="captureBtn" onclick="takeSnapshot()" style="padding: 10px 18px; font-size: 15px; border-radius: 8px; background-color: #10B981; color: white; font-weight: bold; border: none; cursor: pointer;">📸 Capture Photo</button>
                <canvas id="canvas" style="display:none;"></canvas>
            </div>

            <script>
            let videoStream = null;
            let track = null;
            let flashOn = false;

            const constraints = {
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 3840, max: 3840 },
                    height: { ideal: 2160, max: 2160 }
                }
            };

            async function startCamera() {
                try {
                    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
                    const video = document.getElementById('video');
                    video.srcObject = videoStream;
                    track = videoStream.getVideoTracks()[0];
                } catch (err) {
                    console.error("Camera access error:", err);
                }
            }

            async function toggleFlash() {
                if (!track) return;
                const capabilities = track.getCapabilities();
                if (capabilities.torch) {
                    flashOn = !flashOn;
                    await track.applyConstraints({
                        advanced: [{ torch: flashOn }]
                    });
                } else {
                    alert("Torch/Flash is not supported on this camera/browser.");
                }
            }

            function takeSnapshot() {
                const video = document.getElementById('video');
                const canvas = document.getElementById('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
                window.parent.postMessage({type: "streamlit:setComponentValue", value: dataUrl}, "*");
            }

            startCamera();
            </script>
            """
      image_data = components.html(camera_html, height=430)

      if image_data:
        header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        captured_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    elif cam_source == "Standard Camera Input":
      photo = st.camera_input("Take a photo of fabric surface")
      if photo:
        image = Image.open(photo)
        captured_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    else:
      file_up = st.file_uploader(
          "Select Fabric Image", type=["jpg", "jpeg", "png"]
      )
      if file_up:
        image = Image.open(file_up)
        captured_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if captured_img is not None:
      # Applying OpenCV Filters for Clarity
      if enable_denoise:
        captured_img = cv2.fastNlMeansDenoisingColored(
            captured_img, None, 10, 10, 7, 21
        )

      if enable_sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        captured_img = cv2.filter2D(captured_img, -1, kernel)

      img_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
      h, w, _ = img_rgb.shape
      roi = img_rgb[int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)]
      mean_rgb = cv2.mean(roi)[:3]
      current_lab = rgb_to_lab(mean_rgb)

      st.image(
          img_rgb,
          caption=f"Captured Surface - {position}",
          use_container_width=True,
      )
      st.info(
          f"📊 Extracted CIELAB Parameters: L*={current_lab[0]},"
          f" a*={current_lab[1]}, b*={current_lab[2]}"
      )

      if st.button("💾 Save Scan Point to MySQL", use_container_width=True):
        current_scans = st.session_state.batch_data[batch_no]["rolls"][roll_no]
        base_lab = (
            current_scans[0]["lab"] if len(current_scans) > 0 else current_lab
        )
        delta_e_calc = calculate_ciede2000(base_lab, current_lab)

        db_success = save_scan_to_mysql(
            st.session_state.user_name,
            batch_no,
            roll_no,
            position,
            current_lab,
            delta_e_calc,
        )

        scan_data = {
            "position": position,
            "lab": current_lab,
            "rgb": mean_rgb,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.batch_data[batch_no]["rolls"][roll_no].append(
            scan_data
        )

        if db_success:
          speak_voice(
              f"Scan point saved to database for position {position}."
          )
          st.success(
              f"✅ Position [{position}] saved directly to MySQL Database!"
          )
          time.sleep(1)
          st.rerun()

  current_scans = st.session_state.batch_data[batch_no]["rolls"][roll_no]
  if len(current_scans) > 0:
    st.markdown("---")
    st.subheader(f"📈 Real-time Roll Variation Analysis ({roll_no})")

    df_roll = pd.DataFrame(current_scans)
    base_lab = df_roll.iloc[0]["lab"]

    df_roll["Delta_E_Base"] = df_roll["lab"].apply(
        lambda x: calculate_ciede2000(base_lab, x)
    )

    avg_de = df_roll["Delta_E_Base"].mean()
    max_de = df_roll["Delta_E_Base"].max()
    std_de = (
        df_roll["Delta_E_Base"].std() if len(df_roll) > 1 else 0.0
    )
    grade_text, badge_class, icon = get_quality_grade(avg_de, max_de)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Average ΔE2000", f"{avg_de:.3f}")
    m2.metric("Max Variation", f"{max_de:.3f}")
    m3.metric("Standard Deviation (σ)", f"{std_de:.3f}")
    m4.markdown(
        "**Roll Quality Grade**<br><span"
        f" class='{badge_class}'>{icon} {grade_text}</span>",
        unsafe_allow_html=True,
    )

    fig = px.line(
        df_roll,
        x="position",
        y="Delta_E_Base",
        markers=True,
        title=f"Color Variation Profile Across Roll ({roll_no})",
        labels={"Delta_E_Base": "CIEDE2000 (ΔE)", "position": "Position"},
        template="plotly_dark",
    )
    fig.add_hline(
        y=0.75,
        line_dash="dash",
        line_color="#EAB308",
        annotation_text="Warning Threshold (0.75)",
    )
    fig.add_hline(
        y=1.20,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Tolerance Limit (1.20)",
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Module 2: Batch Analytics ---
elif nav_choice == "📊 Batch Quality Analytics":
  st.subheader(f"📦 Overall Batch Color Uniformity Dashboard [{batch_no}]")

  all_rolls = st.session_state.batch_data[batch_no]["rolls"]
  batch_records = []

  for r_id, scans in all_rolls.items():
    if len(scans) > 0:
      first_lab = scans[0]["lab"]
      for s in scans:
        de_val = calculate_ciede2000(first_lab, s["lab"])
        batch_records.append({
            "Roll Number": r_id,
            "Position": s["position"],
            "L*": s["lab"][0],
            "a*": s["lab"][1],
            "b*": s["lab"][2],
            "Delta E": de_val,
        })

  if len(batch_records) > 0:
    df_batch = pd.DataFrame(batch_records)

    c1, c2 = st.columns(2)
    with c1:
      fig_box = px.box(
          df_batch,
          x="Roll Number",
          y="Delta E",
          color="Roll Number",
          title="Batch Delta E Dispersion Across Rolls",
          template="plotly_dark",
      )
      st.plotly_chart(fig_box, use_container_width=True)

    with c2:
      fig_bar = px.bar(
          df_batch,
          x="Position",
          y="Delta E",
          color="Roll Number",
          barmode="group",
          title="Position-wise Delta E Comparison",
          template="plotly_dark",
      )
      st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 📋 Tabular Batch Quality Records")
    st.dataframe(df_batch, use_container_width=True)
  else:
    st.warning(
        "No inspection records found for this batch. Scan points in Module 1"
        " first."
    )


# --- Module 3: Live Database Records ---
elif nav_choice == "📜 Live MySQL Database Records":
  st.subheader(f"📄 Live MySQL Inspection Records [{batch_no}]")

  conn = get_db_connection()
  if conn:
    try:
      query = (
          "SELECT * FROM color_inspections WHERE batch_no = %s ORDER BY"
          " scan_timestamp DESC"
      )
      df_db = pd.read_sql(query, conn, params=[batch_no])
      conn.close()

      if not df_db.empty:
        st.success(
            f"📊 Loaded {len(df_db)} records directly from MySQL Database."
        )
        st.dataframe(df_db, use_container_width=True)
      else:
        st.warning(
            "No records found in MySQL Database for this Batch Number."
        )
    except Exception as e:
      st.error(f"❌ Failed to fetch data from Database: {e}")
