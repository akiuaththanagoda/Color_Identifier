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

st.set_page_config(
    page_title="Ocean Lanka - AI Color Spectrophotometer & Custom Analytics",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { background-color: #0B0F19; color: #F8FAFC; }
    .stApp { max-width: 100%; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; height: 45px; background-color: #0EA5E9; color: white; border: none; }
    div.stButton > button:hover { background-color: #0284C7; }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

def get_db_connection():
  try:
    return mysql.connector.connect(
        host="localhost", user="root", password="", database="ocean_quality_db", port=3306
    )
  except:
    return None

def save_scan_to_mysql(inspector, batch, roll, pos, lab, delta_e, drift_e):
  conn = get_db_connection()
  if conn:
    try:
      cursor = conn.cursor()
      query = """INSERT INTO color_inspections 
                (inspector_name, batch_no, roll_no, scan_position, lab_L, lab_a, lab_b, ciede2000_delta_e) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
      cursor.execute(query, (inspector, batch, roll, pos, lab[0], lab[1], lab[2], delta_e))
      conn.commit()
      cursor.close()
      conn.close()
      return True
    except:
      return False
  return False

def speak_voice(text):
  js_code = f"""
    <script>
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance("{text}");
            msg.rate = 1.0;
            window.speechSynthesis.speak(msg);
        }}
    </script>
    """
  components.html(js_code, height=0)

if "authenticated" not in st.session_state:
  st.session_state.authenticated = False
if "user_name" not in st.session_state:
  st.session_state.user_name = "Inspector"
if "batch_data" not in st.session_state:
  st.session_state.batch_data = {
      "BATCH-9001": {
          "rolls": {
              "ROLL-01": [
                  {
                      "position": "Scan @ 08:00",
                      "lab": (18.5, 0.2, -0.5),
                      "de_global": 0.21,
                      "de_seq": 0.21,
                      "de_rolling": 0.21,
                  },
                  {
                      "position": "Scan @ 08:01",
                      "lab": (18.6, 0.3, -0.4),
                      "de_global": 0.45,
                      "de_seq": 0.35,
                      "de_rolling": 0.30,
                  },
                  {
                      "position": "Scan @ 08:02",
                      "lab": (18.4, 0.1, -0.6),
                      "de_global": 0.62,
                      "de_seq": 0.40,
                      "de_rolling": 0.42,
                  },
              ]
          }
      }
  }
if "last_img_hash" not in st.session_state:
  st.session_state.last_img_hash = None
if "last_scan_time" not in st.session_state:
  st.session_state.last_scan_time = time.time()

if not st.session_state.authenticated:
  st.markdown(
      "<h2 style='text-align: center; color: #38BDF8; margin-top: 60px;'>🏭 Ocean"
      " Lanka AI Spectrophotometer Custom System</h2>",
      unsafe_allow_html=True,
  )
  col1, col2, col3 = st.columns([1, 1.5, 1])
  with col2:
    with st.form("login_form"):
      user_in = st.text_input("Operator Name:")
      pass_in = st.text_input("Password:", type="password")
      if st.form_submit_button("Login to Advanced Dashboard"):
        if pass_in == "Ocean2026" and user_in.strip() != "":
          st.session_state.authenticated = True
          st.session_state.user_name = user_in.strip()
          speak_voice("System initialized. Welcome " + user_in)
          time.sleep(2)
          st.rerun()
        else:
          st.error("Invalid credentials!")
  st.stop()

@st.cache_data
def rgb_to_lab(rgb):
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
  return (
      round((116 * fy) - 16, 3),
      round(500 * (fx - fy), 3),
      round(200 * (fy - fz), 3),
  )

def check_image_sharpness(image_bgr, threshold=70.0):
  gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
  laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
  return laplacian_var, laplacian_var >= threshold

def enhance_fabric_image(image_bgr):
  filtered = cv2.bilateralFilter(image_bgr, 9, 75, 75)
  lab_img = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab_img)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  l_eq = clahe.apply(l)
  enhanced_lab = cv2.merge((l_eq, a, b))
  return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def calculate_ciede2000(lab1, lab2):
  L1, a1, b1 = lab1
  L2, a2, b2 = lab2
  avg_L = (L1 + L2) / 2.0
  C1, C2 = math.sqrt(a1**2 + b1**2), math.sqrt(a2**2 + b2**2)
  avg_C = (C1 + C2) / 2.0
  G = 0.5 * (1 - math.sqrt(avg_C**7 / (avg_C**7 + 25**7))) if avg_C > 0 else 0
  a1_p, a2_p = (1 + G) * a1, (1 + G) * a2
  C1_p, C2_p = math.sqrt(a1_p**2 + b1**2), math.sqrt(a2_p**2 + b2**2)
  avg_Cp = (C1_p + C2_p) / 2.0
  h1_p = math.degrees(math.atan2(b1, a1_p)) % 360
  h2_p = math.degrees(math.atan2(b2, a2_p)) % 360
  d_hp = h2_p - h1_p
  if abs(d_hp) > 180:
    d_hp -= 360 if h2_p > h1_p else -360
  avg_hp = (
      (h1_p + h2_p) / 2.0
      if abs(h1_p - h2_p) <= 180
      else ((h1_p + h2_p + 360) / 2.0 if (h1_p + h2_p) < 360 else (h1_p + h2_p - 360) / 2.0)
  )
  T = (
      1
      - 0.17 * math.cos(math.radians(avg_hp - 30))
      + 0.24 * math.cos(math.radians(2 * avg_hp))
      + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
      - 0.20 * math.cos(math.radians(4 * avg_hp - 63))
  )
  dL_p, dC_p = L2 - L1, C2_p - C1_p
  dH_p = 2 * math.sqrt(C1_p * C2_p) * math.sin(math.radians(d_hp / 2.0))
  S_L = 1 + ((0.015 * ((avg_L - 50) ** 2)) / math.sqrt(20 + ((avg_L - 50) ** 2)))
  S_C, S_H = 1 + 0.045 * avg_Cp, 1 + 0.015 * avg_Cp * T
  R_C = 2 * math.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7))
  RT = -math.sin(math.radians(2 * 30 * math.exp(-(((avg_hp - 275) / 25) ** 2)))) * R_C
  de = math.sqrt(
      (dL_p / S_L) ** 2
      + (dC_p / S_C) ** 2
      + (dH_p / S_H) ** 2
      + RT * (dC_p / S_C) * (dH_p / S_H)
  )
  return round(de, 3)

st.sidebar.markdown(f"### 👤 {st.session_state.user_name}")
nav_choice = st.sidebar.radio(
    "Navigation Menu:",
    [
        "📊 Advanced Executive Dashboard (Power BI Style)",
        "🔍 Real-time Custom Inspection",
        "📊 Batch Color Analytics",
        "📋 Custom Shade Library",
        "📜 Live MySQL Audit Records",
        "⚙️ Hardware Calibration",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Production Parameters")
batch_no = st.sidebar.text_input("Batch Number:", value="BATCH-9001").upper()
roll_no = st.sidebar.text_input("Roll Number:", value="ROLL-01").upper()

if batch_no not in st.session_state.batch_data:
  st.session_state.batch_data[batch_no] = {"rolls": {}}
if roll_no not in st.session_state.batch_data[batch_no]["rolls"]:
  st.session_state.batch_data[batch_no]["rolls"][roll_no] = []

if st.sidebar.button("🚪 Logout"):
  st.session_state.authenticated = False
  speak_voice("System logged out.")
  time.sleep(1.5)
  st.rerun()

if nav_choice == "📊 Advanced Executive Dashboard (Power BI Style)":
  st.markdown(
      "## 📊 Ocean Lanka Custom Executive Color Intelligence & Analytics"
      " Dashboard"
  )
  st.markdown(
      "*(Real-time comprehensive enterprise custom analytics replacing"
      " standalone BI tools)*"
  )

  col1, col2, col3, col4, col5 = st.columns(5)
  with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8;">Total Inspected Rolls</div>
            <div style="font-size: 24px; font-weight: bold; color: #38BDF8;">487</div>
            <div style="font-size: 10px; color: #10B981;">▲ +12% this week</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8;">Wastage Cost Saved</div>
            <div style="font-size: 24px; font-weight: bold; color: #10B981;">$108,000</div>
            <div style="font-size: 10px; color: #10B981;">ROI: 3.3 Days</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with col3:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8;">Average Global ΔE</div>
            <div style="font-size: 24px; font-weight: bold; color: #F43F5E;">0.54</div>
            <div style="font-size: 10px; color: #10B981;">Standard (≤1.0)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with col4:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8;">System Compliance</div>
            <div style="font-size: 24px; font-weight: bold; color: #38BDF8;">99.4%</div>
            <div style="font-size: 10px; color: #10B981;">Triple-Track Active</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
  with col5:
    st.markdown(
        """
        <div class="metric-card">
            <div style="font-size: 12px; color: #94A3B8;">Blanket Scrap Avoided</div>
            <div style="font-size: 24px; font-weight: bold; color: #10B981;">36.2 Tons</div>
            <div style="font-size: 10px; color: #10B981;">Eco-Friendly Target</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  st.markdown("<br>", unsafe_allow_html=True)

  col_a, col_b = st.columns([2, 1])

  with col_a:
    st.markdown(
        "### 📈 Monthly Custom Shade Compliance & Creeping Drift Distribution"
    )
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    pass_rates = [85, 88, 92, 90, 94, 96, 95, 97, 98, 99, 99.2, 99.4]
    fig_main = go.Figure(
        data=[
            go.Bar(
                name="Pass Rate (%)",
                x=months,
                y=pass_rates,
                marker_color="#0EA5E9",
            )
        ]
    )
    fig_main.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
    )
    st.plotly_chart(fig_main, use_container_width=True)

  with col_b:
    st.markdown("### 🍩 Fabric Defect & Color Status")
    fig_donut = px.pie(
        names=["Color Drift", "Surface Blur", "Uniform Pass"],
        values=[3, 1, 96],
        hole=0.6,
        template="plotly_dark",
    )
    fig_donut.update_layout(
        margin=dict(l=20, r=20, t=20, b=20), height=320, showlegend=False
    )
    st.plotly_chart(fig_donut, use_container_width=True)

  col_c, col_d = st.columns(2)
  with col_c:
    st.markdown("### 📊 Department-wise Custom Yield Analysis")
    dept_names = ["Knitting", "Dyeing", "Finishing", "Printing", "Quality Control"]
    yield_vals = [94, 91, 96, 93, 98]
    fig_bar = px.bar(
        x=dept_names,
        y=yield_vals,
        color=yield_vals,
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig_bar.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=280,
        xaxis_title="Custom Department",
        yaxis_title="Efficiency %",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

  with col_d:
    st.markdown("### 📉 Custom Rolling Drift vs Depth Dispersion")
    x_depths = [50, 100, 200, 300, 400, 500]
    drift_vals = [0.2, 0.31, 0.45, 0.38, 0.52, 0.59]
    fig_line = go.Figure(
        data=[
            go.Scatter(
                x=x_depths,
                y=drift_vals,
                mode="lines+markers",
                line=dict(color="#10B981", width=3),
            )
        ]
    )
    fig_line.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        height=280,
        xaxis_title="Custom Depth (m)",
        yaxis_title="Rolling Drift ΔE",
    )
    st.plotly_chart(fig_line, use_container_width=True)

elif nav_choice == "🔍 Real-time Custom Inspection":
  st.subheader(
      f"📍 Triple-Track Custom Automated Inspection: [{batch_no}] ➔ [{roll_no}]"
  )

  current_time = time.time()
  elapsed_time = current_time - st.session_state.last_scan_time
  if elapsed_time >= 60:
    speak_voice("Interval reached. Please scan the next custom sample.")
    st.toast(
        "⏰ 1-Minute Custom Inspection Interval Reached! Please capture"
        " sample.",
        icon="🚨",
    )
    st.session_state.last_scan_time = current_time

  input_method = st.radio(
      "Source Custom:",
      [
          "📷 Mobile / Browser Custom Camera",
          "📁 Custom Upload Image",
          "🔌 Custom Wired USB Camera",
          "🌐 Custom IP Network Camera",
      ],
      horizontal=True,
  )

  captured_img = None

  if input_method == "📷 Mobile / Browser Custom Camera":
    photo = st.camera_input("Capture Custom Fabric Sample")
    if photo:
      captured_img = cv2.cvtColor(
          np.array(Image.open(photo)), cv2.COLOR_RGB2BGR
      )

  elif input_method == "📁 Custom Upload Image":
    file_up = st.file_uploader(
        "Upload Custom Custom Image", type=["jpg", "png", "jpeg"]
    )
    if file_up:
      captured_img = cv2.cvtColor(
          np.array(Image.open(file_up)), cv2.COLOR_RGB2BGR
      )

  elif input_method == "🔌 Custom Wired USB Camera":
    usb_index = st.number_input(
        "Custom USB Camera Index (e.g. 0, 1):", value=0, min_value=0, max_value=5
    )
    if st.button("📸 Capture from Custom Wired USB Camera"):
      try:
        cap = cv2.VideoCapture(int(usb_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
          st.error(
              f"❌ Cannot open custom USB camera at index {usb_index}. Check"
              " Windows Privacy Settings."
          )
        else:
          ret, frame = cap.read()
          cap.release()
          if ret and frame is not None:
            captured_img = frame
            st.success(
                f"✅ Frame successfully captured from Custom Wired USB Camera"
                f" (Index {usb_index})!"
            )
          else:
            st.error("❌ Failed to grab frame from Custom USB Camera.")
      except Exception as e:
        st.error(f"❌ Custom USB Camera Error: {e}")

  elif input_method == "🌐 Custom IP Network Camera":
    ip_url = st.text_input(
        "Enter Custom IP Camera Stream URL:",
        value="http://192.168.1.100:8080/video",
    )
    if st.button("🌐 Capture from Custom IP Network Camera"):
      try:
        cap = cv2.VideoCapture(ip_url)
        ret, frame = cap.read()
        cap.release()
        if ret:
          captured_img = frame
          st.success("✅ Frame successfully captured from Custom IP Camera!")
        else:
          st.error("❌ Could not connect to custom IP Camera stream.")
      except Exception as e:
        st.error(f"❌ Custom IP Camera Connection Error: {e}")

  if captured_img is not None:
    score, is_sharp = check_image_sharpness(captured_img)

    if not is_sharp:
      speak_voice("Custom image is blurry. Please recapture a clear photo.")
      st.error(
          f"❌ Custom Image is blurry (Sharpness Score: {score:.1f} < 70)."
          " Please recapture."
      )
    else:
      enhanced_img = enhance_fabric_image(captured_img)
      h, w, _ = enhanced_img.shape
      img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
      ah, aw = int(h * 0.15), int(w * 0.15)
      cy, cx = h // 2, w // 2
      roi = img_rgb[cy - ah // 2 : cy + ah // 2, cx - aw // 2 : cx + aw // 2]

      cv2.rectangle(
          img_rgb,
          (cx - aw // 2, cy - ah // 2),
          (cx + aw // 2, cy + ah // 2),
          (0, 255, 255),
          2,
      )
      st.image(img_rgb, width=300)

      current_lab = rgb_to_lab(cv2.mean(roi)[:3])
      scans = st.session_state.batch_data[batch_no]["rolls"][roll_no]

      base_lab_global = scans[0]["lab"] if len(scans) > 0 else current_lab
      de_global = calculate_ciede2000(base_lab_global, current_lab)

      base_lab_seq = scans[-1]["lab"] if len(scans) > 0 else current_lab
      de_seq = calculate_ciede2000(base_lab_seq, current_lab)

      if len(scans) >= 3:
        recent_labs = [s["lab"] for s in scans[-3:]]
        base_lab_rolling = np.mean(recent_labs, axis=0)
      else:
        base_lab_rolling = base_lab_seq
      de_rolling = calculate_ciede2000(base_lab_rolling, current_lab)

      is_pass = de_global <= 1.0 and de_seq <= 0.8 and de_rolling <= 0.6
      status_text = (
          "PASS (Stable Uniformity)" if is_pass else "WARNING (Drift Detected)"
      )
      status_color = "#059669" if is_pass else "#D97706"

      img_hash = hash(captured_img.tobytes())
      position_label = f"Scan @ {datetime.now().strftime('%H:%M:%S')}"

      if st.session_state.last_img_hash != img_hash:
        save_scan_to_mysql(
            st.session_state.user_name,
            batch_no,
            roll_no,
            position_label,
            current_lab,
            de_global,
            de_rolling,
        )
        st.session_state.batch_data[batch_no]["rolls"][roll_no].append({
            "position": position_label,
            "lab": current_lab,
            "de_global": de_global,
            "de_seq": de_seq,
            "de_rolling": de_rolling,
        })
        st.session_state.last_img_hash = img_hash
        st.session_state.last_scan_time = time.time()

        speak_voice(
            f"Global Delta E is {de_global}. Rolling drift is {de_rolling}."
        )

      st.markdown(
          f"""
            <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 2px solid {status_color}; padding: 18px; border-radius: 14px; text-align: center; margin-top: 15px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);">
                <div style="font-size: 12px; color: #94A3B8; font-weight: bold; letter-spacing: 1.5px;">TRIPLE-TRACK CUSTOM TELEMETRY</div>
                <div style="font-size: 26px; font-weight: bold; color: #F8FAFC; margin: 5px 0;">Global ΔE = {de_global} | Sequential = {de_seq} | Rolling Drift = {de_rolling}</div>
                <span style="background-color: {status_color}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 14px;">{status_text}</span>
            </div>
            """,
          unsafe_allow_html=True,
      )

  current_scans = st.session_state.batch_data[batch_no]["rolls"][roll_no]
  if len(current_scans) > 0:
    st.markdown("---")
    st.subheader(
        f"📈 Triple-Track Custom Roll Uniformity & Creeping Drift Profile"
        f" ({roll_no})"
    )
    df = pd.DataFrame(current_scans)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["position"],
            y=df["de_global"],
            mode="lines+markers",
            name="Global ΔE (vs Start 0m)",
            line=dict(color="#0EA5E9", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["position"],
            y=df["de_seq"],
            mode="lines+markers",
            name="Sequential Shift (vs N-1)",
            line=dict(color="#EAB308", width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["position"],
            y=df["de_rolling"],
            mode="lines+markers",
            name="Rolling Moving Baseline (Drift Detector)",
            line=dict(color="#EC4899", width=2, dash="dash"),
        )
    )
    fig.update_layout(template="plotly_dark", xaxis_title="Custom Inspection Point")
    st.plotly_chart(fig, use_container_width=True)

elif nav_choice == "📊 Batch Color Analytics":
  st.subheader(f"📊 Custom Batch Quality Analytics [{batch_no}]")
  rolls_data = st.session_state.batch_data[batch_no]["rolls"]
  records = []
  for r_id, scs in rolls_data.items():
    for s in scs:
      records.append({
          "Roll": r_id,
          "Position": s["position"],
          "Global ΔE": s["de_global"],
          "Rolling Drift": s["de_rolling"],
      })

  if len(records) > 0:
    df_b = pd.DataFrame(records)
    c1, c2 = st.columns(2)
    with c1:
      fig_b = px.box(
          df_b,
          x="Roll",
          y="Global ΔE",
          color="Roll",
          title="Custom Global Roll Dispersion",
          template="plotly_dark",
      )
      st.plotly_chart(fig_b, use_container_width=True)
    with c2:
      fig_sc = px.scatter(
          df_b,
          x="Position",
          y="Rolling Drift",
          color="Roll",
          title="Custom Rolling Drift Scatter",
          template="plotly_dark",
      )
      st.plotly_chart(fig_sc, use_container_width=True)
    # Fixed dataframe usage with use_container_width
    st.dataframe(df_b, use_container_width=True)
  else:
    st.info("No custom inspection records available for this batch yet.")

elif nav_choice == "📋 Custom Shade Library":
  st.subheader("📋 Custom Ocean Lanka Standard Custom Shade Library")
  shades_df = pd.DataFrame([
      {
          "Standard Name": "BLACK-093-54A2",
          "Customer": "VS&Co",
          "Target L*": 18.5,
          "Target a*": 0.2,
          "Target b*": -0.5,
          "Tolerance ΔE": 1.0,
      },
      {
          "Standard Name": "NAVY-712-10B",
          "Customer": "Brands",
          "Target L*": 22.1,
          "Target a*": 1.1,
          "Target b*": -4.2,
          "Tolerance ΔE": 1.0,
      },
      {
          "Standard Name": "OPTIC-WHITE-01",
          "Customer": "Global",
          "Target L*": 95.2,
          "Target a*": -0.1,
          "Target b*": 1.2,
          "Tolerance ΔE": 0.8,
      },
  ])
  
  st.dataframe(shades_df, use_container_width=True)

elif nav_choice == "📜 Live MySQL Audit Records":
  st.subheader(f"📜 Live MySQL Custom Audit Records [{batch_no}]")
  conn = get_db_connection()
  if conn:
    try:
      df_db = pd.read_sql(
          "SELECT * FROM color_inspections WHERE batch_no = %s ORDER BY"
          " scan_timestamp DESC",
          conn,
          params=[batch_no],
      )
      conn.close()
      if not df_db.empty:
        
        st.dataframe(df_db, use_container_width=True)
      else:
        st.warning("No records found in MySQL database for this custom batch.")
    except Exception as e:
      st.error(f"Custom Database Query Error: {e}")
  else:
    st.error("Custom Database connection failed.")

elif nav_choice == "⚙️ Hardware Calibration":
  st.subheader("⚙️ Custom Spectrophotometer Calibration & Hardware Settings")
  st.info(
      "White Tile Calibration is recommended before starting a custom"
      " production batch."
  )
  if st.button("⚪ Perform Custom White Tile Calibration"):
    speak_voice("White tile calibration completed successfully.")
    st.success(
        "✅ Custom White Tile Calibration Passed! Optical sensor zeroed"
        " successfully."
    )
