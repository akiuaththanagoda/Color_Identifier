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

# --- Page Configuration ---
st.set_page_config(
    page_title="Ocean Lanka - AI Spectrophotometer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Clean Minimalist & Professional Styling ---
st.markdown(
    """
<style>
    .main { background-color: #0B0F19; color: #F8FAFC; }
    .stApp { max-width: 100%; }
    .metric-container {
        background: #1E293B;
        border: 1px solid #334155;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
    }
    .badge-pass { background-color: #059669; color: white; padding: 4px 10px; border-radius: 15px; font-weight: bold; }
    .badge-fail { background-color: #DC2626; color: white; padding: 4px 10px; border-radius: 15px; font-weight: bold; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; height: 42px; background-color: #0EA5E9; color: white; border: none; }
    div.stButton > button:hover { background-color: #0284C7; }
</style>
""",
    unsafe_allow_html=True,
)


# --- Database Helpers ---
def get_db_connection():
  try:
    return mysql.connector.connect(
        host="localhost", user="root", password="", database="ocean_quality_db", port=3306
    )
  except:
    return None


def save_scan_to_mysql(inspector, batch, roll, pos, lab, delta_e):
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


# --- Voice Output Helper ---
def speak_voice(text):
  js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance('{text}');
        msg.rate = 1.0;
        window.speechSynthesis.speak(msg);
    </script>
    """
  st.html(js_code)


# --- Session State ---
if "authenticated" not in st.session_state:
  st.session_state.authenticated = False
if "user_name" not in st.session_state:
  st.session_state.user_name = "Inspector"
if "batch_data" not in st.session_state:
  st.session_state.batch_data = {}

# --- Login Gate ---
if not st.session_state.authenticated:
  st.markdown(
      "<h2 style='text-align: center; color: #38BDF8; margin-top: 50px;'>🏭 Ocean"
      " Lanka AI Spectrophotometer</h2>",
      unsafe_allow_html=True,
  )
  col1, col2, col3 = st.columns([1, 1.5, 1])
  with col2:
    with st.form("login_form"):
      user_in = st.text_input("Operator Name:")
      pass_in = st.text_input("Password:", type="password")
      if st.form_submit_button("Login"):
        if pass_in == "Ocean2026" and user_in.strip() != "":
          st.session_state.authenticated = True
          st.session_state.user_name = user_in.strip()
          speak_voice("System initialized. Welcome " + user_in)
          st.rerun()
        else:
          st.error("Invalid credentials")
  st.stop()


# --- Color Math ---
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


def get_details(l1, l2):
  dL = round(l2[0] - l1[0], 3)
  da = round(l2[1] - l1[1], 3)
  db = round(l2[2] - l1[2], 3)
  C1, C2 = math.sqrt(l1[1] ** 2 + l1[2] ** 2), math.sqrt(l2[1] ** 2 + l2[2] ** 2)
  dC = round(C2 - C1, 3)
  de = calculate_ciede2000(l1, l2)
  dH = round(math.sqrt(max(0.0, de**2 - dL**2 - dC**2)), 3)
  return {
      "dL": dL,
      "da": da,
      "db": db,
      "dC": dC,
      "dH": dH,
      "de": de,
      "l_comm": "Lighter" if dL > 0 else "Darker",
      "rg_comm": "More Red" if da > 0 else "More Green",
      "yb_comm": "More Yellow" if db > 0 else "More Blue",
  }


# --- Top Bar (Context & Clock) ---
top_c1, top_c2, top_c3, top_c4 = st.columns([1.5, 1.5, 2, 1])
with top_c1:
  batch_no = st.text_input("Batch No:", value="BATCH-9001").upper()
with top_c2:
  roll_no = st.text_input("Roll No:", value="ROLL-01").upper()
with top_c3:
  st.markdown(
      f"👤 **{st.session_state.user_name}** | 🟢 **System Ready**",
      unsafe_allow_html=True,
  )
with top_c4:
  if st.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

if batch_no not in st.session_state.batch_data:
  st.session_state.batch_data[batch_no] = {"rolls": {}}
if roll_no not in st.session_state.batch_data[batch_no]["rolls"]:
  st.session_state.batch_data[batch_no]["rolls"][roll_no] = []

st.markdown("---")

# --- Main Inspection Interface ---
col_opt, col_pos = st.columns([1, 1])
with col_opt:
  input_method = st.radio(
      "Select Image Input:",
      ["📷 Mobile / Live Camera", "📁 Upload Image"],
      horizontal=True,
  )
with col_pos:
  position = st.selectbox(
      "Scan Position:",
      [
          "Start (0m)",
          "Middle (50m)",
          "End (100m)",
          "Edge-Left",
          "Center-Point",
          "Edge-Right",
      ],
  )

captured_img = None
if input_method == "📷 Mobile / Live Camera":
  photo = st.camera_input("Capture Fabric Sample")
  if photo:
    captured_img = cv2.cvtColor(
        np.array(Image.open(photo)), cv2.COLOR_RGB2BGR
    )
else:
  file_up = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
  if file_up:
    captured_img = cv2.cvtColor(
        np.array(Image.open(file_up)), cv2.COLOR_RGB2BGR
    )

if captured_img is not None:
  # Instant aperture crop (center 15%)
  h, w, _ = captured_img.shape
  img_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
  ah, aw = int(h * 0.15), int(w * 0.15)
  cy, cx = h // 2, w // 2
  roi = img_rgb[cy - ah // 2 : cy + ah // 2, cx - aw // 2 : cx + aw // 2]

  # Draw reticle
  cv2.rectangle(
      img_rgb,
      (cx - aw // 2, cy - ah // 2),
      (cx + aw // 2, cy + ah // 2),
      (0, 255, 255),
      2,
  )
  st.image(img_rgb, width=350)

  current_lab = rgb_to_lab(cv2.mean(roi)[:3])
  scans = st.session_state.batch_data[batch_no]["rolls"][roll_no]
  base_lab = scans[0]["lab"] if len(scans) > 0 else current_lab
  metrics = get_details(base_lab, current_lab)

  # --- HORIZONTAL VALUES ROW ---
  st.markdown("### 📊 Instant Color Metrics")
  m1, m2, m3, m4, m5, m6 = st.columns(6)
  m1.metric("ΔL*", f"{metrics['dL']}", metrics["l_comm"])
  m2.metric("Δa*", f"{metrics['da']}", metrics["rg_comm"])
  m3.metric("Δb*", f"{metrics['db']}", metrics["yb_comm"])
  m4.metric("ΔC*", f"{metrics['dC']}")
  m5.metric("ΔH*", f"{metrics['dH']}")
  m6.metric(
      "ΔE2000",
      f"{metrics['de']}",
      "PASS" if metrics["de"] <= 1.0 else "FAIL",
      delta_color="inverse",
  )

  if st.button("💾 Save Scan & Record to Database"):
    success = save_scan_to_mysql(
        st.session_state.user_name,
        batch_no,
        roll_no,
        position,
        current_lab,
        metrics["de"],
    )
    st.session_state.batch_data[batch_no]["rolls"][roll_no].append(
        {"position": position, "lab": current_lab, "de": metrics["de"]}
    )
    if success:
      speak_voice("Scan saved successfully")
      st.success("✅ Saved successfully!")
      time.sleep(0.5)
      st.rerun()

# --- Bottom Chart ---
current_scans = st.session_state.batch_data[batch_no]["rolls"][roll_no]
if len(current_scans) > 0:
  st.markdown("---")
  st.subheader(f"📈 Roll Variation Profile ({roll_no})")
  df = pd.DataFrame(current_scans)
  fig = px.line(
      df,
      x="position",
      y="de",
      markers=True,
      template="plotly_dark",
      labels={"de": "ΔE2000", "position": "Position"},
  )
  fig.add_hline(y=1.0, line_dash="dash", line_color="#EF4444")
  st.plotly_chart(fig, width="stretch")
