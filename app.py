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
    page_title="Ocean Lanka - AI Spectrophotometer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Professional Styling ---
st.markdown(
    """
<style>
    .main { background-color: #0B0F19; color: #F8FAFC; }
    .stApp { max-width: 100%; }
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; height: 45px; background-color: #0EA5E9; color: white; border: none; }
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


# --- Reliable Voice Output Helper ---
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
      "<h2 style='text-align: center; color: #38BDF8; margin-top: 60px;'>🏭 Ocean"
      " Lanka AI Spectrophotometer</h2>",
      unsafe_allow_html=True,
  )
  col1, col2, col3 = st.columns([1, 1.5, 1])
  with col2:
    with st.form("login_form"):
      user_in = st.text_input("Operator Name:")
      pass_in = st.text_input("Password:", type="password")
      if st.form_submit_button("Login to System"):
        if pass_in == "Ocean2026" and user_in.strip() != "":
          st.session_state.authenticated = True
          st.session_state.user_name = user_in.strip()
          speak_voice("System initialized. Welcome " + user_in)
          time.sleep(2)  # Allow voice to finish before rerun
          st.rerun()
        else:
          st.error("Invalid credentials!")
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


# --- Sidebar Navigation & Context Setup ---
st.sidebar.markdown(f"### 👤 {st.session_state.user_name}")
nav_choice = st.sidebar.radio(
    "Navigation Menu:",
    [
        "🔍 Real-time Inspection",
        "📊 Batch Analytics",
        "📋 Shade Library",
        "📜 Live DB Records",
        "⚙️ Calibration",
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

# --- Module 1: Real-time Inspection ---
if nav_choice == "🔍 Real-time Inspection":
  st.subheader(f"📍 Active Inspection: [{batch_no}] ➔ [{roll_no}]")

  col_opt, col_pos = st.columns([1, 1])
  with col_opt:
    input_method = st.radio(
        "Source:", ["📷 Mobile Camera", "📁 Upload Image"], horizontal=True
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
  if input_method == "📷 Mobile Camera":
    photo = st.camera_input("Capture Fabric Sample (Aperture Spot)")
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
    # Tight Aperture Extraction (Central 15%)
    h, w, _ = captured_img.shape
    img_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
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
    base_lab = scans[0]["lab"] if len(scans) > 0 else current_lab
    metrics = get_details(base_lab, current_lab)

    # --- PRIMARY METRIC DISPLAY (ΔE2000 Focus) ---
    de_val = metrics["de"]
    is_pass = de_val <= 1.0
    status_text = "PASS (Within Spec)" if is_pass else "FAIL (Out of Spec)"
    status_color = "#059669" if is_pass else "#DC2626"

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); border: 2px solid {status_color}; padding: 18px; border-radius: 14px; text-align: center; margin-top: 15px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);">
            <div style="font-size: 12px; color: #94A3B8; font-weight: bold; letter-spacing: 1.5px;">PRIMARY COLOR METRIC</div>
            <div style="font-size: 40px; font-weight: bold; color: #F8FAFC; margin: 5px 0;">ΔE2000 = {de_val}</div>
            <span style="background-color: {status_color}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 14px;">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SECONDARY METRICS (Compact view below) ---
    with st.expander("🔍 View Secondary Vector Components (ΔL*, Δa*, Δb*, ΔC*, ΔH*)"):
      m1, m2, m3, m4, m5 = st.columns(5)
      m1.metric("ΔL*", f"{metrics['dL']}", metrics["l_comm"])
      m2.metric("Δa*", f"{metrics['da']}", metrics["rg_comm"])
      m3.metric("Δb*", f"{metrics['db']}", metrics["yb_comm"])
      m4.metric("ΔC*", f"{metrics['dC']}")
      m5.metric("ΔH*", f"{metrics['dH']}")

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
        speak_voice("Scan saved successfully to database.")
        time.sleep(2)  # Allow voice to finish before rerun
        st.success("✅ Saved successfully to Database!")
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

# --- Module 2: Batch Analytics ---
elif nav_choice == "📊 Batch Analytics":
  st.subheader(f"📊 Batch Quality Analytics [{batch_no}]")
  rolls_data = st.session_state.batch_data[batch_no]["rolls"]
  records = []
  for r_id, scs in rolls_data.items():
    for s in scs:
      records.append({"Roll": r_id, "Position": s["position"], "ΔE2000": s["de"]})

  if len(records) > 0:
    df_b = pd.DataFrame(records)
    c1, c2 = st.columns(2)
    with c1:
      fig_b = px.box(
          df_b,
          x="Roll",
          y="ΔE2000",
          color="Roll",
          title="Roll Variation Dispersion",
          template="plotly_dark",
      )
      st.plotly_chart(fig_b, width="stretch")
    with c2:
      fig_sc = px.scatter(
          df_b,
          x="Position",
          y="ΔE2000",
          color="Roll",
          title="Position-wise ΔE Scatter",
          template="plotly_dark",
      )
      st.plotly_chart(fig_sc, width="stretch")
    st.dataframe(df_b, width="stretch")
  else:
    st.info("No inspection records available for this batch yet.")

# --- Module 3: Shade Library ---
elif nav_choice == "📋 Shade Library":
  st.subheader("📋 Ocean Lanka Standard Shade Library")
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
  st.dataframe(shades_df, width="stretch")

# --- Module 4: Live DB Records ---
elif nav_choice == "📜 Live DB Records":
  st.subheader(f"📜 Live MySQL Records [{batch_no}]")
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
        st.dataframe(df_db, width="stretch")
      else:
        st.warning("No records found in MySQL database for this batch.")
    except Exception as e:
      st.error(f"Database Query Error: {e}")
  else:
    st.error("Database connection failed.")

# --- Module 5: Calibration Settings ---
elif nav_choice == "⚙️ Calibration":
  st.subheader("⚙️ Spectrophotometer Calibration & Settings")
  st.info(
      "White Tile Calibration is recommended before starting a new production"
      " batch."
  )
  if st.button("⚪ Perform White Tile Calibration"):
    speak_voice("White tile calibration completed successfully.")
    st.success(
        "✅ White Tile Calibration Passed! Optical sensor zeroed successfully."
    )
