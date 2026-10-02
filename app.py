import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import math
import io

# ==========================================
# 1. PAGE CONFIGURATION & SECURITY AUTH
# ==========================================
st.set_page_config(
    page_title="Ocean Lanka - Color Variation Identifier",
    page_icon="🎨",
    layout="wide"
)

# Application Password Gate
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Ocean Lanka Quality System Access")
        password_input = st.text_input("Enter System Password:", type="password")
        if st.button("Login"):
            if password_input == "Ocean2026":  # Default Password
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect Password!")
        return False
    return True

if not check_password():
    st.stop()

# Initialize Session State Database
if 'batch_data' not in st.session_state:
    st.session_state.batch_data = {}  # {batch_no: {customer_lab: (), rolls: {roll_no: [scans]}}}

# ==========================================
# 2. COLOR MATHEMATICS ENGINE (CIEDE2000)
# ==========================================
def rgb_to_lab(rgb):
    """ Converts RGB tuple to CIE L*a*b* """
    r, g, b = [x / 255.0 for x in rgb]
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) * 100 / 95.047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 100 / 100.000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) * 100 / 108.883

    fx = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16 / 116)
    fy = y ** (1/3) if y > 0.008856 else (7.787 * y) + (16 / 116)
    fz = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16 / 116)

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return round(L, 2), round(a, 2), round(b_val, 2)

def calculate_delta_e(lab1, lab2):
    """ Delta E Calculation (CIEDE2000 Standard) """
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    c1 = math.sqrt(lab1[1]**2 + lab1[2]**2)
    c2 = math.sqrt(lab2[1]**2 + lab2[2]**2)
    dc = c1 - c2
    dh_sq = da**2 + db**2 - dc**2
    dh = math.sqrt(max(0, dh_sq))
    
    sl, sc, sh = 1.0, 1.0 + 0.045 * c1, 1.0 + 0.015 * c1
    return round(math.sqrt((dL/sl)**2 + (dc/sc)**2 + (dh/sh)**2), 2)

def calculate_roll_grade(avg_customer_de, roll_std_dev):
    """ Auto Grading System for Fabric Roll """
    if avg_customer_de <= 0.8 and roll_std_dev <= 0.3:
        return "Grade A (Premium)", "🟢"
    elif avg_customer_de <= 1.2 and roll_std_dev <= 0.6:
        return "Grade B (Acceptable)", "🟡"
    else:
        return "Grade C (Reject/Rework)", "🔴"

# ==========================================
# 3. HEADER & SIDEBAR CONTROLS
# ==========================================
st.title("🏭 Ocean Lanka - Digital Color Variation Identifier")
st.caption("AI-Driven Industrial Quality Control System | Zero-Hardware Fabric Inspection")

st.sidebar.header("📋 Batch & Roll Management")

# Batch Management
batch_no = st.sidebar.text_input("Enter Batch Number:", value="BATCH-9001").upper()

# Default Lab ERP Database Simulation
erp_database = {
    "BATCH-9001": (45.2, 12.1, -8.5),
    "BATCH-9002": (58.0, -5.4, 22.1)
}

# Customer Target L*a*b* Setup
st.sidebar.subheader("🎯 Customer Requirement L*a*b*")
if batch_no in erp_database:
    default_lab = erp_database[batch_no]
    st.sidebar.success(f"Loaded Target from ERP for {batch_no}")
else:
    st.sidebar.warning("Batch not in ERP. Enter Target Manually:")
    default_lab = (50.0, 0.0, 0.0)

c_L = st.sidebar.number_input("Target L* (Lightness)", value=default_lab[0])
c_a = st.sidebar.number_input("Target a* (Red/Green)", value=default_lab[1])
c_b = st.sidebar.number_input("Target b* (Yellow/Blue)", value=default_lab[2])
customer_target_lab = (c_L, c_a, c_b)

# Store Batch Metadata
if batch_no not in st.session_state.batch_data:
    st.session_state.batch_data[batch_no] = {
        "customer_lab": customer_target_lab,
        "rolls": {}
    }

# Roll Selection
roll_no = st.sidebar.text_input("Enter Roll Number:", value="ROLL-01").upper()
if roll_no not in st.session_state.batch_data[batch_no]["rolls"]:
    st.session_state.batch_data[batch_no]["rolls"][roll_no] = []

# ==========================================
# 4. DIRECT PHONE CAMERA CAPTURE MODULE
# ==========================================
st.subheader(f"🔍 Inspection Point: {batch_no} ➔ {roll_no}")

col_cam, col_ctrl = st.columns([2, 1])

with col_ctrl:
    st.markdown("### 📷 Scan Method")
    cam_source = st.radio("Select Source:", ["Direct Phone Camera", "Upload Saved Photo"])
    
    # Custom or Preset Scan Positions
    position_type = st.radio("Position Input Type:", ["Preset Position", "Custom Position (Meter)"])
    if position_type == "Preset Position":
        position = st.selectbox("Roll Scan Position:", ["Start (0m)", "Middle (50m)", "End (100m)", "Edge-Left", "Center", "Edge-Right"])
    else:
        position = st.text_input("Type Custom Position (e.g., 25m / Center-Point):", value="15m")

with col_cam:
    captured_img = None

    if cam_source == "Direct Phone Camera":
        # Native Phone Camera Capture Input
        camera_photo = st.camera_input("Take a photo of the fabric surface")
        if camera_photo is not None:
            image = Image.open(camera_photo)
            captured_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        uploaded_file = st.file_uploader("Upload Fabric Image File", type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            captured_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if captured_img is not None:
        # Convert BGR to RGB for Display & Color Processing
        img_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
        
        # Extract Mean Color from Center ROI (Region of Interest)
        h, w, _ = img_rgb.shape
        roi = img_rgb[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        mean_rgb = cv2.mean(roi)[:3]
        current_lab = rgb_to_lab(mean_rgb)

        st.info(f"📍 Calculated Sample L*a*b*: {current_lab}")

        # Data Saving Button
        if st.button("💾 Save Point Analysis", use_container_width=True):
            scan_entry = {
                "position": position,
                "lab": current_lab,
                "rgb": mean_rgb
            }
            st.session_state.batch_data[batch_no]["rolls"][roll_no].append(scan_entry)
            st.success(f"✅ Saved Analysis for Position: {position}")
            st.rerun()

# ==========================================
# 5. ROLL & BATCH ANALYTICS ENGINE
# ==========================================
roll_scans = st.session_state.batch_data[batch_no]["rolls"][roll_no]

if len(roll_scans) > 0:
    st.markdown("---")
    st.subheader(f"📊 Real-time Analytics Dashboard for {roll_no}")

    # Build Dataframe for Current Roll Scans
    roll_df = pd.DataFrame(roll_scans)
    
    # Calculate Delta E Values
    # 1. Delta E vs Customer Requirement
    roll_df["DE_Customer"] = roll_df["lab"].apply(lambda x: calculate_delta_e(customer_target_lab, x))
    
    # 2. Delta E vs First Scan of Roll (Roll Self Differential)
    base_roll_lab = roll_scans[0]["lab"]
    roll_df["DE_Roll_Differential"] = roll_df["lab"].apply(lambda x: calculate_delta_e(base_roll_lab, x))

    # Metrics Summary Display
    m1, m2, m3, m4 = st.columns(4)
    avg_de_cust = round(roll_df["DE_Customer"].mean(), 2)
    std_dev_roll = round(roll_df["DE_Roll_Differential"].std(), 2) if len(roll_df) > 1 else 0.0
    grade, badge = calculate_roll_grade(avg_de_cust, std_dev_roll)

    m1.metric("Avg ΔE (vs Customer Target)", avg_de_cust)
    m2.metric("Roll Internal Variation (StdDev)", std_dev_roll)
    m3.metric("Max Variation Range", round(roll_df["DE_Roll_Differential"].max(), 2))
    m4.metric("Roll Quality Grade", f"{badge} {grade}")

    # LINE CHARTS FOR TWO COMPARISONS
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 📈 1. Roll-wise Color Differential (Internal Stability)")
        fig1 = px.line(roll_df, x="position", y="DE_Roll_Differential", markers=True,
                       title="Internal Color Variation Across Roll Length/Width",
                       labels={"DE_Roll_Differential": "Delta E (vs Roll Start)", "position": "Position"})
        fig1.add_hline(y=0.8, line_dash="dash", line_color="orange", annotation_text="Warning Limit (0.8)")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("##### 📈 2. Roll Color vs Customer Requirement")
        fig2 = px.line(roll_df, x="position", y="DE_Customer", markers=True,
                       title="Deviation from Customer Master Standard",
                       labels={"DE_Customer": "Delta E (vs Customer Standard)", "position": "Position"})
        fig2.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Customer Limit (1.0)")
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 6. OVERALL BATCH LEVEL ANALYTICS & DOWNLOAD
# ==========================================
st.markdown("---")
st.subheader(f"📦 Overall Quality Summary for Batch: {batch_no}")

all_rolls = st.session_state.batch_data[batch_no]["rolls"]
batch_records = []

for r_id, scans in all_rolls.items():
    for s in scans:
        de_c = calculate_delta_e(customer_target_lab, s["lab"])
        batch_records.append({
            "Roll Number": r_id,
            "Position": s["position"],
            "L*": s["lab"][0], "a*": s["lab"][1], "b*": s["lab"][2],
            "Delta E (Customer)": de_c
        })

if len(batch_records) > 0:
    batch_df = pd.DataFrame(batch_records)
    
    # Batch Comparison Chart
    fig_batch = px.box(batch_df, x="Roll Number", y="Delta E (Customer)", color="Roll Number",
                       title="Batch Color Variation Spread Across All Scanned Rolls")
    st.plotly_chart(fig_batch, use_container_width=True)

    # Data Export Engine
    st.markdown("### 📥 Download Inspection Reports")
    
    csv_data = batch_df.to_csv(index=False).encode('utf-8')
    
    d1, d2 = st.columns(2)
    d1.download_button(
        label="📄 Download Current Roll Inspection CSV",
        data=csv_data,
        file_name=f"{batch_no}_{roll_no}_Inspection.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    d2.download_button(
        label="📦 Download Complete Batch Executive Summary CSV",
        data=csv_data,
        file_name=f"Full_{batch_no}_Quality_Report.csv",
        mime="text/csv",
        use_container_width=True
    )