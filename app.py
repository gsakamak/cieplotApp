import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import colour
from colour.plotting import plot_chromaticity_diagram_CIE1931
import warnings
import streamlit as st
import io
from PIL import Image

# --- Hide unnecessary colour-science warnings ---
warnings.filterwarnings('ignore', category=colour.utilities.ColourUsageWarning)

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="CIE 1931 Config Web App", layout="wide")

# ==========================================
# 1. Login Authentication Function
# ==========================================
def check_login():
    """Validates email domain and password (must match email string)"""
    st.title("🔐 Login")
    st.markdown("Please enter your company email and password to access the app.")
    
    # Credentials Inputs
    email = st.text_input("Email Address:").strip()
    password = st.text_input("Password:", type="password").strip()
    
    if st.button("Login"):
        if email and password:
            parts = email.split('@')
            if len(parts) == 2:
                domain = parts[1].lower()
                allowed_domains = ["yitoa.co.jp", "yitoa.com"]
                
                # Condition 1: Domain check
                # Condition 2: Password must exactly match the email address
                if domain in allowed_domains and password == email:
                    st.success("Login successful!")
                    st.session_state['authenticated'] = True
                    st.rerun()
                elif domain not in allowed_domains:
                    st.error("Access Denied: Invalid email domain.")
                else:
                    st.error("Access Denied: Password must exactly match your Email Address.")
            else:
                st.error("Please enter a valid email address.")
        else:
            st.warning("Please fill in both Email Address and Password.")

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# If not authenticated, show login screen and stop further execution
if not st.session_state['authenticated']:
    check_login()
    st.stop()


# ==========================================
# 2. Main Application Features
# ==========================================

def load_color_data_from_bytes(content_bytes):
    try:
        # 1. 文字コードの自動判定・デコード (ExcelのShift-JISやBOM付UTF-8対策)
        try:
            decoded_str = content_bytes.decode('utf-8-sig')
        except UnicodeDecodeError:
            try:
                decoded_str = content_bytes.decode('cp932')
            except UnicodeDecodeError:
                decoded_str = content_bytes.decode('shift_jis', errors='replace')
                
        content = decoded_str.splitlines()
        
        # 2. データのヘッダー行を特定する (Name と R または x がある行)
        header_row_index = 0
        for i, line in enumerate(content):
            if "Name" in line and ("R" in line or "x" in line):
                header_row_index = i
                break
                
        content_stream = io.StringIO("\n".join(content))
        df = pd.read_csv(content_stream, skiprows=header_row_index)
        
        # 3. 列名（ヘッダー）の空白文字を強制削除 (" R " -> "R" に整形)
        if not df.empty:
            df.columns = df.columns.astype(str).str.strip()
        
        # 4. Name列内の空白文字の削除
        if 'Name' in df.columns:
            df['Name'] = df['Name'].astype(str).str.strip()
            
        df = df.dropna(how='all', axis=1)
        df = df.dropna(how='all', axis=0)
        return df
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None

def load_color_data(uploaded_file):
    if uploaded_file is not None:
        return load_color_data_from_bytes(uploaded_file.getvalue())
    return None

def load_local_csv(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return load_color_data_from_bytes(f.read())
        except Exception as e:
             st.error(f"Error loading local default file '{filepath}': {e}")
    return None

def draw_gamut_triangle(ax, df, color, linestyle, label, linewidth):
    if df is not None and 'Name' in df.columns and 'x' in df.columns and 'y' in df.columns:
        df_rgb = df.set_index('Name').reindex(['R', 'G', 'B', 'R'])
        if not df_rgb[['x', 'y']].isna().any().any():
            ax.plot(
                df_rgb['x'], df_rgb['y'], 
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.8, 
                label=label
            )

def plot_chromaticity_customized(df_target, df_before, df_after, color_space, fig_size, show_labels):
    base_size = 8.0
    scale = fig_size / base_size
    
    font_s = 10 * scale
    font_xs = 8 * scale
    font_title = 12 * scale
    
    marker_target = 120 * (scale**2)
    marker_data = 35 * (scale**2)
    marker_cct = 5 * (scale**2)
    
    line_w = 1.0 * scale
    line_w_thin = 0.4 * scale
    line_w_cct = 0.8 * scale
    text_offset = 0.008 * scale

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    plot_chromaticity_diagram_CIE1931(
        axes=ax, show=False, title='',
        bounding_box=(0, 0.9, 0, 0.9), standalone=False, transparent_background=True
    )
    
    ax.set_title('CIE 1931 xy Chromaticity Diagram', fontsize=font_title, color='black', pad=10*scale)
    
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    ax.set_facecolor('white')
    ax.patch.set_alpha(1.0)
    
    ax.tick_params(axis='both', colors='black', labelcolor='black', labelsize=font_s)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(line_w)
        spine.set_visible(True)
        
    ax.set_xlabel('x', color='black', fontsize=font_s)
    ax.set_ylabel('y', color='black', fontsize=font_s)
    
    for child in ax.get_children():
        obj_type = type(child).__name__
        if obj_type == 'Line2D':
            if child.get_color() in ['black', 'k', '#000000', (0.0, 0.0, 0.0, 1.0)]:
                 child.set_visible(False)
            if child.get_marker() not in ['None', '', ' ']:
                child.set_marker('None')
                child.set_markersize(0)
        elif obj_type == 'PathCollection':
            child.set_visible(False)
            child.set_alpha(0.0)
            
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    xyz = cmfs.values
    xy = colour.XYZ_to_xy(xyz)
    ax.plot(xy[:, 0], xy[:, 1], color='#E0E0E0', linewidth=line_w_thin, zorder=1)
    ax.plot([xy[0, 0], xy[-1, 0]], [xy[0, 1], xy[-1, 1]], color='#E0E0E0', linewidth=line_w_thin, zorder=1)

    if color_space.upper() == 'DCI-P3':
        gamut_x, gamut_y = [0.680, 0.265, 0.150, 0.680], [0.320, 0.690, 0.060, 0.320]
        label_text = 'DCI-P3 (Ref)'
    else:
        gamut_x, gamut_y = [0.640, 0.300, 0.150, 0.640], [0.330, 0.600, 0.060, 0.330]
        label_text = 'sRGB (Ref)'
    ax.plot(gamut_x, gamut_y, color='lightgray', linestyle='--', linewidth=line_w, label=label_text, zorder=3)
    
    wp_x, wp_y = [], []
    for cct in range(2000, 21000, 500):
        try: xy_cct = colour.CCT_to_xy(cct)
        except AttributeError: xy_cct = colour.xy_from_CCT(cct)
        wp_x.append(xy_cct[0])
        wp_y.append(xy_cct[1])
    ax.plot(wp_x, wp_y, color='#E8E8E8', alpha=0.7, linestyle='-', linewidth=line_w_cct, zorder=2)
    
    target_ccts = [20000, 15000, 10000, 8000, 6000, 4000, 2000]
    for cct in target_ccts:
        try: xy_cct = colour.CCT_to_xy(cct)
        except AttributeError: xy_cct = colour.xy_from_CCT(cct)
        ax.scatter(xy_cct[0], xy_cct[1], color='#E8E8E8', alpha=0.9, s=marker_cct, zorder=4)
        ax.text(xy_cct[0] + 0.005, xy_cct[1] + 0.005, f'{cct}K', fontsize=7*scale, color='darkgray', alpha=0.9, zorder=5)

    if df_target is not None and not df_target.empty and 'x' in df_target.columns and 'y' in df_target.columns:
        ax.scatter(df_target['x'], df_target['y'], marker='o', color='black', edgecolors='none', s=marker_target, label='Target', zorder=6)
        if show_labels and 'Name' in df_target.columns:
            for _, row in df_target.iterrows():
                ax.text(row['x'] + text_offset, row['y'] + text_offset, str(row['Name']), fontsize=font_xs, color='black', fontweight='bold', zorder=8)

    if df_before is not None and not df_before.empty and 'x' in df_before.columns and 'y' in df_before.columns:
        ax.scatter(df_before['x'], df_before['y'], marker='o', color='#FF40FF', s=marker_data, label='Before', zorder=6)
        draw_gamut_triangle(ax, df_before, color='#FF40FF', linestyle=':', label='Before Gamut', linewidth=line_w*1.2)
        if show_labels and 'Name' in df_before.columns:
            for _, row in df_before.iterrows():
                ax.text(row['x'] + text_offset, row['y'] - text_offset*1.5, str(row['Name']), fontsize=font_xs, color='#D500D5', zorder=8)

    if df_after is not None and not df_after.empty and 'x' in df_after.columns and 'y' in df_after.columns:
        ax.scatter(df_after['x'], df_after['y'], marker='o', color='#00FA00', s=marker_data, label='After', zorder=7)
        draw_gamut_triangle(ax, df_after, color='yellow', linestyle='-', label='After Gamut', linewidth=line_w*1.2)
        if show_labels and 'Name' in df_after.columns:
            for _, row in df_after.iterrows():
                ax.text(row['x'] + text_offset, row['y'] + text_offset, str(row['Name']), fontsize=font_xs, color='#008000', zorder=8)

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(visible=True, which='major', color='lightgray', linestyle='-', linewidth=line_w_thin)
    
    legend = ax.legend(loc='upper right', fontsize=font_s)
    if legend:
        for text in legend.get_texts(): text.set_color('black')
            
    ax.set_xlim(0.0, 0.9)
    ax.set_ylim(0.0, 0.9)
    fig.tight_layout()
    return fig

def get_delta_e_from_csv(row):
    possible_names = ['DeltaE2000', 'DeltaE', 'Delta E', 'dE', 'ΔE', 'Delta_E', 'Delta_E2000']
    for col in possible_names:
        if col in row.index and pd.notna(row[col]):
            try: return f"{float(row[col]):.2f}"
            except ValueError: return str(row[col])
    return "N/A"

# ==========================================
# ★ 追加: ユーザー指定の純粋な手動計算ロジック
# ==========================================
D65_X, D65_Y, D65_Z = 95.047, 100.000, 108.883

def rgb_8bit_to_target_XYZ(r, g, b):
    r_l = (r / 255.0) ** 2.2
    g_l = (g / 255.0) ** 2.2
    b_l = (b / 255.0) ** 2.2

    X = (0.4124564 * r_l + 0.3575761 * g_l + 0.1804375 * b_l) * 100.0
    Y = (0.2126729 * r_l + 0.7151522 * g_l + 0.0721750 * b_l) * 100.0
    Z = (0.0193339 * r_l + 0.1191920 * g_l + 0.9503041 * b_l) * 100.0
    return X, Y, Z

def measured_xyY_to_XYZ(x, y, Y_meas, Y_white_meas):
    Y_norm = (Y_meas / Y_white_meas) * 100.0 if Y_white_meas > 0 else 0
    if y == 0:
        return 0.0, 0.0, 0.0
    X = (x * Y_norm) / y
    Z = ((1.0 - x - y) * Y_norm) / y
    return X, Y_norm, Z

def f_lab(t):
    delta = 6.0 / 29.0
    if t > delta ** 3:
        return math.pow(t, 1.0 / 3.0)
    else:
        return (1.0 / 3.0) * ((29.0 / 6.0) ** 2) * t + (4.0 / 29.0)

def XYZ_to_Lab(X, Y, Z):
    L = 116.0 * f_lab(Y / D65_Y) - 16.0
    a = 500.0 * (f_lab(X / D65_X) - f_lab(Y / D65_Y))
    b = 200.0 * (f_lab(Y / D65_Y) - f_lab(Z / D65_Z))
    return L, a, b

def delta_E_2000(Lab1, Lab2):
    L1, a1, b1 = Lab1
    L2, a2, b2 = Lab2
    k_L, k_C, k_H = 1.0, 1.0, 1.0

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - math.sqrt((C_bar**7) / (C_bar**7 + 25.0**7)))

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = math.sqrt(a1_prime**2 + b1**2)
    C2_prime = math.sqrt(a2_prime**2 + b2**2)

    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360.0 if (b1 != 0 or a1_prime != 0) else 0.0
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360.0 if (b2 != 0 or a2_prime != 0) else 0.0

    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    if C1_prime * C2_prime == 0:
        dh_prime = 0.0
    elif abs(h2_prime - h1_prime) <= 180.0:
        dh_prime = h2_prime - h1_prime
    elif h2_prime <= h1_prime:
        dh_prime = h2_prime - h1_prime + 360.0
    else:
        dh_prime = h2_prime - h1_prime - 360.0

    dH_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2.0))

    L_bar_prime = (L1 + L2) / 2.0
    C_bar_prime = (C1_prime + C2_prime) / 2.0

    if C1_prime * C2_prime == 0:
        H_bar_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180.0:
        H_bar_prime = (h1_prime + h2_prime) / 2.0
    elif (h1_prime + h2_prime) < 360.0:
        H_bar_prime = (h1_prime + h2_prime + 360.0) / 2.0
    else:
        H_bar_prime = (h1_prime + h2_prime - 360.0) / 2.0

    T = 1.0 - 0.17 * math.cos(math.radians(H_bar_prime - 30.0)) \
            + 0.24 * math.cos(math.radians(2.0 * H_bar_prime)) \
            + 0.32 * math.cos(math.radians(3.0 * H_bar_prime + 6.0)) \
            - 0.20 * math.cos(math.radians(4.0 * H_bar_prime - 63.0))

    dTheta = 30.0 * math.exp(-(((H_bar_prime - 275.0) / 25.0) ** 2))
    R_c = 2.0 * math.sqrt((C_bar_prime**7) / (C_bar_prime**7 + 25.0**7))
    
    S_L = 1.0 + ((0.015 * (L_bar_prime - 50.0)**2) / math.sqrt(20.0 + (L_bar_prime - 50.0)**2))
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T
    R_T = -math.sin(math.radians(2.0 * dTheta)) * R_c

    dE00 = math.sqrt(
        (dL_prime / (k_L * S_L))**2 +
        (dC_prime / (k_C * S_C))**2 +
        (dH_prime / (k_H * S_H))**2 +
        R_T * (dC_prime / (k_C * S_C)) * (dH_prime / (k_H * S_H))
    )

    return dE00


def calculate_custom_delta_e(df_meas, row_meas, df_target, row_name, color_space):
    try:
        if df_target is None or df_meas is None: return "N/A"
        
        t_row = df_target[df_target['Name'] == row_name]
        if t_row.empty: return "N/A"
        t_row = t_row.iloc[0]
        
        # 1. ターゲットRGBの取得 (8bit)
        if not all(c in t_row for c in ['R', 'G', 'B']):
            return "N/A (Missing Target RGB)"
            
        R, G, B = float(t_row['R']), float(t_row['G']), float(t_row['B'])
        x_m, y_m = float(row_meas['x']), float(row_meas['y'])
        
        # 2. 測定されたY値（輝度）の取得
        lum_cols = ['Y', 'Lv', 'Luminance', 'L']
        y_col = next((c for c in lum_cols if c in df_meas.columns), None)
        if not y_col:
            return "N/A (Missing Measured Y)"
        
        Y_m = float(row_meas[y_col])
        
        # 3. 測定された白のY値を取得し輝度正規化の基準とする
        white_names = ['white', 'w', '19', 'patch 19', 'neutral 8']
        df_meas_names = df_meas['Name'].astype(str).str.strip().str.lower()
        white_row = df_meas[df_meas_names.isin(white_names)]
        
        if not white_row.empty:
            Y_white = float(white_row.iloc[0][y_col])
        elif all(c in df_meas.columns for c in ['R', 'G', 'B']):
            white_row = df_meas[(df_meas['R']==255) & (df_meas['G']==255) & (df_meas['B']==255)]
            if not white_row.empty:
                Y_white = float(white_row.iloc[0][y_col])
            else:
                Y_white = float(df_meas[y_col].max())
        else:
            Y_white = float(df_meas[y_col].max())
            
        if Y_white <= 0: return "N/A"
        
        # 4. Target XYZ計算とLab変換
        target_XYZ = rgb_8bit_to_target_XYZ(R, G, B)
        target_Lab = XYZ_to_Lab(*target_XYZ)
        
        # 5. Measured XYZ計算とLab変換
        meas_XYZ = measured_xyY_to_XYZ(x_m, y_m, Y_m, Y_white)
        meas_Lab = XYZ_to_Lab(*meas_XYZ)
            
        # 6. CIEDE2000の算出
        de2000 = delta_E_2000(target_Lab, meas_Lab)
        
        return f"{de2000:.4f}"
        
    except Exception as e:
        return "N/A"

# ==========================================
# ★ 追加: UI表示用に正規化Y値を取得するヘルパー関数
# ==========================================
def get_target_y_norm(df_target, row_name):
    try:
        if df_target is None: return "N/A"
        t_row = df_target[df_target['Name'] == row_name]
        if t_row.empty: return "N/A"
        t_row = t_row.iloc[0]
        if not all(c in t_row for c in ['R', 'G', 'B']): return "N/A"
        R, G, B = float(t_row['R']), float(t_row['G']), float(t_row['B'])
        _, Y, _ = rgb_8bit_to_target_XYZ(R, G, B)
        return f"{Y:.2f}"
    except Exception:
        return "N/A"

def get_measured_y_norm(df_meas, row_meas):
    try:
        if df_meas is None: return "N/A"
        lum_cols = ['Y', 'Lv', 'Luminance', 'L']
        y_col = next((c for c in lum_cols if c in df_meas.columns), None)
        if not y_col: return "N/A"
        
        Y_m = float(row_meas[y_col])
        
        white_names = ['white', 'w', '19', 'patch 19', 'neutral 8']
        df_meas_names = df_meas['Name'].astype(str).str.strip().str.lower()
        white_row = df_meas[df_meas_names.isin(white_names)]
        
        if not white_row.empty:
            Y_white = float(white_row.iloc[0][y_col])
        elif all(c in df_meas.columns for c in ['R', 'G', 'B']):
            white_row = df_meas[(df_meas['R']==255) & (df_meas['G']==255) & (df_meas['B']==255)]
            if not white_row.empty:
                Y_white = float(white_row.iloc[0][y_col])
            else:
                Y_white = float(df_meas[y_col].max())
        else:
            Y_white = float(df_meas[y_col].max())
            
        if Y_white <= 0: return "N/A"
        Y_norm = (Y_m / Y_white) * 100.0
        return f"{Y_norm:.2f}"
    except Exception:
        return "N/A"


# --- Layout: Main Page ---
st.title("CIE 1931 Chromaticity Analyzer")
st.markdown("Upload your CSV files to plot the color data on the CIE 1931 chromaticity diagram.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Target")
    file_target = st.file_uploader("Upload Target CSV", type=["csv"], key="target")
    if file_target is None:
        st.info("Using default: target_machbeth.csv")

with col2:
    st.subheader("Before")
    file_before = st.file_uploader("Upload Before CSV", type=["csv"], key="before")

with col3:
    st.subheader("After")
    file_after = st.file_uploader("Upload After CSV", type=["csv"], key="after")
    
color_space = st.selectbox("Reference Gamut:", ["sRGB", "DCI-P3"])

# --- Core Processing Logic ---
if file_target is not None:
    df_t_full = load_color_data(file_target)
else:
    df_t_full = load_local_csv("target_machbeth.csv")
    
df_b_full = load_color_data(file_before)
df_a_full = load_color_data(file_after)

# Extract unique names for inspector
all_names = []
if df_t_full is not None and 'Name' in df_t_full.columns: all_names.extend(df_t_full['Name'].tolist())
if df_b_full is not None and 'Name' in df_b_full.columns: all_names.extend(df_b_full['Name'].tolist())
if df_a_full is not None and 'Name' in df_a_full.columns: all_names.extend(df_a_full['Name'].tolist())
unique_names = sorted(list(set([str(n) for n in all_names if pd.notna(n)])))

df_t_plot = df_t_full
df_b_plot = df_b_full
df_a_plot = df_a_full

# ==========================================
# 3. Sidebar UI Assembly
# ==========================================

# Logo and Copyright
try:
    logo = Image.open("yitoa.png")
    st.sidebar.image(logo, width='stretch')
except FileNotFoundError:
    st.sidebar.warning("Logo image 'yitoa.png' not found.")

st.sidebar.markdown(
    "<div style='text-align: center; font-size: 0.8em; color: gray; margin-bottom: 20px;'>"
    "Copyright(c) YITOA Technology.<br>All rights reserved."
    "</div>",
    unsafe_allow_html=True
)

st.sidebar.header("Graph Settings")
fig_size = st.sidebar.slider("Graph Size (inches)", min_value=2, max_value=20, value=4, step=1)
show_labels = st.sidebar.checkbox("Show Point Names on Graph", value=False)

# Data Inspector
if unique_names:
    st.sidebar.markdown("---")
    st.sidebar.header("Data Inspector")
    
    selected_name = st.sidebar.selectbox("Select Point ID to inspect:", unique_names)
    plot_only_selected = st.sidebar.checkbox("Plot ONLY selected point", value=False)
    
    if plot_only_selected:
        if df_t_full is not None and 'Name' in df_t_full.columns: df_t_plot = df_t_full[df_t_full['Name'] == selected_name]
        if df_b_full is not None and 'Name' in df_b_full.columns: df_b_plot = df_b_full[df_b_full['Name'] == selected_name]
        if df_a_full is not None and 'Name' in df_a_full.columns: df_a_plot = df_a_full[df_a_full['Name'] == selected_name]
    
    st.sidebar.markdown(f"### Metrics for `{selected_name}`")
    
    if df_t_full is not None and 'Name' in df_t_full.columns:
        t_row = df_t_full[df_t_full['Name'] == selected_name]
        if not t_row.empty:
            t_y_norm = get_target_y_norm(df_t_full, selected_name)
            st.sidebar.markdown(f"<span style='color: black; font-size: 1.2em;'>●</span> **Target Point**:<br>x: `{t_row.iloc[0]['x']:.4f}`<br>y: `{t_row.iloc[0]['y']:.4f}`<br>Y (Norm): `{t_y_norm}`", unsafe_allow_html=True)
            
    if df_b_full is not None and 'Name' in df_b_full.columns:
        b_row = df_b_full[df_b_full['Name'] == selected_name]
        if not b_row.empty:
            de_b = get_delta_e_from_csv(b_row.iloc[0])
            calc_de_b = calculate_custom_delta_e(df_b_full, b_row.iloc[0], df_t_full, selected_name, color_space)
            b_y_norm = get_measured_y_norm(df_b_full, b_row.iloc[0])
            st.sidebar.markdown(f"<span style='color: #FF40FF; font-size: 1.2em;'>●</span> **Before Point**:<br>x: `{b_row.iloc[0]['x']:.4f}`<br>y: `{b_row.iloc[0]['y']:.4f}`<br>Y (Norm): `{b_y_norm}`<br>ΔE (CSV Data): **`{de_b}`**<br>ΔE (Calculated): **`{calc_de_b}`**", unsafe_allow_html=True)
            
    if df_a_full is not None and 'Name' in df_a_full.columns:
        a_row = df_a_full[df_a_full['Name'] == selected_name]
        if not a_row.empty:
            de_a = get_delta_e_from_csv(a_row.iloc[0])
            calc_de_a = calculate_custom_delta_e(df_a_full, a_row.iloc[0], df_t_full, selected_name, color_space)
            a_y_norm = get_measured_y_norm(df_a_full, a_row.iloc[0])
            st.sidebar.markdown(f"<span style='color: #00FA00; font-size: 1.2em;'>●</span> **After Point**:<br>x: `{a_row.iloc[0]['x']:.4f}`<br>y: `{a_row.iloc[0]['y']:.4f}`<br>Y (Norm): `{a_y_norm}`<br>ΔE (CSV Data): **`{de_a}`**<br>ΔE (Calculated): **`{calc_de_a}`**", unsafe_allow_html=True)

# ★ Log Out button at the VERY BOTTOM of the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Log Out"):
    st.session_state['authenticated'] = False
    st.rerun()

# ==========================================
# ★ 追加: Data Overview表示用の列追加ロジック
# ==========================================
def prepare_display_df(df_meas, df_target, color_space):
    if df_meas is None: return None
    df_disp = df_meas.copy()
    if df_target is not None and 'Name' in df_disp.columns:
        calc_de_list = []
        csv_de_list = []
        for _, row in df_disp.iterrows():
            name = row['Name']
            calc_de_list.append(calculate_custom_delta_e(df_meas, row, df_target, name, color_space))
            csv_de_list.append(get_delta_e_from_csv(row))
        df_disp['ΔE (CSV Data)'] = csv_de_list
        df_disp['ΔE (Calculated)'] = calc_de_list
    return df_disp

df_b_display = prepare_display_df(df_b_full, df_t_full, color_space)
df_a_display = prepare_display_df(df_a_full, df_t_full, color_space)
df_t_display = df_t_full.copy() if df_t_full is not None else None

# ==========================================
# 4. Rendering & Output Display
# ==========================================
if df_t_full is not None or df_b_full is not None or df_a_full is not None:
    st.markdown("---")
    
    fig = plot_chromaticity_customized(df_t_plot, df_b_plot, df_a_plot, color_space, fig_size, show_labels)
    st.pyplot(fig, width='content')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    
    st.download_button(
        label="📥 Download Graph as PNG",
        data=buf,
        file_name="cie1931_chromaticity_diagram.png",
        mime="image/png"
    )
    st.markdown("---")

    st.markdown("### Data Overview")
    tab1, tab2, tab3 = st.tabs(["Before Data", "After Data", "Target Data"])
    
    with tab1:
        if df_b_display is not None: st.dataframe(df_b_display, width='stretch')
        else: st.info("No Before data uploaded or invalid format.")
            
    with tab2:
        if df_a_display is not None: st.dataframe(df_a_display, width='stretch')
        else: st.info("No After data uploaded or invalid format.")
            
    with tab3:
        if df_t_display is not None: st.dataframe(df_t_display, width='stretch')
        else: st.info("No Target data uploaded or invalid format.")
else:
    st.info("Please upload at least one CSV file to generate the diagram.")