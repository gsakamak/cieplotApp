import os
import math
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import colour
from colour.plotting import plot_chromaticity_diagram_CIE1931
import warnings
import streamlit as st
import io
import joblib  
from PIL import Image
from sklearn.ensemble import RandomForestRegressor  
from scipy.optimize import differential_evolution   

# --- Hide unnecessary colour-science warnings ---
warnings.filterwarnings('ignore', category=colour.utilities.ColourUsageWarning)

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="CIE 1931 Config Web App", layout="wide")

# ==========================================
# 1. Login Authentication & Session State
# ==========================================
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'predicted_df' not in st.session_state:      
    st.session_state['predicted_df'] = None
if 'tuning_results' not in st.session_state:    
    st.session_state['tuning_results'] = None

def check_login():
    """Validates email domain and password"""
    st.title("🔐 Login")
    st.markdown("Please enter your company email and password to access the app.")
    
    email = st.text_input("Email Address:").strip()
    password = st.text_input("Password:", type="password").strip()
    
    if st.button("Login"):
        if email and password:
            parts = email.split('@')
            if len(parts) == 2:
                domain = parts[1].lower()
                allowed_domains = ["yitoa.co.jp", "yitoa.com"]
                
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

if not st.session_state['authenticated']:
    check_login()
    st.stop()

# ==========================================
# 2. Main Application Features
# ==========================================
def load_color_data_from_bytes(content_bytes):
    try:
        try:
            decoded_str = content_bytes.decode('utf-8-sig')
        except UnicodeDecodeError:
            try:
                decoded_str = content_bytes.decode('cp932')
            except UnicodeDecodeError:
                decoded_str = content_bytes.decode('shift_jis', errors='replace')
                
        content = decoded_str.splitlines()
        
        header_row_index = 0
        for i, line in enumerate(content):
            if "Name" in line and ("R" in line or "x" in line):
                header_row_index = i
                break
                
        content_stream = io.StringIO("\n".join(content))
        df = pd.read_csv(content_stream, skiprows=header_row_index)
        
        if not df.empty:
            df.columns = df.columns.astype(str).str.strip()
        
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

def plot_chromaticity_customized(df_target, df_before, df_after, color_space, fig_size, show_labels, df_pred=None, show_pred=False, xlim=(0.0, 0.9), ylim=(0.0, 0.9)):
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

    # ★ FIX: BT.2020 の基準座標を追加
    if color_space.upper() == 'DCI-P3':
        gamut_x, gamut_y = [0.680, 0.265, 0.150, 0.680], [0.320, 0.690, 0.060, 0.320]
        label_text = 'DCI-P3 (Ref)'
    elif color_space.upper() == 'BT.2020':
        gamut_x, gamut_y = [0.708, 0.170, 0.131, 0.708], [0.292, 0.797, 0.046, 0.292]
        label_text = 'BT.2020 (Ref)'
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

    if show_pred and df_pred is not None and not df_pred.empty and 'x' in df_pred.columns:
        ax.scatter(df_pred['x'], df_pred['y'], color='blue', marker='*', s=marker_data*1.5, label='AI Prediction', zorder=9)
        if show_labels:
            for name, row in df_pred.iterrows():
                ax.text(row['x'] + text_offset*1.5, row['y'] - text_offset*1.5, str(name), fontsize=font_xs, color='blue', zorder=10)

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(visible=True, which='major', color='lightgray', linestyle='-', linewidth=line_w_thin)
    
    legend = ax.legend(loc='upper right', fontsize=font_s)
    if legend:
        for text in legend.get_texts(): text.set_color('black')
            
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    fig.tight_layout()
    return fig

def get_delta_e_from_csv(row):
    for col in row.index:
        clean_col = str(col).strip().lower().replace(" ", "").replace("_", "")
        if clean_col in ['deltae2000', 'deltae', 'de', 'δe']:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                try: return f"{float(val):.2f}"
                except ValueError: return str(val)
    return "N/A"

# ==========================================
# ★ Gamut Area Calculation Logic
# ==========================================
def calculate_gamut_area(df):
    if df is None or 'Name' not in df.columns or 'x' not in df.columns or 'y' not in df.columns:
        return None
    try:
        r_row = df[df['Name'].astype(str).str.strip().str.upper().isin(['R', 'RED'])].iloc[0]
        g_row = df[df['Name'].astype(str).str.strip().str.upper().isin(['G', 'GREEN'])].iloc[0]
        b_row = df[df['Name'].astype(str).str.strip().str.upper().isin(['B', 'BLUE'])].iloc[0]
        
        xr, yr = float(r_row['x']), float(r_row['y'])
        xg, yg = float(g_row['x']), float(g_row['y'])
        xb, yb = float(b_row['x']), float(b_row['y'])
        
        return 0.5 * abs(xr * (yg - yb) + xg * (yb - yr) + xb * (yr - yg))
    except Exception:
        return None

# ==========================================
# ★ Math/Processing Logic
# ==========================================
D65_X, D65_Y, D65_Z = 95.047, 100.000, 108.883

def rgb_8bit_to_target_XYZ(r, g, b):
    r_l, g_l, b_l = (r / 255.0) ** 2.2, (g / 255.0) ** 2.2, (b / 255.0) ** 2.2
    X = (0.4124564 * r_l + 0.3575761 * g_l + 0.1804375 * b_l) * 100.0
    Y = (0.2126729 * r_l + 0.7151522 * g_l + 0.0721750 * b_l) * 100.0
    Z = (0.0193339 * r_l + 0.1191920 * g_l + 0.9503041 * b_l) * 100.0
    return X, Y, Z

def measured_xyY_to_XYZ(x, y, Y_meas, Y_white_meas):
    Y_norm = (Y_meas / Y_white_meas) * 100.0 if Y_white_meas > 0 else 0
    if y == 0: return 0.0, 0.0, 0.0
    X, Z = (x * Y_norm) / y, ((1.0 - x - y) * Y_norm) / y
    return X, Y_norm, Z

def f_lab(t):
    delta = 6.0 / 29.0
    return math.pow(t, 1.0 / 3.0) if t > delta ** 3 else (1.0 / 3.0) * ((29.0 / 6.0) ** 2) * t + (4.0 / 29.0)

def XYZ_to_Lab(X, Y, Z):
    L = 116.0 * f_lab(Y / D65_Y) - 16.0
    a = 500.0 * (f_lab(X / D65_X) - f_lab(Y / D65_Y))
    b = 200.0 * (f_lab(Y / D65_Y) - f_lab(Z / D65_Z))
    return L, a, b

def delta_E_2000(Lab1, Lab2):
    L1, a1, b1 = Lab1; L2, a2, b2 = Lab2
    C1 = math.sqrt(a1**2 + b1**2); C2 = math.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - math.sqrt((C_bar**7) / (C_bar**7 + 25.0**7)))
    a1_prime, a2_prime = (1.0 + G) * a1, (1.0 + G) * a2
    C1_prime = math.sqrt(a1_prime**2 + b1**2); C2_prime = math.sqrt(a2_prime**2 + b2**2)
    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360.0 if (b1 != 0 or a1_prime != 0) else 0.0
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360.0 if (b2 != 0 or a2_prime != 0) else 0.0
    dL_prime, dC_prime = L2 - L1, C2_prime - C1_prime
    if C1_prime * C2_prime == 0: dh_prime = 0.0
    elif abs(h2_prime - h1_prime) <= 180.0: dh_prime = h2_prime - h1_prime
    elif h2_prime <= h1_prime: dh_prime = h2_prime - h1_prime + 360.0
    else: dh_prime = h2_prime - h1_prime - 360.0
    dH_prime = 2.0 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(dh_prime / 2.0))
    L_bar_prime, C_bar_prime = (L1 + L2) / 2.0, (C1_prime + C2_prime) / 2.0
    if C1_prime * C2_prime == 0: H_bar_prime = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180.0: H_bar_prime = (h1_prime + h2_prime) / 2.0
    elif (h1_prime + h2_prime) < 360.0: H_bar_prime = (h1_prime + h2_prime + 360.0) / 2.0
    else: H_bar_prime = (h1_prime + h2_prime - 360.0) / 2.0
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
    return math.sqrt((dL_prime / S_L)**2 + (dC_prime / S_C)**2 + (dH_prime / S_H)**2 + R_T * (dC_prime / S_C) * (dH_prime / S_H))

def calculate_custom_delta_e(df_meas, row_meas, df_target, row_name, color_space):
    try:
        if df_target is None or df_meas is None: return "N/A"
        t_row = df_target[df_target['Name'] == row_name]
        if t_row.empty: return "N/A"
        t_row = t_row.iloc[0]
        
        target_Lab = None
        lum_cols = ['Y', 'Lv', 'Luminance', 'L']
        y_col_t = next((c for c in lum_cols if c in t_row.index), None)
        
        if 'x' in t_row.index and 'y' in t_row.index:
            x_t, y_t = float(t_row['x']), float(t_row['y'])
            
            if y_col_t:
                Y_t = float(t_row[y_col_t])
            elif all(c in t_row for c in ['R', 'G', 'B']):
                R, G, B = float(t_row['R']), float(t_row['G']), float(t_row['B'])
                _, Y_t, _ = rgb_8bit_to_target_XYZ(R, G, B)
            else:
                return "N/A"

            if y_t == 0:
                X_t, Z_t = 0.0, 0.0
            else:
                X_t = (x_t * Y_t) / y_t
                Z_t = ((1.0 - x_t - y_t) * Y_t) / y_t
            target_Lab = XYZ_to_Lab(X_t, Y_t, Z_t)
            
        elif all(c in t_row for c in ['R', 'G', 'B']):
            R, G, B = float(t_row['R']), float(t_row['G']), float(t_row['B'])
            target_XYZ = rgb_8bit_to_target_XYZ(R, G, B)
            target_Lab = XYZ_to_Lab(*target_XYZ)
        else:
            return "N/A"
            
        res_m = get_meas_xyz_lab(df_meas, row_meas, df_target)
        if res_m is not None:
            _, meas_Lab = res_m
            return f"{delta_E_2000(target_Lab, meas_Lab):.4f}"
            
        return "N/A"
    except Exception: return "N/A"

def get_target_y_norm(df_target, row_name):
    try:
        if df_target is None: return "N/A"
        t_row = df_target[df_target['Name'] == row_name]
        if t_row.empty: return "N/A"
        t_row = t_row.iloc[0]
        
        lum_cols = ['Y', 'Lv', 'Luminance', 'L']
        y_col_t = next((c for c in lum_cols if c in t_row.index), None)
        
        if 'x' in t_row.index and 'y' in t_row.index:
            if y_col_t:
                return f"{float(t_row[y_col_t]):.2f}"
            elif all(c in t_row for c in ['R', 'G', 'B']):
                R, G, B = float(t_row['R']), float(t_row['G']), float(t_row['B'])
                _, Y, _ = rgb_8bit_to_target_XYZ(R, G, B)
                return f"{Y:.2f}"
            
        if all(c in t_row for c in ['R', 'G', 'B']):
            R, G, B = float(t_row['R']), float(t_row['G']), float(t_row['B'])
            _, Y, _ = rgb_8bit_to_target_XYZ(R, G, B)
            return f"{Y:.2f}"
            
        return "N/A"
    except Exception: return "N/A"

def get_measured_y_norm(df_meas, row_meas, df_target):
    try:
        if df_meas is None: return "N/A"
        lum_cols = ['Y', 'Lv', 'Luminance', 'L']
        y_col = next((c for c in lum_cols if c in df_meas.columns), None)
        if not y_col: return "N/A"
        Y_m = float(row_meas[y_col])
        
        Y_white = None
        if df_target is not None and all(c in df_target.columns for c in ['R', 'G', 'B']):
            t_white = df_target[(df_target['R'] == 255) & (df_target['G'] == 255) & (df_target['B'] == 255)]
            if not t_white.empty:
                true_white_name = str(t_white.iloc[0]['Name']).strip()
                m_white = df_meas[df_meas['Name'].astype(str).str.strip() == true_white_name]
                if not m_white.empty: Y_white = float(m_white.iloc[0][y_col])
        if Y_white is None:
            white_names = ['white', 'w', 'patch 19', 'neutral 8']
            df_meas_names = df_meas['Name'].astype(str).str.strip().str.lower()
            m_white = df_meas[df_meas_names.isin(white_names)]
            if not m_white.empty: Y_white = float(m_white.iloc[0][y_col])
            else: Y_white = float(df_meas[y_col].max())
            
        if Y_white <= 0: return "N/A"
        return f"{(Y_m / Y_white) * 100.0:.2f}"
    except Exception: return "N/A"

def get_meas_xyz_lab(df_meas, row_meas, df_target):
    try:
        if df_meas is None: return None
        lum_cols = ['Y', 'Lv', 'Luminance', 'L']
        y_col = next((c for c in lum_cols if c in df_meas.columns), None)
        if not y_col: return None
        
        Y_m = float(row_meas[y_col])
        x_m, y_m_val = float(row_meas['x']), float(row_meas['y'])
        
        Y_white = None
        if df_target is not None and all(c in df_target.columns for c in ['R', 'G', 'B']):
            t_white = df_target[(df_target['R'] == 255) & (df_target['G'] == 255) & (df_target['B'] == 255)]
            if not t_white.empty:
                true_white_name = str(t_white.iloc[0]['Name']).strip()
                m_white = df_meas[df_meas['Name'].astype(str).str.strip() == true_white_name]
                if not m_white.empty: Y_white = float(m_white.iloc[0][y_col])
        if Y_white is None:
            white_names = ['white', 'w', 'patch 19', 'neutral 8']
            df_meas_names = df_meas['Name'].astype(str).str.strip().str.lower()
            m_white = df_meas[df_meas_names.isin(white_names)]
            if not m_white.empty: Y_white = float(m_white.iloc[0][y_col])
            else: Y_white = float(df_meas[y_col].max())
            
        if Y_white <= 0: return None
        
        m_X, m_Y_norm, m_Z = measured_xyY_to_XYZ(x_m, y_m_val, Y_m, Y_white)
        m_L, m_a, m_b = XYZ_to_Lab(m_X, m_Y_norm, m_Z)
        return (m_X, m_Y_norm, m_Z), (m_L, m_a, m_b)
    except Exception: return None

# ==========================================
# ★ AI / Optimizer Engine
# ==========================================
def Lab_to_XYZ_vec(Lab_array):
    L, a, b = Lab_array[:, 0], Lab_array[:, 1], Lab_array[:, 2]
    fy = (L + 16.0) / 116.0; fx = a / 500.0 + fy; fz = fy - b / 200.0
    delta = 6.0 / 29.0
    f_inv = lambda t: np.where(t > delta, t**3, 3.0 * (delta**2) * (t - 4.0/29.0))
    return np.column_stack((D65_X * f_inv(fx), D65_Y * f_inv(fy), D65_Z * f_inv(fz)))

def XYZ_to_sRGB_vec(XYZ_array):
    x, y, z = XYZ_array[:, 0] / 100.0, XYZ_array[:, 1] / 100.0, XYZ_array[:, 2] / 100.0
    r =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    gamma_corr = lambda c: np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.maximum(c, 0) ** (1/2.4)) - 0.055)
    return np.clip(np.column_stack((gamma_corr(r), gamma_corr(g), gamma_corr(b))) * 255.0, 0, 255)

def sRGB_to_XYZ_vec(RGB_array):
    r, g, b = RGB_array[:, 0]/255.0, RGB_array[:, 1]/255.0, RGB_array[:, 2]/255.0
    inv_gamma = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = inv_gamma(r), inv_gamma(g), inv_gamma(b)
    X = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) * 100.0
    Y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) * 100.0
    Z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) * 100.0
    return np.column_stack((X, Y, Z))

def XYZ_to_Lab_vec(XYZ_array):
    X, Y, Z = XYZ_array[:, 0], XYZ_array[:, 1], XYZ_array[:, 2]
    f_lab_v = lambda t: np.where(t > (6.0/29.0)**3, t**(1.0/3.0), (1.0/3.0) * ((29.0/6.0)**2) * t + (4.0/29.0))
    fx, fy, fz = f_lab_v(X / D65_X), f_lab_v(Y / D65_Y), f_lab_v(Z / D65_Z)
    return np.column_stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)))

def Lab_to_xy_vec(Lab_array):
    XYZ = Lab_to_XYZ_vec(Lab_array)
    sum_XYZ = np.sum(XYZ, axis=1)
    sum_XYZ[sum_XYZ == 0] = 1e-10
    x = XYZ[:, 0] / sum_XYZ
    y = XYZ[:, 1] / sum_XYZ
    return np.column_stack((x, y))

class YT7875_HybridDigitalTwin:
    def __init__(self):
        self.bounds_W = [(0, 252)] * 3
        self.bounds_Colors = [(-128, 127)] * 18
        self.full_bounds = self.bounds_W + self.bounds_Colors
        self.error_corrector = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
        self.has_training_data = False
        self.model_filepath = "yt7875_ai_model.pkl"

    def load_trained_model(self):
        if os.path.exists(self.model_filepath):
            try:
                self.error_corrector = joblib.load(self.model_filepath)
                self.has_training_data = True
                return True
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return False
        return False

    def update_model(self, df_history):
        if df_history is None or len(df_history) == 0: return
        X_train, Y_error = [], []
        for _, row in df_history.iterrows():
            b_lab = row[['L_bef', 'a_bef', 'b_bef']].values.astype(float)
            regs = row[[f'Reg_{i}' for i in range(21)]].values.astype(int)
            actual_aft = row[['L_aft', 'a_aft', 'b_aft']].values.astype(float)
            math_pred = self._simulate_hardware_math_lab_vec(b_lab.reshape(1,3), regs)[0]
            X_train.append(np.concatenate([b_lab, regs]))
            Y_error.append(actual_aft - math_pred)
        
        self.error_corrector.fit(X_train, Y_error)
        self.has_training_data = True
        joblib.dump(self.error_corrector, self.model_filepath)

    def _parse_registers(self, reg_21_vars):
        dW = np.array(reg_21_vars[0:3]) * 0.25 - 63.0
        dR, dG, dB = np.array(reg_21_vars[3:6]), np.array(reg_21_vars[6:9]), np.array(reg_21_vars[9:12])
        dC, dM, dY = np.array(reg_21_vars[12:15]), np.array(reg_21_vars[15:18]), np.array(reg_21_vars[18:21])
        return dW, dR, dG, dB, dC, dM, dY

    def _simulate_hardware_math_vectorized(self, input_rgb_array, reg_21_vars):
        rgb = np.array(input_rgb_array) / 255.0
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        dW, dR, dG, dB, dC, dM, dY = self._parse_registers(reg_21_vars)
        offset = np.zeros_like(rgb)
        c1 = (r >= g) & (g >= b)
        offset[c1] = np.outer(r[c1]-g[c1], dR) + np.outer(g[c1]-b[c1], dY) + np.outer(b[c1], dW)
        c2 = (r >= b) & (b >= g) & ~c1
        offset[c2] = np.outer(r[c2]-b[c2], dR) + np.outer(b[c2]-g[c2], dM) + np.outer(g[c2], dW)
        c3 = (g >= r) & (r >= b) & ~c1 & ~c2
        offset[c3] = np.outer(g[c3]-r[c3], dG) + np.outer(r[c3]-b[c3], dY) + np.outer(b[c3], dW)
        c4 = (g >= b) & (b >= r) & ~c1 & ~c2 & ~c3
        offset[c4] = np.outer(g[c4]-b[c4], dG) + np.outer(b[c4]-r[c4], dC) + np.outer(r[c4], dW)
        c5 = (b >= r) & (r >= g) & ~c1 & ~c2 & ~c3 & ~c4
        offset[c5] = np.outer(b[c5]-r[c5], dB) + np.outer(r[c5]-g[c5], dM) + np.outer(g[c5], dW)
        c6 = (b >= g) & (g >= r) & ~c1 & ~c2 & ~c3 & ~c4 & ~c5
        offset[c6] = np.outer(b[c6]-g[c6], dB) + np.outer(g[c6]-r[c6], dC) + np.outer(r[c6], dW)
        return np.clip(input_rgb_array + offset, 0, 255)

    def _simulate_hardware_math_lab_vec(self, before_lab_array, reg_21_vars):
        XYZ = Lab_to_XYZ_vec(before_lab_array)
        sRGB = XYZ_to_sRGB_vec(XYZ)
        after_sRGB = self._simulate_hardware_math_vectorized(sRGB, reg_21_vars)
        return XYZ_to_Lab_vec(sRGB_to_XYZ_vec(after_sRGB))

    def predict_hybrid_vec(self, before_lab_array, reg_21_vars):
        math_pred = self._simulate_hardware_math_lab_vec(before_lab_array, reg_21_vars)
        if self.has_training_data:
            regs_tiled = np.tile(reg_21_vars, (before_lab_array.shape[0], 1))
            X_input = np.hstack((before_lab_array, regs_tiled))
            math_pred += self.error_corrector.predict(X_input)
        return math_pred

    def optimize_21_registers(self, before_lab_array, target_lab_array):
        N = before_lab_array.shape[0]
        def cost_function(reg_21_vars):
            preds = self.predict_hybrid_vec(before_lab_array, reg_21_vars)
            return sum(delta_E_2000(target_lab_array[i], preds[i]) for i in range(N)) / N

        result = differential_evolution(
            cost_function, 
            self.full_bounds, 
            strategy='best1bin', 
            maxiter=40, 
            popsize=15, 
            tol=0.01, 
            workers=1,
            seed=42
        )
        return np.round(result.x).astype(int), result.fun

def extract_ai_lab_data(df, df_target, is_target=False):
    if df is None: return pd.DataFrame()
    names, labs = [], []
    for _, row in df.iterrows():
        name = str(row['Name']).strip()
        if is_target:
            lum_cols = ['Y', 'Lv', 'Luminance', 'L']
            y_col_t = next((c for c in lum_cols if c in df.columns), None)
            
            if 'x' in df.columns and 'y' in df.columns:
                x_t, y_t = float(row['x']), float(row['y'])
                
                if y_col_t:
                    Y_t = float(row[y_col_t])
                elif all(c in df.columns for c in ['R', 'G', 'B']) and pd.notna(row.get('R')):
                    _, Y_t, _ = rgb_8bit_to_target_XYZ(float(row['R']), float(row['G']), float(row['B']))
                else:
                    continue
                    
                if y_t == 0:
                    X_t, Z_t = 0.0, 0.0
                else:
                    X_t = (x_t * Y_t) / y_t
                    Z_t = ((1.0 - x_t - y_t) * Y_t) / y_t
                labs.append(XYZ_to_Lab(X_t, Y_t, Z_t))
                names.append(name)
            elif all(c in df.columns for c in ['R', 'G', 'B']) and pd.notna(row.get('R')):
                X, Y, Z = rgb_8bit_to_target_XYZ(float(row['R']), float(row['G']), float(row['B']))
                labs.append(XYZ_to_Lab(X, Y, Z))
                names.append(name)
        else:
            res = get_meas_xyz_lab(df, row, df_target)
            if res is not None:
                _, (L, a, b) = res
                labs.append((L, a, b))
                names.append(name)
                
    return pd.DataFrame(labs, columns=['L', 'a', 'b'], index=names)


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
    
# ★ FIX: BT.2020 を追加
color_space = st.selectbox("Reference Gamut:", ["sRGB", "DCI-P3", "BT.2020"])

# --- Core Processing Logic ---
if file_target is not None:
    df_t_full = load_color_data(file_target)
else:
    df_t_full = load_local_csv("target_machbeth.csv")
    
df_b_full = load_color_data(file_before)
df_a_full = load_color_data(file_after)

all_names = []
if df_t_full is not None and 'Name' in df_t_full.columns: all_names.extend(df_t_full['Name'].tolist())
if df_b_full is not None and 'Name' in df_b_full.columns: all_names.extend(df_b_full['Name'].tolist())
if df_a_full is not None and 'Name' in df_a_full.columns: all_names.extend(df_a_full['Name'].tolist())

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

unique_names = sorted(list(set([str(n) for n in all_names if pd.notna(n)])), key=natural_sort_key)

# ==========================================
# 3. Sidebar UI Assembly
# ==========================================
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
show_xyz_lab = st.sidebar.checkbox("Show XYZ & L*a*b* Values", value=False) 
show_ai_pred = st.sidebar.checkbox("Show AI Prediction", value=True)

show_target = st.sidebar.checkbox("Show Target Data on Plot", value=True)
show_before = st.sidebar.checkbox("Show Before Data on Plot", value=True)
show_after = st.sidebar.checkbox("Show After Data on Plot", value=True)

x_axis_range = st.sidebar.slider("X-Axis Range", min_value=0.0, max_value=1.0, value=(0.0, 0.9), step=0.01)
y_axis_range = st.sidebar.slider("Y-Axis Range", min_value=0.0, max_value=1.0, value=(0.0, 0.9), step=0.01)

df_t_plot = df_t_full
df_b_plot = df_b_full
df_a_plot = df_a_full
df_pred_plot = st.session_state['predicted_df']

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
        if df_pred_plot is not None:
            if selected_name in df_pred_plot.index:
                df_pred_plot = df_pred_plot.loc[[selected_name]]
            else:
                df_pred_plot = pd.DataFrame(columns=['x', 'y'])
    
    st.sidebar.markdown(f"### Metrics for `{selected_name}`")
    
    # Target Data
    if df_t_full is not None and 'Name' in df_t_full.columns:
        t_row = df_t_full[df_t_full['Name'] == selected_name]
        if not t_row.empty:
            t_y_norm = get_target_y_norm(df_t_full, selected_name)
            text_t = f"<span style='color: black; font-size: 1.2em;'>●</span> **Target Point**:<br>x: `{t_row.iloc[0]['x']:.4f}`<br>y: `{t_row.iloc[0]['y']:.4f}`<br>Y: `{t_y_norm}`"
            if show_xyz_lab:
                try:
                    lum_cols = ['Y', 'Lv', 'Luminance', 'L']
                    y_col_t = next((c for c in lum_cols if c in t_row.iloc[0].index), None)
                    
                    if 'x' in t_row.iloc[0].index and 'y' in t_row.iloc[0].index:
                        x_t, y_t = float(t_row.iloc[0]['x']), float(t_row.iloc[0]['y'])
                        
                        if y_col_t:
                            Y_t = float(t_row.iloc[0][y_col_t])
                        elif all(c in t_row.iloc[0] for c in ['R', 'G', 'B']):
                            R, G, B = float(t_row.iloc[0]['R']), float(t_row.iloc[0]['G']), float(t_row.iloc[0]['B'])
                            _, Y_t, _ = rgb_8bit_to_target_XYZ(R, G, B)
                        else:
                            Y_t = 0
                            
                        if Y_t > 0:
                            if y_t == 0:
                                X_t, Z_t = 0.0, 0.0
                            else:
                                X_t, Z_t = (x_t * Y_t) / y_t, ((1.0 - x_t - y_t) * Y_t) / y_t
                            t_L, t_a, t_b = XYZ_to_Lab(X_t, Y_t, Z_t)
                            text_t += f"<br>XYZ: `{X_t:.2f}, {Y_t:.2f}, {Z_t:.2f}`<br>L*a*b*: `{t_L:.2f}, {t_a:.2f}, {t_b:.2f}`"
                            
                    elif all(c in t_row.iloc[0] for c in ['R', 'G', 'B']):
                        R, G, B = float(t_row.iloc[0]['R']), float(t_row.iloc[0]['G']), float(t_row.iloc[0]['B'])
                        t_X, t_Y_val, t_Z = rgb_8bit_to_target_XYZ(R, G, B)
                        t_L, t_a, t_b = XYZ_to_Lab(t_X, t_Y_val, t_Z)
                        text_t += f"<br>XYZ: `{t_X:.2f}, {t_Y_val:.2f}, {t_Z:.2f}`<br>L*a*b*: `{t_L:.2f}, {t_a:.2f}, {t_b:.2f}`"
                except Exception: pass
            st.sidebar.markdown(text_t, unsafe_allow_html=True)
            
    # Before Data
    if df_b_full is not None and 'Name' in df_b_full.columns:
        b_row = df_b_full[df_b_full['Name'] == selected_name]
        if not b_row.empty:
            de_b = get_delta_e_from_csv(b_row.iloc[0])
            calc_de_b = calculate_custom_delta_e(df_b_full, b_row.iloc[0], df_t_full, selected_name, color_space)
            b_y_norm = get_measured_y_norm(df_b_full, b_row.iloc[0], df_t_full)
            text_b = f"<span style='color: #FF40FF; font-size: 1.2em;'>●</span> **Before Point**:<br>x: `{b_row.iloc[0]['x']:.4f}`<br>y: `{b_row.iloc[0]['y']:.4f}`<br>Y (Norm): `{b_y_norm}`"
            if show_xyz_lab:
                res = get_meas_xyz_lab(df_b_full, b_row.iloc[0], df_t_full)
                if res:
                    (m_X, m_Y, m_Z), (m_L, m_a, m_b) = res
                    text_b += f"<br>XYZ: `{m_X:.2f}, {m_Y:.2f}, {m_Z:.2f}`<br>L*a*b*: `{m_L:.2f}, {m_a:.2f}, {m_b:.2f}`"
            text_b += f"<br>ΔE (CSV Data): **`{de_b}`**<br>ΔE (Calculated): **`{calc_de_b}`**"
            st.sidebar.markdown(text_b, unsafe_allow_html=True)
            
    # After Data
    if df_a_full is not None and 'Name' in df_a_full.columns:
        a_row = df_a_full[df_a_full['Name'] == selected_name]
        if not a_row.empty:
            de_a = get_delta_e_from_csv(a_row.iloc[0])
            calc_de_a = calculate_custom_delta_e(df_a_full, a_row.iloc[0], df_t_full, selected_name, color_space)
            a_y_norm = get_measured_y_norm(df_a_full, a_row.iloc[0], df_t_full)
            text_a = f"<span style='color: #00FA00; font-size: 1.2em;'>●</span> **After Point**:<br>x: `{a_row.iloc[0]['x']:.4f}`<br>y: `{a_row.iloc[0]['y']:.4f}`<br>Y (Norm): `{a_y_norm}`"
            if show_xyz_lab:
                res = get_meas_xyz_lab(df_a_full, a_row.iloc[0], df_t_full)
                if res:
                    (m_X, m_Y, m_Z), (m_L, m_a, m_b) = res
                    text_a += f"<br>XYZ: `{m_X:.2f}, {m_Y:.2f}, {m_Z:.2f}`<br>L*a*b*: `{m_L:.2f}, {m_a:.2f}, {m_b:.2f}`"
            text_a += f"<br>ΔE (CSV Data): **`{de_a}`**<br>ΔE (Calculated): **`{calc_de_a}`**"
            st.sidebar.markdown(text_a, unsafe_allow_html=True)

    # Simulation Data (AI Prediction)
    if st.session_state['tuning_results'] is not None and selected_name in st.session_state['tuning_results']['common_names']:
        res_ai = st.session_state['tuning_results']
        idx_ai = list(res_ai['common_names']).index(selected_name)
        p_L, p_a, p_b = res_ai['predicted_labs'][idx_ai]
        p_de = res_ai['patch_des'][idx_ai]
        
        if st.session_state['predicted_df'] is not None and selected_name in st.session_state['predicted_df'].index:
            p_x = st.session_state['predicted_df'].loc[selected_name, 'x']
            p_y = st.session_state['predicted_df'].loc[selected_name, 'y']
        else:
            p_x, p_y = np.nan, np.nan
        
        p_XYZ = Lab_to_XYZ_vec(np.array([[p_L, p_a, p_b]]))[0]
        p_X, p_Y_val, p_Z = p_XYZ
        
        text_p = f"<span style='color: blue; font-size: 1.2em;'>★</span> **AI Simulation Point**:<br>x: `{p_x:.4f}`<br>y: `{p_y:.4f}`<br>Y (Norm): `{p_Y_val:.2f}`"
        if show_xyz_lab:
            text_p += f"<br>XYZ: `{p_X:.2f}, {p_Y_val:.2f}, {p_Z:.2f}`<br>L*a*b*: `{p_L:.2f}, {p_a:.2f}, {p_b:.2f}`"
        text_p += f"<br>ΔE (vs Target): **`{p_de:.4f}`**"
        st.sidebar.markdown(text_p, unsafe_allow_html=True)

st.sidebar.markdown("---")
if st.sidebar.button("Log Out"):
    st.session_state['authenticated'] = False
    st.rerun()

# ==========================================
# ★ Data Overview Processing
# ==========================================
def prepare_display_df(df_meas, df_target, color_space, show_ext=False):
    if df_meas is None: return None
    df_disp = df_meas.copy()
    
    if 'Name' in df_disp.columns:
        csv_de_list = []
        for _, row in df_disp.iterrows():
            csv_de_list.append(get_delta_e_from_csv(row))
        df_disp['ΔE (CSV Data)'] = csv_de_list
        
        calc_de_list = []
        for _, row in df_disp.iterrows():
            name = row['Name']
            if df_target is not None:
                calc_de_list.append(calculate_custom_delta_e(df_meas, row, df_target, name, color_space))
            else:
                calc_de_list.append("N/A")
        df_disp['ΔE (Calculated)'] = calc_de_list
        
        if show_ext:
            x_list, y_norm_list, z_list = [], [], []
            l_list, a_list, b_list = [], [], []
            for _, row in df_disp.iterrows():
                res = get_meas_xyz_lab(df_meas, row, df_target)
                if res:
                    (m_X, m_Y, m_Z), (m_L, m_a, m_b) = res
                    x_list.append(round(m_X, 2))
                    y_norm_list.append(round(m_Y, 2))
                    z_list.append(round(m_Z, 2))
                    l_list.append(round(m_L, 2))
                    a_list.append(round(m_a, 2))
                    b_list.append(round(m_b, 2))
                else:
                    x_list.append(np.nan)
                    y_norm_list.append(np.nan)
                    z_list.append(np.nan)
                    l_list.append(np.nan)
                    a_list.append(np.nan)
                    b_list.append(np.nan)
                    
            df_disp['X'] = x_list
            df_disp['Y (Norm)'] = y_norm_list
            df_disp['Z'] = z_list
            df_disp['L*'] = l_list
            df_disp['a*'] = a_list
            df_disp['b*'] = b_list
            
    return df_disp

df_b_display = prepare_display_df(df_b_full, df_t_full, color_space, show_xyz_lab)
df_a_display = prepare_display_df(df_a_full, df_t_full, color_space, show_xyz_lab)

df_t_display = df_t_full.copy() if df_t_full is not None else None

if show_xyz_lab and df_t_display is not None:
    x_list, y_list, z_list = [], [], []
    l_list, a_list, b_list = [], [], []
    for _, row in df_t_display.iterrows():
        try:
            lum_cols = ['Y', 'Lv', 'Luminance', 'L']
            y_col_t = next((c for c in lum_cols if c in row.index), None)
            
            if 'x' in row.index and 'y' in row.index:
                x_t, y_t = float(row['x']), float(row['y'])
                
                if y_col_t:
                    Y_t = float(row[y_col_t])
                elif all(c in df_t_display.columns for c in ['R', 'G', 'B']):
                    R, G, B = float(row['R']), float(row['G']), float(row['B'])
                    _, Y_t, _ = rgb_8bit_to_target_XYZ(R, G, B)
                else:
                    raise ValueError
                
                if y_t == 0:
                    t_X, t_Z = 0.0, 0.0
                else:
                    t_X, t_Z = (x_t * Y_t) / y_t, ((1.0 - x_t - y_t) * Y_t) / y_t
                t_Y_val = Y_t
                t_L, t_a, t_b = XYZ_to_Lab(t_X, t_Y_val, t_Z)
                
            elif all(c in df_t_display.columns for c in ['R', 'G', 'B']):
                R, G, B = float(row['R']), float(row['G']), float(row['B'])
                t_X, t_Y_val, t_Z = rgb_8bit_to_target_XYZ(R, G, B)
                t_L, t_a, t_b = XYZ_to_Lab(t_X, t_Y_val, t_Z)
            else:
                raise ValueError
                
            x_list.append(round(t_X, 2))
            y_list.append(round(t_Y_val, 2))
            z_list.append(round(t_Z, 2))
            l_list.append(round(t_L, 2))
            a_list.append(round(t_a, 2))
            b_list.append(round(t_b, 2))
        except Exception:
            x_list.append(np.nan)
            y_list.append(np.nan)
            z_list.append(np.nan)
            l_list.append(np.nan)
            a_list.append(np.nan)
            b_list.append(np.nan)
            
    df_t_display['X'] = x_list
    df_t_display['Y'] = y_list
    df_t_display['Z'] = z_list
    df_t_display['L*'] = l_list
    df_t_display['a*'] = a_list
    df_t_display['b*'] = b_list

# ==========================================
# 4. Rendering & Output Display
# ==========================================
if df_t_full is not None or df_b_full is not None or df_a_full is not None:
    st.markdown("---")
    
    df_t_render = df_t_plot if show_target else None
    df_b_render = df_b_plot if show_before else None
    df_a_render = df_a_plot if show_after else None
    
    fig = plot_chromaticity_customized(
        df_t_render, df_b_render, df_a_render, color_space, fig_size, 
        show_labels, df_pred=df_pred_plot, show_pred=show_ai_pred,
        xlim=x_axis_range, ylim=y_axis_range
    )
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

    # ==========================================
    # ★ Gamut Ratio Coverage Logic
    # ==========================================
    st.markdown("### 📊 Gamut Coverage Ratio")
    
    def get_gamut_area(df):
        if df is None or 'Name' not in df.columns or 'x' not in df.columns or 'y' not in df.columns:
            return None
        try:
            r_row = df[df['Name'].astype(str).str.strip().str.upper().isin(['R', 'RED'])].iloc[0]
            g_row = df[df['Name'].astype(str).str.strip().str.upper().isin(['G', 'GREEN'])].iloc[0]
            b_row = df[df['Name'].astype(str).str.strip().str.upper().isin(['B', 'BLUE'])].iloc[0]
            
            xr, yr = float(r_row['x']), float(r_row['y'])
            xg, yg = float(g_row['x']), float(g_row['y'])
            xb, yb = float(b_row['x']), float(b_row['y'])
            
            return 0.5 * abs(xr * (yg - yb) + xg * (yb - yr) + xb * (yr - yg))
        except Exception:
            return None
            
    # ★ FIX: BT.2020 の基準面積を追加
    if color_space.upper() == "DCI-P3":
        ref_area = 0.1520
    elif color_space.upper() == "BT.2020":
        ref_area = 0.21187
    else:
        ref_area = 0.11205
        
    area_b = get_gamut_area(df_b_full)
    area_a = get_gamut_area(df_a_full)
    
    ratio_b = (area_b / ref_area) * 100 if area_b else None
    ratio_a = (area_a / ref_area) * 100 if area_a else None
    
    m1, m2 = st.columns(2)
    with m1:
        if ratio_b is not None:
            st.metric(label=f"Before Data ({color_space} Ratio)", value=f"{ratio_b:.1f} %")
        else:
            st.metric(label=f"Before Data ({color_space} Ratio)", value="N/A (Missing R/G/B)")
    with m2:
        if ratio_a is not None:
            delta = ratio_a - ratio_b if ratio_b is not None else None
            st.metric(label=f"After Data ({color_space} Ratio)", value=f"{ratio_a:.1f} %", delta=f"{delta:+.1f} %" if delta is not None else None)
        else:
            st.metric(label=f"After Data ({color_space} Ratio)", value="N/A (Missing R/G/B)")
            
    st.markdown("---")

    # ==========================================
    # ★ ΔE Line Chart
    # ==========================================
    st.markdown("### ΔE Line Chart (Before vs After vs Simulation)")
    
    col_chart_1, col_chart_2 = st.columns([3, 1])
    
    with col_chart_1:
        selected_names_for_plot = st.multiselect(
            "Select Data Points to Plot:", 
            options=unique_names, 
            default=unique_names
        )
        
        has_calculated = False
        if df_b_display is not None and 'ΔE (Calculated)' in df_b_display.columns:
            if any(str(v).strip() != "N/A" for v in df_b_display['ΔE (Calculated)']):
                has_calculated = True
        if df_a_display is not None and 'ΔE (Calculated)' in df_a_display.columns:
            if any(str(v).strip() != "N/A" for v in df_a_display['ΔE (Calculated)']):
                has_calculated = True
                
        default_de_idx = 0 if has_calculated else 1
        de_source = st.radio("Select ΔE Source for Line Chart:", ["Calculated", "CSV Data"], horizontal=True, index=default_de_idx)
        
    with col_chart_2:
        ymax_de = st.number_input("Y-axis Max (ΔE)", min_value=2.0, max_value=200.0, value=20.0, step=1.0)
        show_before_line = st.checkbox("Show Before", value=True)
        show_after_line = st.checkbox("Show After", value=True)
        
    col_name = 'ΔE (Calculated)' if de_source == "Calculated" else 'ΔE (CSV Data)'
    
    de_b_vals = [np.nan] * len(selected_names_for_plot)
    de_a_vals = [np.nan] * len(selected_names_for_plot)
    de_p_vals = [np.nan] * len(selected_names_for_plot)
    
    res_ai = st.session_state.get('tuning_results')

    for i, name in enumerate(selected_names_for_plot):
        if df_b_display is not None:
            r = df_b_display[df_b_display['Name'] == name]
            if not r.empty:
                val = r.iloc[0].get(col_name, "N/A")
                if str(val).strip() != "N/A":
                    try: de_b_vals[i] = float(val)
                    except ValueError: pass
        if df_a_display is not None:
            r = df_a_display[df_a_display['Name'] == name]
            if not r.empty:
                val = r.iloc[0].get(col_name, "N/A")
                if str(val).strip() != "N/A":
                    try: de_a_vals[i] = float(val)
                    except ValueError: pass
        
        if res_ai is not None and name in res_ai['common_names']:
            idx_ai = list(res_ai['common_names']).index(name)
            de_p_vals[i] = float(res_ai['patch_des'][idx_ai])
            
    if selected_names_for_plot:
        fig_line, ax_line = plt.subplots(figsize=(10, 4))
        ax_line.axhline(y=2.0, color='gray', linestyle='-', linewidth=1.5, label='ΔE = 2.0')
        ax_line.axhspan(0, 2.0, color='#00FA00', alpha=0.1)
        
        if show_before_line and any(not np.isnan(v) for v in de_b_vals):
            ax_line.plot(selected_names_for_plot, de_b_vals, marker='o', color='#FF40FF', linestyle='--', label='Before')
        if show_after_line and any(not np.isnan(v) for v in de_a_vals):
            ax_line.plot(selected_names_for_plot, de_a_vals, marker='o', color='#00FA00', linestyle='--', label='After')
        if show_ai_pred and any(not np.isnan(v) for v in de_p_vals):
            ax_line.plot(selected_names_for_plot, de_p_vals, marker='*', color='blue', linestyle=':', markersize=8, label='AI Prediction (Simulation)')
            
        ax_line.set_ylim(0, ymax_de)  
        ax_line.set_xlabel('Image Name')
        
        ax_line.set_ylabel('ΔE')
        
        ax_line.set_xticks(range(len(selected_names_for_plot)))
        ax_line.set_xticklabels(selected_names_for_plot, rotation=45, ha='right')
        ax_line.grid(True, linestyle=':', alpha=0.6)
        ax_line.legend(loc='upper right')
        fig_line.tight_layout()
        st.pyplot(fig_line, width='content')
        
        valid_b = [v for v in de_b_vals if not np.isnan(v)]
        valid_a = [v for v in de_a_vals if not np.isnan(v)]
        valid_p = [v for v in de_p_vals if not np.isnan(v)]

        st.markdown("**ΔE Statistics (Plotted Data)**")
        stat_cols = st.columns(3)
        with stat_cols[0]:
            if show_before_line and valid_b:
                st.markdown(f"<div style='color:#FF40FF;'><b>Before</b><br>Avg: {sum(valid_b)/len(valid_b):.2f}<br>Max: {max(valid_b):.2f} / Min: {min(valid_b):.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#FF40FF;'><b>Before</b><br>N/A</div>", unsafe_allow_html=True)
        with stat_cols[1]:
            if show_after_line and valid_a:
                st.markdown(f"<div style='color:#008000;'><b>After</b><br>Avg: {sum(valid_a)/len(valid_a):.2f}<br>Max: {max(valid_a):.2f} / Min: {min(valid_a):.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#008000;'><b>After</b><br>N/A</div>", unsafe_allow_html=True)
        with stat_cols[2]:
            if show_ai_pred and valid_p:
                st.markdown(f"<div style='color:blue;'><b>AI Prediction</b><br>Avg: {sum(valid_p)/len(valid_p):.2f}<br>Max: {max(valid_p):.2f} / Min: {min(valid_p):.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:blue;'><b>AI Prediction</b><br>N/A</div>", unsafe_allow_html=True)

    else:
        st.info("No data points selected for the line chart.")
        
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

    # ==========================================
    # ★ AI Auto-Tuning Panel
    # ==========================================
    st.markdown("---")
    st.header("🪄 AI Auto-Tuning (Register Optimization)")
    st.markdown("""
    This feature calculates the difference between **Before (Current Panel)** and **Target (Goal Color)**, 
    and uses a digital twin AI strictly compliant with the YT7875 specification to automatically search for the optimal 21 register values that minimize ΔE.
    """)

    if df_b_full is not None and df_t_full is not None:
        if st.button("🚀 Run Auto-Tuning (Optimize Registers)", type="primary"):
            with st.spinner("AI is searching through thousands of register combinations (approx. 10-30 seconds)..."):
                
                df_b_lab = extract_ai_lab_data(df_b_full, df_t_full, is_target=False)
                df_t_lab = extract_ai_lab_data(df_t_full, df_t_full, is_target=True)
                common_names = df_b_lab.index.intersection(df_t_lab.index)
                
                if len(common_names) == 0:
                    st.error("❌ No matching patch names found between Before and Target. Please check the 'Name' columns.")
                else:
                    arr_b = df_b_lab.loc[common_names].values
                    arr_t = df_t_lab.loc[common_names].values

                    twin = YT7875_HybridDigitalTwin()
                    history_path = "training_history_log.csv"
                    model_path = twin.model_filepath
                    status_msg = ""
                    
                    try:
                        if os.path.exists(model_path) and os.path.exists(history_path):
                            if os.path.getmtime(history_path) > os.path.getmtime(model_path):
                                df_hist = pd.read_csv(history_path)
                                twin.update_model(df_hist)
                                status_msg = f"✅ New data detected. AI model (.pkl) has been retrained and updated with {len(df_hist)} records."
                            else:
                                twin.load_trained_model()
                                status_msg = "⚡ Successfully loaded the saved AI model (.pkl) for fast execution."
                        elif os.path.exists(model_path):
                            twin.load_trained_model()
                            status_msg = "⚡ Successfully loaded the saved AI model (.pkl) for fast execution."
                        elif os.path.exists(history_path):
                            df_hist = pd.read_csv(history_path)
                            twin.update_model(df_hist)
                            status_msg = f"✅ Created a new AI model (.pkl) from {len(df_hist)} past records."
                    except Exception as e:
                        status_msg = f"⚠️ Failed to load AI model: {e} (Continuing with math-based simulation only)"
                    
                    best_regs, avg_de = twin.optimize_21_registers(arr_b, arr_t)

                    predicted_labs = twin.predict_hybrid_vec(arr_b, best_regs)
                    predicted_xys = Lab_to_xy_vec(predicted_labs)
                    st.session_state['predicted_df'] = pd.DataFrame(predicted_xys, columns=['x', 'y'], index=common_names)
                    
                    patch_des = [delta_E_2000(arr_t[i], predicted_labs[i]) for i in range(len(common_names))]

                    reg_names = [
                        "W_R", "W_G", "W_B", "R_R", "R_G", "R_B", "G_R", "G_G", "G_B", 
                        "B_R", "B_G", "B_B", "C_R", "C_G", "C_B", "M_R", "M_G", "M_B", "Y_R", "Y_G", "Y_B"
                    ]
                    
                    st.session_state['tuning_results'] = {
                        'best_regs': best_regs,
                        'avg_de': avg_de,
                        'common_names': common_names,
                        'status_msg': status_msg,
                        'reg_names': reg_names,
                        'target_labs': arr_t,
                        'predicted_labs': predicted_labs,
                        'patch_des': patch_des
                    }

            st.rerun()

    # ---------------------------------------------------------
    # AI Tuning Results Area
    # ---------------------------------------------------------
    if st.session_state['tuning_results'] is not None:
        res = st.session_state['tuning_results']
        
        if res['status_msg']:
            st.success(res['status_msg'])
            
        st.success(f"🎉 **Optimization Complete! Predicted Average ΔE(Calc) for {len(res['common_names'])} patches : {res['avg_de']:.4f}**")
        st.info("💡 AI predicted values (blue stars) are plotted on the graph. You can toggle their visibility in the sidebar.")
        
        st.markdown("### 🛠 Recommended Register Settings")
        df_results_ai = pd.DataFrame({
            "Register (Point_Color)": res['reg_names'],
            "Dec": res['best_regs'],
            "Hex": [hex(v & 0xFF).upper().replace('X', 'x') for v in res['best_regs']]
        })
        
        c1, c2, c3 = st.columns(3)
        c1.write("**White / Red / Green Points**")
        c1.dataframe(df_results_ai.iloc[0:9].set_index("Register (Point_Color)"))
        c2.write("**Blue / Cyan / Magenta Points**")
        c2.dataframe(df_results_ai.iloc[9:18].set_index("Register (Point_Color)"))
        c3.write("**Yellow Point**")
        c3.dataframe(df_results_ai.iloc[18:21].set_index("Register (Point_Color)"))

        st.markdown("### 📊 Detailed Patch Report (Target vs Simulation)")
        
        target_labs = res['target_labs']
        predicted_labs = res['predicted_labs']
        patch_des = res['patch_des']
        names_ai = res['common_names']
        
        detail_data = []
        for i in range(len(names_ai)):
            detail_data.append({
                "Patch Name": names_ai[i],
                "Target L*": f"{target_labs[i][0]:.2f}",
                "Target a*": f"{target_labs[i][1]:.2f}",
                "Target b*": f"{target_labs[i][2]:.2f}",
                "Sim. L*": f"{predicted_labs[i][0]:.2f}",
                "Sim. a*": f"{predicted_labs[i][1]:.2f}",
                "Sim. b*": f"{predicted_labs[i][2]:.2f}",
                "ΔE(Calc)": round(patch_des[i], 3)
            })
        
        df_detail = pd.DataFrame(detail_data).set_index("Patch Name")
        
        st.dataframe(
            df_detail.style.background_gradient(subset=['ΔE(Calc)'], cmap='Reds', vmin=0, vmax=5),
            width='stretch'
        )
        st.caption("※ Items with larger ΔE(Calc) are highlighted in red.")

    # ==========================================
    # ★ 7. AI Auto-Update & Learning Panel
    # ==========================================
    st.markdown("---")
    st.header("🧠 AI Auto-Update & Learning")
    st.markdown("""
    Feed the actual measured results back into the AI. By providing the **Before CSV**, **After CSV** (measured after applying registers), and the **21 Register Values** used, the AI will automatically append the data to `training_history_log.csv` and re-train/save the `.pkl` model.
    """)

    with st.expander("📥 Register New Training Data", expanded=False):
        if df_b_full is None or df_a_full is None:
            st.warning("⚠️ Please upload both 'Before CSV' and 'After CSV' at the top of the page.")
        else:
            default_regs = ""
            if st.session_state.get('tuning_results') is not None:
                default_regs = ",".join(map(str, st.session_state['tuning_results']['best_regs']))
                st.info("💡 The 21 registers from the recent Auto-Tuning have been auto-filled.")
                
            reg_input = st.text_input("21 Register Values (Comma-separated / 10-base Decimal):", value=default_regs)
            
            if st.button("💾 Save Data & Update AI Model", type="primary"):
                try:
                    reg_list = [int(r.strip()) for r in reg_input.split(',')]
                    if len(reg_list) != 21:
                        st.error(f"❌ Invalid number of registers. Expected 21, got {len(reg_list)}.")
                    else:
                        with st.spinner("Extracting data and re-training AI..."):
                            df_b_lab = extract_ai_lab_data(df_b_full, df_t_full, is_target=False)
                            df_a_lab = extract_ai_lab_data(df_a_full, df_t_full, is_target=False)
                            
                            common_names = df_b_lab.index.intersection(df_a_lab.index)
                            if len(common_names) == 0:
                                st.error("❌ No matching patch names found between Before and After CSVs.")
                            else:
                                new_data = []
                                for name in common_names:
                                    b_L, b_a, b_b = df_b_lab.loc[name]
                                    a_L, a_a, a_b = df_a_lab.loc[name]
                                    row = [b_L, b_a, b_b] + reg_list + [a_L, a_a, a_b]
                                    new_data.append(row)
                                
                                cols = ['L_bef', 'a_bef', 'b_bef'] + [f'Reg_{i}' for i in range(21)] + ['L_aft', 'a_aft', 'b_aft']
                                df_new = pd.DataFrame(new_data, columns=cols)
                                
                                history_path = "training_history_log.csv"
                                if os.path.exists(history_path):
                                    df_existing = pd.read_csv(history_path)
                                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                                else:
                                    df_combined = df_new
                                    
                                df_combined = df_combined.drop_duplicates(subset=['L_bef', 'a_bef', 'b_bef'], keep='last')
                                    
                                df_combined.to_csv(history_path, index=False)
                                
                                twin = YT7875_HybridDigitalTwin()
                                twin.update_model(df_combined)
                                
                                st.success(f"🎉 Success! Added/Updated {len(common_names)} patches to `training_history_log.csv` and saved the AI model (`yt7875_ai_model.pkl`)!")
                                st.balloons()
                                
                except ValueError:
                    st.error("❌ All register values must be integers.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

else:
    st.info("💡 To run auto-tuning, please upload both 'Target CSV' and 'Before CSV'.")