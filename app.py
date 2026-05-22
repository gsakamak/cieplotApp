import os
import pandas as pd
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
        content = content_bytes.decode("utf-8").splitlines()
        header_row_index = 0
        for i, line in enumerate(content):
            if "Name" in line or ("x" in line and "y" in line):
                header_row_index = i
                break
        content_stream = io.StringIO("\n".join(content))
        df = pd.read_csv(content_stream, skiprows=header_row_index)
        
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
            st.sidebar.markdown(f"<span style='color: black; font-size: 1.2em;'>●</span> **Target Point**:<br>x: `{t_row.iloc[0]['x']:.4f}`<br>y: `{t_row.iloc[0]['y']:.4f}`", unsafe_allow_html=True)
            
    if df_b_full is not None and 'Name' in df_b_full.columns:
        b_row = df_b_full[df_b_full['Name'] == selected_name]
        if not b_row.empty:
            de_b = get_delta_e_from_csv(b_row.iloc[0])
            st.sidebar.markdown(f"<span style='color: #FF40FF; font-size: 1.2em;'>●</span> **Before Point**:<br>x: `{b_row.iloc[0]['x']:.4f}`<br>y: `{b_row.iloc[0]['y']:.4f}`<br>ΔE (CSV Data): **`{de_b}`**", unsafe_allow_html=True)
            
    if df_a_full is not None and 'Name' in df_a_full.columns:
        a_row = df_a_full[df_a_full['Name'] == selected_name]
        if not a_row.empty:
            de_a = get_delta_e_from_csv(a_row.iloc[0])
            st.sidebar.markdown(f"<span style='color: #00FA00; font-size: 1.2em;'>●</span> **After Point**:<br>x: `{a_row.iloc[0]['x']:.4f}`<br>y: `{a_row.iloc[0]['y']:.4f}`<br>ΔE (CSV Data): **`{de_a}`**", unsafe_allow_html=True)

# ★ Log Out button at the VERY BOTTOM of the sidebar
st.sidebar.markdown("---")
if st.sidebar.button("Log Out"):
    st.session_state['authenticated'] = False
    st.rerun()

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
        if df_b_full is not None: st.dataframe(df_b_full, width='stretch')
        else: st.info("No Before data uploaded or invalid format.")
            
    with tab2:
        if df_a_full is not None: st.dataframe(df_a_full, width='stretch')
        else: st.info("No After data uploaded or invalid format.")
            
    with tab3:
        if df_t_full is not None: st.dataframe(df_t_full, width='stretch')
        else: st.info("No Target data uploaded or invalid format.")
else:
    st.info("Please upload at least one CSV file to generate the diagram.")