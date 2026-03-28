import io
import re
import json
import zipfile
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from shapely.geometry import LineString, mapping
from pyproj import CRS, Transformer

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

# =========================
# BRANDING & THEME
# =========================
APP_TITLE = "OLSEM ver.a (Online Line Method Shoreline Evolution)"
BRAND = "by. OCEAN TECHNO"

st.set_page_config(page_title=APP_TITLE, layout="wide")

# Custom CSS for Dark Blue Background & White Tables
st.markdown("""
<style>
/* Dark Blue Background for the Main App */
.stApp {
    background-color: #0b1c2e;
    color: #ffffff;
}

/* Adjust text colors to white for readability on dark background */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: #e2e8f0 !important;
}

/* Sidebar Darker Blue */
[data-testid="stSidebar"] {
    background-color: #07121e;
}

/* Force White Background for Data Editor / Tables */
[data-testid="stDataEditor"] {
    background-color: #ffffff !important;
    border-radius: 8px;
    padding: 5px;
}
/* Ensure table text is dark */
[data-testid="stDataEditor"] * {
    color: #000000 !important;
}

/* Fix Streamlit file uploader text legibility */
.stFileUploader section * {
    color: #ffffff !important;
}

/* Primary Button Styling */
button[kind="primary"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-radius: 5px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SECURITY SYSTEM (LOGIN)
# =========================
def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        # VALIDASI PASSWORD (Ubah di sini jika ingin mengganti password)
        if st.session_state["password"] == "D2102014": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Hapus password dari memori session
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.title("🔒 Restricted Access")
        st.info("Please enter the password to access OLSEM ver.a")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.title("🔒 Restricted Access")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Incorrect password. Please try again.")
        return False
    return True

# Hentikan eksekusi aplikasi di sini jika password belum benar
if not check_password():
    st.stop()

# =====================================================================
# JIKA PASSWORD BENAR, KODE DI BAWAH INI AKAN DIJALANKAN (MAIN APP)
# =====================================================================

# =========================
# PLOT HELPERS
# =========================
def set_plain_axis(ax):
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, alpha=0.22)

def beautify_fig(fig, title=None):
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.set_constrained_layout(True)

def fig_to_png_bytes(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf

def b64_img(img_buf):
    if img_buf is None:
        return ""
    img_buf.seek(0)
    return base64.b64encode(img_buf.read()).decode('utf-8')

def legend_bottom(ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        borderaxespad=0.0,
        frameon=True,
        ncol=ncol
    )

def add_north_arrow(ax, xy=(0.94, 0.12), size=0.09):
    x, y = xy
    ax.annotate(
        "N",
        xy=(x, y + size),
        xytext=(x, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=2, color="black"),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
        zorder=20,
        color="black"
    )

def add_map_note_box(ax, location_name, brand, max_adv, max_ret, s_adv, s_ret, report_year):
    note = (
        f"Location: {location_name}\n"
        f"Horizon: {report_year} years\n"
        f"Max accretion: {max_adv:.2f} m (s≈{s_adv:.0f} m)\n"
        f"Max erosion: {max_ret:.2f} m (s≈{s_ret:.0f} m)\n"
        f"{brand}"
    )
    ax.text(
        0.02, 0.02, note,
        transform=ax.transAxes,
        fontsize=9.5,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.90),
        zorder=20,
        color="black"
    )

def annotate_max_min_dy(ax, x, y, dy, nx, ny, label_prefix=""):
    i_max = int(np.nanargmax(dy))
    dy_max = float(dy[i_max])

    i_min = int(np.nanargmin(dy))
    dy_min = float(dy[i_min])

    x0_max, y0_max = x[i_max], y[i_max]
    x1_max, y1_max = x0_max + nx[i_max] * dy_max, y0_max + ny[i_max] * dy_max

    x0_min, y0_min = x[i_min], y[i_min]
    x1_min, y1_min = x0_min + nx[i_min] * dy_min, y0_min + ny[i_min] * dy_min

    ax.annotate(
        f"{label_prefix}Max advance: {dy_max:.2f} m",
        xy=(x1_max, y1_max),
        xytext=(20, 25), 
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.90),
        zorder=20,
        color="black"
    )

    ax.annotate(
        f"{label_prefix}Max retreat: {dy_min:.2f} m",
        xy=(x1_min, y1_min),
        xytext=(20, -35),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.90),
        zorder=20,
        color="black"
    )

    return dy_max, dy_min, i_max, i_min

def get_wave_rose_png(df):
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1) 
    bins = np.arange(0, 361, 22.5)
    counts, _ = np.histogram(df['Dir_from'].dropna(), bins=bins)
    angles = np.deg2rad(bins[:-1])
    width = np.deg2rad(22.5)
    
    norm = plt.Normalize(counts.min(), counts.max())
    colors_arr = plt.cm.viridis(norm(counts))
    
    ax.bar(angles, counts, width=width, bottom=0.0, color=colors_arr, edgecolor='black', alpha=0.8)
    ax.set_title("Wave Rose (Dominant Wave Direction Distribution)", va='bottom', fontweight='bold', fontsize=11)
    fig.tight_layout()
    buf = fig_to_png_bytes(fig)
    plt.close(fig)
    return buf

# =========================
# GEOMETRY
# =========================
def arc_length(x, y):
    dx = np.diff(x); dy = np.diff(y)
    ds = np.sqrt(dx*dx + dy*dy)
    return np.concatenate([[0.0], np.cumsum(ds)])

def resample_polyline_by_ds(x, y, ds_target=25.0):
    s = arc_length(x, y)
    if len(s) < 2:
        return x, y, s
    s_new = np.arange(0, s[-1] + 1e-9, ds_target)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new, s_new

def tangent_and_normal(x, y):
    dx = np.gradient(x); dy = np.gradient(y)
    mag = np.sqrt(dx*dx + dy*dy) + 1e-12
    tx, ty = dx/mag, dy/mag
    nx, ny = -ty, tx
    return tx, ty, nx, ny

def alongshore_tangent_angle_deg(tx, ty):
    return (np.rad2deg(np.arctan2(ty, tx)) + 360.0) % 360.0

# =========================
# WAVE PARSER
# =========================
WAVE_LINE_RE = re.compile(r"^\s*(\d{8})\s+(\d+)\s+([0-9.+-eE]+)\s+([0-9.+-eE]+)\s+([0-9.+-eE]+)")

def parse_wave_text(file_bytes: bytes):
    text = file_bytes.decode("utf-8", errors="ignore").splitlines()
    rows = []
    for ln in text:
        m = WAVE_LINE_RE.match(ln)
        if not m:
            continue
        d8, hhmm, hs, tp, direc = m.groups()
        rows.append((d8, int(hhmm), float(hs), float(tp), float(direc)))

    if not rows:
        raise ValueError("No valid wave data found. Format: YYYYMMDD HHMM Hs T DirFROM")

    df = pd.DataFrame(rows, columns=["date8", "hhmm", "Hs", "T", "Dir_from"])
    df["date"] = pd.to_datetime(df["date8"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])

    HH = (df["hhmm"] // 100).astype(int)
    MM = (df["hhmm"] % 100).astype(int)
    df["time"] = df["date"] + pd.to_timedelta(HH, unit="h") + pd.to_timedelta(MM, unit="m")

    df = df.drop(columns=["date8", "hhmm", "date"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

def daily_wave(df_wave):
    df = df_wave.copy()
    df = df.set_index("time")

    def circ_mean_deg(series):
        ang = np.deg2rad(series.to_numpy())
        s = np.nanmean(np.sin(ang)); c = np.nanmean(np.cos(ang))
        return (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0

    out = pd.DataFrame()
    out["Hs"] = pd.to_numeric(df["Hs"], errors="coerce").resample("D").mean()
    out["T"] = pd.to_numeric(df["T"], errors="coerce").resample("D").mean()
    out["Dir_from"] = pd.to_numeric(df["Dir_from"], errors="coerce").resample("D").apply(circ_mean_deg)
    out = out.dropna().reset_index()
    return out

# =========================
# BATHY XYZ + IDW
# =========================
def read_xyz(file_bytes: bytes):
    txt = file_bytes.decode("utf-8", errors="ignore")
    data = []
    for ln in txt.splitlines():
        parts = re.split(r"[,\s]+", ln.strip())
        if len(parts) < 3:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            data.append((x, y, z))
        except:
            pass
    if not data:
        raise ValueError("XYZ data is empty/unreadable. Ensure columns: x y z.")
    return pd.DataFrame(data, columns=["x", "y", "z"])

def idw_interpolate(xy_src, z_src, xy_q, k=12, p=2.0):
    out = np.empty(len(xy_q), dtype=float)
    X = xy_src[:, 0]; Y = xy_src[:, 1]
    k = int(min(k, len(xy_src)))
    for i, (qx, qy) in enumerate(xy_q):
        dx = X - qx; dy = Y - qy
        d2 = dx*dx + dy*dy
        idx = np.argpartition(d2, k-1)[:k]
        d2k = d2[idx]; zk = z_src[idx]
        w = 1.0 / (np.power(d2k + 1e-12, p/2.0))
        out[i] = np.sum(w * zk) / np.sum(w)
    return out

# =========================
# WAVE TRANSFORMATION
# =========================
g = 9.81

def wave_number(T, h):
    w = 2.0 * np.pi / max(T, 1e-6)
    k = (w*w) / g
    for _ in range(30):
        kh = k * h
        t = np.tanh(kh)
        f = g * k * t - w*w
        df = g * t + g * k * (1.0 - t*t) * h
        k_new = k - f / (df + 1e-12)
        if abs(k_new - k) < 1e-10:
            break
        k = max(k_new, 1e-12)
    return k

def group_velocity(T, h, k):
    w = 2.0 * np.pi / max(T, 1e-6)
    c = w / (k + 1e-12)
    kh = k*h
    n = 0.5 * (1.0 + (2*kh) / (np.sinh(2*kh) + 1e-12))
    Cg = n * c
    return c, Cg

def transform_wave_local(Hs0, T0, Dir_from0, h_local, shore_tangent_deg):
    h = float(max(abs(h_local), 0.5))
    k0 = wave_number(T0, h=1e6)
    C0, Cg0 = group_velocity(T0, h=1e6, k=k0)
    k = wave_number(T0, h=h)
    C, Cg = group_velocity(T0, h=h, k=k)

    Ks = np.sqrt(max(Cg0, 1e-12) / max(Cg, 1e-12))
    Hs_s = Hs0 * Ks

    Dir_to0 = (Dir_from0 + 180.0) % 360.0
    theta0 = ((Dir_to0 - shore_tangent_deg + 540.0) % 360.0) - 180.0
    theta0_rad = np.deg2rad(theta0)

    s_theta = (C / max(C0, 1e-12)) * np.sin(theta0_rad)
    s_theta = np.clip(s_theta, -0.999999, 0.999999)
    theta_rad = np.arcsin(s_theta)
    theta_deg = np.rad2deg(theta_rad)

    Dir_to_local = (shore_tangent_deg + theta_deg) % 360.0
    Dir_from_local = (Dir_to_local - 180.0) % 360.0
    return Hs_s, Dir_from_local

# =========================
# ONE-LINE MODEL
# =========================
def wave_to_alpha(Dir_from_deg, shoreline_tangent_deg):
    Dir_to = (Dir_from_deg + 180.0) % 360.0
    d = (Dir_to - shoreline_tangent_deg + 540.0) % 360.0 - 180.0
    return np.deg2rad(d)

def compute_Q_indicator(Hs, alpha_rad, K):
    return K * (Hs ** 2) * np.sin(2.0 * alpha_rad)

# =========================
# STRUCTURES
# =========================
def default_structures_df():
    return pd.DataFrame([], columns=["name", "type", "x1", "y1", "x2", "y2", "param1", "param2"])

def dist_point_to_segment(px, py, x1, y1, x2, y2):
    vx = x2 - x1; vy = y2 - y1
    wx = px - x1; wy = py - y1
    vv = vx*vx + vy*vy + 1e-12
    t = (wx*vx + wy*vy) / vv
    t = np.clip(t, 0.0, 1.0)
    cx = x1 + t*vx; cy = y1 + t*vy
    dx = px - cx; dy = py - cy
    return np.sqrt(dx*dx + dy*dy)

def apply_structures(
    xr, yr, s,
    Hs_loc, Dir_loc,
    nx, ny,
    structures: pd.DataFrame,
    strength=1.0,
    seawall_reflection_slope=2.0,
    seawall_erosion_hold=0.5,
):
    if structures is None or len(structures) == 0:
        return Hs_loc.copy(), Dir_loc.copy(), np.zeros_like(Hs_loc)

    H = Hs_loc.copy()
    dy_extra = np.zeros_like(Hs_loc)

    for _, row in structures.iterrows():
        tp = str(row.get("type", "")).strip().lower()
        try:
            x1 = float(row.get("x1", np.nan)); y1 = float(row.get("y1", np.nan))
            x2 = float(row.get("x2", np.nan)); y2 = float(row.get("y2", np.nan))
        except:
            continue
        if np.any(np.isnan([x1,y1,x2,y2])):
            continue

        p1 = float(row.get("param1", 500.0) or 500.0)
        p2 = float(row.get("param2", 0.6) or 0.6)
        p1 = max(p1, 1e-6)
        p2 = np.clip(p2, 0.0, 1.0)

        d = np.array([dist_point_to_segment(px, py, x1,y1,x2,y2) for px,py in zip(xr,yr)])
        w = np.exp(-(d / p1)**2)

        if "detached" in tp or "breakwater" in tp:
            diffract = np.clip(p2, 0.05, 0.95)
            reduce = np.clip((1.0 - diffract) * strength, 0.0, 0.95)
            H = H * (1.0 - w * reduce)
            dy_extra += (3.0e-5 * strength) * (1.0 - diffract) * w

        elif "seawall" in tp or "revet" in tp:
            f_slope = np.clip(1.0 / (1.0 + 0.35*seawall_reflection_slope), 0.15, 1.0)
            Kr = np.clip(p2 * (1.0 - f_slope) * strength, 0.0, 0.8)
            H = H * (1.0 + Kr * w)

    return H, Dir_loc.copy(), dy_extra

def plot_structures(ax, structures_df):
    if structures_df is None or len(structures_df) == 0:
        return
    for _, r in structures_df.iterrows():
        nm = str(r.get("name", "")).strip()
        tp = str(r.get("type", "")).strip()

        try:
            x1 = float(r.get("x1", np.nan)); y1 = float(r.get("y1", np.nan))
            x2 = float(r.get("x2", np.nan)); y2 = float(r.get("y2", np.nan))
        except:
            continue
        if np.any(np.isnan([x1,y1,x2,y2])):
            continue

        label = f"Structure: {tp}" if nm == "" else f"{nm} ({tp})"
        ax.plot([x1, x2], [y1, y2], linewidth=4.5, label=label, zorder=10)

# =========================
# EXPORT GEOJSON
# =========================
def make_geojson_line(x, y, crs_in: CRS, out_wgs84=True, props=None):
    props = props or {}
    line = LineString(np.column_stack([x, y]))
    if out_wgs84:
        transformer = Transformer.from_crs(crs_in, CRS.from_epsg(4326), always_xy=True)
        coords = [transformer.transform(xx, yy) for xx, yy in line.coords]
        line = LineString(coords)
        crs_name = "EPSG:4326"
    else:
        crs_name = crs_in.to_string()

    feat = {"type": "Feature", "properties": {"crs": crs_name, **props}, "geometry": mapping(line)}
    return json.dumps({"type": "FeatureCollection", "features": [feat]}, indent=2).encode("utf-8")

# =========================
# PDF REPORT 
# =========================
def _pdf_header_footer(canvas, doc, title, brand, location):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.2*cm, A4[1] - 1.0*cm, title[:95])
    canvas.drawRightString(A4[0] - 1.2*cm, A4[1] - 1.0*cm, brand)
    canvas.setFillColor(colors.black)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(1.2*cm, 0.8*cm, f"Location: {location}")
    canvas.drawRightString(A4[0] - 1.2*cm, 0.8*cm, f"Page {doc.page}")
    canvas.restoreState()

def build_pdf_report(payload):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, leading=11))
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=8, spaceAfter=4))

    story = []
    story.append(Paragraph(payload["title"], styles["Title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"<b>Location:</b> {payload['location']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Brand:</b> {payload['brand']}", styles["Normal"]))
    story.append(Paragraph(f"<b>CRS:</b> {payload['crs_text']}", styles["Normal"]))
    story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    stt = payload["stats"]
    story.append(Paragraph("Executive Summary", styles["H1"]))
    story.append(Paragraph(
        f"Analysis Horizon: <b>{payload['report_year']} years</b>. "
        f"Max accretion (advance): <b>{stt['max_advance']:.2f} m</b> (s≈{stt['s_advance']:.0f} m). "
        f"Max erosion (retreat): <b>{stt['max_retreat']:.2f} m</b> (s≈{stt['s_retreat']:.0f} m).",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Input Data Summary", styles["H1"]))
    ws = payload.get("wave_stats", {})
    story.append(Paragraph("Wave Data", styles["H2"]))
    if ws:
        story.append(Paragraph(
            f"Observation days: <b>{ws.get('n_days','-')}</b>. "
            f"Average Hs: <b>{ws.get('Hs_mean','-'):.3f} m</b>, "
            f"Average T: <b>{ws.get('T_mean','-'):.3f} s</b>, "
            f"Dominant Dir_from: <b>{ws.get('Dir_med','-'):.1f}°</b>.",
            styles["Normal"]
        ))
    else:
        story.append(Paragraph("Wave statistics unavailable.", styles["Small"]))

    bs = payload.get("bathy_stats", {})
    story.append(Paragraph("Bathymetry Data", styles["H2"]))
    if bs:
        story.append(Paragraph(
            f"Point counts: <b>{bs.get('n_pts','-')}</b>. "
            f"Depth (z) range: <b>{bs.get('z_min','-'):.2f} to {bs.get('z_max','-'):.2f} m</b>.",
            styles["Normal"]
        ))
    else:
        story.append(Paragraph("Bathymetry not used or unavailable.", styles["Small"]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Model Parameters", styles["H1"]))
    p = payload["params"]
    tbl = Table([
        ["Parameter", "Value"],
        ["Dc (Active depth of closure, m)", f"{p['Dc']}"],
        ["K (Transport scale)", f"{p['K']}"],
        ["Resample target ds (m)", f"{p['ds_target']}"],
        ["Bathy affects Q (Shoaling/Refraction)", f"{p['use_bathy']}"],
        ["Structure global strength", f"{p['struct_strength']}"],
        ["Seawall slope (tan β)", f"{p['seawall_slope']}"],
        ["Seawall erosion hold factor", f"{p['seawall_hold']}"],
    ], colWidths=[7.2*cm, 8.8*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1f2a44")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("BACKGROUND",(0,1),(-1,-1),colors.whitesmoke),
        ("FONTSIZE",(0,0),(-1,-1),9),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.35*cm))

    if payload.get("structures_table") is not None and len(payload["structures_table"]) > 0:
        story.append(Paragraph("Coastal Structures (Input)", styles["H1"]))
        stbl = payload["structures_table"].copy().fillna("")
        data = [stbl.columns.tolist()] + stbl.values.tolist()
        t = Table(data, colWidths=[2.6*cm, 3.4*cm, 2.0*cm,2.0*cm,2.0*cm,2.0*cm,1.8*cm,1.8*cm])
        t.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e6f2ff")),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),7),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.35*cm))

    story.append(PageBreak())
    story.append(Paragraph("Analysis & Visualization Results", styles["H1"]))
    story.append(Spacer(1, 0.25*cm))

    for key in ["map_bathy", "Q_compare", "dy_plots", "map_final"]:
        if key in payload["images"] and payload["images"][key] is not None:
            story.append(Paragraph(payload["images_caption"][key], styles["H2"]))
            
            img_buf = payload["images"][key]
            img_buf.seek(0)
            img_reader = ImageReader(img_buf)
            img_w, img_h = img_reader.getSize()
            aspect = img_h / float(img_w)
            
            target_width = 17.0 * cm
            target_height = target_width * aspect
            
            story.append(RLImage(img_buf, width=target_width, height=target_height))
            story.append(Spacer(1, 0.3*cm))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.2*cm, rightMargin=1.2*cm,
        topMargin=1.6*cm, bottomMargin=1.4*cm
    )
    doc.build(
        story,
        onFirstPage=lambda c, d: _pdf_header_footer(c, d, payload["title"], payload["brand"], payload["location"]),
        onLaterPages=lambda c, d: _pdf_header_footer(c, d, payload["title"], payload["brand"], payload["location"]),
    )
    buf.seek(0)
    return buf

# =========================
# UI SIDEBAR
# =========================
st.title(APP_TITLE)
st.markdown(f"**{BRAND}**")

st.sidebar.header("Settings")
location_name = st.sidebar.text_input("Location Name", value="Location-1")

utm_zone = st.sidebar.number_input("UTM Zone", min_value=1, max_value=60, value=51, step=1)
hemisphere = st.sidebar.selectbox("Hemisphere", ["N", "S"], index=1)
epsg = (32600 + int(utm_zone)) if hemisphere == "N" else (32700 + int(utm_zone))
crs_utm = CRS.from_epsg(epsg)
st.sidebar.success(f"UTM CRS: EPSG:{epsg} (Zone {utm_zone}{hemisphere})")

Dc = st.sidebar.slider("Dc (Active Depth, m)", 1.0, 25.0, 8.0, 0.5)
K = st.sidebar.slider("K (Transport Scale)", 0.00001, 0.05, 0.01, 0.00001, format="%.6f")
reverse_normal = st.sidebar.checkbox("Reverse shore normal (if flipped)", value=False)
out_wgs84 = st.sidebar.checkbox("Output GeoJSON in WGS84 (EPSG:4326)", value=True)

st.sidebar.markdown("---")
use_bathy_transform = st.sidebar.checkbox("Bathymetry affects Q (Shoaling & Refraction)", value=True)
idw_k = st.sidebar.slider("IDW k-neighbor", 6, 30, 12, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Structure Effects (Global)")
struct_strength = st.sidebar.slider("Strength (Effect multiplier)", 0.0, 10.0, 2.0, 0.1)

st.sidebar.subheader("Seawall/Revetment")
seawall_slope = st.sidebar.slider("Slope (tan β)", 0.5, 10.0, 2.0, 0.1)
seawall_hold = st.sidebar.slider("Erosion hold (0=full hold, 1=no hold)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
with st.sidebar.expander("⚠️ DISCLAIMER & LEGAL", expanded=False):
    st.caption("""
    **Limitation of Liability:**
    This application is provided "AS IS". It is exclusively intended for **Rapid Assessment**, pre-feasibility studies, and academic purposes. It is NOT a substitute for Detail Engineering Design (DED) of physical coastal structures. 
    
    Simulation results are indicative based on empirical *One-Line Model* assumptions and do not substitute official regulatory or legal decisions. The developer is not liable for any damages, construction failures, or spatial conflicts arising from decisions based on this application's output. Usage risk relies entirely on the user.
    """)

# =========================
# DATA UPLOAD UI
# =========================
c1, c2 = st.columns(2)
with c1:
    st.subheader("1) Upload Shoreline (CSV x,y)")
    shore_file = st.file_uploader("Shoreline CSV", type=["csv"])
    ds_target = st.slider("Resample point distance (m)", 5.0, 200.0, 25.0, 1.0)
with c2:
    st.subheader("2) Upload Wave Data (txt/csv/dat)")
    st.caption("Format: YYYYMMDD HHMM Hs T DirFROM")
    wave_file = st.file_uploader("Wave file", type=["txt", "csv", "dat"])

st.subheader("3) Upload Bathymetry (XYZ x y z)")
bathy_file = st.file_uploader("Bathy XYZ", type=["xyz", "txt", "dat"])

# =========================
# STRUCTURES UI (Moved here)
# =========================
if "structures_df" not in st.session_state:
    st.session_state["structures_df"] = default_structures_df()

st.markdown("## 4) STRUCTURE INPUT (Optional)")
st.caption(
    "Columns: name, type, x1, y1, x2, y2, param1, param2.\n"
    "- Detached Breakwater: param1=influence radius (m; e.g., 300–800), param2=diffraction factor (0..1; smaller=stronger block)\n"
    "- Seawall/Revetment: param1=influence radius (m), param2=reflection strength (0..1)\n"
)

colA, colB, colC, colD = st.columns([1.2, 1.2, 1.2, 1.2])
with colA:
    if st.button("➕ Add Structure"):
        idx = len(st.session_state["structures_df"]) + 1
        new_row = {
            "name": f"Structure-{idx}",
            "type": "Detached Breakwater",
            "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan,
            "param1": 500.0,
            "param2": 0.60
        }
        st.session_state["structures_df"] = pd.concat(
            [st.session_state["structures_df"], pd.DataFrame([new_row])],
            ignore_index=True
        )
with colB:
    if st.button("🧹 Reset Structures"):
        st.session_state["structures_df"] = default_structures_df()
with colC:
    st.download_button(
        "💾 Save Structures (JSON)",
        data=st.session_state["structures_df"].to_json(orient="records", indent=2),
        file_name=f"structures_{location_name}.json",
        mime="application/json",
        use_container_width=True
    )
with colD:
    up = st.file_uploader("📂 Load Structures (JSON)", type=["json"], label_visibility="collapsed")
    if up is not None:
        try:
            arr = json.loads(up.read().decode("utf-8"))
            st.session_state["structures_df"] = pd.DataFrame(arr)
            st.success("Structures successfully loaded.")
        except Exception as e:
            st.error(f"Failed to load JSON: {e}")

edited_struct = st.data_editor(
    st.session_state["structures_df"],
    num_rows="fixed",
    use_container_width=True,
    key="structures_editor",
    column_config={
        "name": st.column_config.TextColumn("name", help="Structure name for legend"),
        "type": st.column_config.SelectboxColumn(
            "type", options=["Detached Breakwater", "Seawall/Revetment"], required=True
        ),
        "param1": st.column_config.NumberColumn("param1", help="Influence radius (m)", format="%.1f"),
        "param2": st.column_config.NumberColumn("param2", help="Factor (see notes)", format="%.2f"),
    },
)
st.session_state["structures_df"] = edited_struct

# =========================
# READ INPUTS
# =========================
shore_ok = False
xr = yr = s = None
x0 = y0 = None

if shore_file is not None:
    df = pd.read_csv(shore_file)
    df.columns = [c.strip().lower() for c in df.columns]
    if not set(["x", "y"]).issubset(df.columns):
        st.error("CSV shoreline must contain x,y columns")
    else:
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["x", "y"])
        x0 = df["x"].to_numpy()
        y0 = df["y"].to_numpy()
        xr, yr, s = resample_polyline_by_ds(x0, y0, ds_target=ds_target)
        shore_ok = True

wave_ok = False
wdaily = None
wave_stats = {}
wave_rose_png = None

if wave_file is not None:
    try:
        wdf = parse_wave_text(wave_file.read())
        wdaily = daily_wave(wdf)
        wave_ok = True
        
        wave_rose_png = get_wave_rose_png(wdaily)
        
        st.markdown("### Wave Daily Preview")
        st.dataframe(wdaily.head(25), use_container_width=True)

        wave_stats = {
            "n_days": int(len(wdaily)),
            "Hs_mean": float(wdaily["Hs"].mean()),
            "T_mean": float(wdaily["T"].mean()),
            "Dir_med": float(wdaily["Dir_from"].median()),
        }
    except Exception as e:
        st.error(f"Failed to read wave file: {e}")

bathy_ok = False
bdf = None
bathy_stats = {}
if bathy_file is not None:
    try:
        bdf = read_xyz(bathy_file.read())
        bathy_ok = True
        st.success(f"Bathymetry loaded: {len(bdf):,} points")
        bathy_stats = {
            "n_pts": int(len(bdf)),
            "z_min": float(bdf["z"].min()),
            "z_max": float(bdf["z"].max()),
        }
    except Exception as e:
        st.error(f"Failed to read bathymetry: {e}")

# =========================
# PLOTS INPUT
# =========================
if shore_ok:
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.plot(x0, y0, label="Original Shoreline", linewidth=1.6, zorder=2)
    ax.plot(xr, yr, label=f"Resampled Shoreline (ds={ds_target:.0f}m)", linewidth=2.6, zorder=3)
    plot_structures(ax, st.session_state["structures_df"])
    add_north_arrow(ax) 
    ax.set_xlabel("X (UTM)")
    ax.set_ylabel("Y (UTM)")
    ax.set_aspect("equal", adjustable="box")
    set_plain_axis(ax)
    
    legend_bottom(ax, ncol=2) 
    beautify_fig(fig, title=f"Shoreline + Structures (Input) — {location_name}")
    
    png_input = fig_to_png_bytes(fig)
    st.pyplot(fig, clear_figure=True)
    
    st.download_button(
        label="⬇️ Download Input Map (PNG)",
        data=png_input,
        file_name=f"Input_Shoreline_{location_name}.png",
        mime="image/png"
    )

map_bathy_png = None
if bathy_ok and shore_ok:
    bplot = bdf.sample(min(20000, len(bdf)), random_state=2) if len(bdf) > 20000 else bdf
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    
    try:
        sc = ax.tricontourf(bplot["x"], bplot["y"], bplot["z"], levels=20, cmap="viridis", alpha=0.85, zorder=1)
        cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.04)
        cb.set_label("z (m)")
    except Exception as e:
        sc = ax.scatter(bplot["x"], bplot["y"], c=bplot["z"], s=4, cmap="viridis", zorder=1)
        cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.04)
        cb.set_label("z (m)")

    ax.plot(xr, yr, linewidth=2.6, label="Shoreline", color="red", zorder=5)
    plot_structures(ax, st.session_state["structures_df"])
    
    add_north_arrow(ax)
    ax.set_xlabel("X (UTM)")
    ax.set_ylabel("Y (UTM)")
    ax.set_aspect("equal", adjustable="box")
    set_plain_axis(ax)
    
    legend_bottom(ax, ncol=2) 
    beautify_fig(fig, title=f"Bathymetry + Shoreline + Structures — {location_name}")
    
    map_bathy_png = fig_to_png_bytes(fig)
    st.pyplot(fig, clear_figure=True)
    
    st.download_button(
        label="⬇️ Download Bathymetry Map (PNG)",
        data=map_bathy_png,
        file_name=f"Bathymetry_{location_name}.png",
        mime="image/png"
    )

# =========================
# SIMULATION
# =========================
results = None
final_map_png = None
Q_png = None
dy_png = None

if shore_ok and wave_ok:
    st.markdown("## Simulation")
    years = st.multiselect("Select horizon (years)", [1, 2, 5, 10], default=[5, 10])
    years = sorted(list(set(years)))
    report_year = st.selectbox(
        "Horizon for MAX annotation (Plot & Report)",
        options=years, index=len(years)-1 if years else 0
    )

    run = st.button("▶️ Run Simulation", type="primary")

    if run:
        tx, ty, nx, ny = tangent_and_normal(xr, yr)
        if reverse_normal:
            nx, ny = -nx, -ny
        shore_tan = alongshore_tangent_angle_deg(tx, ty)

        Hs0 = float(wdaily["Hs"].mean())
        T0  = float(wdaily["T"].mean())
        Dir0 = float(wdaily["Dir_from"].median())

        if use_bathy_transform and bathy_ok and len(bdf) > 10:
            bsub = bdf.sample(min(50000, len(bdf)), random_state=1) if len(bdf) > 50000 else bdf
            xy_src = bsub[["x", "y"]].to_numpy()
            z_src  = bsub["z"].to_numpy()
            z_shore = idw_interpolate(xy_src, z_src, np.column_stack([xr, yr]), k=int(idw_k), p=2.0)
            h_shore = np.maximum(np.abs(z_shore), 0.5)

            Hs_loc = np.zeros_like(xr, dtype=float)
            Dir_loc = np.zeros_like(xr, dtype=float)
            for i in range(len(xr)):
                Hs_loc[i], Dir_loc[i] = transform_wave_local(Hs0, T0, Dir0, h_shore[i], shore_tan[i])
        else:
            Hs_loc = np.ones_like(xr) * Hs0
            Dir_loc = np.ones_like(xr) * Dir0

        # BASELINE
        alpha_base = wave_to_alpha(Dir_loc, shore_tan)
        Q_base = compute_Q_indicator(Hs_loc, alpha_base, K)
        dQds_base = np.gradient(Q_base, s + 1e-9)
        dy_dt_base = -(1.0 / max(Dc, 1e-6)) * dQds_base

        # STRUCTURES MODIFY
        Hs_mod, Dir_mod, dy_extra = apply_structures(
            xr, yr, s,
            Hs_loc, Dir_loc,
            nx, ny,
            st.session_state["structures_df"],
            strength=float(struct_strength),
            seawall_reflection_slope=float(seawall_slope),
            seawall_erosion_hold=float(seawall_hold),
        )

        alpha_mod = wave_to_alpha(Dir_mod, shore_tan)
        Q_mod = compute_Q_indicator(Hs_mod, alpha_mod, K)
        dQds_mod = np.gradient(Q_mod, s + 1e-9)
        dy_dt_mod = -(1.0 / max(Dc, 1e-6)) * dQds_mod

        dy_dt_final = dy_dt_mod + dy_extra

        sim = {}
        for Y in years:
            ndays = int(round(365.0 * Y))
            sim[Y] = dy_dt_final * ndays

        results = {
            "Hs0": Hs0, "T0": T0, "Dir0": Dir0,
            "Hs_loc": Hs_loc, "Dir_loc": Dir_loc,
            "Q_base": Q_base, "Q_mod": Q_mod,
            "dy_dt_final": dy_dt_final,
            "sim": sim,
            "nx": nx, "ny": ny,
        }

        st.success("Simulation Complete.")

        # Q plot
        figQ, axQ = plt.subplots(figsize=(10.2, 4.6))
        axQ.plot(s, Q_base, label="Q without structures")
        axQ.plot(s, Q_mod, label="Q with structures")
        axQ.set_xlabel("s (m)")
        axQ.set_ylabel("Q (scale)")
        axQ.grid(True, alpha=0.25)
        legend_bottom(axQ, ncol=2)
        beautify_fig(figQ, title=f"Sediment Transport Rate, Q(s) — {location_name}")
        
        Q_png = fig_to_png_bytes(figQ)
        st.pyplot(figQ, clear_figure=True)
        st.download_button("⬇️ Download Q(s) Chart (PNG)", data=Q_png, file_name=f"Q_plot_{location_name}.png", mime="image/png")

        # dy plot
        figDY, axDY = plt.subplots(figsize=(10.2, 4.6))
        dy_r = sim[int(report_year)]
        axDY.plot(s, dy_r, linewidth=2)
        axDY.axhline(0, linewidth=1, color="red", linestyle="--")
        axDY.set_xlabel("s (m)")
        axDY.set_ylabel("Δy (m)")
        axDY.grid(True, alpha=0.25)
        beautify_fig(figDY, title=f"Shoreline Change normal vs s ({report_year} years) — {location_name}")
        
        dy_png = fig_to_png_bytes(figDY)
        st.pyplot(figDY, clear_figure=True)
        st.download_button("⬇️ Download Δy Chart (PNG)", data=dy_png, file_name=f"dy_plot_{location_name}.png", mime="image/png")

        # Final map
        figM, axM = plt.subplots(figsize=(10.2, 6.2))
        if bathy_ok:
            bplot = bdf.sample(min(20000, len(bdf)), random_state=4) if len(bdf) > 20000 else bdf
            try:
                sc_map = axM.tricontourf(bplot["x"], bplot["y"], bplot["z"], levels=20, cmap="viridis", alpha=0.85, zorder=1)
                cb = figM.colorbar(sc_map, ax=axM, shrink=0.82, pad=0.04)
                cb.set_label("z (m)")
            except Exception:
                sc_map = axM.scatter(bplot["x"], bplot["y"], c=bplot["z"], s=4, alpha=0.95, zorder=1)
                cb = figM.colorbar(sc_map, ax=axM, shrink=0.82, pad=0.04)
                cb.set_label("z (m)")

        axM.plot(xr, yr, label="Initial Shoreline", linewidth=3, zorder=5)

        for Y in years:
            dy = sim[Y]
            xY = xr + results["nx"] * dy
            yY = yr + results["ny"] * dy
            axM.plot(xY, yY, label=f"Result {Y} years", linewidth=2.3, zorder=6)

        plot_structures(axM, st.session_state["structures_df"])

        dy_r = sim[int(report_year)]
        max_adv, max_ret, i_max, i_min = annotate_max_min_dy(
            axM, xr, yr, dy_r, results["nx"], results["ny"], label_prefix=f"({report_year}y) "
        )

        s_adv = float(s[i_max])
        s_ret = float(s[i_min])

        add_north_arrow(axM)  
        add_map_note_box(axM, location_name, BRAND, max_adv, max_ret, s_adv, s_ret, int(report_year))  

        axM.set_xlabel("X (UTM)")
        axM.set_ylabel("Y (UTM)")
        axM.set_aspect("equal", adjustable="box")
        set_plain_axis(axM)
        
        legend_bottom(axM, ncol=3) 
        beautify_fig(figM, title=f"Shoreline Evolution Map ({report_year} years) — {location_name}")
        
        final_map_png = fig_to_png_bytes(figM)
        st.pyplot(figM, clear_figure=True)
        st.download_button("⬇️ Download Final Map (PNG)", data=final_map_png, file_name=f"Result_{location_name}_{report_year}y.png", mime="image/png")

        # Export GeoJSON
        st.markdown("## Export GeoJSON")
        pick = st.selectbox("Select layer to export", ["Initial Shoreline"] + [f"{Y} years result" for Y in years])
        if pick == "Initial Shoreline":
            xexp, yexp = xr, yr
        else:
            Y = int(pick.split()[0])
            dy = sim[Y]
            xexp = xr + results["nx"] * dy
            yexp = yr + results["ny"] * dy

        props = {"location": location_name, "brand": BRAND, "model": "One-Line + Bathy + Structures"}
        gj = make_geojson_line(xexp, yexp, crs_in=crs_utm, out_wgs84=bool(out_wgs84), props=props)
        st.download_button(
            "Download GeoJSON",
            data=gj,
            file_name=f"shoreline_{location_name}_{pick.replace(' ', '_')}.geojson",
            mime="application/geo+json",
        )

        # PDF report
        st.markdown("## Professional PDF Report")
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Compiling PDF report..."):
                images = {
                    "map_bathy": map_bathy_png if map_bathy_png else None,
                    "Q_compare": Q_png,
                    "dy_plots": dy_png,
                    "map_final": final_map_png,
                }
                captions = {
                    "map_bathy": "Bathymetry Map + Shoreline + Coastal Structures",
                    "Q_compare": "Comparison of Q(s) with vs without structures",
                    "dy_plots": f"Δy(s) Shoreline displacement at {report_year} years horizon",
                    "map_final": "Shoreline Evolution Map + Conclusion (accretion/erosion indicators)",
                }

                stbl = st.session_state["structures_df"].copy()
                if len(stbl):
                    stbl = stbl[["name","type","x1","y1","x2","y2","param1","param2"]]

                pdf_payload = {
                    "title": APP_TITLE,
                    "brand": BRAND,
                    "location": location_name,
                    "crs_text": f"EPSG:{epsg} (UTM Zone {utm_zone}{hemisphere})",
                    "report_year": int(report_year),
                    "params": {
                        "Dc": float(Dc),
                        "K": float(K),
                        "ds_target": float(ds_target),
                        "use_bathy": bool(use_bathy_transform and bathy_ok),
                        "struct_strength": float(struct_strength),
                        "seawall_slope": float(seawall_slope),
                        "seawall_hold": float(seawall_hold),
                    },
                    "structures_table": stbl if len(stbl) else None,
                    "stats": {
                        "max_advance": float(max_adv),
                        "max_retreat": float(max_ret),
                        "s_advance": float(s_adv),
                        "s_retreat": float(s_ret),
                    },
                    "images": images,
                    "images_caption": captions,
                    "wave_stats": wave_stats,
                    "bathy_stats": bathy_stats if bathy_ok else {},
                }

                pdf_buf = build_pdf_report(pdf_payload)
                st.session_state["pdf_data"] = pdf_buf.getvalue()

        if "pdf_data" in st.session_state:
            st.success("PDF generated successfully! Click below to download.")
            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state["pdf_data"],
                file_name=f"Report_Shoreline_{location_name}_{report_year}y.pdf",
                mime="application/pdf",
            )
            
        # =========================
        # TEXT / WORD REPORT (WITH EMBEDDED IMAGES)
        # =========================
        st.markdown("---")
        st.markdown("## Comprehensive Analysis Report (Narrative)")
        
        st_df = st.session_state["structures_df"]
        num_struct = len(st_df.dropna(subset=['x1','y1','x2','y2'])) if len(st_df) > 0 else 0
        struct_text = f"There are {num_struct} artificial coastal structures introduced in this simulation to evaluate the effects of wave energy attenuation (diffraction) and wave reflection on sediment dynamics." if num_struct > 0 else "In this computational simulation, no artificial coastal protection structures were introduced."
        
        bathy_text = f"The seabed elevation (bathymetry) data is obtained from the integration of *in-situ* bathymetric measurements and spatial secondary data from the *General Bathymetric Chart of the Oceans* (GEBCO). In this computation, the bathymetry dataset consisting of {bathy_stats['n_pts']:,} points with a depth range of {bathy_stats['z_min']:.2f} meters to {bathy_stats['z_max']:.2f} meters is interpolated using the *Inverse Distance Weighting* (IDW) method to transform wave propagation properties." if bathy_ok else "In this analysis, a uniform depth condition is assumed without any intervention from detailed bathymetric contour variations."

        # Menyiapkan base64 string
        wr_b64 = b64_img(wave_rose_png)
        q_b64 = b64_img(Q_png)
        dy_b64 = b64_img(dy_png)
        map_b64 = b64_img(final_map_png)

        report_markdown = rf"""
# SHORELINE EVOLUTION ANALYSIS REPORT: {location_name.upper()}

## 1. BACKGROUND
Coastal dynamics is a natural process that is intensively influenced by the interaction of wave hydrodynamics, currents, and marine sediment movement. This shoreline evolution modeling analysis is conducted in the coastal area of {location_name} by projecting the morphological time horizon for the next {report_year} years. This numerical modeling aims to technically identify the coastal vulnerability level against potential abrasion (erosion) as well as the probability of accretion (sedimentation).

The baseline shoreline data has been validated and spatially resampled using a spatial resolution interval ($ds$) of {ds_target} meters. {bathy_text}

The hydrodynamic wave data acting as the main driving force in this model is obtained directly from the *Global Ocean Physics Analysis and Forecast* provided by the Copernicus Marine Service (https://data.marine.copernicus.eu/). Based on the time-series analysis spanning {wave_stats.get('n_days', 0)} days of observation, the average significant wave height ($H_s$) is recorded at {wave_stats.get('Hs_mean', 0):.2f} meters with an average period of {wave_stats.get('T_mean', 0):.2f} seconds. The dominant wave direction (*Dir_from*) originates from an angle of {wave_stats.get('Dir_med', 0):.1f}° relative to True North. {struct_text} The following wave rose diagram illustrates the directional distribution of the incident waves in the study area:

<br>
<img src="data:image/png;base64,{wr_b64}" width="500" />
<br>

## 2. BASIC THEORY
This coastal evolution modeling relies on the spatial sediment continuity equation, widely recognized as the *One-Line Model* (Pelnard-Considère, 1956). The fundamental concept of this model formulation assumes that the cross-shore profile moves autonomously and parallel to itself, maintaining a relatively constant geometric profile, where massive sediment movement occurs predominantly along the shoreline (*Longshore Sediment Transport* / LST).

The differential continuity equation for shoreline change is denoted as:
$$ \frac{{\partial y}}{{\partial t}} + \frac{{1}}{{D_c}} \frac{{\partial Q}}{{\partial s}} = 0 $$
Where:
- $y$ = Shoreline position perpendicular to the initial reference line (m)
- $t$ = Computational simulation time interval
- $D_c$ = Depth of Closure (m), determined in this computation as {Dc} m
- $Q$ = Longshore sediment transport rate (m³/time)
- $s$ = Alongshore spatial coordinate index (m)

To calculate the magnitude of the sediment transport rate ($Q$), the empirical formulation modified from the CERC (Coastal Engineering Research Center) method stated in the USACE *Coastal Engineering Manual* (2002) is applied:
$$ Q = K \cdot H_{{sb}}^2 \cdot \sin(2\alpha_{{bs}}) $$
The $K$ value represents an empirical transport scale calibration constant set at {K:.5f}, $H_{{sb}}$ indicates the breaking wave height, and $\alpha_{{bs}}$ represents the wave breaking angle deflection relative to the local shoreline normal. The wave transformation process (refraction and shoaling) from deep water to shallow water zones is computed using Snell's Law:
$$ \frac{{\sin \theta}}{{C}} = \text{{constant}} $$

## 3. ANALYSIS AND CALCULATION
The resolution of the entire *One-Line Model* equation is executed based on iterative numerical computation (finite difference method) through an algorithm design compiled in Python. 

**Numerical Algorithm Procedure:**
1.  **Coastal Geometry Initialization:** The coordinate matrix is calculated for its arc length ($s$), and the normal-tangential gradient vectors are derived.
2.  **Wave Refraction Transformation:** Transforming deep water wave ($H_0$) into the local wave height value ($H_{{s,\text{{local}}}}$) by factoring in the bathymetry.
3.  **LST Indicator Calculation ($Q$):** Distributing the sediment rate $Q$. If the wave trajectory is obstructed by a structural profile such as a Breakwater or Seawall, a mathematical diffraction/reflection correction is applied $\rightarrow H_{{s,\text{{mod}}}} = H_s \cdot (1 - K_d)$.

<br>
<img src="data:image/png;base64,{q_b64}" width="800" />
<br>

*Graph Explanation:* The $Q(s)$ function graph above describes the dynamics of the sediment transport rate indicator along the shoreline index. Highly fluctuating values signify areas of accelerated longshore drift due to critical incident angles, while the structure-modified graph (orange curve) depicts the energy attenuation effect when coastal protection structures are involved.

4.  **Spatial Evolution ($\Delta y$):** The equilibrium rate is calculated from the spatial derivative $\frac{{\Delta Q}}{{\Delta s}}$ accumulated linearly over the simulation time ({report_year} years). Taking computational samples at the extreme point of the array index {i_max}, a radically declining $\frac{{dQ}}{{ds}}$ gradient results in a massive positive $\Delta y$ displacement (sediment run-up).

<br>
<img src="data:image/png;base64,{dy_b64}" width="800" />
<br>

*Graph Explanation:* This $\Delta y$ (delta y) graph shows the deviation of shoreline advance (positive values) or retreat (negative values) over the {report_year}-year timeframe, measured perpendicularly (shore normal).

## 4. CONCLUSION
As a comprehensive output, the spatial distribution of coastal evolution is mapped onto the following projection:

<br>
<img src="data:image/png;base64,{map_b64}" width="800" />
<br>

Based on the algorithm computation of the *One-Line Model* against bathymetric data and hydrodynamic observation (*Copernicus*) processed numerically over a {report_year}-year horizon, it is technically and descriptively concluded that:
1.  **Coastal Accretion:** The most extreme sediment accumulation point (landward advance) is recorded at **{max_adv:.2f} meters**, centered around the coastal observation station $s \approx {s_adv:.0f}$ meters.
2.  **Coastal Abrasion:** The most severe dominant coastal retreat (material erosion) occurs up to **{max_ret:.2f} meters**, located at the edge station $s \approx {s_ret:.0f}$ meters.
3.  **Spatial Technical Application:** Technically, the trend calculation of abrasion and accretion rates derived from this coastal numerical regression computation can be used comprehensively as a descriptive reference basis for coastal spatial planning. This is particularly crucial in justifying the determination of the **coastal buffer zone** boundary, ensuring the sustainable safeguard of aquatic environmental utilities from natural dynamics.
"""
        with st.expander("📄 Show Text / Word Report (Markdown)", expanded=True):
            st.markdown(report_markdown, unsafe_allow_html=True)
            
            st.download_button(
                label="⬇️ Download Report (.MD)",
                data=report_markdown,
                file_name=f"Narrative_Report_{location_name}.md",
                mime="text/markdown"
            )

else:
    st.info("Upload shoreline + wave data to run the simulation (bathymetry & structures are optional).")
