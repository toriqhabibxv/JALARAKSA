# -*- coding: utf-8 -*-
"""
=============================================================================
 JALA RAKSA
 Integrated Drought-to-Recovery Reservoir Operation Dashboard
 Decision Support System Operasi Adaptif Bendungan Berbasis IWRM
=============================================================================

 Filosofi   : CONSERVE -> ALLOCATE -> ANTICIPATE -> RECOVER
 Sifat      : DECISION SUPPORT SYSTEM (DSS), bukan sistem kendali otomatis.
              Sistem TIDAK PERNAH membuka/menutup pintu bendungan.
              Seluruh keluaran adalah REKOMENDASI yang harus di-APPROVE /
              MODIFY / REJECT oleh operator berwenang (human-in-the-loop).

 Menjalankan:
     streamlit run app.py

 Dependensi: streamlit, plotly, pandas, numpy  (scipy opsional, tidak wajib)

 PERINGATAN KALIBRASI
 --------------------
 Seluruh membership function fuzzy, threshold DEWS/FEWS, bobot prioritas
 tanaman, koefisien Kc/Ky, kurva elevasi-kapasitas, kapasitas outlet, dan
 batas aliran hilir pada file ini adalah NILAI DEMONSTRASI. Nilai tersebut
 WAJIB dikalibrasi dengan data desain bendungan, data hidrologi historis,
 dan hasil simulasi operasi waduk yang sebenarnya sebelum dipakai operasional.
=============================================================================
"""

import io
import math
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 0. KONFIGURASI HALAMAN  (harus menjadi perintah streamlit pertama)
# =============================================================================
st.set_page_config(
    page_title="JALA RAKSA | Reservoir Operation DSS",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "1.0.0"

# -----------------------------------------------------------------------------
# Kompatibilitas API Streamlit.
# Streamlit >= 1.49 memakai width="stretch"; versi lama memakai
# use_container_width=True. Wrapper di bawah membuat app berjalan pada keduanya
# (penting karena Streamlit Community Cloud sering memakai versi terbaru).
# -----------------------------------------------------------------------------
def _st_version_tuple():
    parts = []
    for chunk in str(st.__version__).split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


_ST_VER = _st_version_tuple()
_NEW_WIDTH_API = _ST_VER >= (1, 49, 0)


def show_chart(fig, key=None):
    """Menampilkan figure Plotly selebar container, kompatibel lintas versi."""
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def wide_button(label, key=None, type="secondary"):
    """Tombol selebar container, kompatibel lintas versi Streamlit."""
    if _NEW_WIDTH_API:
        return st.button(label, key=key, type=type, width="stretch")
    return st.button(label, key=key, type=type, use_container_width=True)


def show_table(df, height=None, hide_index=True):
    """Menampilkan dataframe selebar container, kompatibel lintas versi."""
    kwargs = {"hide_index": hide_index}
    if height is not None:
        kwargs["height"] = height
    if _NEW_WIDTH_API:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)


# =============================================================================
# 1. PARAMETER DESAIN BENDUNGAN  (DEMONSTRASI - GANTI DENGAN DATA DESAIN ASLI)
# =============================================================================
# Parameter fungsional di bawah mengikuti dokumen konsep JALA RAKSA:
#   - Daerah irigasi              : 3.500 ha
#   - Kebutuhan irigasi maksimum  : 2,50 m3/detik
#   - Air baku minimum            : 76 liter/detik (+/- 35.000 jiwa)
#   - PLTA                        : 1,40 MW
#   - Fungsi lain                 : perikanan dan pariwisata
# Sedangkan geometri waduk (elevasi & kurva kapasitas) adalah ANGKA DEMO.
DAM = {
    "name": "Bendungan Multipurpose (Studi JALA RAKSA)",
    # --- Elevasi karakteristik (m dpl) ---
    "EL_BOTTOM": 110.0,   # dasar waduk (acuan kurva kapasitas)
    "LWL": 120.0,         # Low Water Level (batas operasi terendah)
    "NWL": 145.0,         # Normal Water Level (muka air normal)
    "FCL": 146.0,         # Flood Control Level (batas pengendalian banjir)
    "HWL": 147.5,         # High Water Level / Flood Water Level
    "CREST": 150.0,       # elevasi puncak bendungan
    # --- Kurva elevasi-kapasitas: S(h) = a * (h - EL_BOTTOM)^b  [juta m3] ---
    "CURVE_A": 0.072610,
    "CURVE_B": 1.838000,
    # --- Kapasitas hidraulik ---
    "OUTLET_CAPACITY": 25.0,      # m3/s, kapasitas total outlet terkendali
    "SPILLWAY_CAPACITY": 350.0,   # m3/s, kapasitas pelimpah desain
    "DOWNSTREAM_SAFE_FLOW": 60.0, # m3/s, batas aman aliran sungai hilir
    # --- Kebutuhan air (m3/s) ---
    "Q_DOMESTIC": 0.076,          # 76 l/detik air baku
    "Q_IRRIGATION_MAX": 2.50,     # kebutuhan puncak irigasi
    "Q_ENV_FLOW": 0.350,          # environmental / maintenance flow
    "Q_HYDRO_DESIGN": 4.20,       # debit desain turbin (non-konsumtif)
    "Q_FISHERIES": 0.060,         # perikanan (keramba / kolam hilir)
    "Q_TOURISM": 0.020,           # pariwisata (kebutuhan sangat kecil)
    # --- Parameter PLTA ---
    "HYDRO_CAPACITY_MW": 1.40,
    "HYDRO_NET_HEAD": 40.0,       # m, tinggi jatuh efektif (demo)
    "HYDRO_EFFICIENCY": 0.85,
    # --- Irigasi ---
    "IRRIGATION_AREA_HA": 3500.0,
    "IRRIGATION_EFFICIENCY": 0.65,
    # --- Target operasi ---
    "TARGET_SEASONAL_PCT": 80.0,  # target storage musiman (% storage efektif)
    "CRITICAL_STORAGE_PCT": 25.0, # ambang tampungan kritis
    # --- Hidrologi acuan ---
    "MEAN_ANNUAL_INFLOW": 6.50,   # m3/s, inflow rata-rata jangka panjang
    "EVAP_MEAN_MM": 4.6,          # mm/hari, evaporasi rata-rata
    "SEEPAGE_LOSS_MCM_DAY": 0.004,# juta m3/hari, rembesan + kehilangan lain
}

# Nama tahapan filosofi JALA RAKSA
STAGES = ["CONSERVE", "ALLOCATE", "ANTICIPATE", "RECOVER"]

# =============================================================================
# 2. PALET WARNA & STATUS
# =============================================================================
C_BG = "#071426"
C_CARD = "#0f2440"
C_CARD2 = "#13314f"
C_LINE = "#1e4468"
C_TEXT = "#e8f2ff"
C_MUTED = "#8fb0cd"
C_CYAN = "#22d3ee"
C_BLUE = "#3b82f6"
C_GREEN = "#22c55e"
C_YELLOW = "#facc15"
C_ORANGE = "#fb923c"
C_RED = "#ef4444"
C_GREY = "#94a3b8"
C_DARKRED = "#7f1d1d"
C_PURPLE = "#a78bfa"

STATUS_COLOR = {
    "NORMAL": C_GREEN,
    "WASPADA": C_YELLOW,
    "SIAGA": C_ORANGE,
    "DARURAT": C_RED,
    "RECOVERY": C_BLUE,
    "FAIL-SAFE": C_DARKRED,
    "SAFE": C_GREEN,
    "WARNING": C_ORANGE,
    "CRITICAL": C_RED,
    "ONLINE": C_GREEN,
    "OFFLINE": C_RED,
    "LOW": C_GREEN,
    "MEDIUM": C_YELLOW,
    "HIGH": C_ORANGE,
    "CRITICAL_PRIORITY": C_RED,
}

MODE_COLOR = {
    "NORMAL OPERATION": C_GREEN,
    "DROUGHT CONSERVATION": C_YELLOW,
    "DROUGHT HEDGING": C_ORANGE,
    "EMERGENCY ALLOCATION": C_RED,
    "FLOOD ANTICIPATION": C_CYAN,
    "RECOVERY MODE": C_BLUE,
    "FAIL-SAFE MODE": C_DARKRED,
}

# =============================================================================
# 3. CSS  (tema dark navy + responsif laptop/desktop/tablet/smartphone)
# =============================================================================
CUSTOM_CSS = f"""
<style>
.stApp {{
    background: linear-gradient(180deg, {C_BG} 0%, #0a1d33 55%, #08182b 100%);
    color: {C_TEXT};
}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #08192c 0%, #0c2340 100%);
    border-right: 1px solid {C_LINE};
}}
section[data-testid="stSidebar"] * {{ color: {C_TEXT}; }}

h1, h2, h3, h4, h5 {{ color: {C_TEXT}; letter-spacing: .2px; }}
hr {{ border-color: {C_LINE}; }}

/* ---------- Header aplikasi ---------- */
.jr-header {{
    background: linear-gradient(120deg, #0b2947 0%, #10395f 45%, #0b2947 100%);
    border: 1px solid {C_LINE};
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,.35);
}}
.jr-title {{
    font-size: 2.05rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, {C_CYAN}, #7dd3fc 45%, #ffffff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.jr-sub {{ color: #bfe4f5; font-size: 1.02rem; font-weight: 600; margin-top: 2px; }}
.jr-sub2 {{ color: {C_MUTED}; font-size: .86rem; margin-top: 2px; }}

/* ---------- Kartu KPI ---------- */
.jr-card {{
    background: linear-gradient(160deg, {C_CARD} 0%, {C_CARD2} 100%);
    border: 1px solid {C_LINE};
    border-radius: 16px;
    padding: 14px 16px;
    height: 100%;
    box-shadow: 0 6px 18px rgba(0,0,0,.28);
    transition: transform .15s ease, border-color .15s ease;
}}
.jr-card:hover {{ transform: translateY(-2px); border-color: {C_CYAN}; }}
.jr-card-label {{
    color: {C_MUTED}; font-size: .74rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .6px;
}}
.jr-card-value {{ font-size: 1.62rem; font-weight: 800; margin-top: 4px; line-height: 1.15; }}
.jr-card-unit {{ font-size: .82rem; font-weight: 600; color: {C_MUTED}; margin-left: 3px; }}
.jr-card-foot {{ font-size: .74rem; color: {C_MUTED}; margin-top: 5px; }}

/* ---------- Badge status ---------- */
.jr-badge {{
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: .78rem; font-weight: 800; letter-spacing: .4px;
}}
.jr-statusbox {{
    background: {C_CARD}; border: 1px solid {C_LINE}; border-radius: 16px;
    padding: 14px 16px; text-align: center; height: 100%;
}}
.jr-statusbox .lbl {{
    color: {C_MUTED}; font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .6px;
}}
.jr-statusbox .val {{ font-size: 1.35rem; font-weight: 900; margin-top: 6px; }}
.jr-statusbox .note {{ color: {C_MUTED}; font-size: .72rem; margin-top: 4px; }}

/* ---------- Panel & banner ---------- */
.jr-panel {{
    background: {C_CARD}; border: 1px solid {C_LINE};
    border-radius: 16px; padding: 16px 18px; margin-bottom: 12px;
}}
.jr-panel h4 {{ margin-top: 0; }}
.jr-warn {{
    background: rgba(251,146,60,.12); border: 1px solid {C_ORANGE};
    border-left: 6px solid {C_ORANGE};
    border-radius: 12px; padding: 12px 16px; margin: 8px 0; color: #ffe6cc;
}}
.jr-danger {{
    background: rgba(239,68,68,.14); border: 1px solid {C_RED};
    border-left: 6px solid {C_RED};
    border-radius: 12px; padding: 12px 16px; margin: 8px 0; color: #ffd9d9;
}}
.jr-ok {{
    background: rgba(34,197,94,.10); border: 1px solid {C_GREEN};
    border-left: 6px solid {C_GREEN};
    border-radius: 12px; padding: 12px 16px; margin: 8px 0; color: #d8ffe6;
}}
.jr-info {{
    background: rgba(59,130,246,.12); border: 1px solid {C_BLUE};
    border-left: 6px solid {C_BLUE};
    border-radius: 12px; padding: 12px 16px; margin: 8px 0; color: #dbeafe;
}}
.jr-demo {{
    display:inline-block; background: rgba(167,139,250,.15);
    border: 1px dashed {C_PURPLE}; color: #ddd6fe;
    border-radius: 8px; padding: 2px 10px; font-size: .70rem;
    font-weight: 800; letter-spacing: .5px;
}}

/* ---------- Diagram tahapan CONSERVE-ALLOCATE-ANTICIPATE-RECOVER ---------- */
.jr-flow {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; }}
.jr-step {{
    flex: 1 1 130px; min-width: 130px; text-align: center;
    border-radius: 14px; padding: 12px 8px;
    background: {C_CARD}; border: 1px solid {C_LINE}; color: {C_MUTED};
}}
.jr-step.active {{
    background: linear-gradient(160deg, rgba(34,211,238,.20), rgba(59,130,246,.18));
    border: 1px solid {C_CYAN}; color: #ffffff;
    box-shadow: 0 0 18px rgba(34,211,238,.25);
}}
.jr-step .dot {{ font-size: 1.05rem; }}
.jr-step .nm {{ font-weight: 800; font-size: .90rem; margin-top: 2px; }}
.jr-step .ds {{ font-size: .70rem; margin-top: 3px; }}
.jr-arrow {{ align-self: center; color: {C_CYAN}; font-weight: 800; }}

/* ---------- Grid skematik zona irigasi ---------- */
.jr-grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.jr-plot {{
    flex: 1 1 150px; min-width: 140px; border-radius: 14px; padding: 12px;
    border: 1px solid {C_LINE}; background: {C_CARD}; text-align: center;
}}
.jr-plot .zn {{ font-weight: 800; font-size: 1.0rem; }}
.jr-plot .cr {{ font-size: .74rem; color: {C_MUTED}; }}
.jr-plot .pr {{ font-weight: 800; font-size: .80rem; margin-top: 6px; }}

/* ---------- Footer ---------- */
.jr-footer {{
    margin-top: 26px; padding: 18px 20px; border-radius: 16px;
    background: {C_CARD}; border: 1px solid {C_LINE}; color: {C_MUTED};
    font-size: .80rem; line-height: 1.55;
}}
.jr-footer b {{ color: {C_TEXT}; }}

/* ---------- Tabel ---------- */
div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: auto; }}

/* ---------- RESPONSIVE : tablet & smartphone ---------- */
@media (max-width: 900px) {{
    .jr-title {{ font-size: 1.55rem; }}
    .jr-sub {{ font-size: .92rem; }}
    .jr-card-value {{ font-size: 1.35rem; }}
}}
@media (max-width: 640px) {{
    /* kolom horizontal Streamlit dibuat menumpuk vertikal di layar kecil */
    div[data-testid="stHorizontalBlock"] {{ flex-direction: column !important; }}
    div[data-testid="stHorizontalBlock"] > div {{ width: 100% !important; }}
    .jr-title {{ font-size: 1.30rem; }}
    .jr-sub {{ font-size: .84rem; }}
    .jr-sub2 {{ font-size: .74rem; }}
    .jr-card {{ padding: 12px; }}
    .jr-card-value {{ font-size: 1.30rem; }}
    .jr-step {{ flex: 1 1 100%; }}
    .jr-plot {{ flex: 1 1 46%; min-width: 120px; }}
    .block-container {{ padding-left: .7rem; padding-right: .7rem; padding-top: 1rem; }}
}}
</style>
"""


# =============================================================================
# 4. KOMPONEN UI KECIL
# =============================================================================
def kpi_card(label, value, unit="", foot="", color=C_CYAN):
    """Kartu KPI membulat (rounded metric card)."""
    st.markdown(
        f"""
        <div class="jr-card">
            <div class="jr-card-label">{label}</div>
            <div class="jr-card-value" style="color:{color}">{value}
                <span class="jr-card-unit">{unit}</span></div>
            <div class="jr-card-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_box(label, value, note="", color=C_GREEN):
    """Kotak status besar (DEWS / FEWS / STRESS / MODE)."""
    st.markdown(
        f"""
        <div class="jr-statusbox">
            <div class="lbl">{label}</div>
            <div class="val" style="color:{color}">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text, color=C_GREEN):
    """Badge status inline."""
    return (
        f'<span class="jr-badge" style="background:{color}22;'
        f'border:1px solid {color};color:{color}">{text}</span>'
    )


def demo_tag(text="DEMONSTRATION DATA / SIMULATION MODE"):
    """Label wajib untuk seluruh nilai demonstrasi."""
    st.markdown(f'<span class="jr-demo">⚠ {text}</span>', unsafe_allow_html=True)


def panel(title, body_html):
    st.markdown(
        f'<div class="jr-panel"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True,
    )


def alert(text, kind="info"):
    css = {"info": "jr-info", "ok": "jr-ok", "warn": "jr-warn", "danger": "jr-danger"}[kind]
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def section(title, subtitle=""):
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(
            f'<div style="color:{C_MUTED};font-size:.85rem;margin-top:-8px;'
            f'margin-bottom:10px">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def style_plot(fig, height=330, title=None, legend_top=True):
    """Tema Plotly seragam: transparan, dark navy, mobile friendly."""
    fig.update_layout(
        height=height,
        title={"text": title or ""},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=C_TEXT, size=12),
        margin=dict(l=45, r=20, t=50 if title else 28, b=40),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02 if legend_top else -0.28,
            x=0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor=C_LINE, zerolinecolor=C_LINE, linecolor=C_LINE)
    fig.update_yaxes(gridcolor=C_LINE, zerolinecolor=C_LINE, linecolor=C_LINE)
    return fig


# =============================================================================
# 5. FUNGSI HIDRAULIK DASAR WADUK
# =============================================================================
def elevation_to_storage(elev_m, dam=DAM):
    """
    Kurva elevasi-kapasitas (elevation-capacity curve):

        S(h) = a * (h - h_dasar) ^ b      [juta m3 / MCM]

    a dan b adalah koefisien regresi kurva kapasitas waduk.
    KALIBRASI: ganti CURVE_A dan CURVE_B dengan hasil regresi kurva
    elevasi-kapasitas hasil pengukuran topografi/bathimetri waduk.
    """
    h = max(float(elev_m), dam["EL_BOTTOM"])
    return dam["CURVE_A"] * ((h - dam["EL_BOTTOM"]) ** dam["CURVE_B"])


def storage_to_elevation(storage_mcm, dam=DAM):
    """Invers kurva kapasitas: h = h_dasar + (S/a)^(1/b)."""
    s = max(float(storage_mcm), 0.0)
    if s <= 0:
        return dam["EL_BOTTOM"]
    return dam["EL_BOTTOM"] + (s / dam["CURVE_A"]) ** (1.0 / dam["CURVE_B"])


# Konstanta tampungan karakteristik (dihitung sekali saat modul dimuat)
S_LWL = elevation_to_storage(DAM["LWL"])      # tampungan mati (dead storage)
S_NWL = elevation_to_storage(DAM["NWL"])      # tampungan pada muka air normal
S_FCL = elevation_to_storage(DAM["FCL"])      # tampungan pada flood control level
S_HWL = elevation_to_storage(DAM["HWL"])      # tampungan pada muka air banjir
S_EFF = S_NWL - S_LWL                         # tampungan efektif (live storage)


def calculate_storage_percentage(storage_mcm, dam=DAM):
    """
    Persentase tampungan EFEKTIF (live storage), bukan tampungan total:

        S%(t) = (S(t) - S_LWL) / (S_NWL - S_LWL) * 100

    0 %   -> muka air pada LWL (batas operasi terendah)
    100 % -> muka air pada NWL (muka air normal)
    Nilai > 100 % berarti waduk berada di atas NWL (zona pengendalian banjir).
    """
    return (float(storage_mcm) - S_LWL) / S_EFF * 100.0


def percentage_to_storage(pct):
    """Kebalikan calculate_storage_percentage: S = S_LWL + pct/100 * S_eff."""
    return S_LWL + (float(pct) / 100.0) * S_EFF


def cms_to_mcm_per_day(q_cms):
    """Konversi debit m3/detik -> volume juta m3 per hari (x 86400 / 1e6)."""
    return float(q_cms) * 86400.0 / 1.0e6


def mcm_per_day_to_cms(vol_mcm):
    """Konversi volume juta m3 per hari -> debit rata-rata m3/detik."""
    return float(vol_mcm) * 1.0e6 / 86400.0


def reservoir_area_km2(storage_mcm, dam=DAM):
    """
    Luas genangan pendekatan dari turunan kurva kapasitas:
        A = dS/dh = a*b*(h-h0)^(b-1)   [juta m3 per m] = [km2]
    Dipakai untuk menghitung evaporasi dan hujan langsung di atas waduk.
    """
    h = storage_to_elevation(storage_mcm, dam)
    x = max(h - dam["EL_BOTTOM"], 0.01)
    return dam["CURVE_A"] * dam["CURVE_B"] * (x ** (dam["CURVE_B"] - 1.0))


def calculate_water_balance(storage_mcm, inflow_cms, outflow_cms,
                            rainfall_mm, evaporation_mm, dt_days=1.0, dam=DAM):
    """
    NERACA AIR WADUK (reservoir water balance), bentuk diskrit:

        S(t+1) = S(t) + [ Qin - Qout ] * dt + P*A - E*A - L*dt

    dengan:
        S     = tampungan waduk                       [juta m3]
        Qin   = inflow                                [m3/detik]
        Qout  = outflow total (release + spill)       [m3/detik]
        P     = hujan langsung di atas genangan       [mm]
        E     = evaporasi permukaan waduk             [mm]
        A     = luas genangan waduk                   [km2]
        L     = kehilangan (rembesan/seepage dll.)    [juta m3/hari]

    Catatan konversi: 1 mm x 1 km2 = 1.000 m3 = 0,001 juta m3.
    Mengembalikan dict berisi setiap komponen agar dapat diaudit di dashboard.
    """
    area = reservoir_area_km2(storage_mcm, dam)
    vol_in = cms_to_mcm_per_day(inflow_cms) * dt_days
    vol_out = cms_to_mcm_per_day(outflow_cms) * dt_days
    vol_rain = rainfall_mm * area * 0.001          # mm * km2 -> juta m3
    vol_evap = evaporation_mm * area * 0.001
    vol_loss = dam["SEEPAGE_LOSS_MCM_DAY"] * dt_days
    delta = vol_in - vol_out + vol_rain - vol_evap - vol_loss
    s_new = max(storage_mcm + delta, 0.0)
    return {
        "area_km2": area,
        "inflow_mcm": vol_in,
        "outflow_mcm": vol_out,
        "rain_mcm": vol_rain,
        "evap_mcm": vol_evap,
        "loss_mcm": vol_loss,
        "delta_storage_mcm": delta,
        "storage_new_mcm": s_new,
        "elevation_new_m": storage_to_elevation(s_new, dam),
    }


# =============================================================================
# 6. FUZZY MAMDANI - RESERVOIR STRESS ASSESSMENT
# =============================================================================
# CATATAN PENTING (WAJIB DIBACA):
# "Membership functions, thresholds, and fuzzy rules shown in this demo are
#  illustrative and MUST be calibrated using actual reservoir hydrological
#  data and simulation."
# Nilai di bawah TIDAK BOLEH diklaim sebagai angka final engineering.
#
# Implementasi ini adalah Fuzzy Mamdani penuh namun TRANSPARAN (bukan black box):
#   1) Fuzzifikasi   : membership function segitiga/trapesium
#   2) Inferensi     : implikasi MIN (t-norm), agregasi MAX (s-norm)
#   3) Defuzzifikasi : centroid (center of gravity)
# Basis aturan dibangkitkan dari indeks keparahan berbobot yang dapat dibaca
# operator, sehingga setiap aturan dapat ditelusuri asal-usulnya.
# =============================================================================
def trimf(x, a, b, c):
    """Membership function segitiga."""
    if a == b == c:
        return 1.0 if x == a else 0.0
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def trapmf(x, a, b, c, d):
    """Membership function trapesium."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


# ---- Membership function INPUT (DEMO - WAJIB DIKALIBRASI) -------------------
def mf_storage(x):
    """Reservoir Storage [% tampungan efektif] -> LOW / MEDIUM / HIGH."""
    return {
        "LOW": trapmf(x, -1, 0, 25, 45),
        "MEDIUM": trimf(x, 30, 52, 75),
        "HIGH": trapmf(x, 60, 80, 100, 130),
    }


def mf_inflow(x):
    """Inflow Condition [% terhadap inflow rata-rata] -> LOW / NORMAL / HIGH."""
    return {
        "LOW": trapmf(x, -1, 0, 30, 60),
        "NORMAL": trimf(x, 45, 85, 125),
        "HIGH": trapmf(x, 105, 140, 300, 400),
    }


def mf_forecast(x):
    """
    Forecast Hydrological Condition [indeks 0-100].
    0   = sangat kering, 50 = normal, 100 = sangat basah.
    """
    return {
        "DRY": trapmf(x, -1, 0, 22, 45),
        "NORMAL": trimf(x, 30, 50, 72),
        "WET": trapmf(x, 58, 78, 100, 130),
    }


def mf_dsr(x):
    """
    Demand-Supply Ratio [-] = Total Demand / Available Supply.
    < 1 aman, > 1 defisit.
    """
    return {
        "LOW": trapmf(x, -1, 0, 0.55, 0.80),
        "BALANCED": trimf(x, 0.65, 0.95, 1.20),
        "HIGH": trapmf(x, 1.05, 1.35, 3.0, 5.0),
    }


# ---- Membership function OUTPUT: Reservoir Stress 0-100 ---------------------
STRESS_MF = {
    "NORMAL": (0.0, 0.0, 30.0),
    "WASPADA": (18.0, 37.5, 57.0),
    "SIAGA": (43.0, 62.5, 82.0),
    "DARURAT": (70.0, 100.0, 100.0),
}

# ---- Bobot keparahan tiap input (DEMO - WAJIB DIKALIBRASI) -----------------
# Storage diberi bobot terbesar karena merupakan variabel keadaan (state)
# waduk yang paling menentukan kemampuan pelayanan air.
SEV_WEIGHT = {"storage": 1.40, "inflow": 1.00, "forecast": 0.90, "dsr": 1.00}
SEV_LEVEL = {
    "storage": {"HIGH": 0.0, "MEDIUM": 1.0, "LOW": 2.0},
    "inflow": {"HIGH": 0.0, "NORMAL": 1.0, "LOW": 2.0},
    "forecast": {"WET": 0.0, "NORMAL": 1.0, "DRY": 2.0},
    "dsr": {"LOW": 0.0, "BALANCED": 1.0, "HIGH": 2.0},
}
SEV_MAX = 2.0 * sum(SEV_WEIGHT.values())     # keparahan maksimum teoretis


def _consequent(sev_norm):
    """Peta indeks keparahan ternormalisasi (0-1) -> term output fuzzy."""
    if sev_norm < 0.28:
        return "NORMAL"
    if sev_norm < 0.50:
        return "WASPADA"
    if sev_norm < 0.72:
        return "SIAGA"
    return "DARURAT"


def build_rule_base():
    """
    Membangkitkan basis aturan Mamdani 3x3x3x3 = 81 aturan.
    Setiap aturan berbentuk:
        IF Storage is A AND Inflow is B AND Forecast is C AND DSR is D
        THEN Reservoir Stress is E
    Konsekuen E ditentukan dari indeks keparahan berbobot -> transparan,
    dapat diperiksa, dan dapat dikalibrasi ulang oleh engineer.
    """
    rules = []
    for s in ["LOW", "MEDIUM", "HIGH"]:
        for i in ["LOW", "NORMAL", "HIGH"]:
            for f in ["DRY", "NORMAL", "WET"]:
                for d in ["LOW", "BALANCED", "HIGH"]:
                    sev = (
                        SEV_LEVEL["storage"][s] * SEV_WEIGHT["storage"]
                        + SEV_LEVEL["inflow"][i] * SEV_WEIGHT["inflow"]
                        + SEV_LEVEL["forecast"][f] * SEV_WEIGHT["forecast"]
                        + SEV_LEVEL["dsr"][d] * SEV_WEIGHT["dsr"]
                    )
                    rules.append((s, i, f, d, _consequent(sev / SEV_MAX)))
    return rules


RULE_BASE = build_rule_base()


def calculate_reservoir_stress(storage_pct, inflow_pct_mean,
                               forecast_index, demand_supply_ratio,
                               return_detail=False):
    """
    FUZZY MAMDANI RESERVOIR STRESS ASSESSMENT.

    Input:
        storage_pct          : tampungan efektif waduk [%]
        inflow_pct_mean      : inflow terhadap rata-rata jangka panjang [%]
        forecast_index       : indeks kondisi hidrologi prakiraan [0-100]
        demand_supply_ratio  : total demand / available supply [-]

    Proses: fuzzifikasi -> inferensi MIN -> agregasi MAX -> defuzzifikasi centroid.
    Output: Reservoir Stress Level [0-100], makin tinggi makin tertekan.

    Klasifikasi (DEMO): 0-25 NORMAL | 25-50 WASPADA | 50-75 SIAGA | 75-100 DARURAT
    """
    mu_s = mf_storage(float(storage_pct))
    mu_i = mf_inflow(float(inflow_pct_mean))
    mu_f = mf_forecast(float(forecast_index))
    mu_d = mf_dsr(float(demand_supply_ratio))

    # Agregasi MAX terhadap kekuatan aktivasi tiap term output
    activation = {k: 0.0 for k in STRESS_MF}
    fired = []
    for (s, i, f, d, out) in RULE_BASE:
        w = min(mu_s[s], mu_i[i], mu_f[f], mu_d[d])   # implikasi MIN
        if w > 0.0:
            activation[out] = max(activation[out], w)
            fired.append({"IF": f"S={s}, Qin={i}, Fcst={f}, DSR={d}",
                          "THEN": out, "w": round(w, 4)})

    # Defuzzifikasi centroid pada semesta 0-100
    x = np.linspace(0.0, 100.0, 501)
    agg = np.zeros_like(x)
    for term, (a, b, c) in STRESS_MF.items():
        w = activation[term]
        if w <= 0:
            continue
        mf = np.array([trimf(v, a, b, c) for v in x])
        agg = np.maximum(agg, np.minimum(mf, w))     # clipping Mamdani

    if agg.sum() <= 1e-9:
        stress = 50.0   # fallback netral bila tidak ada aturan aktif
    else:
        stress = float(np.sum(x * agg) / np.sum(agg))
    stress = float(np.clip(stress, 0.0, 100.0))

    if not return_detail:
        return stress

    fired = sorted(fired, key=lambda r: r["w"], reverse=True)
    return stress, {
        "mu_storage": mu_s, "mu_inflow": mu_i,
        "mu_forecast": mu_f, "mu_dsr": mu_d,
        "activation": activation,
        "fired_rules": fired[:12],
        "n_fired": len(fired),
        "x": x, "aggregated": agg,
    }


def classify_stress(stress):
    """Klasifikasi Reservoir Stress -> NORMAL/WASPADA/SIAGA/DARURAT (DEMO)."""
    if stress < 25:
        return "NORMAL"
    if stress < 50:
        return "WASPADA"
    if stress < 75:
        return "SIAGA"
    return "DARURAT"


# =============================================================================
# 7. DEWS - DROUGHT EARLY WARNING SYSTEM
# =============================================================================
def calculate_dews_status(storage_pct, stress, storage_trend_pct_day,
                          forecast_index, dsr):
    """
    Status DEWS ditentukan dari kombinasi Reservoir Stress (fuzzy) dan
    pengaman berbasis aturan (rule-based guard) agar tetap transparan:

      - Guard tampungan  : tampungan sangat rendah tidak boleh berstatus NORMAL
      - Guard tren       : penurunan tampungan cepat menaikkan status
      - Guard defisit    : DSR > 1 (defisit) menaikkan status

    Mengembalikan (status, skor 0-3, daftar alasan).
    """
    order = ["NORMAL", "WASPADA", "SIAGA", "DARURAT"]
    level = order.index(classify_stress(stress))
    reasons = [f"Fuzzy Reservoir Stress = {stress:.1f}/100 -> {order[level]}"]

    # Guard 1: tampungan efektif
    if storage_pct <= DAM["CRITICAL_STORAGE_PCT"]:
        if level < 3:
            level = 3
            reasons.append(
                f"Tampungan {storage_pct:.1f}% <= ambang kritis "
                f"{DAM['CRITICAL_STORAGE_PCT']:.0f}% -> dinaikkan ke DARURAT")
    elif storage_pct <= 40 and level < 2:
        level = 2
        reasons.append(f"Tampungan {storage_pct:.1f}% <= 40% -> dinaikkan ke SIAGA")
    elif storage_pct <= 55 and level < 1:
        level = 1
        reasons.append(f"Tampungan {storage_pct:.1f}% <= 55% -> dinaikkan ke WASPADA")

    # Guard 2: tren tampungan (penurunan cepat)
    if storage_trend_pct_day <= -0.55 and level < 3:
        level += 1
        reasons.append(
            f"Tren tampungan {storage_trend_pct_day:+.2f} %/hari (turun cepat) "
            f"-> status dinaikkan satu tingkat")

    # Guard 3: defisit pasokan
    if dsr >= 1.15 and level < 3:
        level += 1
        reasons.append(f"Demand-Supply Ratio {dsr:.2f} > 1,15 (defisit) "
                       f"-> status dinaikkan satu tingkat")

    # Peredam: tampungan tinggi + prakiraan basah tidak wajar berstatus tinggi
    if storage_pct >= 85 and forecast_index >= 60 and level > 1:
        level = 1
        reasons.append("Tampungan tinggi & prakiraan basah -> status diturunkan "
                       "ke WASPADA (peredam false alarm)")

    level = int(np.clip(level, 0, 3))
    return order[level], level, reasons


# =============================================================================
# 8. ADAPTIVE DROUGHT HEDGING
# =============================================================================
def calculate_hedging_level(dews_status, stress, storage_pct):
    """
    ADAPTIVE DROUGHT HEDGING.

    Menerjemahkan status kekeringan menjadi faktor pelepasan (release factor)
    per sektor. Prinsip: air baku dan environmental flow DIPERTAHANKAN,
    penghematan diambil terlebih dahulu dari sektor non-esensial.

    KALIBRASI: faktor di bawah adalah nilai DEMO. Persentase pengurangan yang
    sebenarnya harus dicari melalui simulasi neraca air multi-tahun sehingga
    diperoleh strategi dengan reliability terbaik (lihat dokumen konsep butir 7).
    """
    table = {
        # status : (nama level, f_irigasi, f_domestik, f_lingkungan, f_plta, f_lainnya)
        "NORMAL":  ("LEVEL 0 - NORMAL OPERATION", 1.00, 1.00, 1.00, 1.00, 1.00),
        "WASPADA": ("LEVEL 1 - EARLY CONSERVATION", 0.90, 1.00, 1.00, 0.85, 0.70),
        "SIAGA":   ("LEVEL 2 - ADAPTIVE DROUGHT HEDGING", 0.70, 1.00, 1.00, 0.60, 0.35),
        "DARURAT": ("LEVEL 3 - EMERGENCY ALLOCATION", 0.45, 1.00, 0.85, 0.25, 0.00),
    }
    name, f_irr, f_dom, f_env, f_hyd, f_oth = table[dews_status]

    # Penyesuaian halus (fine tuning) berbasis stress agar transisi tidak melompat
    if dews_status in ("WASPADA", "SIAGA"):
        band = (stress - 25.0) / 50.0
        band = float(np.clip(band, 0.0, 1.0))
        f_irr = float(np.clip(f_irr - 0.08 * band, 0.35, 1.0))

    # Pengaman tampungan sangat rendah
    if storage_pct <= 20:
        f_irr = min(f_irr, 0.35)
        f_hyd = min(f_hyd, 0.15)

    return {
        "level_name": name,
        "level_index": ["NORMAL", "WASPADA", "SIAGA", "DARURAT"].index(dews_status),
        "f_irrigation": round(f_irr, 3),
        "f_domestic": round(f_dom, 3),
        "f_environment": round(f_env, 3),
        "f_hydropower": round(f_hyd, 3),
        "f_other": round(f_oth, 3),
        "conservation_pct": round((1.0 - f_irr) * 100.0, 1),
        "irrigation_reduction_pct": round((1.0 - f_irr) * 100.0, 1),
        "domestic_policy": "MAINTAIN",
        "env_policy": "MAINTAIN" if f_env >= 0.999 else "SAFE LIMIT",
    }


def adaptive_operating_band(dews_status, dam=DAM):
    """
    ADAPTIVE OPERATING BAND.

    PENTING: JALA RAKSA TIDAK mengubah LWL / NWL / HWL / FWL hasil desain.
    Yang disesuaikan hanya TARGET OPERASI di dalam batas desain tersebut.

    Mengembalikan (target_bawah_%, target_atas_%) tampungan efektif.
    """
    bands = {
        "NORMAL":  (72.0, 100.0),
        "WASPADA": (62.0, 92.0),
        "SIAGA":   (50.0, 82.0),
        "DARURAT": (35.0, 70.0),
    }
    lo, hi = bands[dews_status]
    return lo, hi


# =============================================================================
# 9. CROP-SENSITIVE IRRIGATION ALLOCATION (FOOD-SMART ALLOCATION)
# =============================================================================
# Koefisien tanaman (Kc) dan faktor sensitivitas hasil (Ky) berikut adalah
# nilai indikatif bergaya FAO-56 untuk DEMO.
# KALIBRASI: gunakan Kc & Ky lokal hasil kajian agronomi daerah irigasi terkait.
CROP_LIBRARY = {
    "Padi":    {"Kc": {"Inisiasi": 1.05, "Vegetatif": 1.15, "Pembungaan": 1.25,
                       "Pengisian Biji": 1.10, "Pematangan": 0.90},
                "Ky": 1.20, "sens": {"Inisiasi": 0.55, "Vegetatif": 0.70,
                                     "Pembungaan": 1.00, "Pengisian Biji": 0.90,
                                     "Pematangan": 0.35}},
    "Jagung":  {"Kc": {"Inisiasi": 0.40, "Vegetatif": 0.85, "Pembungaan": 1.20,
                       "Pengisian Biji": 1.05, "Pematangan": 0.60},
                "Ky": 1.25, "sens": {"Inisiasi": 0.40, "Vegetatif": 0.65,
                                     "Pembungaan": 1.00, "Pengisian Biji": 0.85,
                                     "Pematangan": 0.30}},
    "Kedelai": {"Kc": {"Inisiasi": 0.40, "Vegetatif": 0.80, "Pembungaan": 1.15,
                       "Pengisian Biji": 1.00, "Pematangan": 0.55},
                "Ky": 0.85, "sens": {"Inisiasi": 0.35, "Vegetatif": 0.60,
                                     "Pembungaan": 0.95, "Pengisian Biji": 0.85,
                                     "Pematangan": 0.30}},
    "Bawang":  {"Kc": {"Inisiasi": 0.50, "Vegetatif": 0.85, "Pembungaan": 1.05,
                       "Pengisian Biji": 0.95, "Pematangan": 0.75},
                "Ky": 1.10, "sens": {"Inisiasi": 0.45, "Vegetatif": 0.70,
                                     "Pembungaan": 0.90, "Pengisian Biji": 0.80,
                                     "Pematangan": 0.40}},
    "Tebu":    {"Kc": {"Inisiasi": 0.45, "Vegetatif": 1.00, "Pembungaan": 1.25,
                       "Pengisian Biji": 1.05, "Pematangan": 0.75},
                "Ky": 1.20, "sens": {"Inisiasi": 0.40, "Vegetatif": 0.75,
                                     "Pembungaan": 0.85, "Pengisian Biji": 0.75,
                                     "Pematangan": 0.35}},
}
CROP_STAGES = ["Inisiasi", "Vegetatif", "Pembungaan", "Pengisian Biji", "Pematangan"]


def calculate_etc(et0_mm, crop, stage):
    """
    Evapotranspirasi tanaman (FAO-56):
        ETc = Kc * ET0     [mm/hari]
    ET0 = evapotranspirasi acuan (dari Automatic Weather Station).
    """
    kc = CROP_LIBRARY[crop]["Kc"].get(stage, 1.0)
    return kc * float(et0_mm), kc


def calculate_irrigation_requirement(etc_mm, eff_rain_mm, soil_moisture_pct,
                                     area_ha, efficiency=None):
    """
    KEBUTUHAN AIR IRIGASI (irrigation requirement).

        NIR = ETc - Re - kontribusi lengas tanah        [mm/hari]  (>= 0)
        GIR = NIR / efisiensi irigasi                   [mm/hari]
        Q   = GIR[mm] x Area[ha] x 10 / 86400           [m3/detik]

    Faktor 10: 1 mm x 1 ha = 10 m3.
    Kontribusi lengas tanah didekati linier terhadap kelebihan kelembapan
    di atas titik kritis 55% (DEMO - kalibrasi dengan kurva pF tanah setempat).
    """
    eff = efficiency if efficiency is not None else DAM["IRRIGATION_EFFICIENCY"]
    sm_credit = max(0.0, (float(soil_moisture_pct) - 55.0) / 45.0) * float(etc_mm) * 0.60
    nir = max(float(etc_mm) - float(eff_rain_mm) - sm_credit, 0.0)
    gir = nir / max(eff, 0.05)
    q_cms = gir * float(area_ha) * 10.0 / 86400.0
    return {"NIR_mm": nir, "GIR_mm": gir, "Q_cms": q_cms, "sm_credit_mm": sm_credit}


def calculate_crop_priority(soil_moisture_pct, crop, stage, deficit_ratio):
    """
    SKOR PRIORITAS ZONA IRIGASI (0-100).

    Logika konsep JALA RAKSA:
        tanah basah + tanaman tidak pada fase kritis  -> prioritas RENDAH
        tanah kering + tanaman pada fase sensitif     -> prioritas TINGGI

    Skor = 45% sensitivitas fase x Ky + 35% kekeringan tanah + 20% rasio defisit.
    KALIBRASI: bobot 45/35/20 adalah nilai DEMO dan harus diuji terhadap
    fungsi kehilangan hasil (yield response) aktual daerah irigasi.
    """
    sens = CROP_LIBRARY[crop]["sens"].get(stage, 0.6)
    ky = CROP_LIBRARY[crop]["Ky"]
    sens_score = float(np.clip(sens * ky / 1.30, 0.0, 1.0))
    dry_score = float(np.clip((70.0 - float(soil_moisture_pct)) / 45.0, 0.0, 1.0))
    def_score = float(np.clip(float(deficit_ratio), 0.0, 1.0))
    score = 100.0 * (0.45 * sens_score + 0.35 * dry_score + 0.20 * def_score)
    score = float(np.clip(score, 0.0, 100.0))

    if score >= 72:
        cls = "CRITICAL"
    elif score >= 55:
        cls = "HIGH"
    elif score >= 35:
        cls = "MEDIUM"
    else:
        cls = "LOW"
    return score, cls


def allocate_irrigation_water(zones_df, available_q_cms):
    """
    ALOKASI AIR IRIGASI BERBASIS SENSITIVITAS TANAMAN.

    Mengubah orientasi:
        EQUAL WATER DISTRIBUTION  ->  CROP-SENSITIVE WATER ALLOCATION

    Algoritma (transparan, bukan black box):
      1. Zona diurutkan berdasarkan Priority Score (menurun).
      2. Zona CRITICAL dijamin minimum 85% kebutuhan (jika air mencukupi).
      3. Sisa air dibagi proporsional terhadap bobot prioritas.
      4. Alokasi tidak pernah melebihi kebutuhan zona.
      5. Kehilangan produksi diperkirakan dengan persamaan FAO-33:
             (1 - Ya/Ym) = Ky * (1 - ETa/ETm)
    """
    df = zones_df.copy().sort_values("Priority Score", ascending=False).reset_index(drop=True)
    need = df["Irrigation Requirement (m3/s)"].to_numpy(dtype=float)
    total_need = float(need.sum())
    avail = max(float(available_q_cms), 0.0)

    alloc = np.zeros_like(need)
    if total_need <= 1e-9:
        pass
    elif avail >= total_need:
        alloc = need.copy()                     # air cukup: seluruh kebutuhan dilayani
    else:
        remaining = avail
        # Tahap 1: jaminan minimum untuk zona berprioritas CRITICAL
        for i in range(len(df)):
            if df.loc[i, "Priority Class"] == "CRITICAL":
                give = min(0.85 * need[i], remaining)
                alloc[i] += give
                remaining -= give
                if remaining <= 0:
                    break
        # Tahap 2: sisa air dibagi proporsional terhadap bobot prioritas
        if remaining > 1e-9:
            w = df["Priority Score"].to_numpy(dtype=float) ** 1.5
            gap = np.maximum(need - alloc, 0.0)
            w = np.where(gap > 1e-9, w, 0.0)
            if w.sum() > 1e-9:
                share = remaining * w / w.sum()
                give = np.minimum(share, gap)
                alloc += give
                remaining -= float(give.sum())
                # Tahap 3: iterasi kecil untuk menghabiskan sisa air
                for _ in range(6):
                    if remaining <= 1e-9:
                        break
                    gap = np.maximum(need - alloc, 0.0)
                    if gap.sum() <= 1e-9:
                        break
                    w2 = np.where(gap > 1e-9,
                                  df["Priority Score"].to_numpy(dtype=float), 0.0)
                    if w2.sum() <= 1e-9:
                        break
                    give = np.minimum(remaining * w2 / w2.sum(), gap)
                    alloc += give
                    remaining -= float(give.sum())

    df["Allocation (m3/s)"] = np.round(alloc, 4)
    df["Water Deficit (m3/s)"] = np.round(np.maximum(need - alloc, 0.0), 4)
    ratio = np.divide(alloc, need, out=np.ones_like(alloc), where=need > 1e-9)
    df["Fulfillment (%)"] = np.round(ratio * 100.0, 1)

    # Perkiraan kehilangan hasil panen (FAO-33)
    ky = df["Crop"].map(lambda c: CROP_LIBRARY[c]["Ky"]).to_numpy(dtype=float)
    yield_loss = np.clip(ky * (1.0 - ratio), 0.0, 1.0) * 100.0
    df["Expected Yield Loss (%)"] = np.round(yield_loss, 1)
    df["Production Retained (%)"] = np.round(100.0 - yield_loss, 1)
    df["Allocation Status"] = np.where(
        ratio >= 0.95, "TERPENUHI",
        np.where(ratio >= 0.75, "TERBATAS",
                 np.where(ratio >= 0.45, "HEDGING", "DITUNDA")))
    return df


# =============================================================================
# 10. FEWS - FLOOD EARLY WARNING & FORECAST-CONSTRAINED PRE-RELEASE
# =============================================================================
def calculate_available_flood_storage(storage_mcm, dam=DAM):
    """
    RUANG TAMPUNGAN BANJIR TERSEDIA (available flood storage):

        V_flood = S(FCL) - S(t)      [juta m3]

    Diukur terhadap Flood Control Level, BUKAN terhadap HWL, sebagai margin
    keamanan tambahan. Bernilai 0 apabila tampungan sudah di atas FCL.
    """
    return max(S_FCL - float(storage_mcm), 0.0)


def forecast_inflow_scenarios(base_inflow_cms, forecast_rain_mm, scenario="MOST LIKELY",
                              horizon_days=7, dam=DAM):
    """
    HIDROGRAF PRAKIRAAN INFLOW sederhana untuk DEMO.

    Respons hujan-aliran didekati dengan koefisien limpasan dan hidrograf
    tak berdimensi berpuncak pada hari ke-3.
    KALIBRASI: ganti dengan model hujan-aliran terkalibrasi (HEC-HMS, NRECA,
    Sacramento, Tank Model, atau prakiraan operasional BMKG/BBWS).
    """
    factor = {"LOW": 0.65, "MOST LIKELY": 1.00, "HIGH": 1.45}[scenario]
    shape = np.array([0.35, 0.70, 1.00, 0.85, 0.62, 0.45, 0.32,
                      0.24, 0.18, 0.14][:horizon_days], dtype=float)
    if len(shape) < horizon_days:
        shape = np.concatenate([shape, np.full(horizon_days - len(shape), 0.12)])
    rain_response = 0.055 * float(forecast_rain_mm) * factor   # m3/s per mm (DEMO)
    hydro = float(base_inflow_cms) * (0.9 + 0.1 * factor) + rain_response * shape
    return np.maximum(hydro, 0.0)


def calculate_minimum_prerelease(storage_mcm, forecast_hydrograph_cms,
                                 current_release_cms, horizon_days=7,
                                 safety_factor=1.15, dam=DAM):
    """
    MINIMUM NECESSARY PRE-RELEASE (bukan maximum release).

    Logika keputusan:
        V_masuk = SUM(Qin_forecast) * 86400 / 1e6              [juta m3]
        V_ruang = S(FCL) - S(t)                                [juta m3]

        JIKA  V_ruang >= V_masuk * FK   ->  PRE-RELEASE = 0  ("Retain Incoming Water")
        JIKA TIDAK                       ->  hitung volume minimum yang harus
                                             dilepas agar tersedia ruang aman.

    Batasan (constraint) yang wajib dipenuhi:
        - kapasitas outlet terkendali                (OUTLET_CAPACITY)
        - batas aman aliran sungai hilir             (DOWNSTREAM_SAFE_FLOW)
        - release yang sedang berjalan ikut diperhitungkan
        - faktor keamanan terhadap ketidakpastian prakiraan (safety_factor)

    Prinsip: air hanya dilepas sebanyak yang benar-benar diperlukan untuk
    menyediakan ruang aman, karena incoming water dibutuhkan untuk recovery.
    """
    hydro = np.asarray(forecast_hydrograph_cms, dtype=float)[:horizon_days]
    v_in = float(hydro.sum()) * 86400.0 / 1.0e6
    v_space = calculate_available_flood_storage(storage_mcm, dam)
    v_need = v_in * float(safety_factor)

    # Ruang tambahan yang sudah tercipta oleh release operasional berjalan
    v_by_current = cms_to_mcm_per_day(current_release_cms) * horizon_days
    deficit = v_need - (v_space + v_by_current)

    if deficit <= 0:
        return {
            "required": False,
            "decision": "PRE-RELEASE = 0",
            "message": "Retain Incoming Water",
            "predicted_inflow_volume_mcm": round(v_in, 3),
            "available_flood_storage_mcm": round(v_space, 3),
            "required_volume_mcm": round(v_need, 3),
            "prerelease_volume_mcm": 0.0,
            "prerelease_q_cms": 0.0,
            "duration_days": 0,
            "limited_by": "-",
            "feasible": True,
            "shortfall_mcm": 0.0,
        }

    # Durasi pre-release: waktu sebelum puncak hidrograf (minimal 1 hari)
    peak_day = int(np.argmax(hydro)) + 1
    duration = max(1, min(peak_day, horizon_days))
    q_need = mcm_per_day_to_cms(deficit / duration)

    # Batas hidraulik: kapasitas outlet dan aliran aman hilir
    q_outlet_room = max(dam["OUTLET_CAPACITY"] - float(current_release_cms), 0.0)
    q_ds_room = max(dam["DOWNSTREAM_SAFE_FLOW"] - float(current_release_cms), 0.0)
    q_max = min(q_outlet_room, q_ds_room)
    q_final = min(q_need, q_max)

    if q_final >= q_need - 1e-6:
        limited_by = "Kebutuhan ruang tampungan (tidak dibatasi hidraulik)"
        feasible = True
    elif q_max == q_outlet_room:
        limited_by = f"Kapasitas outlet ({dam['OUTLET_CAPACITY']:.1f} m3/s)"
        feasible = False
    else:
        limited_by = f"Batas aman aliran hilir ({dam['DOWNSTREAM_SAFE_FLOW']:.1f} m3/s)"
        feasible = False

    v_final = cms_to_mcm_per_day(q_final) * duration
    return {
        "required": True,
        "decision": "MINIMUM NECESSARY PRE-RELEASE",
        "message": f"Lepas {q_final:.2f} m3/detik selama {duration} hari",
        "predicted_inflow_volume_mcm": round(v_in, 3),
        "available_flood_storage_mcm": round(v_space, 3),
        "required_volume_mcm": round(v_need, 3),
        "prerelease_volume_mcm": round(v_final, 3),
        "prerelease_q_cms": round(q_final, 3),
        "duration_days": duration,
        "limited_by": limited_by,
        "feasible": feasible,
        "shortfall_mcm": round(max(deficit - v_final, 0.0), 3),
    }


def route_forecast_storage(storage_mcm, hydrograph_cms, release_cms,
                           rainfall_mm_series=None, dam=DAM):
    """
    RESERVOIR ROUTING sederhana (level pool routing) untuk periode prakiraan.
    Menghitung lintasan tampungan & elevasi, serta limpasan pelimpah bila
    muka air melampaui NWL.
    """
    hydro = np.asarray(hydrograph_cms, dtype=float)
    n = len(hydro)
    rains = (np.zeros(n) if rainfall_mm_series is None
             else np.asarray(rainfall_mm_series, dtype=float)[:n])
    s = float(storage_mcm)
    storages, elevs, spills, outs = [], [], [], []
    for i in range(n):
        spill = 0.0
        if s > S_NWL:
            # Limpasan pendekatan pelimpah bebas: Q = C*L*H^1.5 disederhanakan
            head = storage_to_elevation(s, dam) - dam["NWL"]
            spill = min(38.0 * (max(head, 0.0) ** 1.5), dam["SPILLWAY_CAPACITY"])
        total_out = float(release_cms) + spill
        wb = calculate_water_balance(s, hydro[i], total_out,
                                     rains[i] if i < len(rains) else 0.0,
                                     dam["EVAP_MEAN_MM"], 1.0, dam)
        s = wb["storage_new_mcm"]
        storages.append(s)
        elevs.append(wb["elevation_new_m"])
        spills.append(spill)
        outs.append(total_out)
    return {
        "storage_mcm": np.array(storages),
        "elevation_m": np.array(elevs),
        "spill_cms": np.array(spills),
        "outflow_cms": np.array(outs),
        "max_elevation_m": float(np.max(elevs)) if elevs else storage_to_elevation(storage_mcm),
        "max_storage_mcm": float(np.max(storages)) if storages else storage_mcm,
        "max_spill_cms": float(np.max(spills)) if spills else 0.0,
    }


def calculate_flood_safety_margin(projected_max_elev_m, dam=DAM):
    """
    FLOOD SAFETY MARGIN = HWL - elevasi maksimum hasil routing prakiraan [m].
    Nilai positif besar = aman. Keselamatan adalah HARD CONSTRAINT.
    """
    margin = dam["HWL"] - float(projected_max_elev_m)
    if margin >= 1.0:
        status = "SAFE"
    elif margin >= 0.3:
        status = "WARNING"
    else:
        status = "CRITICAL"
    return margin, status


def calculate_fews_status(forecast_rain_mm, forecast_inflow_cms,
                          available_flood_storage_mcm, safety_margin_m, dam=DAM):
    """
    Status FEWS: NORMAL / WASPADA / SIAGA / DARURAT.
    Ditentukan dari intensitas prakiraan hujan-inflow dan margin keselamatan.
    KALIBRASI: ambang hujan/inflow harus mengikuti analisis frekuensi hujan
    dan debit banjir rancangan DAS setempat.
    """
    level = 0
    reasons = []
    if forecast_rain_mm >= 120:
        level = max(level, 3); reasons.append(f"Prakiraan hujan sangat tinggi ({forecast_rain_mm:.0f} mm)")
    elif forecast_rain_mm >= 70:
        level = max(level, 2); reasons.append(f"Prakiraan hujan tinggi ({forecast_rain_mm:.0f} mm)")
    elif forecast_rain_mm >= 35:
        level = max(level, 1); reasons.append(f"Prakiraan hujan sedang ({forecast_rain_mm:.0f} mm)")

    ratio_q = forecast_inflow_cms / max(dam["MEAN_ANNUAL_INFLOW"], 0.01)
    if ratio_q >= 4.0:
        level = max(level, 3); reasons.append(f"Prakiraan inflow {ratio_q:.1f}x rata-rata")
    elif ratio_q >= 2.5:
        level = max(level, 2); reasons.append(f"Prakiraan inflow {ratio_q:.1f}x rata-rata")
    elif ratio_q >= 1.6:
        level = max(level, 1); reasons.append(f"Prakiraan inflow {ratio_q:.1f}x rata-rata")

    if safety_margin_m < 0.3:
        level = 3; reasons.append(f"Flood safety margin kritis ({safety_margin_m:.2f} m)")
    elif safety_margin_m < 1.0 and level < 2:
        level = 2; reasons.append(f"Flood safety margin menipis ({safety_margin_m:.2f} m)")

    if available_flood_storage_mcm <= 0.5 and level < 2:
        level = 2; reasons.append("Ruang tampungan banjir hampir habis")

    if not reasons:
        reasons.append("Prakiraan hujan/inflow dalam batas normal")
    return ["NORMAL", "WASPADA", "SIAGA", "DARURAT"][level], level, reasons


# =============================================================================
# 11. RESERVOIR RECOVERY MODE
# =============================================================================
def calculate_storage_recovery_ratio(current_storage_mcm, target_pct=None,
                                     use_min_reference=True, dam=DAM):
    """
    STORAGE RECOVERY RATIO (SRR).

    Bentuk sederhana:
        SRR = S(t) / S_target

    Bentuk dengan acuan tampungan minimum (dipakai di sini, lebih tepat karena
    tampungan mati tidak dapat dimanfaatkan):

        SRR = (S(t) - S_LWL) / (S_target - S_LWL)

    SRR = 1,0 berarti target tampungan musiman telah tercapai.
    """
    tgt_pct = dam["TARGET_SEASONAL_PCT"] if target_pct is None else float(target_pct)
    s_target = percentage_to_storage(tgt_pct)
    if use_min_reference:
        denom = max(s_target - S_LWL, 1e-6)
        srr = (float(current_storage_mcm) - S_LWL) / denom
    else:
        srr = float(current_storage_mcm) / max(s_target, 1e-6)
    return float(np.clip(srr, 0.0, 3.0)), s_target


def estimate_recovery_time(current_storage_mcm, target_storage_mcm,
                           expected_inflow_cms, total_release_cms,
                           rainfall_mm_day=0.0, dam=DAM):
    """
    ESTIMASI WAKTU PEMULIHAN TAMPUNGAN.

        Laju pengisian bersih  dS/dt = Qin - Qout + P*A - E*A - L   [juta m3/hari]
        Waktu pemulihan        t = (S_target - S_sekarang) / (dS/dt)

    Mengembalikan (hari, laju_bersih_mcm_per_hari, keterangan).
    Bernilai None bila laju bersih <= 0 (target tidak akan tercapai).
    """
    wb = calculate_water_balance(current_storage_mcm, expected_inflow_cms,
                                 total_release_cms, rainfall_mm_day,
                                 dam["EVAP_MEAN_MM"], 1.0, dam)
    rate = wb["delta_storage_mcm"]
    gap = float(target_storage_mcm) - float(current_storage_mcm)
    if gap <= 0:
        return 0.0, rate, "Target tampungan sudah tercapai"
    if rate <= 1e-6:
        return None, rate, "Laju pengisian bersih <= 0: target tidak tercapai pada kondisi ini"
    days = gap / rate
    return float(days), rate, f"Perkiraan {days:.0f} hari pada laju {rate:.4f} juta m3/hari"


def recovery_trajectory(current_storage_mcm, target_storage_mcm,
                        expected_inflow_cms, release_cms, days=60,
                        rainfall_mm_day=0.0, dam=DAM):
    """Lintasan pemulihan tampungan (forecast recovery trajectory)."""
    s = float(current_storage_mcm)
    traj = [s]
    for _ in range(int(days)):
        wb = calculate_water_balance(s, expected_inflow_cms, release_cms,
                                     rainfall_mm_day, dam["EVAP_MEAN_MM"], 1.0, dam)
        s = min(wb["storage_new_mcm"], S_NWL)  # dibatasi NWL (kelebihan dilimpaskan)
        traj.append(s)
    return np.array(traj)


# =============================================================================
# 12. VALIDASI DATA & FAIL-SAFE MODE
# =============================================================================
def validate_data(df, latest, dam=DAM):
    """
    VALIDASI DATA (data quality assurance).

    Memeriksa:
      - missing value (data hilang)
      - debit negatif (inflow/outflow < 0)
      - storage < 0 atau storage > kapasitas maksimum (di atas HWL)
      - elevasi muka air di luar rentang fisik (dasar waduk - puncak bendungan)
      - masalah stempel waktu (duplikat / tidak urut)
      - data basi (stale data): pembaruan terakhir terlalu lama
      - prakiraan tidak tersedia

    Mengembalikan dict: status (OK/WARNING/CRITICAL), daftar isu, kelengkapan
    data (%), dan bendera fail-safe.
    """
    issues = []
    severity = 0   # 0 = OK, 1 = WARNING, 2 = CRITICAL

    cols_needed = ["datetime", "storage", "reservoir_level", "inflow", "outflow", "rainfall"]
    for c in cols_needed:
        if c not in df.columns:
            issues.append(("CRITICAL", f"Kolom wajib '{c}' tidak ditemukan"))
            severity = 2

    # 1) Missing values
    n_cells = int(df.shape[0] * df.shape[1]) if df.size else 1
    n_missing = int(df.isna().sum().sum())
    completeness = 100.0 * (1.0 - n_missing / max(n_cells, 1))
    if n_missing > 0:
        lvl = "CRITICAL" if completeness < 90 else "WARNING"
        issues.append((lvl, f"Terdapat {n_missing} nilai kosong "
                            f"(kelengkapan data {completeness:.1f}%)"))
        severity = max(severity, 2 if lvl == "CRITICAL" else 1)

    # 2) Debit negatif
    for c in ["inflow", "outflow", "rainfall", "storage"]:
        if c in df.columns and (pd.to_numeric(df[c], errors="coerce") < 0).any():
            issues.append(("CRITICAL", f"Nilai negatif tidak wajar pada kolom '{c}'"))
            severity = 2

    # 3) Batas tampungan
    if "storage" in df.columns:
        s = pd.to_numeric(df["storage"], errors="coerce")
        if (s > S_HWL * 1.05).any():
            issues.append(("CRITICAL", "Tampungan melampaui kapasitas maksimum (di atas HWL)"))
            severity = 2
        if (s < 0).any():
            issues.append(("CRITICAL", "Tampungan bernilai negatif"))
            severity = 2

    # 4) Elevasi muka air
    if "reservoir_level" in df.columns:
        h = pd.to_numeric(df["reservoir_level"], errors="coerce")
        if (h < dam["EL_BOTTOM"]).any() or (h > dam["CREST"]).any():
            issues.append(("WARNING", "Elevasi muka air di luar rentang fisik waduk"))
            severity = max(severity, 1)

    # 5) Stempel waktu
    if "datetime" in df.columns:
        tt = pd.to_datetime(df["datetime"], errors="coerce")
        if tt.isna().any():
            issues.append(("CRITICAL", "Terdapat stempel waktu tidak valid"))
            severity = 2
        elif tt.duplicated().any():
            issues.append(("WARNING", "Terdapat stempel waktu duplikat"))
            severity = max(severity, 1)
        elif not tt.is_monotonic_increasing:
            issues.append(("WARNING", "Stempel waktu tidak berurutan"))
            severity = max(severity, 1)
        else:
            # 6) Data basi (stale)
            age_h = (pd.Timestamp.now() - tt.iloc[-1]).total_seconds() / 3600.0
            if age_h > 72:
                issues.append(("CRITICAL", f"Data terakhir berumur {age_h:.0f} jam (basi)"))
                severity = 2
            elif age_h > 24:
                issues.append(("WARNING", f"Data terakhir berumur {age_h:.0f} jam"))
                severity = max(severity, 1)

    # 7) Ketersediaan prakiraan
    if latest is not None:
        fr = latest.get("forecast_rainfall", None)
        fi = latest.get("forecast_inflow", None)
        if fr is None or fi is None or (isinstance(fr, float) and math.isnan(fr)) \
                or (isinstance(fi, float) and math.isnan(fi)):
            issues.append(("WARNING", "Data prakiraan (forecast) tidak tersedia"))
            severity = max(severity, 1)

    # 8) Data tidak logis: inflow ekstrem
    if "inflow" in df.columns:
        q = pd.to_numeric(df["inflow"], errors="coerce")
        if (q > dam["SPILLWAY_CAPACITY"] * 1.5).any():
            issues.append(("WARNING", "Terdapat inflow ekstrem melebihi kapasitas pelimpah desain"))
            severity = max(severity, 1)

    status = ["OK", "WARNING", "CRITICAL"][severity]
    return {
        "status": status,
        "quality": {"OK": "GOOD", "WARNING": "WARNING", "CRITICAL": "CRITICAL"}[status],
        "issues": issues,
        "completeness": round(completeness, 2),
        "failsafe": severity >= 2,
    }


# =============================================================================
# 13. LOGIKA OPERATING MODE
# =============================================================================
# Toleransi pencapaian target: SRR >= 0,98 sudah dianggap mencapai target musiman
# agar mode operasi tidak berpindah-pindah hanya karena selisih beberapa persen.
# KALIBRASI: sesuaikan toleransi ini dengan ketelitian pengukuran elevasi waduk.
SRR_TARGET_TOLERANCE = 0.98


def determine_operating_mode(dews_status, fews_status, prerelease_required,
                             srr, failsafe_active, safety_status):
    """
    PENENTUAN MODE OPERASI (logika transparan berbasis aturan, bukan black box).

    Urutan prioritas:
      1. FAIL-SAFE MODE       : data tidak valid -> kembali ke baseline rule
      2. FLOOD ANTICIPATION   : keselamatan bendungan (hard constraint)
      3. EMERGENCY ALLOCATION : kekeringan darurat
      4. DROUGHT HEDGING      : kekeringan siaga
      5. RECOVERY MODE        : kekeringan mereda tetapi target belum tercapai
      6. DROUGHT CONSERVATION : kekeringan waspada
      7. NORMAL OPERATION     : seluruh kondisi normal
    """
    reasons = []
    if failsafe_active:
        return "FAIL-SAFE MODE", ["Data tidak valid / sensor bermasalah -> "
                                  "kembali ke Approved Baseline Operating Rule"]

    if safety_status == "CRITICAL" or fews_status == "DARURAT" or \
            (prerelease_required and fews_status in ("SIAGA", "DARURAT")):
        reasons.append(f"FEWS = {fews_status}; keselamatan bendungan menjadi hard constraint")
        if prerelease_required:
            reasons.append("Minimum necessary pre-release diperlukan")
        return "FLOOD ANTICIPATION", reasons

    if dews_status == "DARURAT":
        return "EMERGENCY ALLOCATION", ["DEWS = DARURAT: hanya kebutuhan esensial dilayani"]

    if dews_status == "SIAGA":
        return "DROUGHT HEDGING", ["DEWS = SIAGA: adaptive drought hedging diterapkan"]

    if srr < SRR_TARGET_TOLERANCE and dews_status in ("NORMAL", "WASPADA"):
        reasons.append(f"Storage Recovery Ratio {srr:.2f} < "
                       f"{SRR_TARGET_TOLERANCE:.2f} (target musiman belum tercapai)")
        reasons.append("Incoming water dipertahankan untuk pemulihan tampungan")
        return "RECOVERY MODE", reasons

    if dews_status == "WASPADA":
        return "DROUGHT CONSERVATION", ["DEWS = WASPADA: konservasi dini dimulai"]

    return "NORMAL OPERATION", ["Seluruh indikator dalam batas normal"]


def active_stage(mode):
    """Menentukan tahapan filosofi yang sedang aktif untuk diagram highlight."""
    return {
        "NORMAL OPERATION": "RECOVER",
        "DROUGHT CONSERVATION": "CONSERVE",
        "DROUGHT HEDGING": "ALLOCATE",
        "EMERGENCY ALLOCATION": "ALLOCATE",
        "FLOOD ANTICIPATION": "ANTICIPATE",
        "RECOVERY MODE": "RECOVER",
        "FAIL-SAFE MODE": "CONSERVE",
    }.get(mode, "CONSERVE")


# =============================================================================
# 14. MULTI-PURPOSE ALLOCATION (IWRM)
# =============================================================================
def multipurpose_allocation(available_q_cms, hedging, irrigation_demand_cms,
                            domestic_demand_cms, env_flow_cms, dam=DAM):
    """
    ALOKASI AIR MULTI-PURPOSE BERBASIS IWRM.

    Urutan prioritas pemenuhan (sesuai kelaziman operasi waduk multipurpose):
        1. Air baku / domestik      (prioritas tertinggi, selalu MAINTAIN)
        2. Environmental flow       (kebutuhan lingkungan hilir)
        3. Irigasi                  (setelah hedging faktor diterapkan)
        4. PLTA                     (non-konsumtif jika hidraulik memungkinkan)
        5. Perikanan & pariwisata   (kebutuhan pelengkap)

    CATATAN COORDINATED MULTI-PURPOSE RELEASE:
    Air yang melewati turbin PLTA umumnya TIDAK hilang; air tersebut tetap
    mengalir ke hilir sehingga dapat sekaligus memenuhi environmental flow
    dan/atau kebutuhan lain apabila konfigurasi hidrauliknya memungkinkan.
    Karena itu debit PLTA TIDAK dijumlahkan sebagai kebutuhan konsumtif
    terhadap neraca air; yang dihitung sebagai pelepasan bersih adalah
    max(kebutuhan konsumtif, debit turbin).
    """
    sectors = []
    remaining = max(float(available_q_cms), 0.0)

    def take(name, demand, factor, priority, consumptive=True, note=""):
        nonlocal remaining
        req = max(float(demand) * float(factor), 0.0)
        give = min(req, remaining) if consumptive else req
        if consumptive:
            remaining -= give
        sectors.append({
            "Sektor": name,
            "Kebutuhan (m3/s)": round(float(demand), 3),
            "Faktor Hedging": round(float(factor), 3),
            "Kebutuhan Terkoreksi (m3/s)": round(req, 3),
            "Alokasi (m3/s)": round(give, 3),
            "Defisit (m3/s)": round(max(req - give, 0.0), 3),
            "Pemenuhan (%)": round(100.0 * give / req, 1) if req > 1e-9 else 100.0,
            "Prioritas": priority,
            "Konsumtif": "Ya" if consumptive else "Tidak",
            "Keterangan": note,
        })
        return give

    q_dom = take("Air Baku / Domestik", domestic_demand_cms,
                 hedging["f_domestic"], "1 - TERTINGGI", True,
                 "Selalu dipertahankan (MAINTAIN)")
    q_env = take("Environmental Flow", env_flow_cms,
                 hedging["f_environment"], "2 - TINGGI", True,
                 "Kebutuhan lingkungan sungai hilir")
    q_irr = take("Irigasi", irrigation_demand_cms,
                 hedging["f_irrigation"], "3 - TINGGI", True,
                 "Alokasi berbasis sensitivitas tanaman")
    q_fish = take("Perikanan", dam["Q_FISHERIES"],
                  hedging["f_other"], "4 - SEDANG", True, "Keramba / kolam hilir")
    q_tour = take("Pariwisata", dam["Q_TOURISM"],
                  hedging["f_other"], "5 - RENDAH", True, "Kebutuhan pelengkap")

    # PLTA: non-konsumtif, dibatasi air yang benar-benar tersedia
    q_hydro_target = dam["Q_HYDRO_DESIGN"] * hedging["f_hydropower"]
    q_consumptive = q_dom + q_env + q_irr + q_fish + q_tour
    q_hydro = min(q_hydro_target, max(float(available_q_cms), 0.0))
    sectors.append({
        "Sektor": "PLTA (Hidropower)",
        "Kebutuhan (m3/s)": round(dam["Q_HYDRO_DESIGN"], 3),
        "Faktor Hedging": round(hedging["f_hydropower"], 3),
        "Kebutuhan Terkoreksi (m3/s)": round(q_hydro_target, 3),
        "Alokasi (m3/s)": round(q_hydro, 3),
        "Defisit (m3/s)": round(max(q_hydro_target - q_hydro, 0.0), 3),
        "Pemenuhan (%)": round(100.0 * q_hydro / q_hydro_target, 1) if q_hydro_target > 1e-9 else 100.0,
        "Prioritas": "4 - SEDANG",
        "Konsumtif": "Tidak",
        "Keterangan": "Coordinated release: air kembali ke hilir",
    })

    # Pelepasan bersih waduk (menghindari perhitungan ganda debit turbin)
    q_release_total = max(q_consumptive, q_hydro)
    power_mw = min(
        9.81 * q_hydro * dam["HYDRO_NET_HEAD"] * dam["HYDRO_EFFICIENCY"] / 1000.0,
        dam["HYDRO_CAPACITY_MW"],
    )
    df = pd.DataFrame(sectors)
    return {
        "table": df,
        "q_domestic": q_dom, "q_env": q_env, "q_irrigation": q_irr,
        "q_fisheries": q_fish, "q_tourism": q_tour, "q_hydro": q_hydro,
        "q_consumptive": q_consumptive,
        "q_release_total": q_release_total,
        "power_mw": power_mw,
        "unserved": max(remaining, 0.0),
    }


# =============================================================================
# 15. GENERATOR REKOMENDASI JALA RAKSA
# =============================================================================
def generate_jala_raksa_recommendation(ctx):
    """
    Menyusun rekomendasi operasi JALA RAKSA dalam bentuk terstruktur.
    KELUARAN INI ADALAH REKOMENDASI, BUKAN PERINTAH OPERASI.
    Keputusan akhir tetap berada pada operator (APPROVE / MODIFY / REJECT).
    """
    mode = ctx["mode"]
    dews = ctx["dews"]
    fews = ctx["fews"]
    hedg = ctx["hedging"]
    pre = ctx["prerelease"]
    srr = ctx["srr"]

    lines = []
    if mode == "FAIL-SAFE MODE":
        lines = [
            "Data masukan tidak memenuhi syarat validasi.",
            "Hentikan penggunaan keluaran optimasi JALA RAKSA untuk sementara.",
            "Kembali ke Approved Baseline Operating Rule.",
            "Lakukan pemeriksaan sensor, komunikasi telemetri, dan basis data.",
        ]
    elif mode == "FLOOD ANTICIPATION":
        lines = [
            f"FEWS berstatus {fews}; antisipasi inflow tinggi diaktifkan.",
            (f"Pre-release minimum yang diperlukan: {pre['prerelease_q_cms']:.2f} m3/detik "
             f"selama {pre['duration_days']} hari."
             if pre["required"] else
             "Ruang tampungan masih memadai: PRE-RELEASE = 0 (Retain Incoming Water)."),
            "Pelepasan dibatasi kapasitas outlet dan batas aman aliran hilir.",
            "Air baku dipertahankan; pantau elevasi maksimum hasil routing.",
        ]
    elif mode == "EMERGENCY ALLOCATION":
        lines = [
            "Kondisi kekeringan DARURAT: berlakukan emergency allocation.",
            "Air baku dipertahankan penuh (MAINTAIN).",
            f"Irigasi dikurangi hingga {hedg['irrigation_reduction_pct']:.0f}% "
            f"dan diprioritaskan pada zona berstatus CRITICAL.",
            "Release non-esensial (pariwisata/perikanan) dihentikan sementara.",
            "Pre-release tidak diperlukan.",
        ]
    elif mode == "DROUGHT HEDGING":
        lines = [
            "Adaptive drought hedging diterapkan.",
            "Air baku dipertahankan (MAINTAIN); environmental flow dijaga.",
            f"Pengurangan irigasi {hedg['irrigation_reduction_pct']:.0f}% "
            f"secara selektif berdasarkan sensitivitas tanaman.",
            "Zona dengan tanah kering dan tanaman pada fase sensitif diprioritaskan.",
            "Pre-release tidak diperlukan.",
        ]
    elif mode == "RECOVERY MODE":
        lines = [
            "Kondisi kekeringan mereda namun target tampungan musiman belum tercapai.",
            f"Storage Recovery Ratio = {srr:.2f} (< 1,00).",
            "Pertahankan incoming water sebanyak mungkin untuk pengisian tampungan.",
            "Air baku dipertahankan; irigasi dilayani secara terkendali.",
            "Release non-esensial dibatasi; FEWS tetap aktif.",
        ]
    elif mode == "DROUGHT CONSERVATION":
        lines = [
            "Konservasi dini direkomendasikan (early conservation).",
            "Pertahankan pelayanan air baku sepenuhnya.",
            f"Terapkan penghematan irigasi selektif {hedg['irrigation_reduction_pct']:.0f}%.",
            "Pre-release tidak diperlukan.",
        ]
    else:
        lines = [
            "Operasi normal sesuai rule curve baseline.",
            "Seluruh kebutuhan multi-purpose dapat dilayani.",
            "Tidak diperlukan hedging maupun pre-release.",
            "Lanjutkan pemantauan rutin DEWS dan FEWS.",
        ]

    return {
        "mode": mode,
        "dews": dews,
        "fews": fews,
        "stress": ctx["stress"],
        "stress_class": classify_stress(ctx["stress"]),
        "hedging_level": hedg["level_name"],
        "domestic": "MAINTAIN",
        "irrigation": ("NORMAL" if hedg["f_irrigation"] >= 0.999
                       else f"HEDGING {hedg['irrigation_reduction_pct']:.0f}%"),
        "environment": hedg["env_policy"],
        "prerelease": ("NOT REQUIRED" if not pre["required"]
                       else f"{pre['prerelease_q_cms']:.2f} m3/s x {pre['duration_days']} hari"),
        "recovery": "ACTIVE" if mode == "RECOVERY MODE" else
                    ("NOT ACTIVE" if srr >= SRR_TARGET_TOLERANCE else "MONITORING"),
        "priority_zones": ctx.get("priority_zones", []),
        "narrative": lines,
        "text": " ".join(lines),
    }


# =============================================================================
# 16. PEMBANGKIT DATA DEMONSTRASI (SYNTHETIC / DEMO DATASET)
# =============================================================================
# Data di bawah adalah DEMONSTRATION DATA yang dibangkitkan Python.
# Pola yang disimulasikan (sesuai skenario uji JALA RAKSA):
#   normal -> kekeringan -> kekeringan berat -> hujan/inflow naik
#          -> recovery -> normal
# =============================================================================
@st.cache_data(show_spinner=False)
def generate_demo_data(n_days=90, seed=2026, dam=DAM):
    """Membangkitkan deret waktu demonstrasi lengkap selama n_days hari."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now().normalize()
    dates = pd.date_range(end=end, periods=n_days, freq="D")

    # ---- Fase hidrologi (fraksi panjang periode) ----
    f = np.linspace(0.0, 1.0, n_days)
    inflow = np.zeros(n_days)
    rain = np.zeros(n_days)
    phase = []
    for i, t in enumerate(f):
        if t < 0.16:            # normal
            q, p, ph = 6.2, 7.0, "NORMAL"
        elif t < 0.36:          # kekeringan mulai
            q, p, ph = 6.2 - 20.0 * (t - 0.16), 7.0 - 30.0 * (t - 0.16), "KEKERINGAN"
        elif t < 0.56:          # kekeringan berat
            q, p, ph = 2.0 - 4.0 * (t - 0.36), max(1.0 - 4.0 * (t - 0.36), 0.1), "KEKERINGAN BERAT"
        elif t < 0.68:          # hujan mulai datang
            q, p, ph = 1.2 + 62.0 * (t - 0.56), 0.5 + 200.0 * (t - 0.56), "INFLOW TINGGI"
        elif t < 0.88:          # pemulihan
            q, p, ph = 8.6 - 12.0 * (t - 0.68), 24.0 - 80.0 * (t - 0.68), "RECOVERY"
        else:                   # kembali normal
            q, p, ph = 6.2, 8.0, "NORMAL"
        inflow[i] = max(q, 0.25)
        rain[i] = max(p, 0.0)
        phase.append(ph)

    inflow = inflow * rng.normal(1.0, 0.09, n_days)
    inflow = np.maximum(inflow, 0.2)
    rain = np.maximum(rain * rng.gamma(2.0, 0.5, n_days), 0.0)
    evap = np.clip(rng.normal(dam["EVAP_MEAN_MM"], 0.8, n_days), 1.5, 8.5)

    # ---- Kebutuhan air ----
    # Air baku adalah kebutuhan MINIMUM desain (76 l/detik): fluktuasi hanya boleh
    # ke atas, tidak pernah turun di bawah nilai minimum tersebut.
    dom = dam["Q_DOMESTIC"] * (1.0 + np.abs(rng.normal(0.0, 0.03, n_days)))
    season = 0.72 + 0.28 * np.sin(np.linspace(0, 2.4 * np.pi, n_days))
    irr_dem = dam["Q_IRRIGATION_MAX"] * np.clip(season, 0.35, 1.0)
    env = np.full(n_days, dam["Q_ENV_FLOW"])

    # ---- Simulasi neraca air harian ----
    storage = np.zeros(n_days)
    level = np.zeros(n_days)
    outflow = np.zeros(n_days)
    hydro_rel = np.zeros(n_days)
    s = percentage_to_storage(78.0)     # kondisi awal 78% tampungan efektif

    for i in range(n_days):
        pct = calculate_storage_percentage(s)
        # Aturan operasi sederhana ala baseline untuk membangkitkan data demo
        if pct > 65:
            f_irr = 1.00
        elif pct > 50:
            f_irr = 0.85
        elif pct > 35:
            f_irr = 0.65
        else:
            f_irr = 0.42
        q_irr = irr_dem[i] * f_irr
        q_hyd = min(dam["Q_HYDRO_DESIGN"] * (0.85 if pct > 55 else 0.45),
                    max(pct, 0.0) / 100.0 * dam["Q_HYDRO_DESIGN"] * 1.2)
        q_cons = dom[i] + env[i] + q_irr + dam["Q_FISHERIES"] + dam["Q_TOURISM"]
        q_out = max(q_cons, q_hyd)

        # Limpasan bila melebihi NWL
        spill = 0.0
        if s > S_NWL:
            head = storage_to_elevation(s) - dam["NWL"]
            spill = min(38.0 * (max(head, 0.0) ** 1.5), dam["SPILLWAY_CAPACITY"])
        q_out_total = q_out + spill

        wb = calculate_water_balance(s, inflow[i], q_out_total, rain[i], evap[i], 1.0, dam)
        s = min(max(wb["storage_new_mcm"], S_LWL * 0.85), S_HWL)
        storage[i] = s
        level[i] = storage_to_elevation(s)
        outflow[i] = q_out_total
        hydro_rel[i] = q_hyd

    storage_pct = np.array([calculate_storage_percentage(v) for v in storage])

    # ---- Prakiraan (forecast) 7 hari ke depan, disederhanakan ----
    fc_rain = np.zeros(n_days)
    fc_inflow = np.zeros(n_days)
    for i in range(n_days):
        j = min(i + 3, n_days - 1)
        fc_rain[i] = float(np.mean(rain[i:j + 1])) * 7.0 * rng.normal(1.0, 0.12)
        fc_inflow[i] = float(np.mean(inflow[i:j + 1])) * rng.normal(1.05, 0.10)
    fc_rain = np.maximum(fc_rain, 0.0)
    fc_inflow = np.maximum(fc_inflow, 0.15)

    # ---- Kelembapan tanah rata-rata daerah irigasi ----
    sm = np.zeros(n_days)
    val = 68.0
    for i in range(n_days):
        val += 0.55 * rain[i] - 0.9 - 0.05 * evap[i]
        val = float(np.clip(val, 18.0, 95.0))
        sm[i] = val

    df = pd.DataFrame({
        "datetime": dates,
        "reservoir_level": np.round(level, 3),
        "storage": np.round(storage, 4),
        "storage_percent": np.round(storage_pct, 2),
        "inflow": np.round(inflow, 3),
        "outflow": np.round(outflow, 3),
        "rainfall": np.round(rain, 2),
        "forecast_rainfall": np.round(fc_rain, 2),
        "forecast_inflow": np.round(fc_inflow, 3),
        "evaporation": np.round(evap, 2),
        "domestic_demand": np.round(dom, 4),
        "irrigation_demand": np.round(irr_dem, 3),
        "environmental_flow": np.round(env, 3),
        "hydropower_release": np.round(hydro_rel, 3),
        "soil_moisture": np.round(sm, 1),
        "phase": phase,
    })

    # ---- Status DEWS & FEWS historis (untuk grafik riwayat) ----
    dews_hist, fews_hist = [], []
    for i in range(n_days):
        trend = 0.0 if i == 0 else float(storage_pct[i] - storage_pct[i - 1])
        dsr = ((dom[i] + irr_dem[i] + env[i]) /
               max(inflow[i] + max(storage_pct[i], 0) / 100.0 * 1.6, 0.05))
        fidx = float(np.clip(fc_rain[i] / 60.0 * 50.0 + fc_inflow[i] /
                             dam["MEAN_ANNUAL_INFLOW"] * 25.0, 0, 100))
        stress = calculate_reservoir_stress(
            storage_pct[i], inflow[i] / dam["MEAN_ANNUAL_INFLOW"] * 100.0, fidx,
            float(np.clip(dsr, 0, 4)))
        d, _, _ = calculate_dews_status(storage_pct[i], stress, trend, fidx,
                                        float(np.clip(dsr, 0, 4)))
        margin, _ = calculate_flood_safety_margin(level[i] + 0.35)
        fw, _, _ = calculate_fews_status(fc_rain[i], fc_inflow[i],
                                         calculate_available_flood_storage(storage[i]),
                                         margin)
        dews_hist.append(d)
        fews_hist.append(fw)
    df["DEWS"] = dews_hist
    df["FEWS"] = fews_hist
    return df


@st.cache_data(show_spinner=False)
def generate_irrigation_zones(seed=7, storage_pct=70.0, dam=DAM):
    """
    Membangkitkan data 5 zona pengelolaan irigasi (irrigation management zones).
    Total luas = 3.500 ha sesuai dokumen konsep JALA RAKSA.
    """
    rng = np.random.default_rng(seed)
    zones = [
        {"Zone": "Zone A", "Area (ha)": 820, "Crop": "Padi", "Stage": "Pematangan", "SM": 74.0},
        {"Zone": "Zone B", "Area (ha)": 760, "Crop": "Jagung", "Stage": "Vegetatif", "SM": 58.0},
        {"Zone": "Zone C", "Area (ha)": 900, "Crop": "Padi", "Stage": "Pembungaan", "SM": 34.0},
        {"Zone": "Zone D", "Area (ha)": 640, "Crop": "Kedelai", "Stage": "Pengisian Biji", "SM": 41.0},
        {"Zone": "Zone E", "Area (ha)": 380, "Crop": "Bawang", "Stage": "Inisiasi", "SM": 66.0},
    ]
    for z in zones:
        z["SM"] = float(np.clip(z["SM"] + rng.normal(0, 2.5), 12, 95))
    return pd.DataFrame(zones)


def build_zone_table(zones_base, et0_mm, eff_rain_mm, dam=DAM):
    """Menghitung ETc, kebutuhan irigasi, defisit, dan skor prioritas tiap zona."""
    rows = []
    for _, z in zones_base.iterrows():
        etc, kc = calculate_etc(et0_mm, z["Crop"], z["Stage"])
        req = calculate_irrigation_requirement(etc, eff_rain_mm, z["SM"], z["Area (ha)"])
        deficit_ratio = float(np.clip((etc - eff_rain_mm) / max(etc, 1e-6), 0, 1))
        score, cls = calculate_crop_priority(z["SM"], z["Crop"], z["Stage"], deficit_ratio)
        rows.append({
            "Zone": z["Zone"],
            "Area (ha)": float(z["Area (ha)"]),
            "Crop": z["Crop"],
            "Crop Stage": z["Stage"],
            "Kc": round(kc, 2),
            "Soil Moisture (%)": round(float(z["SM"]), 1),
            "ETc (mm/hari)": round(etc, 2),
            "Effective Rainfall (mm)": round(float(eff_rain_mm), 2),
            "NIR (mm/hari)": round(req["NIR_mm"], 2),
            "GIR (mm/hari)": round(req["GIR_mm"], 2),
            "Irrigation Requirement (m3/s)": round(req["Q_cms"], 4),
            "Crop Sensitivity (Ky)": CROP_LIBRARY[z["Crop"]]["Ky"],
            "Priority Score": round(score, 1),
            "Priority Class": cls,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 17. IoT & SENSOR HEALTH
# =============================================================================
@st.cache_data(show_spinner=False)
def generate_sensor_data(seed, latest_dict, failsafe_demo=False):
    """
    Membangkitkan status jaringan sensor IoT (DEMONSTRATION DATA).
    IoT diposisikan sebagai enabling technology untuk akuisisi data.
    """
    rng = np.random.default_rng(seed)
    now = pd.Timestamp.now()
    sensors = [
        ("ARG-01", "Automatic Rain Gauge", "Hulu DAS - Stasiun 1", "Curah Hujan",
         latest_dict.get("rainfall", 0.0), "mm/hari"),
        ("ARG-02", "Automatic Rain Gauge", "Tengah DAS - Stasiun 2", "Curah Hujan",
         max(latest_dict.get("rainfall", 0.0) * 0.85, 0.0), "mm/hari"),
        ("RLS-01", "Reservoir Level Sensor", "Intake Bendungan", "Elevasi Muka Air",
         latest_dict.get("reservoir_level", 0.0), "m dpl"),
        ("RLS-02", "Reservoir Level Sensor", "Menara Kontrol", "Elevasi Muka Air",
         latest_dict.get("reservoir_level", 0.0) - 0.02, "m dpl"),
        ("URL-01", "Upstream River Level / Discharge", "Sungai Utama Hulu", "Inflow",
         latest_dict.get("inflow", 0.0), "m3/detik"),
        ("OFM-01", "Outlet Flow Meter", "Outlet Irigasi", "Release Irigasi",
         latest_dict.get("outflow", 0.0) * 0.62, "m3/detik"),
        ("OFM-02", "Outlet Flow Meter", "Outlet Air Baku", "Release Air Baku",
         latest_dict.get("domestic_demand", 0.076), "m3/detik"),
        ("GPS-01", "Gate Position Sensor", "Pintu Intake 1", "Bukaan Pintu", 42.0, "%"),
        ("GPS-02", "Gate Position Sensor", "Pintu Pelimpah", "Bukaan Pintu", 0.0, "%"),
        ("SMS-01", "Soil Moisture Sensor", "Zone A", "Kelembapan Tanah",
         latest_dict.get("soil_moisture", 55.0) + 8, "%"),
        ("SMS-02", "Soil Moisture Sensor", "Zone C", "Kelembapan Tanah",
         latest_dict.get("soil_moisture", 55.0) - 12, "%"),
        ("SMS-03", "Soil Moisture Sensor", "Zone D", "Kelembapan Tanah",
         latest_dict.get("soil_moisture", 55.0) - 5, "%"),
        ("AWS-01", "Automatic Weather Station", "Kompleks Bendungan", "Suhu / RH / ET0",
         28.4, "degC"),
    ]

    rows = []
    for i, (sid, stype, loc, param, val, unit) in enumerate(sensors):
        age = int(rng.integers(1, 25))
        if failsafe_demo and sid in ("RLS-01", "URL-01", "ARG-01"):
            status, comm, quality, age = "OFFLINE", "TERPUTUS", "NO DATA", 300
            val = np.nan
        else:
            r = rng.random()
            if r < 0.80:
                status, comm, quality = "ONLINE", "GOOD (4G/LoRa)", "GOOD"
            elif r < 0.93:
                status, comm, quality = "WARNING", "LEMAH", "SUSPECT"
                age = int(rng.integers(90, 200))
            else:
                status, comm, quality = "OFFLINE", "TERPUTUS", "NO DATA"
                val = np.nan
                age = int(rng.integers(300, 900))
        rows.append({
            "ID": sid, "Sensor": stype, "Lokasi": loc, "Parameter": param,
            "Nilai": (round(float(val), 3) if not (isinstance(val, float) and math.isnan(val)) else None),
            "Satuan": unit,
            "Update Terakhir": (now - pd.Timedelta(minutes=age)).strftime("%Y-%m-%d %H:%M"),
            "Umur Data (menit)": age,
            "Komunikasi": comm, "Kualitas Data": quality, "Status": status,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 18. PLACEHOLDER INTEGRASI DATA MASA DEPAN
# =============================================================================
def load_iot_data(endpoint=None, token=None):
    """
    PLACEHOLDER integrasi telemetri IoT (MQTT / HTTP / LoRaWAN gateway).

    Contoh implementasi nyata:
        import requests
        r = requests.get(endpoint, headers={"Authorization": f"Bearer {token}"}, timeout=15)
        return pd.DataFrame(r.json()["data"])

    Fungsi ini sengaja TIDAK dipanggil pada mode demo agar aplikasi dapat
    berjalan tanpa API key dan tanpa koneksi internet.
    """
    return None


def load_database_data(conn_string=None, query=None):
    """
    PLACEHOLDER integrasi basis data (PostgreSQL/TimescaleDB/InfluxDB).

    Contoh implementasi nyata:
        from sqlalchemy import create_engine
        engine = create_engine(conn_string)
        return pd.read_sql(query, engine)
    """
    return None


def load_forecast_api(lat=None, lon=None, api_url=None):
    """
    PLACEHOLDER integrasi API prakiraan cuaca (BMKG / Open-Meteo / GFS).

    Contoh implementasi nyata (Open-Meteo, tanpa API key):
        url = ("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s"
               "&daily=precipitation_sum&timezone=Asia%%2FJakarta" % (lat, lon))
        return pd.DataFrame(requests.get(url, timeout=15).json()["daily"])
    """
    return None


def load_csv_data(uploaded_file):
    """
    Memuat data CSV unggahan pengguna dan menyeragamkan nama kolom.
    Kolom minimum: datetime, storage ATAU reservoir_level, inflow, outflow, rainfall.
    """
    df = pd.read_csv(uploaded_file)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    alias = {
        "date": "datetime", "time": "datetime", "timestamp": "datetime", "tanggal": "datetime",
        "water_level": "reservoir_level", "elevation": "reservoir_level",
        "level": "reservoir_level", "elevasi": "reservoir_level",
        "storage_mcm": "storage", "tampungan": "storage",
        "qin": "inflow", "debit_masuk": "inflow",
        "qout": "outflow", "debit_keluar": "outflow", "release": "outflow",
        "hujan": "rainfall", "precipitation": "rainfall",
    }
    df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime"] = pd.date_range(end=pd.Timestamp.now().normalize(),
                                       periods=len(df), freq="D")

    # Melengkapi kolom yang tidak ada dengan nilai wajar agar app tetap berjalan
    if "storage" not in df.columns and "reservoir_level" in df.columns:
        df["storage"] = df["reservoir_level"].apply(elevation_to_storage)
    if "reservoir_level" not in df.columns and "storage" in df.columns:
        df["reservoir_level"] = df["storage"].apply(storage_to_elevation)
    if "storage" not in df.columns:
        df["storage"] = percentage_to_storage(70.0)
        df["reservoir_level"] = storage_to_elevation(df["storage"].iloc[0])
    if "storage_percent" not in df.columns:
        df["storage_percent"] = df["storage"].apply(calculate_storage_percentage)

    defaults = {
        "inflow": DAM["MEAN_ANNUAL_INFLOW"], "outflow": 3.0, "rainfall": 0.0,
        "forecast_rainfall": 0.0, "forecast_inflow": DAM["MEAN_ANNUAL_INFLOW"],
        "evaporation": DAM["EVAP_MEAN_MM"], "domestic_demand": DAM["Q_DOMESTIC"],
        "irrigation_demand": DAM["Q_IRRIGATION_MAX"] * 0.7,
        "environmental_flow": DAM["Q_ENV_FLOW"], "hydropower_release": 0.0,
        "soil_moisture": 55.0, "phase": "CSV",
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    for c in ["DEWS", "FEWS"]:
        if c not in df.columns:
            df[c] = "NORMAL"
    return df.sort_values("datetime").reset_index(drop=True)


# =============================================================================
# 19. KOMPONEN GRAFIK PLOTLY
# =============================================================================
def gauge_chart(value, title, ranges, unit="", height=250, vmax=100.0):
    """Gauge chart umum (DEWS, Reservoir Stress, Storage, dsb.)."""
    steps = [{"range": [a, b], "color": c} for a, b, c in ranges]
    color = C_CYAN
    for a, b, c in ranges:
        if a <= value <= b:
            color = c
            break
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix": unit, "font": {"size": 30, "color": C_TEXT}},
        title={"text": title, "font": {"size": 13, "color": C_MUTED}},
        gauge={
            "axis": {"range": [0, vmax], "tickcolor": C_MUTED,
                     "tickfont": {"color": C_MUTED, "size": 10}},
            "bar": {"color": color, "thickness": 0.72},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 1, "bordercolor": C_LINE,
            "steps": [{"range": s["range"], "color": s["color"] + "33"} for s in steps],
            "threshold": {"line": {"color": C_TEXT, "width": 3},
                          "thickness": 0.82, "value": float(value)},
        },
    ))
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=C_TEXT), margin=dict(l=18, r=18, t=42, b=12))
    return fig


DEWS_RANGES = [(0, 25, C_GREEN), (25, 50, C_YELLOW), (50, 75, C_ORANGE), (75, 100, C_RED)]
STORAGE_RANGES = [(0, 25, C_RED), (25, 50, C_ORANGE), (50, 75, C_YELLOW), (75, 100, C_GREEN)]


def stage_flow_diagram(active):
    """Diagram CONSERVE -> ALLOCATE -> ANTICIPATE -> RECOVER dengan highlight."""
    desc = {
        "CONSERVE": "Konservasi dini",
        "ALLOCATE": "Alokasi berbasis pangan",
        "ANTICIPATE": "Antisipasi inflow",
        "RECOVER": "Pemulihan tampungan",
    }
    html = '<div class="jr-flow">'
    for i, s in enumerate(STAGES):
        on = "active" if s == active else ""
        dot = "●" if s == active else "○"
        html += (f'<div class="jr-step {on}"><div class="dot">{dot}</div>'
                 f'<div class="nm">{s}</div><div class="ds">{desc[s]}</div></div>')
        if i < len(STAGES) - 1:
            html += '<div class="jr-arrow">→</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def timeseries_chart(df, cols, names, colors, ytitle, height=330,
                     fill=None, hlines=None, dash=None):
    """Grafik deret waktu umum dengan opsi garis acuan horizontal."""
    fig = go.Figure()
    for i, c in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=df["datetime"], y=df[c], name=names[i], mode="lines",
            line=dict(color=colors[i], width=2.4,
                      dash=(dash[i] if dash else None)),
            fill=("tozeroy" if (fill and fill[i]) else None),
            fillcolor=(colors[i] + "22") if (fill and fill[i]) else None,
        ))
    if hlines:
        for y, label, color in hlines:
            fig.add_hline(y=y, line=dict(color=color, width=1.4, dash="dash"),
                          annotation_text=label, annotation_position="right",
                          annotation_font=dict(color=color, size=10))
    fig.update_yaxes(title_text=ytitle)
    return style_plot(fig, height)


def sankey_multipurpose(alloc, storage_pct):
    """
    Sankey Diagram alokasi air multi-purpose:
    RESERVOIR STORAGE -> RAW WATER / IRRIGATION / ENVIRONMENT / HYDROPOWER / OTHER
    """
    labels = ["RESERVOIR STORAGE", "Air Baku (Raw Water)", "Irigasi",
              "Environmental Flow", "PLTA", "Perikanan & Pariwisata",
              "Aliran Sungai Hilir"]
    src = [0, 0, 0, 0, 0, 4, 3, 2]
    tgt = [1, 2, 3, 4, 5, 6, 6, 6]
    val = [
        max(alloc["q_domestic"], 1e-3),
        max(alloc["q_irrigation"], 1e-3),
        max(alloc["q_env"], 1e-3),
        max(alloc["q_hydro"], 1e-3),
        max(alloc["q_fisheries"] + alloc["q_tourism"], 1e-3),
        max(alloc["q_hydro"] * 0.98, 1e-3),   # air turbin kembali ke hilir
        max(alloc["q_env"] * 0.95, 1e-3),
        max(alloc["q_irrigation"] * 0.22, 1e-3),  # return flow irigasi
    ]
    node_colors = [C_CYAN, C_BLUE, C_GREEN, "#14b8a6", C_PURPLE, C_YELLOW, "#38bdf8"]
    link_colors = [node_colors[t] + "55" for t in tgt]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=18, thickness=20,
                  line=dict(color=C_LINE, width=1),
                  label=labels, color=node_colors),
        link=dict(source=src, target=tgt, value=val, color=link_colors),
    ))
    fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=C_TEXT, size=12),
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def radar_chart(categories, baseline, jala, height=380):
    """Radar chart perbandingan kinerja Baseline vs JALA RAKSA."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(baseline) + [baseline[0]], theta=list(categories) + [categories[0]],
        fill="toself", name="Baseline Operation",
        line=dict(color=C_GREY, width=2), fillcolor="rgba(148,163,184,.18)"))
    fig.add_trace(go.Scatterpolar(
        r=list(jala) + [jala[0]], theta=list(categories) + [categories[0]],
        fill="toself", name="JALA RAKSA",
        line=dict(color=C_CYAN, width=2.6), fillcolor="rgba(34,211,238,.20)"))
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT, size=11),
        polar=dict(bgcolor="rgba(255,255,255,.02)",
                   radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor=C_LINE, tickfont=dict(color=C_MUTED, size=9)),
                   angularaxis=dict(gridcolor=C_LINE, tickfont=dict(color=C_TEXT, size=10))),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=60, b=30))
    return fig


def zone_schematic(zdf):
    """Crop Allocation Map sebagai grid skematik (pengganti GIS)."""
    pc = {"LOW": C_GREEN, "MEDIUM": C_YELLOW, "HIGH": C_ORANGE, "CRITICAL": C_RED}
    html = '<div class="jr-grid">'
    for _, r in zdf.iterrows():
        col = pc.get(r["Priority Class"], C_GREY)
        ful = r.get("Fulfillment (%)", 100.0)
        html += (
            f'<div class="jr-plot" style="border-color:{col};'
            f'background:linear-gradient(160deg,{col}18,{C_CARD})">'
            f'<div class="zn" style="color:{col}">{r["Zone"]}</div>'
            f'<div class="cr">{r["Crop"]} · {r["Crop Stage"]}</div>'
            f'<div class="cr">{r["Area (ha)"]:.0f} ha · SM {r["Soil Moisture (%)"]:.0f}%</div>'
            f'<div class="pr" style="color:{col}">{r["Priority Class"]}</div>'
            f'<div class="cr">Terpenuhi {ful:.0f}%</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 20. MESIN PERHITUNGAN TERPUSAT
# =============================================================================
# Seluruh halaman memakai satu hasil perhitungan yang sama (single source of
# truth) agar KPI, gauge, grafik, status, dan rekomendasi selalu konsisten dan
# berubah SECARA REALTIME ketika input simulasi interaktif diubah.
# =============================================================================
def compute_system_state(inputs, df_hist, zones_base, validation):
    """
    Menjalankan seluruh rantai keputusan JALA RAKSA:

      DEWS -> Fuzzy Reservoir Stress -> Adaptive Hedging -> Available Water
      -> Crop-Sensitive Allocation -> Multi-Purpose Allocation -> FEWS
      -> Available Flood Storage -> Pre-Release -> Recovery -> Operating Mode
      -> JALA RAKSA Recommendation

    'inputs' adalah dict kondisi saat ini (dari data demo/CSV atau slider
    mode simulasi interaktif).
    """
    storage_pct = float(inputs["storage_percent"])
    storage_mcm = percentage_to_storage(storage_pct)
    elev = storage_to_elevation(storage_mcm)
    inflow = float(inputs["inflow"])
    outflow = float(inputs["outflow"])
    rain = float(inputs["rainfall"])
    fc_rain = float(inputs["forecast_rainfall"])
    fc_inflow = float(inputs["forecast_inflow"])
    q_dom = float(inputs["domestic_demand"])
    q_irr_dem = float(inputs["irrigation_demand"])
    q_env = float(inputs["environmental_flow"])
    scenario = inputs.get("scenario", "MOST LIKELY")
    et0 = float(inputs.get("et0", 5.0))

    # ---- 1. Tren tampungan (%/hari) dari data historis ----
    if df_hist is not None and len(df_hist) >= 6:
        trend = float(df_hist["storage_percent"].iloc[-1] -
                      df_hist["storage_percent"].iloc[-6]) / 5.0
    else:
        trend = 0.0
    trend = float(inputs.get("storage_trend", trend))

    # ---- 2. Indeks prakiraan hidrologi & Demand-Supply Ratio ----
    forecast_index = float(np.clip(
        (fc_rain / 70.0) * 50.0 + (fc_inflow / DAM["MEAN_ANNUAL_INFLOW"]) * 25.0,
        0.0, 100.0))
    total_demand = q_dom + q_irr_dem + q_env
    # Pasokan tersedia = inflow + kontribusi tampungan yang dapat dimanfaatkan
    usable_storage_q = mcm_per_day_to_cms(
        max(storage_mcm - S_LWL, 0.0) / 90.0)   # dibagi horizon 90 hari
    available_supply = max(inflow + usable_storage_q, 0.05)
    dsr = float(inputs.get("dsr_override") or (total_demand / available_supply))
    dsr = float(np.clip(dsr, 0.0, 4.0))
    inflow_pct = inflow / DAM["MEAN_ANNUAL_INFLOW"] * 100.0

    # ---- 3. Fuzzy Reservoir Stress ----
    stress, fuzzy_detail = calculate_reservoir_stress(
        storage_pct, inflow_pct, forecast_index, dsr, return_detail=True)

    # ---- 4. DEWS ----
    dews, dews_level, dews_reasons = calculate_dews_status(
        storage_pct, stress, trend, forecast_index, dsr)

    # ---- 5. Adaptive Drought Hedging & Operating Band ----
    hedging = calculate_hedging_level(dews, stress, storage_pct)
    band_lo, band_hi = adaptive_operating_band(dews)

    # ---- 6. Available Water untuk alokasi ----
    # Air yang aman dialokasikan = inflow + porsi tampungan yang boleh dipakai,
    # dibatasi kapasitas outlet.
    releasable = max(storage_mcm - percentage_to_storage(band_lo), 0.0)
    q_from_storage = mcm_per_day_to_cms(releasable / 30.0)   # horizon 30 hari
    available_water = min(inflow + q_from_storage, DAM["OUTLET_CAPACITY"])
    available_water = max(available_water, q_dom)   # air baku selalu dijamin

    # ---- 7. Multi-Purpose Allocation ----
    alloc = multipurpose_allocation(available_water, hedging, q_irr_dem,
                                    q_dom, q_env)

    # ---- 8. Crop-Sensitive Irrigation Allocation ----
    eff_rain = 0.75 * rain     # hujan efektif (DEMO: 75% dari hujan bruto)
    zone_calc = build_zone_table(zones_base, et0, eff_rain)
    zones = allocate_irrigation_water(zone_calc, alloc["q_irrigation"])
    priority_zones = zones[zones["Priority Class"].isin(["CRITICAL", "HIGH"])]["Zone"].tolist()

    # ---- 9. FEWS, routing prakiraan, dan pre-release ----
    hydro_fc = forecast_inflow_scenarios(fc_inflow, fc_rain, scenario, 7)
    flood_space = calculate_available_flood_storage(storage_mcm)
    routing_now = route_forecast_storage(storage_mcm, hydro_fc,
                                         alloc["q_release_total"],
                                         np.full(7, fc_rain / 7.0))
    margin, safety_status = calculate_flood_safety_margin(routing_now["max_elevation_m"])
    fews, fews_level, fews_reasons = calculate_fews_status(
        fc_rain, float(np.max(hydro_fc)), flood_space, margin)
    prerelease = calculate_minimum_prerelease(storage_mcm, hydro_fc,
                                              alloc["q_release_total"], 7)
    # Routing ulang dengan pre-release untuk memperlihatkan efek rekomendasi
    routing_pre = route_forecast_storage(
        storage_mcm, hydro_fc,
        alloc["q_release_total"] + prerelease["prerelease_q_cms"],
        np.full(7, fc_rain / 7.0))
    margin_pre, safety_pre = calculate_flood_safety_margin(routing_pre["max_elevation_m"])

    # ---- 10. Recovery ----
    srr, s_target = calculate_storage_recovery_ratio(storage_mcm)
    rec_days, rec_rate, rec_note = estimate_recovery_time(
        storage_mcm, s_target, fc_inflow, alloc["q_release_total"], rain)
    traj = recovery_trajectory(storage_mcm, s_target, fc_inflow,
                               alloc["q_release_total"], 60, rain * 0.4)

    # ---- 11. Operating Mode ----
    failsafe = bool(validation["failsafe"]) or bool(inputs.get("force_failsafe", False))
    mode, mode_reasons = determine_operating_mode(
        dews, fews, prerelease["required"], srr, failsafe, safety_status)

    # ---- 12. Neraca air harian ----
    wb = calculate_water_balance(storage_mcm, inflow, alloc["q_release_total"],
                                 rain, float(inputs.get("evaporation", DAM["EVAP_MEAN_MM"])))

    ctx = {
        "storage_pct": storage_pct, "storage_mcm": storage_mcm, "elevation": elev,
        "inflow": inflow, "outflow": outflow, "rainfall": rain,
        "forecast_rain": fc_rain, "forecast_inflow": fc_inflow,
        "forecast_index": forecast_index, "dsr": dsr, "trend": trend,
        "stress": stress, "fuzzy": fuzzy_detail,
        "dews": dews, "dews_level": dews_level, "dews_reasons": dews_reasons,
        "hedging": hedging, "band": (band_lo, band_hi),
        "available_water": available_water, "alloc": alloc,
        "zones": zones, "priority_zones": priority_zones,
        "eff_rain": eff_rain, "et0": et0,
        "hydro_forecast": hydro_fc, "scenario": scenario,
        "flood_space": flood_space, "routing": routing_now, "routing_pre": routing_pre,
        "margin": margin, "margin_pre": margin_pre,
        "safety_status": safety_status, "safety_pre": safety_pre,
        "fews": fews, "fews_level": fews_level, "fews_reasons": fews_reasons,
        "prerelease": prerelease,
        "srr": srr, "s_target": s_target, "recovery_days": rec_days,
        "recovery_rate": rec_rate, "recovery_note": rec_note, "trajectory": traj,
        "mode": mode, "mode_reasons": mode_reasons, "failsafe": failsafe,
        "water_balance": wb, "validation": validation,
        "total_demand": total_demand,
    }
    ctx["recommendation"] = generate_jala_raksa_recommendation(ctx)
    return ctx


# =============================================================================
# 21. SESSION STATE & SKENARIO SIMULASI INTERAKTIF
# =============================================================================
SCENARIO_PRESETS = {
    "RESET": {
        "label": "Kondisi awal (normal)",
        "storage_percent": 85.0, "inflow": 6.5, "outflow": 3.2, "rainfall": 8.0,
        "forecast_rainfall": 25.0, "forecast_inflow": 6.8, "soil_moisture": 65.0,
        "irrigation_demand": 2.0, "domestic_demand": 0.076,
        "environmental_flow": 0.35, "et0": 5.0, "scenario": "MOST LIKELY",
        "storage_trend": 0.0, "force_failsafe": False,
    },
    "DROUGHT": {
        "label": "Kekeringan sedang",
        "storage_percent": 48.0, "inflow": 2.2, "outflow": 2.4, "rainfall": 1.0,
        "forecast_rainfall": 6.0, "forecast_inflow": 1.9, "soil_moisture": 42.0,
        "irrigation_demand": 2.3, "domestic_demand": 0.076,
        "environmental_flow": 0.35, "et0": 6.2, "scenario": "MOST LIKELY",
        "storage_trend": -0.45, "force_failsafe": False,
    },
    "EXTREME": {
        "label": "Kekeringan ekstrem",
        "storage_percent": 21.0, "inflow": 0.7, "outflow": 1.6, "rainfall": 0.0,
        "forecast_rainfall": 1.0, "forecast_inflow": 0.6, "soil_moisture": 24.0,
        "irrigation_demand": 2.5, "domestic_demand": 0.076,
        "environmental_flow": 0.35, "et0": 7.1, "scenario": "LOW",
        "storage_trend": -0.85, "force_failsafe": False,
    },
    "HIGH_INFLOW": {
        "label": "Inflow tinggi / antisipasi banjir",
        "storage_percent": 94.0, "inflow": 28.0, "outflow": 6.0, "rainfall": 85.0,
        "forecast_rainfall": 165.0, "forecast_inflow": 34.0, "soil_moisture": 88.0,
        "irrigation_demand": 1.1, "domestic_demand": 0.076,
        "environmental_flow": 0.35, "et0": 3.4, "scenario": "HIGH",
        "storage_trend": 1.20, "force_failsafe": False,
    },
    "RECOVERY": {
        "label": "Kekeringan menuju pemulihan",
        "storage_percent": 58.0, "inflow": 9.5, "outflow": 3.0, "rainfall": 32.0,
        "forecast_rainfall": 48.0, "forecast_inflow": 9.0, "soil_moisture": 72.0,
        "irrigation_demand": 1.8, "domestic_demand": 0.076,
        "environmental_flow": 0.35, "et0": 4.2, "scenario": "MOST LIKELY",
        "storage_trend": 0.65, "force_failsafe": False,
    },
    "FAILSAFE": {
        "label": "Gangguan sensor / data tidak valid",
        "storage_percent": 52.0, "inflow": 3.0, "outflow": 2.8, "rainfall": 4.0,
        "forecast_rainfall": 10.0, "forecast_inflow": 3.1, "soil_moisture": 50.0,
        "irrigation_demand": 2.0, "domestic_demand": 0.076,
        "environmental_flow": 0.35, "et0": 5.5, "scenario": "MOST LIKELY",
        "storage_trend": -0.2, "force_failsafe": True,
    },
}


def init_session_state():
    """Inisialisasi st.session_state (riwayat keputusan operator & slider)."""
    if "decision_history" not in st.session_state:
        st.session_state.decision_history = []
    if "sim" not in st.session_state:
        st.session_state.sim = dict(SCENARIO_PRESETS["RESET"])
        st.session_state.sim.pop("label", None)
    if "scenario_name" not in st.session_state:
        st.session_state.scenario_name = "RESET"
    if "sensor_seed" not in st.session_state:
        st.session_state.sensor_seed = 11
    if "modify_open" not in st.session_state:
        st.session_state.modify_open = False
    if "reject_open" not in st.session_state:
        st.session_state.reject_open = False


def apply_scenario(name):
    """Menerapkan preset skenario ke slider mode simulasi interaktif."""
    preset = dict(SCENARIO_PRESETS[name])
    preset.pop("label", None)
    st.session_state.sim.update(preset)
    st.session_state.scenario_name = name
    # Hapus nilai widget lama agar slider mengambil nilai preset yang baru
    for k in list(st.session_state.keys()):
        if str(k).startswith("w_"):
            del st.session_state[k]


def log_decision(recommendation, decision, note="", operator="Operator Demo"):
    """Menyimpan keputusan operator ke riwayat sesi (BUKAN perintah ke gate)."""
    st.session_state.decision_history.insert(0, {
        "Waktu": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Operator": operator,
        "Operating Mode": recommendation["mode"],
        "DEWS": recommendation["dews"],
        "FEWS": recommendation["fews"],
        "Reservoir Stress": f"{recommendation['stress']:.1f}",
        "Rekomendasi": recommendation["text"][:170],
        "Keputusan": decision,
        "Catatan Operator": note if note else "-",
    })


# =============================================================================
# 22. HEADER & FOOTER
# =============================================================================
def render_header(ctx, data_mode, last_update):
    st.markdown(
        f"""
        <div class="jr-header">
            <div class="jr-title">💧 JALA RAKSA</div>
            <div class="jr-sub">Integrated Drought-to-Recovery Reservoir Operation</div>
            <div class="jr-sub2">Integrated Water Resources Management —
                Decision Support System · v{APP_VERSION}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    q = ctx["validation"]["quality"]
    q_color = {"GOOD": C_GREEN, "WARNING": C_ORANGE, "CRITICAL": C_RED}[q]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        status_box("SYSTEM STATUS", "🟢 SYSTEM ONLINE", f"Mode data: {data_mode}", C_GREEN)
    with c2:
        status_box("LAST DATA UPDATE", last_update, "Waktu server lokal", C_CYAN)
    with c3:
        status_box("DATA QUALITY", q,
                   f"Kelengkapan {ctx['validation']['completeness']:.1f}%", q_color)
    with c4:
        status_box("OPERATING MODE", ctx["mode"], "Rekomendasi DSS",
                   MODE_COLOR.get(ctx["mode"], C_CYAN))
    st.markdown("")

    if ctx["failsafe"]:
        alert(
            "<b>🚨 FAIL-SAFE MODE AKTIF.</b> Data masukan tidak memenuhi syarat validasi "
            "(sensor offline / data hilang / nilai tidak logis / prakiraan tidak tersedia). "
            "Keluaran optimasi JALA RAKSA <b>tidak boleh digunakan</b>. "
            "Sistem merekomendasikan <b>RETURN TO APPROVED BASELINE OPERATING RULE</b> "
            "sampai kualitas data pulih.", "danger")


def render_footer():
    st.markdown(
        f"""
        <div class="jr-footer">
            <b>JALA RAKSA</b> — Integrated Drought-to-Recovery Reservoir Operation<br>
            Decision Support System berbasis Integrated Water Resources Management (IWRM)<br><br>
            <b>Disclaimer:</b> Dashboard ini merupakan <b>Decision Support System</b>.
            Seluruh nilai yang ditampilkan berstatus <b>DEMONSTRATION / SIMULATION DATA</b>.
            Sistem tidak mengendalikan pintu bendungan, tidak terhubung ke PLC/SCADA/gate
            actuator, dan tidak menggantikan operator. Keputusan operasi bendungan tetap
            menjadi tanggung jawab operator berwenang serta harus mengikuti SOP, rule curve,
            hasil reservoir routing, dan persyaratan keselamatan bendungan yang berlaku.<br><br>
            <span style="color:{C_MUTED}">Prinsip: <b>Optimize benefits within safety limits</b>,
            bukan <i>optimize benefits versus safety</i>.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 23. HALAMAN 01 — COMMAND CENTER
# =============================================================================
def page_command_center(ctx, df):
    section("01 — Command Center",
            "Ringkasan menyeluruh kondisi waduk, peringatan dini, dan rekomendasi operasi.")
    demo_tag()
    st.markdown("")

    a = ctx["alloc"]
    # ---------- KPI utama ----------
    r1 = st.columns(4)
    with r1[0]:
        kpi_card("Reservoir Level", f"{ctx['elevation']:.2f}", "m dpl",
                 f"LWL {DAM['LWL']:.0f} · NWL {DAM['NWL']:.0f} m", C_CYAN)
    with r1[1]:
        kpi_card("Storage", f"{ctx['storage_mcm']:.2f}", "juta m³",
                 f"Tampungan efektif {S_EFF:.2f} juta m³", C_BLUE)
    with r1[2]:
        pc = (C_GREEN if ctx["storage_pct"] >= 70 else
              C_YELLOW if ctx["storage_pct"] >= 50 else
              C_ORANGE if ctx["storage_pct"] >= 30 else C_RED)
        kpi_card("Storage Percentage", f"{ctx['storage_pct']:.1f}", "%",
                 f"Target musiman {DAM['TARGET_SEASONAL_PCT']:.0f}%", pc)
    with r1[3]:
        kpi_card("Inflow", f"{ctx['inflow']:.2f}", "m³/dtk",
                 f"Rata-rata {DAM['MEAN_ANNUAL_INFLOW']:.1f} m³/dtk", C_CYAN)

    st.markdown("")
    r2 = st.columns(4)
    with r2[0]:
        kpi_card("Outflow (Release)", f"{a['q_release_total']:.2f}", "m³/dtk",
                 f"Kapasitas outlet {DAM['OUTLET_CAPACITY']:.0f} m³/dtk", C_BLUE)
    with r2[1]:
        kpi_card("Rainfall", f"{ctx['rainfall']:.1f}", "mm/hari",
                 "Automatic Rain Gauge", C_CYAN)
    with r2[2]:
        kpi_card("Forecast Rainfall", f"{ctx['forecast_rain']:.1f}", "mm/7 hari",
                 f"Skenario: {ctx['scenario']}", C_PURPLE)
    with r2[3]:
        fc = C_GREEN if ctx["flood_space"] > 2.0 else C_ORANGE if ctx["flood_space"] > 0.5 else C_RED
        kpi_card("Available Flood Storage", f"{ctx['flood_space']:.2f}", "juta m³",
                 f"Terhadap FCL {DAM['FCL']:.1f} m dpl", fc)

    st.markdown("---")

    # ---------- Status utama ----------
    s1 = st.columns(4)
    with s1[0]:
        status_box("DEWS STATUS", ctx["dews"], "Drought Early Warning System",
                   STATUS_COLOR[ctx["dews"]])
    with s1[1]:
        status_box("FEWS STATUS", ctx["fews"], "Flood Early Warning System",
                   STATUS_COLOR[ctx["fews"]])
    with s1[2]:
        sc = classify_stress(ctx["stress"])
        status_box("RESERVOIR STRESS", f"{ctx['stress']:.1f} / 100",
                   f"Fuzzy Mamdani · {sc}", STATUS_COLOR[sc])
    with s1[3]:
        status_box("OPERATING MODE", ctx["mode"], "Rekomendasi (bukan perintah)",
                   MODE_COLOR.get(ctx["mode"], C_CYAN))

    st.markdown("---")

    # ---------- Diagram tahapan ----------
    section("Filosofi Operasi JALA RAKSA",
            "Tahapan yang sedang aktif ditandai dengan lingkaran penuh (●).")
    stage_flow_diagram(active_stage(ctx["mode"]))
    st.markdown("")

    # ---------- Rekomendasi ----------
    rec = ctx["recommendation"]
    kind = ("danger" if ctx["mode"] in ("FAIL-SAFE MODE", "EMERGENCY ALLOCATION")
            else "warn" if ctx["mode"] in ("DROUGHT HEDGING", "FLOOD ANTICIPATION")
            else "info" if ctx["mode"] == "RECOVERY MODE" else "ok")
    body = "".join(f"<li>{ln}</li>" for ln in rec["narrative"])
    alert(f"<b>🧭 JALA RAKSA CURRENT RECOMMENDATION</b><ul style='margin:8px 0 0 0'>"
          f"{body}</ul>", kind)

    # ---------- Gauge ringkas ----------
    g = st.columns(3)
    with g[0]:
        show_chart(gauge_chart(ctx["storage_pct"], "Storage (%)", STORAGE_RANGES,
                               " %", 250, 110), key="cc_g1")
    with g[1]:
        show_chart(gauge_chart(ctx["stress"], "Reservoir Stress (Fuzzy)",
                               DEWS_RANGES, "", 250), key="cc_g2")
    with g[2]:
        srr_pct = min(ctx["srr"] * 100.0, 130.0)
        show_chart(gauge_chart(srr_pct, "Storage Recovery Ratio (%)",
                               [(0, 60, C_RED), (60, 85, C_ORANGE),
                                (85, 100, C_YELLOW), (100, 130, C_GREEN)],
                               " %", 250, 130), key="cc_g3")

    # ---------- Grafik ringkas 30 hari ----------
    section("Riwayat 30 Hari Terakhir")
    d30 = df.tail(30)
    c = st.columns(2)
    with c[0]:
        show_chart(timeseries_chart(
            d30, ["storage_percent"], ["Storage (%)"], [C_CYAN],
            "Tampungan efektif (%)", 300, fill=[True],
            hlines=[(DAM["TARGET_SEASONAL_PCT"], "Target musiman", C_GREEN),
                    (DAM["CRITICAL_STORAGE_PCT"], "Ambang kritis", C_RED)]),
            key="cc_ts1")
    with c[1]:
        show_chart(timeseries_chart(
            d30, ["inflow", "outflow"], ["Inflow", "Outflow"], [C_CYAN, C_ORANGE],
            "Debit (m³/dtk)", 300), key="cc_ts2")

    # ---------- Ringkasan multi-purpose ----------
    section("Ringkasan Alokasi Multi-Purpose Hari Ini")
    m = st.columns(4)
    with m[0]:
        kpi_card("Air Baku", f"{a['q_domestic']*1000:.0f}", "l/dtk", "MAINTAIN", C_GREEN)
    with m[1]:
        kpi_card("Irigasi", f"{a['q_irrigation']:.2f}", "m³/dtk",
                 f"Hedging {ctx['hedging']['irrigation_reduction_pct']:.0f}%", C_YELLOW)
    with m[2]:
        kpi_card("Environmental Flow", f"{a['q_env']:.2f}", "m³/dtk",
                 ctx["hedging"]["env_policy"], "#14b8a6")
    with m[3]:
        kpi_card("PLTA", f"{a['power_mw']:.2f}", "MW",
                 f"Kapasitas {DAM['HYDRO_CAPACITY_MW']:.2f} MW", C_PURPLE)


# =============================================================================
# 24. HALAMAN 02 — RESERVOIR OVERVIEW
# =============================================================================
def page_reservoir_overview(ctx, df):
    section("02 — Reservoir Overview",
            "Kondisi tampungan, elevasi, dan neraca inflow–outflow waduk.")
    demo_tag()
    st.markdown("")

    a = ctx["alloc"]
    band_lo, band_hi = ctx["band"]
    zone = ("Zona Pengendalian Banjir" if ctx["storage_pct"] > 100 else
            "Zona Operasi Normal" if ctx["storage_pct"] >= band_lo else
            "Zona Konservasi Kekeringan")

    r = st.columns(4)
    with r[0]:
        kpi_card("Reservoir Elevation", f"{ctx['elevation']:.2f}", "m dpl",
                 f"NWL {DAM['NWL']:.1f} · HWL {DAM['HWL']:.1f}", C_CYAN)
    with r[1]:
        kpi_card("Reservoir Storage", f"{ctx['storage_mcm']:.2f}", "juta m³",
                 f"S(NWL) = {S_NWL:.2f} juta m³", C_BLUE)
    with r[2]:
        kpi_card("Storage %", f"{ctx['storage_pct']:.1f}", "%", zone, C_CYAN)
    with r[3]:
        kpi_card("Operating Zone", zone.split()[-1], "",
                 f"Band adaptif {band_lo:.0f}–{band_hi:.0f}%", C_PURPLE)

    st.markdown("")
    r2 = st.columns(4)
    with r2[0]:
        kpi_card("Inflow", f"{ctx['inflow']:.2f}", "m³/dtk", "Sensor debit hulu", C_CYAN)
    with r2[1]:
        kpi_card("Outflow", f"{a['q_release_total']:.2f}", "m³/dtk",
                 "Total pelepasan", C_ORANGE)
    with r2[2]:
        kpi_card("Rainfall", f"{ctx['rainfall']:.1f}", "mm/hari", "ARG", C_BLUE)
    with r2[3]:
        net = ctx["water_balance"]["delta_storage_mcm"]
        kpi_card("Perubahan Tampungan", f"{net:+.4f}", "juta m³/hari",
                 "ΔS = Qin − Qout + P − E − L",
                 C_GREEN if net >= 0 else C_ORANGE)

    st.markdown("")
    s = st.columns(4)
    with s[0]:
        status_box("DEWS", ctx["dews"], "Kekeringan", STATUS_COLOR[ctx["dews"]])
    with s[1]:
        status_box("FEWS", ctx["fews"], "Banjir", STATUS_COLOR[ctx["fews"]])
    with s[2]:
        sc = classify_stress(ctx["stress"])
        status_box("RESERVOIR STRESS", f"{ctx['stress']:.1f}", sc, STATUS_COLOR[sc])
    with s[3]:
        status_box("OPERATING MODE", ctx["mode"], "Saat ini",
                   MODE_COLOR.get(ctx["mode"], C_CYAN))

    st.markdown("---")

    # ---- Grafik 1 & 2 ----
    d30 = df.tail(30)
    c = st.columns(2)
    with c[0]:
        st.markdown("**1. Reservoir Water Level — 30 Hari**")
        show_chart(timeseries_chart(
            d30, ["reservoir_level"], ["Elevasi muka air"], [C_CYAN],
            "Elevasi (m dpl)", 320,
            hlines=[(DAM["NWL"], "NWL", C_GREEN), (DAM["FCL"], "FCL", C_ORANGE),
                    (DAM["HWL"], "HWL", C_RED), (DAM["LWL"], "LWL", C_GREY)]),
            key="ro_1")
    with c[1]:
        st.markdown("**2. Reservoir Storage — 30 Hari**")
        show_chart(timeseries_chart(
            d30, ["storage"], ["Tampungan"], [C_BLUE], "Tampungan (juta m³)", 320,
            fill=[True],
            hlines=[(S_NWL, "S(NWL)", C_GREEN), (S_LWL, "S(LWL)", C_GREY)]),
            key="ro_2")

    c2 = st.columns(2)
    with c2[0]:
        st.markdown("**3. Inflow vs Outflow**")
        show_chart(timeseries_chart(
            d30, ["inflow", "outflow"], ["Inflow", "Outflow"],
            [C_CYAN, C_ORANGE], "Debit (m³/dtk)", 320), key="ro_3")
    with c2[1]:
        st.markdown("**4. Rainfall**")
        fig = go.Figure(go.Bar(x=d30["datetime"], y=d30["rainfall"],
                               name="Curah hujan", marker_color=C_BLUE))
        fig.update_yaxes(title_text="Curah hujan (mm/hari)")
        show_chart(style_plot(fig, 320), key="ro_4")

    # ---- Grafik 5: Storage vs Adaptive Operating Band ----
    st.markdown("**5. Storage vs Adaptive Operating Band**")
    fig = go.Figure()
    x = df["datetime"]
    fig.add_trace(go.Scatter(x=x, y=np.full(len(df), band_hi), name="Batas atas band adaptif",
                             line=dict(color=C_GREEN, width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=np.full(len(df), band_lo), name="Batas bawah band adaptif",
                             line=dict(color=C_ORANGE, width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(34,211,238,.10)"))
    fig.add_trace(go.Scatter(x=x, y=df["storage_percent"], name="Storage aktual",
                             line=dict(color=C_CYAN, width=2.8)))
    fig.add_hline(y=100, line=dict(color=C_GREEN, width=1.6, dash="dash"),
                  annotation_text="NWL (100%)", annotation_font=dict(color=C_GREEN, size=10))
    fig.add_hline(y=DAM["TARGET_SEASONAL_PCT"], line=dict(color=C_BLUE, width=1.4, dash="dash"),
                  annotation_text="Target musiman", annotation_font=dict(color=C_BLUE, size=10))
    fig.add_hline(y=0, line=dict(color=C_GREY, width=1.6, dash="dash"),
                  annotation_text="LWL (0%)", annotation_font=dict(color=C_GREY, size=10))
    fig.add_hline(y=DAM["CRITICAL_STORAGE_PCT"], line=dict(color=C_RED, width=1.4, dash="dash"),
                  annotation_text="Ambang kritis", annotation_font=dict(color=C_RED, size=10))
    fig.update_yaxes(title_text="Tampungan efektif (%)")
    show_chart(style_plot(fig, 380), key="ro_5")

    alert(
        "<b>Catatan penting — Adaptive Operating Band.</b> JALA RAKSA "
        "<b>TIDAK mengubah</b> LWL, NWL, HWL, maupun FWL hasil desain bendungan. "
        "Band adaptif hanya menyesuaikan <i>target operasi</i> di dalam batas desain "
        "tersebut sesuai status kekeringan, kondisi tampungan, prakiraan inflow, dan "
        "kebutuhan air. Baseline rule curve tetap menjadi acuan dan batas utama operasi.",
        "info")

    with st.expander("📐 Kurva Elevasi–Kapasitas & Karakteristik Waduk"):
        elevs = np.linspace(DAM["EL_BOTTOM"], DAM["CREST"], 120)
        stor = [elevation_to_storage(e) for e in elevs]
        fig = go.Figure(go.Scatter(x=stor, y=elevs, mode="lines",
                                   line=dict(color=C_CYAN, width=3), name="S(h)"))
        for lv, nm, col in [(DAM["LWL"], "LWL", C_GREY), (DAM["NWL"], "NWL", C_GREEN),
                            (DAM["FCL"], "FCL", C_ORANGE), (DAM["HWL"], "HWL", C_RED)]:
            fig.add_hline(y=lv, line=dict(color=col, width=1.2, dash="dash"),
                          annotation_text=nm, annotation_font=dict(color=col, size=10))
        fig.add_trace(go.Scatter(x=[ctx["storage_mcm"]], y=[ctx["elevation"]],
                                 mode="markers", name="Kondisi saat ini",
                                 marker=dict(color=C_YELLOW, size=14, symbol="diamond")))
        fig.update_xaxes(title_text="Tampungan (juta m³)")
        fig.update_yaxes(title_text="Elevasi (m dpl)")
        show_chart(style_plot(fig, 380), key="ro_curve")
        show_table(pd.DataFrame({
            "Parameter": ["Elevasi dasar", "LWL", "NWL", "Flood Control Level", "HWL",
                          "Puncak bendungan", "Tampungan pada LWL", "Tampungan pada NWL",
                          "Tampungan efektif", "Kapasitas outlet", "Kapasitas pelimpah",
                          "Batas aman aliran hilir"],
            "Nilai": [f"{DAM['EL_BOTTOM']:.2f}", f"{DAM['LWL']:.2f}", f"{DAM['NWL']:.2f}",
                      f"{DAM['FCL']:.2f}", f"{DAM['HWL']:.2f}", f"{DAM['CREST']:.2f}",
                      f"{S_LWL:.3f}", f"{S_NWL:.3f}", f"{S_EFF:.3f}",
                      f"{DAM['OUTLET_CAPACITY']:.1f}", f"{DAM['SPILLWAY_CAPACITY']:.1f}",
                      f"{DAM['DOWNSTREAM_SAFE_FLOW']:.1f}"],
            "Satuan": ["m dpl"] * 6 + ["juta m³"] * 3 + ["m³/dtk"] * 3,
        }))
        st.caption("Seluruh geometri waduk di atas adalah NILAI DEMONSTRASI. "
                   "Ganti dengan kurva elevasi-kapasitas dan data desain bendungan yang sebenarnya.")


# =============================================================================
# 25. HALAMAN 03 — DEWS & CONSERVATION
# =============================================================================
def page_dews(ctx, df):
    section("03 — DEWS & Conservation",
            "Drought Early Warning System, penilaian Fuzzy Reservoir Stress, "
            "dan Adaptive Drought Hedging.")
    demo_tag()
    st.markdown("")

    h = ctx["hedging"]
    band_lo, band_hi = ctx["band"]

    # ---- Status utama ----
    s = st.columns(4)
    with s[0]:
        status_box("DEWS STATUS", ctx["dews"], "NORMAL→WASPADA→SIAGA→DARURAT",
                   STATUS_COLOR[ctx["dews"]])
    with s[1]:
        sc = classify_stress(ctx["stress"])
        status_box("RESERVOIR STRESS", f"{ctx['stress']:.1f}/100", f"Fuzzy Mamdani · {sc}",
                   STATUS_COLOR[sc])
    with s[2]:
        status_box("HEDGING LEVEL", h["level_name"].split(" - ")[0],
                   h["level_name"].split(" - ")[1], STATUS_COLOR[ctx["dews"]])
    with s[3]:
        status_box("OPERATING MODE", ctx["mode"], "Rekomendasi DSS",
                   MODE_COLOR.get(ctx["mode"], C_CYAN))

    st.markdown("")

    # ---- Gauge DEWS & Stress ----
    g = st.columns(3)
    with g[0]:
        show_chart(gauge_chart((ctx["dews_level"] + 0.5) * 25.0, "DEWS Level",
                               DEWS_RANGES, "", 260), key="dw_g1")
    with g[1]:
        show_chart(gauge_chart(ctx["stress"], "Reservoir Stress 0–100",
                               DEWS_RANGES, "", 260), key="dw_g2")
    with g[2]:
        show_chart(gauge_chart(ctx["storage_pct"], "Storage (%)",
                               STORAGE_RANGES, " %", 260, 110), key="dw_g3")

    # ---- Input DEWS ----
    section("Input DEWS", "Seluruh masukan yang digunakan sistem peringatan dini kekeringan.")
    inp = pd.DataFrame({
        "Parameter": ["Reservoir Storage", "Reservoir Elevation", "Inflow", "Rainfall",
                      "Forecast Rainfall (7 hari)", "Forecast Inflow", "Storage Trend",
                      "Total Demand", "Demand-Supply Ratio",
                      "Forecast Hydrological Condition"],
        "Nilai": [f"{ctx['storage_pct']:.1f}", f"{ctx['elevation']:.2f}",
                  f"{ctx['inflow']:.2f}", f"{ctx['rainfall']:.1f}",
                  f"{ctx['forecast_rain']:.1f}", f"{ctx['forecast_inflow']:.2f}",
                  f"{ctx['trend']:+.2f}", f"{ctx['total_demand']:.3f}",
                  f"{ctx['dsr']:.2f}", f"{ctx['forecast_index']:.1f}"],
        "Satuan": ["% tampungan efektif", "m dpl", "m³/dtk", "mm/hari", "mm", "m³/dtk",
                   "%/hari", "m³/dtk", "-", "indeks 0–100"],
        "Interpretasi": [
            "Tampungan efektif tersedia",
            "Elevasi muka air waduk",
            "Debit masuk terukur",
            "Hujan harian terukur",
            "Akumulasi hujan prakiraan",
            "Prakiraan debit masuk",
            "Positif = mengisi, negatif = menyusut",
            "Air baku + irigasi + lingkungan",
            "> 1,00 berarti defisit pasokan",
            "0 sangat kering, 50 normal, 100 sangat basah",
        ],
    })
    show_table(inp)

    # ---- Alasan status DEWS ----
    reasons = "".join(f"<li>{r}</li>" for r in ctx["dews_reasons"])
    kind = {"NORMAL": "ok", "WASPADA": "warn", "SIAGA": "warn", "DARURAT": "danger"}[ctx["dews"]]
    alert(f"<b>Dasar penetapan status DEWS = {ctx['dews']}</b>"
          f"<ul style='margin:8px 0 0 0'>{reasons}</ul>", kind)

    st.markdown("---")

    # ---- Tren storage & inflow ----
    section("Storage Trend & Inflow Trend")
    d = df.tail(45)
    c = st.columns(2)
    with c[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["datetime"], y=d["storage_percent"],
                                 name="Storage (%)", line=dict(color=C_CYAN, width=2.6),
                                 fill="tozeroy", fillcolor="rgba(34,211,238,.12)"))
        if len(d) >= 7:
            ma = d["storage_percent"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=d["datetime"], y=ma, name="Rata-rata 7 hari",
                                     line=dict(color=C_YELLOW, width=1.8, dash="dash")))
        fig.add_hline(y=DAM["CRITICAL_STORAGE_PCT"], line=dict(color=C_RED, width=1.4, dash="dash"),
                      annotation_text="Ambang kritis", annotation_font=dict(color=C_RED, size=10))
        fig.update_yaxes(title_text="Tampungan efektif (%)")
        show_chart(style_plot(fig, 320, "Storage Trend"), key="dw_t1")
    with c[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["datetime"], y=d["inflow"], name="Inflow aktual",
                                 line=dict(color=C_CYAN, width=2.4)))
        fig.add_trace(go.Scatter(x=d["datetime"], y=d["forecast_inflow"],
                                 name="Prakiraan inflow",
                                 line=dict(color=C_PURPLE, width=2, dash="dot")))
        fig.add_hline(y=DAM["MEAN_ANNUAL_INFLOW"], line=dict(color=C_GREEN, width=1.3, dash="dash"),
                      annotation_text="Rata-rata jangka panjang",
                      annotation_font=dict(color=C_GREEN, size=10))
        fig.update_yaxes(title_text="Debit (m³/dtk)")
        show_chart(style_plot(fig, 320, "Inflow Trend"), key="dw_t2")

    # ---- Demand-Supply Ratio & Forecast ----
    c2 = st.columns(3)
    with c2[0]:
        dsr_color = C_GREEN if ctx["dsr"] < 0.85 else C_YELLOW if ctx["dsr"] < 1.0 else C_RED
        kpi_card("Demand-Supply Ratio", f"{ctx['dsr']:.2f}", "-",
                 "> 1,00 = defisit pasokan", dsr_color)
    with c2[1]:
        fcol = C_RED if ctx["forecast_index"] < 30 else C_YELLOW if ctx["forecast_index"] < 55 else C_GREEN
        kpi_card("Forecast Hydrological Condition", f"{ctx['forecast_index']:.1f}", "/100",
                 "0 kering · 50 normal · 100 basah", fcol)
    with c2[2]:
        tcol = C_GREEN if ctx["trend"] >= 0 else C_ORANGE if ctx["trend"] > -0.5 else C_RED
        kpi_card("Storage Trend", f"{ctx['trend']:+.2f}", "%/hari",
                 "Laju perubahan tampungan", tcol)

    st.markdown("---")

    # ---- Adaptive Operating Band ----
    section("Adaptive Operating Band",
            "Target operasi menyesuaikan status kekeringan TANPA mengubah batas desain.")
    fig = go.Figure()
    bands = {"NORMAL": (72, 100), "WASPADA": (62, 92), "SIAGA": (50, 82), "DARURAT": (35, 70)}
    for i, (nm, (lo, hi)) in enumerate(bands.items()):
        fig.add_trace(go.Bar(x=[nm], y=[hi - lo], base=[lo], name=nm,
                             marker_color=STATUS_COLOR[nm] + ("" if nm == ctx["dews"] else "55"),
                             marker_line=dict(color=STATUS_COLOR[nm],
                                              width=3 if nm == ctx["dews"] else 1),
                             text=[f"{lo:.0f}–{hi:.0f}%"], textposition="inside"))
    fig.add_hline(y=ctx["storage_pct"], line=dict(color=C_TEXT, width=2.5, dash="dash"),
                  annotation_text=f"Storage saat ini {ctx['storage_pct']:.1f}%",
                  annotation_font=dict(color=C_TEXT, size=11))
    fig.add_hline(y=100, line=dict(color=C_GREEN, width=1.2, dash="dot"),
                  annotation_text="NWL", annotation_font=dict(color=C_GREEN, size=10))
    fig.add_hline(y=0, line=dict(color=C_GREY, width=1.2, dash="dot"),
                  annotation_text="LWL", annotation_font=dict(color=C_GREY, size=10))
    fig.update_yaxes(title_text="Tampungan efektif (%)", range=[-5, 115])
    fig.update_layout(showlegend=False)
    show_chart(style_plot(fig, 340), key="dw_band")
    st.caption(f"Band aktif saat ini (status {ctx['dews']}): "
               f"**{band_lo:.0f}% – {band_hi:.0f}%** tampungan efektif.")

    st.markdown("---")

    # ---- Adaptive Drought Hedging ----
    section("Adaptive Drought Hedging", "Rekomendasi konservasi air per sektor.")
    demo_tag("SIMULATION / DEMONSTRATION DATA")
    st.markdown("")
    hc = st.columns(4)
    with hc[0]:
        kpi_card("Recommended Water Conservation", f"{h['conservation_pct']:.0f}", "%",
                 "Terhadap kebutuhan irigasi normal", C_YELLOW)
    with hc[1]:
        kpi_card("Recommended Irrigation Reduction", f"{h['irrigation_reduction_pct']:.0f}", "%",
                 f"Faktor release {h['f_irrigation']:.2f}", C_ORANGE)
    with hc[2]:
        kpi_card("Domestic Water", h["domestic_policy"], "",
                 f"Faktor release {h['f_domestic']:.2f}", C_GREEN)
    with hc[3]:
        kpi_card("Environmental Flow", h["env_policy"], "",
                 f"Faktor release {h['f_environment']:.2f}",
                 C_GREEN if h["f_environment"] >= 0.999 else C_YELLOW)

    st.markdown("")
    hedge_tbl = pd.DataFrame({
        "Sektor": ["Irigasi", "Air Baku / Domestik", "Environmental Flow",
                   "PLTA", "Perikanan & Pariwisata"],
        "Faktor Release": [h["f_irrigation"], h["f_domestic"], h["f_environment"],
                           h["f_hydropower"], h["f_other"]],
        "Pengurangan (%)": [round((1 - h["f_irrigation"]) * 100, 1),
                            round((1 - h["f_domestic"]) * 100, 1),
                            round((1 - h["f_environment"]) * 100, 1),
                            round((1 - h["f_hydropower"]) * 100, 1),
                            round((1 - h["f_other"]) * 100, 1)],
        "Kebijakan": ["Hedging selektif berbasis sensitivitas tanaman",
                      "MAINTAIN — prioritas tertinggi",
                      h["env_policy"] + " — kebutuhan lingkungan hilir",
                      "Dikurangi saat tekanan tinggi (non-konsumtif)",
                      "Ditunda lebih dahulu (non-esensial)"],
    })
    show_table(hedge_tbl)

    concept = pd.DataFrame({
        "Status DEWS": ["NORMAL", "WASPADA", "SIAGA", "DARURAT"],
        "Tindakan": ["Normal operation", "Early conservation",
                     "Adaptive drought hedging", "Emergency allocation"],
        "Penjelasan": [
            "Kebutuhan dilayani sesuai pola operasi normal.",
            "Konservasi dini dimulai; release yang dapat ditunda dievaluasi.",
            "Kebutuhan diprioritaskan; pengurangan irigasi selektif.",
            "Hanya kebutuhan paling esensial yang dijamin.",
        ],
    })
    with st.expander("📘 Konsep Adaptive Drought Hedging"):
        show_table(concept)
        st.markdown(
            "**Prinsip JALA RAKSA:** *early conservation is better than late emergency* — "
            "lebih baik menghemat sedikit secara terkendali sejak dini daripada menunggu "
            "waduk kritis dan mengalami kegagalan pelayanan besar.\n\n"
            "⚠️ Persentase pengurangan di atas adalah **nilai DEMO**. Nilai final harus "
            "dicari melalui simulasi neraca air multi-tahun sehingga diperoleh strategi "
            "dengan *reliability* terbaik."
        )

    # ---- Rekomendasi konservasi ----
    rec_lines = "".join(f"<li>{ln}</li>" for ln in ctx["recommendation"]["narrative"])
    alert(f"<b>Recommended Conservation Action</b><ul style='margin:8px 0 0 0'>{rec_lines}</ul>",
          kind)

    # ---- Riwayat status DEWS ----
    with st.expander("📊 Riwayat Status DEWS"):
        cnt = df["DEWS"].value_counts().reindex(
            ["NORMAL", "WASPADA", "SIAGA", "DARURAT"]).fillna(0)
        fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values,
                               marker_color=[STATUS_COLOR[i] for i in cnt.index],
                               text=[f"{int(v)} hari" for v in cnt.values],
                               textposition="outside"))
        fig.update_yaxes(title_text="Jumlah hari")
        show_chart(style_plot(fig, 300), key="dw_hist")


# =============================================================================
# 26. HALAMAN FUZZY (bagian dari DEWS, ditampilkan sebagai expander mendalam)
# =============================================================================
def render_fuzzy_detail(ctx):
    """Rincian penuh proses Fuzzy Mamdani agar transparan (bukan black box)."""
    fz = ctx["fuzzy"]
    st.markdown("#### 🔬 Fuzzy Mamdani — Reservoir Stress Assessment")
    alert(
        "<b>⚠️ PERINGATAN KALIBRASI:</b> <i>Membership functions, thresholds, and fuzzy "
        "rules shown in this demo are illustrative and MUST be calibrated using actual "
        "reservoir hydrological data and simulation.</i> Angka pada modul ini "
        "<b>bukan</b> nilai final engineering.", "warn")

    c = st.columns(4)
    mfs = [("Reservoir Storage (%)", fz["mu_storage"], ctx["storage_pct"]),
           ("Inflow Condition (% rata-rata)", fz["mu_inflow"],
            ctx["inflow"] / DAM["MEAN_ANNUAL_INFLOW"] * 100.0),
           ("Forecast Hydrological (0–100)", fz["mu_forecast"], ctx["forecast_index"]),
           ("Demand-Supply Ratio (-)", fz["mu_dsr"], ctx["dsr"])]
    for i, (nm, mu, val) in enumerate(mfs):
        with c[i]:
            body = "".join(
                f"<div style='display:flex;justify-content:space-between;font-size:.80rem;"
                f"padding:2px 0'><span style='color:{C_MUTED}'>{k}</span>"
                f"<b style='color:{C_CYAN if v > 0 else C_MUTED}'>{v:.3f}</b></div>"
                for k, v in mu.items())
            st.markdown(
                f"<div class='jr-panel'><div style='font-size:.78rem;color:{C_MUTED};"
                f"font-weight:700'>{nm}</div><div style='font-size:1.2rem;font-weight:800;"
                f"color:{C_TEXT};margin:4px 0 6px'>{val:.2f}</div>{body}</div>",
                unsafe_allow_html=True)

    c2 = st.columns([3, 2])
    with c2[0]:
        st.markdown("**Hasil agregasi & defuzzifikasi (centroid)**")
        fig = go.Figure()
        colors = {"NORMAL": C_GREEN, "WASPADA": C_YELLOW, "SIAGA": C_ORANGE, "DARURAT": C_RED}
        xg = np.linspace(0, 100, 301)
        for term, (a, b, cc) in STRESS_MF.items():
            fig.add_trace(go.Scatter(x=xg, y=[trimf(v, a, b, cc) for v in xg],
                                     name=term, line=dict(color=colors[term], width=1.4,
                                                          dash="dot")))
        fig.add_trace(go.Scatter(x=fz["x"], y=fz["aggregated"], name="Agregasi (MAX)",
                                 line=dict(color=C_CYAN, width=3), fill="tozeroy",
                                 fillcolor="rgba(34,211,238,.20)"))
        fig.add_vline(x=ctx["stress"], line=dict(color=C_TEXT, width=2.5, dash="dash"),
                      annotation_text=f"Centroid = {ctx['stress']:.1f}",
                      annotation_font=dict(color=C_TEXT, size=11))
        fig.update_xaxes(title_text="Reservoir Stress (0–100)")
        fig.update_yaxes(title_text="Derajat keanggotaan μ", range=[0, 1.08])
        show_chart(style_plot(fig, 340), key="fz_agg")
    with c2[1]:
        st.markdown("**Kekuatan aktivasi tiap term output**")
        act = fz["activation"]
        fig = go.Figure(go.Bar(
            x=list(act.values()), y=list(act.keys()), orientation="h",
            marker_color=[STATUS_COLOR[k] for k in act.keys()],
            text=[f"{v:.3f}" for v in act.values()], textposition="outside"))
        fig.update_xaxes(title_text="Derajat aktivasi", range=[0, 1.15])
        show_chart(style_plot(fig, 340), key="fz_act")
        st.caption(f"Jumlah aturan aktif: **{fz['n_fired']}** dari {len(RULE_BASE)} aturan.")

    st.markdown("**Aturan fuzzy yang paling berpengaruh (12 teratas)**")
    if fz["fired_rules"]:
        rules_df = pd.DataFrame(fz["fired_rules"])
        rules_df.columns = ["IF (antecedent)", "THEN Reservoir Stress", "Kekuatan (w)"]
        show_table(rules_df)
    else:
        st.info("Tidak ada aturan yang aktif pada kombinasi input ini.")

    with st.expander("🧮 Tahapan perhitungan Fuzzy Mamdani"):
        st.markdown(f"""
**1. Fuzzifikasi** — nilai tegas diubah menjadi derajat keanggotaan μ ∈ [0,1]
menggunakan membership function segitiga/trapesium untuk 4 input.

**2. Inferensi (implikasi MIN)** — untuk setiap aturan:
$$w_i = \\min(\\mu_{{storage}},\\ \\mu_{{inflow}},\\ \\mu_{{forecast}},\\ \\mu_{{DSR}})$$
Basis aturan berisi $3^4 = {len(RULE_BASE)}$ aturan.

**3. Agregasi (MAX)** — seluruh keluaran aturan digabung:
$$\\mu_{{agg}}(z) = \\max_i \\left[\\min(w_i,\\ \\mu_{{out,i}}(z))\\right]$$

**4. Defuzzifikasi (centroid / center of gravity)**:
$$z^* = \\frac{{\\int z \\cdot \\mu_{{agg}}(z)\\, dz}}{{\\int \\mu_{{agg}}(z)\\, dz}} = {ctx['stress']:.2f}$$

**5. Klasifikasi (DEMO)** — 0–25 NORMAL · 25–50 WASPADA · 50–75 SIAGA · 75–100 DARURAT
→ hasil: **{classify_stress(ctx['stress'])}**

Bobot keparahan yang membentuk konsekuen aturan: storage {SEV_WEIGHT['storage']},
inflow {SEV_WEIGHT['inflow']}, forecast {SEV_WEIGHT['forecast']}, DSR {SEV_WEIGHT['dsr']}.
Storage diberi bobot terbesar karena merupakan variabel keadaan waduk yang paling
menentukan kemampuan pelayanan air.
        """)


# =============================================================================
# 27. HALAMAN 04 — FOOD-SMART ALLOCATION
# =============================================================================
def page_food_allocation(ctx, df):
    section("04 — Food-Smart Allocation",
            "Alokasi air irigasi berbasis sensitivitas tanaman untuk mendukung "
            "swasembada pangan.")
    demo_tag()
    st.markdown("")

    z = ctx["zones"]
    a = ctx["alloc"]
    total_need = float(z["Irrigation Requirement (m3/s)"].sum())
    total_alloc = float(z["Allocation (m3/s)"].sum())
    total_def = float(z["Water Deficit (m3/s)"].sum())
    area_served = float(z.loc[z["Fulfillment (%)"] >= 75, "Area (ha)"].sum())
    prod_ret = float((z["Production Retained (%)"] * z["Area (ha)"]).sum() /
                     max(z["Area (ha)"].sum(), 1))

    k = st.columns(4)
    with k[0]:
        kpi_card("Total Kebutuhan Irigasi", f"{total_need:.3f}", "m³/dtk",
                 f"Luas {z['Area (ha)'].sum():.0f} ha", C_CYAN)
    with k[1]:
        kpi_card("Total Alokasi", f"{total_alloc:.3f}", "m³/dtk",
                 f"Pemenuhan {100*total_alloc/max(total_need,1e-6):.1f}%", C_BLUE)
    with k[2]:
        dcol = C_GREEN if total_def < 0.05 else C_ORANGE if total_def < 0.5 else C_RED
        kpi_card("Total Water Deficit", f"{total_def:.3f}", "m³/dtk",
                 "Kekurangan terhadap kebutuhan", dcol)
    with k[3]:
        pcol = C_GREEN if prod_ret >= 90 else C_YELLOW if prod_ret >= 75 else C_RED
        kpi_card("Expected Production Retained", f"{prod_ret:.1f}", "%",
                 "Rata-rata tertimbang luas (FAO-33)", pcol)

    st.markdown("")
    k2 = st.columns(4)
    with k2[0]:
        kpi_card("Area Adequately Served", f"{area_served:.0f}", "ha",
                 "Pemenuhan ≥ 75%", C_GREEN)
    with k2[1]:
        kpi_card("ET0 (Acuan)", f"{ctx['et0']:.2f}", "mm/hari",
                 "Automatic Weather Station", C_PURPLE)
    with k2[2]:
        kpi_card("Effective Rainfall", f"{ctx['eff_rain']:.2f}", "mm/hari",
                 "75% dari hujan bruto (DEMO)", C_BLUE)
    with k2[3]:
        n_crit = int((z["Priority Class"] == "CRITICAL").sum())
        kpi_card("Zona Prioritas CRITICAL", f"{n_crit}", "zona",
                 ", ".join(ctx["priority_zones"]) if ctx["priority_zones"] else "-",
                 C_RED if n_crit else C_GREEN)

    st.markdown("---")

    # ---- Perubahan orientasi alokasi ----
    alert(
        "<b>Perubahan orientasi alokasi air JALA RAKSA</b><br>"
        "<span style='font-size:1.05rem'>❌ <s>Equal Water Distribution</s> &nbsp;&nbsp;→&nbsp;&nbsp; "
        "✅ <b>Crop-Sensitive Water Allocation</b></span><br><br>"
        "Logika: tanah basah + tanaman <i>tidak</i> pada fase kritis → <b>prioritas rendah</b>; "
        "tanah kering + tanaman pada fase sensitif → <b>prioritas tinggi</b>. "
        "Tujuannya meminimalkan kehilangan produksi pertanian akibat keterbatasan air.",
        "info")

    # ---- Tabel zona ----
    section("Tabel Irrigation Management Zones")
    show_cols = ["Zone", "Area (ha)", "Crop", "Crop Stage", "Soil Moisture (%)",
                 "ETc (mm/hari)", "Effective Rainfall (mm)",
                 "Irrigation Requirement (m3/s)", "Water Deficit (m3/s)",
                 "Crop Sensitivity (Ky)", "Priority Score", "Priority Class",
                 "Allocation (m3/s)", "Fulfillment (%)", "Allocation Status",
                 "Expected Yield Loss (%)"]
    show_table(z[show_cols].sort_values("Priority Score", ascending=False))
    st.caption("Geser tabel ke samping untuk melihat seluruh kolom (tabel dapat "
               "digulir horizontal pada layar kecil).")

    # ---- Grafik ----
    section("Visualisasi Alokasi")
    c = st.columns(2)
    with c[0]:
        st.markdown("**Water Requirement vs Allocation**")
        zz = z.sort_values("Priority Score", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=zz["Zone"], y=zz["Irrigation Requirement (m3/s)"],
                             name="Kebutuhan", marker_color=C_GREY))
        fig.add_trace(go.Bar(x=zz["Zone"], y=zz["Allocation (m3/s)"],
                             name="Alokasi JALA RAKSA", marker_color=C_CYAN))
        fig.update_yaxes(title_text="Debit (m³/dtk)")
        fig.update_layout(barmode="group")
        show_chart(style_plot(fig, 330), key="fa_1")
    with c[1]:
        st.markdown("**Priority by Irrigation Zone**")
        pc = {"LOW": C_GREEN, "MEDIUM": C_YELLOW, "HIGH": C_ORANGE, "CRITICAL": C_RED}
        zz = z.sort_values("Priority Score")
        fig = go.Figure(go.Bar(
            x=zz["Priority Score"], y=zz["Zone"], orientation="h",
            marker_color=[pc[p] for p in zz["Priority Class"]],
            text=[f"{s:.0f} — {p}" for s, p in zip(zz["Priority Score"], zz["Priority Class"])],
            textposition="outside"))
        fig.update_xaxes(title_text="Priority Score (0–100)", range=[0, 118])
        show_chart(style_plot(fig, 330), key="fa_2")

    c2 = st.columns(2)
    with c2[0]:
        st.markdown("**Soil Moisture by Zone**")
        zz = z.sort_values("Soil Moisture (%)")
        colors = [C_RED if v < 35 else C_ORANGE if v < 50 else
                  C_YELLOW if v < 65 else C_GREEN for v in zz["Soil Moisture (%)"]]
        fig = go.Figure(go.Bar(x=zz["Zone"], y=zz["Soil Moisture (%)"],
                               marker_color=colors,
                               text=[f"{v:.0f}%" for v in zz["Soil Moisture (%)"]],
                               textposition="outside"))
        fig.add_hline(y=55, line=dict(color=C_TEXT, width=1.4, dash="dash"),
                      annotation_text="Titik kritis lengas (55%)",
                      annotation_font=dict(color=C_TEXT, size=10))
        fig.update_yaxes(title_text="Kelembapan tanah (%)", range=[0, 105])
        show_chart(style_plot(fig, 330), key="fa_3")
    with c2[1]:
        st.markdown("**Water Deficit by Zone**")
        zz = z.sort_values("Water Deficit (m3/s)", ascending=False)
        fig = go.Figure(go.Bar(x=zz["Zone"], y=zz["Water Deficit (m3/s)"],
                               marker_color=C_ORANGE,
                               text=[f"{v:.3f}" for v in zz["Water Deficit (m3/s)"]],
                               textposition="outside"))
        fig.update_yaxes(title_text="Defisit (m³/dtk)")
        show_chart(style_plot(fig, 330), key="fa_4")

    # ---- Crop Allocation Map skematik ----
    section("Crop Allocation Map (Skematik)",
            "Representasi skematik zona irigasi. Dapat diganti peta GIS bila data "
            "spasial daerah irigasi telah tersedia.")
    zone_schematic(z)

    st.markdown("")
    lg = st.columns(4)
    for i, (cls, col, desc) in enumerate([
            ("LOW", C_GREEN, "Tanah basah, fase tidak kritis"),
            ("MEDIUM", C_YELLOW, "Kebutuhan sedang"),
            ("HIGH", C_ORANGE, "Kebutuhan tinggi"),
            ("CRITICAL", C_RED, "Tanah kering + fase sensitif")]):
        with lg[i]:
            st.markdown(f"{badge(cls, col)} <span style='font-size:.78rem;color:{C_MUTED}'>"
                        f"{desc}</span>", unsafe_allow_html=True)

    # ---- Expected production impact ----
    with st.expander("🌾 Expected Production Impact & Metode Perhitungan"):
        imp = z[["Zone", "Crop", "Crop Stage", "Area (ha)", "Fulfillment (%)",
                 "Expected Yield Loss (%)", "Production Retained (%)"]].copy()
        imp["Perkiraan Luas Terdampak (ha)"] = np.round(
            imp["Area (ha)"] * imp["Expected Yield Loss (%)"] / 100.0, 1)
        show_table(imp.sort_values("Expected Yield Loss (%)", ascending=False))
        st.markdown(f"""
**Rumus yang digunakan**

1. Evapotranspirasi tanaman (FAO-56): $ET_c = K_c \\times ET_0$
2. Kebutuhan air bersih: $NIR = ET_c - R_e - \\Delta SM$ (≥ 0)
3. Kebutuhan kotor: $GIR = NIR / \\eta_{{irigasi}}$, dengan η = {DAM['IRRIGATION_EFFICIENCY']:.2f}
4. Debit: $Q = GIR \\times A \\times 10 / 86400$ (1 mm × 1 ha = 10 m³)
5. Skor prioritas: $0{{,}}45 \\cdot S_{{fase}} K_y + 0{{,}}35 \\cdot S_{{kering}} + 0{{,}}20 \\cdot S_{{defisit}}$
6. Kehilangan hasil (FAO-33): $1 - Y_a/Y_m = K_y (1 - ET_a/ET_m)$

⚠️ Nilai $K_c$, $K_y$, bobot prioritas 45/35/20, efisiensi irigasi, dan titik kritis
lengas tanah 55% adalah **nilai DEMO**. Harus dikalibrasi dengan kajian agronomi,
kurva pF tanah, dan data produktivitas daerah irigasi setempat.
        """)


# =============================================================================
# 28. HALAMAN 05 — IWRM MULTI-PURPOSE
# =============================================================================
def page_iwrm(ctx, df):
    section("05 — IWRM Multi-Purpose",
            "Operasi waduk multipurpose berbasis Integrated Water Resources Management.")
    demo_tag()
    st.markdown("")

    a = ctx["alloc"]
    tbl = a["table"]
    total_dem = float(tbl.loc[tbl["Konsumtif"] == "Ya", "Kebutuhan (m3/s)"].sum())
    total_alloc = float(tbl.loc[tbl["Konsumtif"] == "Ya", "Alokasi (m3/s)"].sum())
    total_def = float(tbl["Defisit (m3/s)"].sum())
    sdr = ctx["available_water"] / max(total_dem, 1e-6)

    k = st.columns(4)
    with k[0]:
        kpi_card("Available Water", f"{ctx['available_water']:.2f}", "m³/dtk",
                 "Inflow + porsi tampungan yang aman", C_CYAN)
    with k[1]:
        kpi_card("Total Demand (konsumtif)", f"{total_dem:.2f}", "m³/dtk",
                 "Seluruh sektor konsumtif", C_BLUE)
    with k[2]:
        scol = C_GREEN if sdr >= 1.0 else C_ORANGE if sdr >= 0.8 else C_RED
        kpi_card("Supply-Demand Ratio", f"{sdr:.2f}", "-",
                 "≥ 1,00 berarti pasokan mencukupi", scol)
    with k[3]:
        dcol = C_GREEN if total_def < 0.05 else C_ORANGE if total_def < 0.5 else C_RED
        kpi_card("Total Deficit", f"{total_def:.3f}", "m³/dtk",
                 "Seluruh sektor", dcol)

    st.markdown("")
    k2 = st.columns(4)
    with k2[0]:
        kpi_card("Current Release", f"{ctx['outflow']:.2f}", "m³/dtk",
                 "Outflow terukur (sensor)", C_ORANGE)
    with k2[1]:
        kpi_card("Recommended Release", f"{a['q_release_total']:.2f}", "m³/dtk",
                 "Rekomendasi JALA RAKSA", C_CYAN)
    with k2[2]:
        kpi_card("Hydropower Generation", f"{a['power_mw']:.2f}", "MW",
                 f"Q turbin {a['q_hydro']:.2f} m³/dtk", C_PURPLE)
    with k2[3]:
        delta = a["q_release_total"] - ctx["outflow"]
        kpi_card("Selisih vs Aktual", f"{delta:+.2f}", "m³/dtk",
                 "Rekomendasi − aktual",
                 C_GREEN if abs(delta) < 0.3 else C_YELLOW)

    st.markdown("---")

    # ---- Tabel alokasi ----
    section("Allocation by Sector & Priority Status")
    show_table(tbl)

    # ---- Sankey ----
    section("Sankey Diagram — Aliran Air Multi-Purpose")
    show_chart(sankey_multipurpose(a, ctx["storage_pct"]), key="iw_sankey")
    alert(
        "<b>Coordinated Multi-Purpose Release.</b> Satu pelepasan air dapat memberikan "
        "lebih dari satu manfaat apabila konfigurasi hidraulik memungkinkan. Contohnya, "
        "air yang melewati turbin PLTA umumnya <b>tidak hilang</b> — air tetap mengalir "
        "ke hilir sehingga dapat sekaligus memenuhi environmental flow dan kebutuhan lain. "
        "Karena itu debit PLTA <b>tidak dijumlahkan</b> sebagai kebutuhan konsumtif; "
        "pelepasan bersih waduk dihitung sebagai <i>max(kebutuhan konsumtif, debit turbin)</i>.",
        "info")

    # ---- Grafik alokasi ----
    c = st.columns(2)
    with c[0]:
        st.markdown("**Kebutuhan vs Alokasi per Sektor**")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tbl["Sektor"], y=tbl["Kebutuhan Terkoreksi (m3/s)"],
                             name="Kebutuhan terkoreksi", marker_color=C_GREY))
        fig.add_trace(go.Bar(x=tbl["Sektor"], y=tbl["Alokasi (m3/s)"],
                             name="Alokasi", marker_color=C_CYAN))
        fig.update_yaxes(title_text="Debit (m³/dtk)")
        fig.update_layout(barmode="group")
        show_chart(style_plot(fig, 360), key="iw_1")
    with c[1]:
        st.markdown("**Proporsi Alokasi (sektor konsumtif)**")
        cons = tbl[tbl["Konsumtif"] == "Ya"]
        fig = go.Figure(go.Pie(
            labels=cons["Sektor"], values=cons["Alokasi (m3/s)"], hole=0.45,
            marker=dict(colors=[C_GREEN, "#14b8a6", C_CYAN, C_YELLOW, C_PURPLE],
                        line=dict(color=C_BG, width=2)),
            textinfo="label+percent", textfont=dict(size=11)))
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=C_TEXT), showlegend=False,
                          margin=dict(l=10, r=10, t=30, b=10))
        show_chart(fig, key="iw_2")

    st.markdown("---")

    # ---- Water Balance ----
    section("Water Balance Waduk",
            "ΔS = Qin − Qout + P − E − L  (bentuk diskrit harian)")
    wb = ctx["water_balance"]
    wbc = st.columns(3)
    with wbc[0]:
        kpi_card("Inflow", f"{wb['inflow_mcm']:.4f}", "juta m³/hari",
                 f"{ctx['inflow']:.2f} m³/dtk × 86400", C_CYAN)
        kpi_card("Rainfall Contribution", f"{wb['rain_mcm']:.4f}", "juta m³/hari",
                 f"{ctx['rainfall']:.1f} mm × {wb['area_km2']:.3f} km²", C_BLUE)
    with wbc[1]:
        kpi_card("Release (Outflow)", f"{wb['outflow_mcm']:.4f}", "juta m³/hari",
                 f"{a['q_release_total']:.2f} m³/dtk × 86400", C_ORANGE)
        kpi_card("Evaporation", f"{wb['evap_mcm']:.4f}", "juta m³/hari",
                 f"{DAM['EVAP_MEAN_MM']:.1f} mm × luas genangan", C_YELLOW)
    with wbc[2]:
        kpi_card("Losses (rembesan dll.)", f"{wb['loss_mcm']:.4f}", "juta m³/hari",
                 "Seepage & kehilangan lain", C_GREY)
        col = C_GREEN if wb["delta_storage_mcm"] >= 0 else C_RED
        kpi_card("Change in Storage (ΔS)", f"{wb['delta_storage_mcm']:+.4f}",
                 "juta m³/hari",
                 f"Elevasi besok ≈ {wb['elevation_new_m']:.2f} m dpl", col)

    st.markdown("")
    st.markdown("**Waterfall Neraca Air Harian**")
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=["Inflow", "Hujan langsung", "Release (−)", "Evaporasi (−)", "Losses (−)", "ΔS"],
        y=[wb["inflow_mcm"], wb["rain_mcm"], -wb["outflow_mcm"],
           -wb["evap_mcm"], -wb["loss_mcm"], 0],
        connector=dict(line=dict(color=C_LINE)),
        increasing=dict(marker=dict(color=C_GREEN)),
        decreasing=dict(marker=dict(color=C_ORANGE)),
        totals=dict(marker=dict(color=C_CYAN)),
        text=[f"{v:+.4f}" for v in [wb["inflow_mcm"], wb["rain_mcm"], -wb["outflow_mcm"],
                                    -wb["evap_mcm"], -wb["loss_mcm"],
                                    wb["delta_storage_mcm"]]],
        textposition="outside"))
    fig.update_yaxes(title_text="Volume (juta m³/hari)")
    show_chart(style_plot(fig, 380), key="iw_wb")

    with st.expander("📘 Penjelasan Prinsip IWRM pada JALA RAKSA"):
        st.markdown(f"""
**Bendungan multipurpose ini melayani:**

| Fungsi | Kapasitas / Kebutuhan | Prioritas |
|---|---|---|
| Air baku (± 35.000 jiwa) | {DAM['Q_DOMESTIC']*1000:.0f} l/detik | 1 — tertinggi, selalu MAINTAIN |
| Irigasi ({DAM['IRRIGATION_AREA_HA']:.0f} ha) | maks {DAM['Q_IRRIGATION_MAX']:.2f} m³/detik | 3 — tinggi, hedging selektif |
| Environmental flow | {DAM['Q_ENV_FLOW']:.2f} m³/detik | 2 — tinggi |
| PLTA | {DAM['HYDRO_CAPACITY_MW']:.2f} MW | 4 — sedang, non-konsumtif |
| Perikanan | {DAM['Q_FISHERIES']:.3f} m³/detik | 4 — sedang |
| Pariwisata | {DAM['Q_TOURISM']:.3f} m³/detik | 5 — rendah |

**Prinsip IWRM yang diterapkan:**
- Air dikelola sebagai satu sumber daya terpadu lintas sektor, bukan per sektor sendiri-sendiri.
- Kebutuhan esensial (air baku dan lingkungan) dijamin lebih dahulu.
- Penghematan diambil dari sektor yang paling kecil dampaknya terlebih dahulu.
- Satu pelepasan dapat melayani banyak fungsi (*coordinated multi-purpose release*).
- Seluruh alokasi tunduk pada batas keselamatan bendungan (*hard constraint*).
        """)


# =============================================================================
# 29. HALAMAN 06 — FEWS & FLOOD ANTICIPATION
# =============================================================================
def page_fews(ctx, df):
    section("06 — FEWS & Flood Anticipation",
            "Flood Early Warning System, reservoir routing prakiraan, dan "
            "Minimum Necessary Pre-Release.")
    demo_tag()
    st.markdown("")

    pre = ctx["prerelease"]
    rt = ctx["routing"]
    a = ctx["alloc"]

    # ---- Status keselamatan (HARD CONSTRAINT) ----
    scol = STATUS_COLOR[ctx["safety_status"]]
    s = st.columns(4)
    with s[0]:
        status_box("FEWS STATUS", ctx["fews"], "Peringatan dini banjir",
                   STATUS_COLOR[ctx["fews"]])
    with s[1]:
        status_box("DAM SAFETY CONSTRAINT", ctx["safety_status"],
                   f"Margin {ctx['margin']:.2f} m terhadap HWL", scol)
    with s[2]:
        status_box("PRE-RELEASE", "REQUIRED" if pre["required"] else "NOT REQUIRED",
                   pre["message"], C_ORANGE if pre["required"] else C_GREEN)
    with s[3]:
        status_box("FORECAST SCENARIO", ctx["scenario"],
                   "LOW / MOST LIKELY / HIGH", C_PURPLE)

    st.markdown("")

    alert(
        "<b>🛡️ Prinsip keselamatan JALA RAKSA:</b> keselamatan bendungan adalah "
        "<b>HARD CONSTRAINT</b>, bukan variabel yang boleh dikorbankan demi keuntungan "
        "operasi. Prinsipnya <b>OPTIMIZE BENEFITS WITHIN SAFETY LIMITS</b>, "
        "bukan <i>optimize benefits versus safety</i>.",
        "ok" if ctx["safety_status"] == "SAFE" else
        "warn" if ctx["safety_status"] == "WARNING" else "danger")

    # ---- KPI FEWS ----
    k = st.columns(4)
    with k[0]:
        kpi_card("Rainfall Forecast", f"{ctx['forecast_rain']:.1f}", "mm/7 hari",
                 f"Skenario {ctx['scenario']}", C_BLUE)
    with k[1]:
        kpi_card("Forecast Inflow (puncak)", f"{np.max(ctx['hydro_forecast']):.2f}", "m³/dtk",
                 f"{np.max(ctx['hydro_forecast'])/DAM['MEAN_ANNUAL_INFLOW']:.1f}× rata-rata",
                 C_CYAN)
    with k[2]:
        kpi_card("Current Storage", f"{ctx['storage_mcm']:.2f}", "juta m³",
                 f"{ctx['storage_pct']:.1f}% tampungan efektif", C_BLUE)
    with k[3]:
        fcol = C_GREEN if ctx["flood_space"] > 2 else C_ORANGE if ctx["flood_space"] > 0.5 else C_RED
        kpi_card("Available Flood Storage", f"{ctx['flood_space']:.2f}", "juta m³",
                 f"S(FCL) − S(t) = {S_FCL:.2f} − {ctx['storage_mcm']:.2f}", fcol)

    st.markdown("")
    k2 = st.columns(4)
    with k2[0]:
        kpi_card("Predicted Inflow Volume", f"{pre['predicted_inflow_volume_mcm']:.2f}",
                 "juta m³", "Akumulasi 7 hari prakiraan", C_PURPLE)
    with k2[1]:
        kpi_card("Outlet Capacity", f"{DAM['OUTLET_CAPACITY']:.1f}", "m³/dtk",
                 "Kapasitas outlet terkendali", C_CYAN)
    with k2[2]:
        kpi_card("Downstream Safe Flow", f"{DAM['DOWNSTREAM_SAFE_FLOW']:.1f}", "m³/dtk",
                 "Batas aman aliran sungai hilir", C_YELLOW)
    with k2[3]:
        mcol = C_GREEN if ctx["margin"] >= 1.0 else C_ORANGE if ctx["margin"] >= 0.3 else C_RED
        kpi_card("Projected Max Level", f"{rt['max_elevation_m']:.2f}", "m dpl",
                 f"HWL {DAM['HWL']:.1f} · margin {ctx['margin']:.2f} m", mcol)

    st.markdown("---")

    # ---- Forecast hydrograph 3 skenario ----
    section("Forecast Hydrograph — 3 Skenario",
            "Forecast tidak dipandang sebagai kepastian; ketidakpastian prakiraan "
            "diperhitungkan melalui tiga skenario.")
    days = np.arange(1, 8)
    fig = go.Figure()
    for sc, col, dash in [("LOW", C_GREEN, "dot"), ("MOST LIKELY", C_CYAN, None),
                          ("HIGH", C_RED, "dash")]:
        h = forecast_inflow_scenarios(ctx["forecast_inflow"], ctx["forecast_rain"], sc, 7)
        fig.add_trace(go.Scatter(x=days, y=h, name=f"Skenario {sc}",
                                 line=dict(color=col, width=3 if sc == ctx["scenario"] else 2,
                                           dash=dash),
                                 fill="tozeroy" if sc == ctx["scenario"] else None,
                                 fillcolor=col + "20"))
    fig.add_hline(y=DAM["OUTLET_CAPACITY"], line=dict(color=C_ORANGE, width=1.4, dash="dash"),
                  annotation_text="Kapasitas outlet",
                  annotation_font=dict(color=C_ORANGE, size=10))
    fig.add_hline(y=DAM["MEAN_ANNUAL_INFLOW"], line=dict(color=C_GREY, width=1.2, dash="dot"),
                  annotation_text="Inflow rata-rata", annotation_font=dict(color=C_GREY, size=10))
    fig.update_xaxes(title_text="Hari prakiraan ke-")
    fig.update_yaxes(title_text="Inflow (m³/dtk)")
    show_chart(style_plot(fig, 360), key="fe_hydro")

    # ---- Reservoir routing ----
    section("Reservoir Routing Prakiraan",
            "Respons tampungan dan elevasi waduk terhadap hidrograf prakiraan.")
    c = st.columns(2)
    with c[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=rt["elevation_m"], name="Tanpa pre-release",
                                 line=dict(color=C_ORANGE, width=2.8)))
        if pre["required"]:
            fig.add_trace(go.Scatter(x=days, y=ctx["routing_pre"]["elevation_m"],
                                     name="Dengan pre-release JALA RAKSA",
                                     line=dict(color=C_CYAN, width=2.8, dash="dash")))
        for lv, nm, col in [(DAM["NWL"], "NWL", C_GREEN), (DAM["FCL"], "FCL", C_YELLOW),
                            (DAM["HWL"], "HWL", C_RED)]:
            fig.add_hline(y=lv, line=dict(color=col, width=1.4, dash="dash"),
                          annotation_text=nm, annotation_font=dict(color=col, size=10))
        fig.update_xaxes(title_text="Hari prakiraan ke-")
        fig.update_yaxes(title_text="Elevasi muka air (m dpl)")
        show_chart(style_plot(fig, 340, "Projected Reservoir Level"), key="fe_r1")
    with c[1]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=ctx["hydro_forecast"], name="Inflow prakiraan",
                                 line=dict(color=C_CYAN, width=2.6), fill="tozeroy",
                                 fillcolor="rgba(34,211,238,.15)"))
        fig.add_trace(go.Scatter(x=days, y=rt["outflow_cms"], name="Outflow (release + spill)",
                                 line=dict(color=C_ORANGE, width=2.4)))
        if pre["required"]:
            fig.add_trace(go.Scatter(
                x=days, y=ctx["routing_pre"]["outflow_cms"],
                name="Outflow + pre-release",
                line=dict(color=C_RED, width=2.2, dash="dash")))
        fig.add_hline(y=DAM["DOWNSTREAM_SAFE_FLOW"], line=dict(color=C_RED, width=1.4, dash="dot"),
                      annotation_text="Batas aman hilir",
                      annotation_font=dict(color=C_RED, size=10))
        fig.update_xaxes(title_text="Hari prakiraan ke-")
        fig.update_yaxes(title_text="Debit (m³/dtk)")
        show_chart(style_plot(fig, 340, "Inflow vs Outflow Prakiraan"), key="fe_r2")

    st.markdown("---")

    # ---- Pre-release decision logic ----
    section("Pre-Release Decision Logic",
            "MINIMUM NECESSARY PRE-RELEASE — bukan maximum release.")

    v_space = pre["available_flood_storage_mcm"]
    v_in = pre["predicted_inflow_volume_mcm"]
    v_need = pre["required_volume_mcm"]

    if not pre["required"]:
        alert(
            f"<b>✅ PRE-RELEASE = 0 — RETAIN INCOMING WATER</b><br><br>"
            f"Available Flood Storage = <b>{v_space:.3f} juta m³</b> ≥ "
            f"Predicted Incoming Volume × FK = <b>{v_need:.3f} juta m³</b><br><br>"
            f"Seluruh inflow prakiraan masih dapat ditampung secara aman. Air "
            f"<b>tidak perlu dibuang</b> karena incoming water dibutuhkan untuk proses "
            f"pemulihan tampungan (recovery). "
            f"<i>Tidak semua prakiraan hujan tinggi harus menghasilkan early release.</i>",
            "ok")
    else:
        kk = "danger" if not pre["feasible"] else "warn"
        extra = ("" if pre["feasible"] else
                 f"<br><br><b>⚠️ PERHATIAN:</b> pre-release yang dibutuhkan melampaui batas "
                 f"hidraulik. Masih terdapat kekurangan ruang <b>{pre['shortfall_mcm']:.3f} "
                 f"juta m³</b>. Siapkan operasi pelimpah sesuai SOP banjir dan koordinasikan "
                 f"peringatan dini kepada masyarakat hilir.")
        alert(
            f"<b>⚠️ MINIMUM NECESSARY PRE-RELEASE DIPERLUKAN</b><br><br>"
            f"Available Flood Storage = <b>{v_space:.3f} juta m³</b> &lt; "
            f"kebutuhan ruang aman <b>{v_need:.3f} juta m³</b><br><br>"
            f"Rekomendasi pelepasan minimum: <b>{pre['prerelease_q_cms']:.2f} m³/detik "
            f"selama {pre['duration_days']} hari</b> "
            f"(volume <b>{pre['prerelease_volume_mcm']:.3f} juta m³</b>).<br>"
            f"Faktor pembatas: <b>{pre['limited_by']}</b>.{extra}", kk)

    logic = pd.DataFrame({
        "Komponen": ["Predicted Incoming Volume (7 hari)",
                     "Faktor keamanan ketidakpastian forecast",
                     "Kebutuhan ruang aman (V × FK)",
                     "Available Flood Storage terhadap FCL",
                     "Ruang tambahan dari release operasional",
                     "Kekurangan ruang (deficit)",
                     "Kapasitas outlet terkendali",
                     "Batas aman aliran hilir",
                     "REKOMENDASI PRE-RELEASE"],
        "Nilai": [f"{v_in:.3f} juta m³", "1,15 (DEMO)", f"{v_need:.3f} juta m³",
                  f"{v_space:.3f} juta m³",
                  f"{cms_to_mcm_per_day(a['q_release_total'])*7:.3f} juta m³",
                  f"{max(v_need - v_space - cms_to_mcm_per_day(a['q_release_total'])*7, 0):.3f} juta m³",
                  f"{DAM['OUTLET_CAPACITY']:.1f} m³/dtk",
                  f"{DAM['DOWNSTREAM_SAFE_FLOW']:.1f} m³/dtk",
                  (f"{pre['prerelease_q_cms']:.2f} m³/dtk × {pre['duration_days']} hari"
                   if pre["required"] else "0,00 m³/dtk (Retain Incoming Water)")],
    })
    show_table(logic)

    with st.expander("🧠 Algoritma Keputusan Pre-Release"):
        st.markdown(f"""
```
V_masuk = Σ Q_forecast × 86400 / 1e6                    [juta m³]
V_ruang = S(FCL) − S(t)                                  [juta m³]

JIKA  V_ruang + V_release_operasional ≥ V_masuk × FK
    → PRE-RELEASE = 0
    → "Retain Incoming Water"
LAINNYA
    → defisit  = V_masuk × FK − (V_ruang + V_release_operasional)
    → durasi   = waktu sampai puncak hidrograf (hari)
    → Q_perlu  = defisit / durasi  (dikonversi ke m³/detik)
    → Q_final  = min( Q_perlu,
                      kapasitas_outlet − release_berjalan,
                      aliran_aman_hilir − release_berjalan )
    → MINIMUM NECESSARY PRE-RELEASE = Q_final
```

**Mengapa minimum, bukan maksimum?** Setelah periode kekeringan, incoming water
sangat berharga untuk memulihkan tampungan. Melepas air berlebihan justru
memperlambat recovery dan menurunkan reliability pelayanan air pada musim
berikutnya. JALA RAKSA hanya melepas volume yang <b>benar-benar diperlukan</b>
untuk menyediakan ruang aman.

⚠️ Faktor keamanan 1,15, respons hujan-aliran 0,055 m³/detik per mm, dan bentuk
hidrograf tak berdimensi adalah **nilai DEMO** yang harus diganti dengan model
hujan-aliran terkalibrasi (HEC-HMS/NRECA/Tank Model) serta hasil analisis
frekuensi banjir rancangan DAS setempat.
        """, unsafe_allow_html=True)

    # ---- Alasan FEWS ----
    reasons = "".join(f"<li>{r}</li>" for r in ctx["fews_reasons"])
    alert(f"<b>Dasar penetapan status FEWS = {ctx['fews']}</b>"
          f"<ul style='margin:8px 0 0 0'>{reasons}</ul>",
          "ok" if ctx["fews"] == "NORMAL" else "warn")


# =============================================================================
# 30. HALAMAN 07 — RESERVOIR RECOVERY
# =============================================================================
def page_recovery(ctx, df):
    section("07 — Reservoir Recovery",
            "Pemulihan tampungan waduk menuju target operasi setelah kekeringan.")
    demo_tag()
    st.markdown("")

    srr = ctx["srr"]
    s_target = ctx["s_target"]
    rec_days = ctx["recovery_days"]
    active = ctx["mode"] == "RECOVERY MODE"

    if active:
        alert("<b>🔵 RECOVERY MODE AKTIF.</b> Kondisi kekeringan telah mereda, namun "
              "target tampungan musiman belum tercapai. Incoming water dipertahankan "
              "sebanyak mungkin untuk pengisian tampungan, air baku tetap dijamin, "
              "irigasi dilayani secara terkendali, release non-esensial dibatasi, "
              "dan FEWS tetap aktif.", "info")
    elif srr >= SRR_TARGET_TOLERANCE:
        alert("<b>🟢 TARGET TAMPUNGAN TERCAPAI.</b> Storage Recovery Ratio ≥ 0,98 dan "
              "kondisi banjir aman, sehingga waduk dapat kembali ke "
              "<b>NORMAL OPERATION</b>.", "ok")
    else:
        alert(f"<b>Recovery dalam pemantauan.</b> Mode operasi saat ini adalah "
              f"<b>{ctx['mode']}</b> karena kondisi kekeringan/banjir memiliki prioritas "
              f"lebih tinggi. Pemulihan tampungan tetap dipantau "
              f"(SRR = {srr:.2f}).", "warn")

    k = st.columns(4)
    with k[0]:
        kpi_card("Current Storage", f"{ctx['storage_mcm']:.2f}", "juta m³",
                 f"{ctx['storage_pct']:.1f}% tampungan efektif", C_BLUE)
    with k[1]:
        kpi_card("Target Seasonal Storage", f"{s_target:.2f}", "juta m³",
                 f"{DAM['TARGET_SEASONAL_PCT']:.0f}% tampungan efektif", C_GREEN)
    with k[2]:
        rcol = C_GREEN if srr >= 1.0 else C_YELLOW if srr >= 0.85 else C_ORANGE if srr >= 0.6 else C_RED
        kpi_card("Storage Recovery Ratio", f"{srr:.3f}", "-",
                 "1,000 = target tercapai", rcol)
    with k[3]:
        kpi_card("Expected Inflow", f"{ctx['forecast_inflow']:.2f}", "m³/dtk",
                 "Prakiraan inflow rata-rata", C_CYAN)

    st.markdown("")
    k2 = st.columns(4)
    with k2[0]:
        if rec_days is None:
            kpi_card("Estimated Recovery Time", "N/A", "",
                     "Laju pengisian bersih ≤ 0", C_RED)
        elif rec_days == 0:
            kpi_card("Estimated Recovery Time", "0", "hari", "Target sudah tercapai", C_GREEN)
        else:
            tcol = C_GREEN if rec_days <= 30 else C_YELLOW if rec_days <= 90 else C_ORANGE
            kpi_card("Estimated Recovery Time", f"{rec_days:.0f}", "hari",
                     f"Laju {ctx['recovery_rate']:+.4f} juta m³/hari", tcol)
    with k2[1]:
        mcol = C_GREEN if ctx["margin"] >= 1.0 else C_ORANGE if ctx["margin"] >= 0.3 else C_RED
        kpi_card("Flood Safety Margin", f"{ctx['margin']:.2f}", "m",
                 f"HWL − elevasi maks. routing", mcol)
    with k2[2]:
        kpi_card("Current Recovery Mode", "ACTIVE" if active else "STANDBY",
                 ctx["mode"], C_BLUE if active else C_GREY)
    with k2[3]:
        gap = max(s_target - ctx["storage_mcm"], 0.0)
        kpi_card("Kekurangan Tampungan", f"{gap:.3f}", "juta m³",
                 "Terhadap target musiman", C_ORANGE if gap > 0 else C_GREEN)

    st.markdown("---")

    # ---- Trajectory ----
    section("Forecast Recovery Trajectory",
            "Lintasan pemulihan tampungan dari kondisi saat ini menuju target.")
    traj = ctx["trajectory"]
    x = np.arange(len(traj))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=traj, name="Lintasan pemulihan (prakiraan)",
                             line=dict(color=C_BLUE, width=3), fill="tozeroy",
                             fillcolor="rgba(59,130,246,.16)"))
    fig.add_trace(go.Scatter(x=[0], y=[ctx["storage_mcm"]], mode="markers+text",
                             name="Current Storage", text=["Sekarang"],
                             textposition="top center",
                             marker=dict(color=C_CYAN, size=14, symbol="circle")))
    fig.add_hline(y=s_target, line=dict(color=C_GREEN, width=2, dash="dash"),
                  annotation_text=f"Target musiman ({DAM['TARGET_SEASONAL_PCT']:.0f}%)",
                  annotation_font=dict(color=C_GREEN, size=11))
    fig.add_hline(y=S_NWL, line=dict(color=C_GREY, width=1.4, dash="dot"),
                  annotation_text="S(NWL)", annotation_font=dict(color=C_GREY, size=10))
    fig.add_hline(y=S_LWL, line=dict(color=C_RED, width=1.4, dash="dot"),
                  annotation_text="S(LWL)", annotation_font=dict(color=C_RED, size=10))
    if rec_days is not None and 0 < rec_days <= len(traj):
        fig.add_vline(x=rec_days, line=dict(color=C_YELLOW, width=2, dash="dash"),
                      annotation_text=f"Target tercapai ≈ hari ke-{rec_days:.0f}",
                      annotation_font=dict(color=C_YELLOW, size=10))
    fig.update_xaxes(title_text="Hari ke depan")
    fig.update_yaxes(title_text="Tampungan (juta m³)")
    show_chart(style_plot(fig, 400), key="rc_traj")

    # ---- Gauge & strategi ----
    c = st.columns([1, 2])
    with c[0]:
        show_chart(gauge_chart(min(srr * 100, 130), "Storage Recovery Ratio (%)",
                               [(0, 60, C_RED), (60, 85, C_ORANGE),
                                (85, 100, C_YELLOW), (100, 130, C_GREEN)],
                               " %", 300, 130), key="rc_g")
    with c[1]:
        st.markdown("**Recommended Release Strategy — Recovery Mode**")
        strat = pd.DataFrame({
            "Komponen": ["Air baku / domestik", "Irigasi", "Environmental flow",
                         "PLTA", "Release non-esensial", "Incoming water", "FEWS"],
            "Strategi Recovery": [
                "MAINTAIN — dijamin penuh, tidak boleh dikurangi",
                "TERKONTROL — dilayani berbasis sensitivitas tanaman",
                "MAINTAIN — kebutuhan lingkungan hilir dijaga",
                "DIBATASI — disesuaikan agar tidak menghambat pengisian",
                "DIBATASI / DITUNDA — perikanan & pariwisata",
                "DIPERTAHANKAN sebanyak mungkin untuk refill tampungan",
                "TETAP AKTIF — keselamatan tetap hard constraint",
            ],
            "Status": ["AKTIF" if active else "STANDBY"] * 7,
        })
        show_table(strat)

    # ---- Logika recovery ----
    with st.expander("📘 Logika & Formula Reservoir Recovery Mode"):
        st.markdown(f"""
**Storage Recovery Ratio (SRR)** — digunakan bentuk dengan acuan tampungan minimum
karena tampungan mati tidak dapat dimanfaatkan:

$$SRR = \\frac{{S(t) - S_{{LWL}}}}{{S_{{target}} - S_{{LWL}}}}
= \\frac{{{ctx['storage_mcm']:.3f} - {S_LWL:.3f}}}{{{s_target:.3f} - {S_LWL:.3f}}}
= {srr:.3f}$$

Bentuk sederhana $SRR = S(t)/S_{{target}}$ juga tersedia pada fungsi
`calculate_storage_recovery_ratio(..., use_min_reference=False)`.

**Estimated Recovery Time:**

$$t = \\frac{{S_{{target}} - S(t)}}{{dS/dt}}, \\quad
dS/dt = Q_{{in}} - Q_{{out}} + P \\cdot A - E \\cdot A - L$$

Laju pengisian bersih saat ini = **{ctx['recovery_rate']:+.4f} juta m³/hari**.
{ctx['recovery_note']}

**Flood Safety Margin** = HWL − elevasi maksimum hasil routing prakiraan
= {DAM['HWL']:.2f} − {ctx['routing']['max_elevation_m']:.2f} = **{ctx['margin']:.2f} m**

**Logika transisi mode:**

```
JIKA drought membaik DAN target storage belum tercapai (SRR < 1,0)
    → OPERATING MODE = RECOVERY MODE
    → incoming water dipertahankan sebanyak mungkin
    → air baku maintained, irigasi terkontrol
    → release non-esensial dibatasi, FEWS tetap aktif

JIKA target seasonal storage tercapai (SRR ≥ 1,0) DAN flood risk aman
    → OPERATING MODE = NORMAL OPERATION
```
        """)


# =============================================================================
# 31. HALAMAN 08 — PERFORMANCE
# =============================================================================
def compute_performance_demo(df, ctx):
    """
    Perbandingan kinerja BASELINE vs JALA RAKSA.

    ⚠️ DEMONSTRATION ONLY — REPLACE WITH SIMULATION RESULTS.
    Angka di bawah TIDAK BOLEH dijadikan klaim peningkatan kinerja. Nilai final
    WAJIB berasal dari simulasi operasi waduk aktual dengan bendungan, hidrologi,
    kebutuhan, kapasitas outlet, pelimpah, tampungan awal, dan periode simulasi
    yang SAMA. Perbedaan hanya boleh pada strategi operasi.
    """
    z = ctx["zones"]
    served = float(z.loc[z["Fulfillment (%)"] >= 75, "Area (ha)"].sum())
    prod = float((z["Production Retained (%)"] * z["Area (ha)"]).sum() /
                 max(z["Area (ha)"].sum(), 1))
    min_stor = float(df["storage_percent"].min())
    crit_days = int((df["storage_percent"] < DAM["CRITICAL_STORAGE_PCT"]).sum())
    max_elev = float(df["reservoir_level"].max())

    rows = [
        # (indikator, baseline, jala raksa, satuan, arah baik: 'up'/'down')
        ("Water Supply Reliability", 88.0, 97.5, "%", "up"),
        ("Irrigation Reliability", 71.0, 84.0, "%", "up"),
        ("Total Water Deficit", 12.40, 7.10, "juta m³", "down"),
        ("Maximum Water Deficit", 1.35, 0.78, "m³/dtk", "down"),
        ("Critical Deficit Days", 46.0, 24.0, "hari", "down"),
        ("Agricultural Water Deficit", 9.80, 5.40, "juta m³", "down"),
        ("Area Adequately Served", max(served * 0.78, 1.0), served, "ha", "up"),
        ("Expected Crop Production Retained", max(prod - 11.0, 0.0), prod, "%", "up"),
        ("Water Productivity", 0.92, 1.18, "kg/m³", "up"),
        ("Minimum Reservoir Storage", max(min_stor - 6.5, 0.0), min_stor, "%", "up"),
        ("Critical Storage Days", crit_days + 13.0, float(crit_days), "hari", "down"),
        ("Recovery Time", 74.0, 48.0, "hari", "down"),
        ("Storage Recovery Ratio", max(ctx["srr"] - 0.14, 0.0), ctx["srr"], "-", "up"),
        ("Maximum Reservoir Elevation", max_elev + 0.28, max_elev, "m dpl", "down"),
        ("Peak Downstream Release", 31.5, 22.4, "m³/dtk", "down"),
        ("Maximum Spill", 18.6, 11.2, "m³/dtk", "down"),
        ("Flood Safety Margin", max(ctx["margin"] - 0.42, 0.0), ctx["margin"], "m", "up"),
        ("Hydropower Generation", 2.15, 2.46, "GWh", "up"),
    ]
    recs = []
    for nm, b, j, unit, direction in rows:
        diff = j - b
        if abs(b) > 1e-9:
            imp = (j - b) / abs(b) * 100.0
            if direction == "down":
                imp = -imp
        else:
            imp = 0.0
        recs.append({
            "Indicator": nm, "Baseline": round(b, 3), "JALA RAKSA": round(j, 3),
            "Satuan": unit, "Difference": round(diff, 3),
            "Improvement (%)": round(imp, 1),
            "Arah Baik": "Lebih tinggi" if direction == "up" else "Lebih rendah",
        })
    return pd.DataFrame(recs)


def page_performance(ctx, df):
    section("08 — Performance",
            "Perbandingan kinerja Baseline Reservoir Operation vs JALA RAKSA.")

    alert(
        "<b>🚨 DEMONSTRATION ONLY — REPLACE WITH SIMULATION RESULTS.</b><br><br>"
        "Seluruh angka pada halaman ini adalah <b>nilai demonstrasi</b> untuk "
        "memperlihatkan format pembuktian inovasi. JALA RAKSA <b>TIDAK mengklaim</b> "
        "peningkatan kinerja sebesar persentase mana pun. Angka final <b>WAJIB</b> "
        "berasal dari simulasi operasi waduk aktual dengan bendungan, hidrologi, "
        "kebutuhan, kapasitas outlet, pelimpah, tampungan awal, dan periode simulasi "
        "yang <b>identik</b> — perbedaan hanya pada strategi operasi.", "danger")

    perf = compute_performance_demo(df, ctx)

    # ---- Kartu perbandingan utama ----
    section("Comparison Cards — Indikator Kunci")
    key_idx = ["Water Supply Reliability", "Irrigation Reliability",
               "Expected Crop Production Retained", "Recovery Time"]
    cc = st.columns(4)
    for i, nm in enumerate(key_idx):
        row = perf[perf["Indicator"] == nm].iloc[0]
        col = C_GREEN if row["Improvement (%)"] > 0 else C_ORANGE
        with cc[i]:
            st.markdown(
                f"""
                <div class="jr-card">
                    <div class="jr-card-label">{nm}</div>
                    <div style="display:flex;justify-content:space-between;margin-top:8px">
                        <div><div style="font-size:.68rem;color:{C_MUTED}">BASELINE</div>
                        <div style="font-size:1.15rem;font-weight:800;color:{C_GREY}">
                        {row['Baseline']:.2f}</div></div>
                        <div style="align-self:center;color:{C_CYAN};font-size:1.2rem">→</div>
                        <div style="text-align:right"><div style="font-size:.68rem;color:{C_MUTED}">
                        JALA RAKSA</div>
                        <div style="font-size:1.15rem;font-weight:800;color:{C_CYAN}">
                        {row['JALA RAKSA']:.2f}</div></div>
                    </div>
                    <div class="jr-card-foot" style="color:{col};font-weight:700">
                        {row['Improvement (%)']:+.1f}% · {row['Satuan']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- Bar chart & radar ----
    c = st.columns([3, 2])
    with c[0]:
        st.markdown("**Bar Chart — Perbandingan Indikator (dinormalisasi)**")
        sub = perf.head(10).copy()
        maxv = np.maximum(np.abs(sub["Baseline"]), np.abs(sub["JALA RAKSA"]))
        maxv = np.where(maxv < 1e-9, 1.0, maxv)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=sub["Indicator"], x=sub["Baseline"] / maxv * 100,
                             name="Baseline", orientation="h", marker_color=C_GREY))
        fig.add_trace(go.Bar(y=sub["Indicator"], x=sub["JALA RAKSA"] / maxv * 100,
                             name="JALA RAKSA", orientation="h", marker_color=C_CYAN))
        fig.update_xaxes(title_text="Nilai relatif (% terhadap nilai terbesar tiap indikator)")
        fig.update_layout(barmode="group")
        show_chart(style_plot(fig, 460), key="pf_bar")
    with c[1]:
        st.markdown("**Radar Chart — Profil Kinerja**")
        cats = ["Water Supply\nReliability", "Irrigation\nReliability",
                "Crop Production\nRetained", "Storage\nRecovery", "Flood\nSafety"]
        base = [88.0, 71.0,
                float(perf.loc[perf["Indicator"] == "Expected Crop Production Retained",
                               "Baseline"].iloc[0]),
                float(np.clip(perf.loc[perf["Indicator"] == "Storage Recovery Ratio",
                                       "Baseline"].iloc[0] * 100, 0, 100)),
                float(np.clip(perf.loc[perf["Indicator"] == "Flood Safety Margin",
                                       "Baseline"].iloc[0] / 2.5 * 100, 0, 100))]
        jala = [97.5, 84.0,
                float(perf.loc[perf["Indicator"] == "Expected Crop Production Retained",
                               "JALA RAKSA"].iloc[0]),
                float(np.clip(ctx["srr"] * 100, 0, 100)),
                float(np.clip(ctx["margin"] / 2.5 * 100, 0, 100))]
        show_chart(radar_chart(cats, base, jala, 460), key="pf_radar")

    # ---- Tabel performa ----
    section("Performance Table")
    show_table(perf[["Indicator", "Baseline", "JALA RAKSA", "Satuan",
                     "Difference", "Improvement (%)", "Arah Baik"]])
    st.caption("⚠️ DEMONSTRATION ONLY — REPLACE WITH SIMULATION RESULTS. "
               "Kolom Improvement (%) sudah memperhitungkan arah perbaikan "
               "(indikator seperti defisit dan recovery time lebih baik bila lebih kecil).")

    with st.expander("📘 Syarat Pembuktian Inovasi yang Sah"):
        st.markdown(f"""
**Baseline WAJIB menggunakan bendungan yang sama.** Parameter yang harus identik:

| Parameter | Keterangan |
|---|---|
| Reservoir geometry | kurva elevasi-kapasitas yang sama |
| LWL / NWL / HWL / FWL | batas desain identik |
| Inflow | deret hidrologi yang sama |
| Demand | kebutuhan air yang sama |
| Outlet capacity | {DAM['OUTLET_CAPACITY']:.1f} m³/detik |
| Spillway | {DAM['SPILLWAY_CAPACITY']:.1f} m³/detik |
| Initial storage | tampungan awal sama |
| Periode simulasi | rentang waktu sama |

**Perbedaan hanya pada strategi operasi:**

| Baseline | JALA RAKSA |
|---|---|
| Seasonal operating rule | Early drought conservation |
| Standard irrigation allocation | Adaptive hedging |
| Conventional drought response | Crop-sensitive allocation |
| DEWS/FEWS dasar | Forecast-constrained pre-release |
| — | Reservoir recovery operation |

**Skenario uji minimum yang harus dijalankan:** tahun normal, kekeringan sedang,
kekeringan berat, kekeringan panjang, kekeringan → hujan normal, kekeringan →
inflow tinggi, forecast terlalu tinggi, forecast terlalu rendah, gangguan
sensor/data, dan banjir desain.
        """)


# =============================================================================
# 32. HALAMAN 09 — IoT & SENSOR HEALTH
# =============================================================================
def page_iot(ctx, df, sensors):
    section("09 — IoT & Sensor Health",
            "Kesehatan jaringan sensor IoT sebagai enabling technology akuisisi data.")
    demo_tag()
    st.markdown("")

    n_total = len(sensors)
    n_online = int((sensors["Status"] == "ONLINE").sum())
    n_warn = int((sensors["Status"] == "WARNING").sum())
    n_off = int((sensors["Status"] == "OFFLINE").sum())
    completeness = 100.0 * n_online / max(n_total, 1)

    k = st.columns(4)
    with k[0]:
        col = C_GREEN if n_online >= n_total * 0.9 else C_ORANGE if n_online >= n_total * 0.7 else C_RED
        kpi_card("IoT Sensor Network Health", f"{n_online} / {n_total}", "sensor online",
                 f"{completeness:.1f}% jaringan aktif", col)
    with k[1]:
        kpi_card("Data Completeness", f"{ctx['validation']['completeness']:.1f}", "%",
                 "Kelengkapan deret data", C_CYAN)
    with k[2]:
        kpi_card("Sensor Warning / Offline", f"{n_warn} / {n_off}", "sensor",
                 "Perlu pemeriksaan lapangan",
                 C_GREEN if (n_warn + n_off) == 0 else C_ORANGE)
    with k[3]:
        kpi_card("Last Synchronization",
                 dt.datetime.now().strftime("%H:%M:%S"), "",
                 dt.datetime.now().strftime("%d %B %Y"), C_PURPLE)

    st.markdown("")

    if n_off > 0:
        alert(f"<b>⚠️ {n_off} sensor OFFLINE.</b> Sensor: "
              f"<b>{', '.join(sensors.loc[sensors['Status']=='OFFLINE','ID'].tolist())}</b>. "
              f"Periksa catu daya, modul komunikasi, dan gateway telemetri. "
              f"Bila sensor utama (level waduk / inflow / hujan) offline, sistem akan "
              f"masuk <b>FAIL-SAFE MODE</b>.", "warn")

    # ---- Donut & bar ----
    c = st.columns([2, 3])
    with c[0]:
        fig = go.Figure(go.Pie(
            labels=["ONLINE", "WARNING", "OFFLINE"], values=[n_online, n_warn, n_off],
            hole=0.58, marker=dict(colors=[C_GREEN, C_ORANGE, C_RED],
                                   line=dict(color=C_BG, width=3)),
            textinfo="value+percent", textfont=dict(size=12)))
        fig.update_layout(
            height=330, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT),
            margin=dict(l=10, r=10, t=40, b=10),
            title="Status Jaringan Sensor",
            annotations=[dict(text=f"<b>{completeness:.0f}%</b><br>ONLINE",
                              x=0.5, y=0.5, font=dict(size=17, color=C_TEXT),
                              showarrow=False)],
            legend=dict(orientation="h", y=-0.12, x=0.1))
        show_chart(fig, key="io_pie")
    with c[1]:
        grp = sensors.groupby(["Sensor", "Status"]).size().reset_index(name="n")
        fig = go.Figure()
        for stt, col in [("ONLINE", C_GREEN), ("WARNING", C_ORANGE), ("OFFLINE", C_RED)]:
            sub = grp[grp["Status"] == stt]
            if len(sub):
                fig.add_trace(go.Bar(y=sub["Sensor"], x=sub["n"], name=stt,
                                     orientation="h", marker_color=col))
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title_text="Jumlah sensor")
        show_chart(style_plot(fig, 330, "Status per Jenis Sensor"), key="io_bar")

    # ---- Tabel sensor ----
    section("Tabel Sensor IoT")
    def _style_status(v):
        return v
    show_table(sensors)
    st.caption("Kolom Status: ONLINE (normal) · WARNING (komunikasi lemah / data suspect) · "
               "OFFLINE (tidak mengirim data). Geser tabel untuk melihat seluruh kolom.")

    # ---- Peran sensor ----
    with st.expander("📡 Peran Setiap Sensor dalam JALA RAKSA"):
        role = pd.DataFrame({
            "Sensor": ["Automatic Rain Gauge", "Reservoir Level Sensor",
                       "Upstream River Level / Discharge", "Outlet Flow Meter",
                       "Gate Position Sensor", "Soil Moisture Sensor",
                       "Automatic Weather Station"],
            "Data": ["Curah hujan", "Elevasi waduk", "Inflow", "Release / outflow",
                     "Bukaan pintu", "Kelembapan tanah", "Suhu, RH, hujan, radiasi"],
            "Kegunaan dalam JALA RAKSA": [
                "Masukan DEWS & FEWS serta hujan efektif irigasi",
                "Monitoring storage melalui kurva elevasi-kapasitas",
                "Neraca air dan verifikasi prakiraan inflow",
                "Monitoring realisasi operasi dan neraca air",
                "Validasi operasi saja — BUKAN kendali otomatis",
                "Smart irrigation & penilaian prioritas zona",
                "Dukungan perhitungan ET0 dan forecast",
            ],
        })
        show_table(role)
        st.markdown(
            "Penempatan *soil moisture sensor* dilakukan **secara representatif pada "
            "irrigation management zones**, bukan setiap hektare, agar biaya dan "
            "pemeliharaan tetap wajar.\n\n"
            "⚠️ **Gate Position Sensor hanya untuk validasi operasi.** JALA RAKSA "
            "**tidak** mengirim perintah ke PLC/SCADA/gate actuator.")

    # ---- Validasi data ----
    section("Hasil Validasi Data — validate_data()")
    v = ctx["validation"]
    vcol = {"OK": C_GREEN, "WARNING": C_ORANGE, "CRITICAL": C_RED}[v["status"]]
    vc = st.columns(3)
    with vc[0]:
        status_box("DATA VALIDATION", v["status"],
                   f"{len(v['issues'])} isu terdeteksi", vcol)
    with vc[1]:
        status_box("DATA COMPLETENESS", f"{v['completeness']:.1f}%",
                   "Terhadap seluruh sel data", C_CYAN)
    with vc[2]:
        status_box("FAIL-SAFE", "ACTIVE" if ctx["failsafe"] else "STANDBY",
                   "Kembali ke baseline rule" if ctx["failsafe"] else "Data memenuhi syarat",
                   C_RED if ctx["failsafe"] else C_GREEN)

    st.markdown("")
    if v["issues"]:
        show_table(pd.DataFrame(v["issues"], columns=["Tingkat", "Temuan"]))
    else:
        alert("✅ Seluruh pemeriksaan validasi data terlampaui: tidak ada nilai kosong, "
              "tidak ada debit negatif, tampungan dan elevasi berada dalam rentang fisik, "
              "stempel waktu konsisten, dan data tidak basi.", "ok")

    with st.expander("🔍 Daftar Pemeriksaan validate_data()"):
        st.markdown("""
| # | Pemeriksaan | Konsekuensi bila gagal |
|---|---|---|
| 1 | Missing values | WARNING / CRITICAL (bergantung kelengkapan) |
| 2 | Debit negatif (inflow/outflow/hujan) | CRITICAL |
| 3 | Storage < 0 | CRITICAL |
| 4 | Storage > kapasitas maksimum (di atas HWL) | CRITICAL |
| 5 | Elevasi muka air di luar rentang fisik | WARNING |
| 6 | Stempel waktu tidak valid / duplikat / tidak urut | CRITICAL / WARNING |
| 7 | Data basi (> 24 jam / > 72 jam) | WARNING / CRITICAL |
| 8 | Prakiraan tidak tersedia | WARNING |
| 9 | Inflow ekstrem melebihi kapasitas pelimpah desain | WARNING |

Status **CRITICAL** mengaktifkan **FAIL-SAFE MODE**: sistem berhenti memberikan
keluaran optimasi dan merekomendasikan kembali ke *Approved Baseline Operating Rule*.
        """)


# =============================================================================
# 33. HALAMAN 10 — DECISION CENTER
# =============================================================================
def page_decision_center(ctx, df):
    section("10 — Decision Center",
            "Human-in-the-loop operator decision support. Sistem hanya memberikan "
            "rekomendasi; keputusan akhir tetap pada operator.")

    alert(
        "<b>⚠️ SIMULASI DECISION-SUPPORT LOG.</b> Halaman ini <b>tidak</b> mengirim "
        "perintah apa pun ke PLC, SCADA, maupun gate actuator. Seluruh keputusan hanya "
        "dicatat pada riwayat sesi sebagai simulasi pendukung keputusan.", "warn")

    rec = ctx["recommendation"]

    # ---- Panel rekomendasi ----
    section("JALA RAKSA OPERATION RECOMMENDATION")
    items = [
        ("DEWS Status", rec["dews"], STATUS_COLOR[rec["dews"]]),
        ("Reservoir Stress", f"{rec['stress']:.1f} — {rec['stress_class']}",
         STATUS_COLOR[rec["stress_class"]]),
        ("Operating Mode", rec["mode"], MODE_COLOR.get(rec["mode"], C_CYAN)),
        ("Hedging Level", rec["hedging_level"], C_YELLOW),
        ("Domestic Water", rec["domestic"], C_GREEN),
        ("Irrigation", rec["irrigation"],
         C_GREEN if rec["irrigation"] == "NORMAL" else C_ORANGE),
        ("Environmental Flow", rec["environment"], "#14b8a6"),
        ("Priority Zones", ", ".join(rec["priority_zones"]) if rec["priority_zones"] else "—",
         C_RED if rec["priority_zones"] else C_GREY),
        ("Non-Critical Irrigation",
         "REDUCE / DELAY" if ctx["hedging"]["f_irrigation"] < 0.999 else "NORMAL",
         C_ORANGE if ctx["hedging"]["f_irrigation"] < 0.999 else C_GREEN),
        ("FEWS", rec["fews"], STATUS_COLOR[rec["fews"]]),
        ("Forecast Inflow",
         ("LOW" if ctx["forecast_inflow"] < DAM["MEAN_ANNUAL_INFLOW"] * 0.7 else
          "HIGH" if ctx["forecast_inflow"] > DAM["MEAN_ANNUAL_INFLOW"] * 1.6 else "NORMAL"),
         C_CYAN),
        ("Pre-Release", rec["prerelease"],
         C_ORANGE if ctx["prerelease"]["required"] else C_GREEN),
        ("Recovery", rec["recovery"], C_BLUE),
        ("Dam Safety", ctx["safety_status"], STATUS_COLOR[ctx["safety_status"]]),
    ]
    cols = st.columns(2)
    for i, (lbl, val, col) in enumerate(items):
        with cols[i % 2]:
            st.markdown(
                f"""<div style="display:flex;justify-content:space-between;
                    align-items:center;background:{C_CARD};border:1px solid {C_LINE};
                    border-left:4px solid {col};border-radius:10px;
                    padding:9px 14px;margin-bottom:7px">
                    <span style="color:{C_MUTED};font-size:.82rem;font-weight:600">{lbl}</span>
                    <b style="color:{col};font-size:.90rem;text-align:right">{val}</b>
                    </div>""", unsafe_allow_html=True)

    st.markdown("")
    body = "".join(f"<li>{ln}</li>" for ln in rec["narrative"])
    alert(f"<b>Uraian Rekomendasi</b><ul style='margin:8px 0 0 0'>{body}</ul>", "info")

    with st.expander("🔎 Dasar penetapan Operating Mode"):
        st.markdown("\n".join(f"- {r}" for r in ctx["mode_reasons"]))

    st.markdown("---")

    # ---- OPERATOR DECISION ----
    section("OPERATOR DECISION", "Pilih tindakan terhadap rekomendasi di atas.")
    operator = st.text_input("Nama / ID Operator", value="Operator Demo",
                             key="operator_name")

    b = st.columns(3)
    with b[0]:
        if wide_button("✅  APPROVE", key="btn_approve", type="primary"):
            log_decision(rec, "APPROVE", "Rekomendasi disetujui tanpa perubahan.", operator)
            st.session_state.modify_open = False
            st.session_state.reject_open = False
            st.success("✅ Rekomendasi **DISETUJUI** dan tercatat pada riwayat keputusan. "
                       "(Simulasi — tidak ada perintah dikirim ke gate.)")
    with b[1]:
        if wide_button("✏️  MODIFY", key="btn_modify"):
            st.session_state.modify_open = True
            st.session_state.reject_open = False
    with b[2]:
        if wide_button("❌  REJECT", key="btn_reject"):
            st.session_state.reject_open = True
            st.session_state.modify_open = False

    # ---- Form MODIFY ----
    if st.session_state.modify_open:
        st.markdown("")
        with st.form("form_modify"):
            st.markdown("#### ✏️ Modifikasi Rekomendasi Operator")
            m1, m2 = st.columns(2)
            with m1:
                m_mode = st.selectbox(
                    "Operating Mode yang dipilih operator",
                    list(MODE_COLOR.keys()),
                    index=list(MODE_COLOR.keys()).index(rec["mode"]))
                m_irr = st.slider("Alokasi irigasi (m³/detik)", 0.0,
                                  float(DAM["Q_IRRIGATION_MAX"]),
                                  float(min(ctx["alloc"]["q_irrigation"],
                                            DAM["Q_IRRIGATION_MAX"])), 0.05)
                m_dom = st.selectbox("Air baku / domestik",
                                     ["MAINTAIN", "REDUCE (perlu justifikasi)"], 0)
            with m2:
                m_pre = st.number_input("Pre-release yang ditetapkan (m³/detik)",
                                        0.0, float(DAM["OUTLET_CAPACITY"]),
                                        float(ctx["prerelease"]["prerelease_q_cms"]), 0.1)
                m_env = st.selectbox("Environmental flow", ["MAINTAIN", "SAFE LIMIT"], 0)
                m_dur = st.number_input("Durasi penerapan (hari)", 1, 30,
                                        max(int(ctx["prerelease"]["duration_days"]), 1))
            m_note = st.text_area(
                "Catatan / justifikasi operator (wajib)",
                placeholder="Contoh: alokasi irigasi Zone C ditambah karena laporan "
                            "lapangan menunjukkan tanaman padi memasuki fase pembungaan "
                            "lebih cepat dari data sistem.")
            submitted = st.form_submit_button("💾 Simpan Modifikasi")
            if submitted:
                if not m_note.strip():
                    st.error("Catatan/justifikasi operator wajib diisi untuk keputusan MODIFY.")
                else:
                    note = (f"MODE={m_mode}; Irigasi={m_irr:.2f} m³/dtk; Air baku={m_dom}; "
                            f"Env={m_env}; Pre-release={m_pre:.2f} m³/dtk × {m_dur} hari. "
                            f"Justifikasi: {m_note.strip()}")
                    log_decision(rec, "MODIFY", note, operator)
                    st.session_state.modify_open = False
                    st.success("✏️ Modifikasi operator **TERSIMPAN** pada riwayat keputusan. "
                               "(Simulasi — tidak ada perintah dikirim ke gate.)")

    # ---- Form REJECT ----
    if st.session_state.reject_open:
        st.markdown("")
        with st.form("form_reject"):
            st.markdown("#### ❌ Penolakan Rekomendasi")
            r_reason = st.selectbox("Alasan penolakan", [
                "Data sensor diragukan / tidak sesuai kondisi lapangan",
                "Bertentangan dengan SOP / rule curve yang berlaku",
                "Terdapat instruksi dari instansi berwenang",
                "Kondisi lapangan berbeda dari asumsi model",
                "Prakiraan cuaca dinilai tidak andal",
                "Lainnya",
            ])
            r_note = st.text_area("Catatan operator (wajib)",
                                  placeholder="Jelaskan alasan penolakan dan tindakan "
                                              "operasi yang akan diambil sebagai gantinya.")
            submitted = st.form_submit_button("💾 Simpan Penolakan")
            if submitted:
                if not r_note.strip():
                    st.error("Catatan operator wajib diisi untuk keputusan REJECT.")
                else:
                    log_decision(rec, "REJECT", f"{r_reason}. {r_note.strip()}", operator)
                    st.session_state.reject_open = False
                    st.warning("❌ Rekomendasi **DITOLAK** dan tercatat pada riwayat keputusan. "
                               "Operasi mengikuti keputusan operator / SOP yang berlaku.")

    st.markdown("---")

    # ---- Riwayat keputusan ----
    section("Operator Decision History")
    hist = st.session_state.decision_history
    if not hist:
        st.info("Belum ada keputusan yang tercatat pada sesi ini. "
                "Gunakan tombol APPROVE / MODIFY / REJECT di atas.")
    else:
        hc = st.columns(4)
        n_app = sum(1 for h in hist if h["Keputusan"] == "APPROVE")
        n_mod = sum(1 for h in hist if h["Keputusan"] == "MODIFY")
        n_rej = sum(1 for h in hist if h["Keputusan"] == "REJECT")
        with hc[0]:
            kpi_card("Total Keputusan", f"{len(hist)}", "catatan", "Sesi berjalan", C_CYAN)
        with hc[1]:
            kpi_card("APPROVE", f"{n_app}", "", f"{100*n_app/len(hist):.0f}% dari total", C_GREEN)
        with hc[2]:
            kpi_card("MODIFY", f"{n_mod}", "", f"{100*n_mod/len(hist):.0f}% dari total", C_YELLOW)
        with hc[3]:
            kpi_card("REJECT", f"{n_rej}", "", f"{100*n_rej/len(hist):.0f}% dari total", C_RED)

        st.markdown("")
        hdf = pd.DataFrame(hist)
        show_table(hdf)

        d = st.columns(2)
        with d[0]:
            csv = hdf.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Unduh Riwayat Keputusan (CSV)", csv,
                               file_name=f"jala_raksa_decision_log_"
                                         f"{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", key="dl_hist")
        with d[1]:
            if st.button("🗑️ Bersihkan Riwayat Sesi", key="btn_clear"):
                st.session_state.decision_history = []
                st.rerun()


# =============================================================================
# 34. HALAMAN 11 — ABOUT JALA RAKSA
# =============================================================================
def page_about(ctx):
    section("11 — About JALA RAKSA",
            "Integrated Drought-to-Recovery Reservoir Operation.")

    st.markdown(f"""
### Apa itu JALA RAKSA?

**JALA RAKSA** adalah *Decision Support System* (DSS) operasi bendungan adaptif yang
mengelola **satu siklus operasi penuh dari kekeringan hingga pemulihan tampungan**
melalui konservasi dini, alokasi air berbasis kebutuhan pangan, antisipasi inflow
tinggi, dan operasi pemulihan waduk.

Sistem ini **tidak menggantikan operator** dan **tidak membuka pintu bendungan secara
otomatis**. JALA RAKSA mengolah data hidrologi, kondisi waduk, kebutuhan air, kondisi
pertanian, serta prakiraan hujan dan inflow menjadi rekomendasi operasi yang
selanjutnya dapat di-**APPROVE**, **MODIFY**, atau **REJECT** oleh operator.

### Tujuan Utama

1. Mendeteksi tekanan kekeringan **sebelum** tampungan mencapai kondisi kritis
2. Melakukan konservasi air lebih awal melalui *adaptive drought hedging*
3. Menjaga kebutuhan air baku dan kebutuhan esensial lainnya
4. Mengalokasikan air irigasi terbatas berdasarkan risiko kehilangan produksi tanaman
5. Menentukan perlu/tidaknya *pre-release* ketika inflow tinggi diprediksi datang
6. Mempertahankan incoming water setelah kekeringan tanpa mengurangi keamanan banjir
7. Mempercepat pemulihan tampungan menuju target operasi
8. Meningkatkan *reliability* dan *resilience* operasi bendungan multipurpose
    """)

    st.markdown("---")
    section("Filosofi: CONSERVE → ALLOCATE → ANTICIPATE → RECOVER")
    stage_flow_diagram(active_stage(ctx["mode"]))
    st.markdown("")

    fc = st.columns(4)
    phil = [
        ("CONSERVE", C_YELLOW, "Jangan menunggu waduk kritis untuk mulai menghemat. "
                               "Penghematan kecil yang terkendali sejak dini lebih baik "
                               "daripada kegagalan pelayanan besar di kemudian hari."),
        ("ALLOCATE", C_GREEN, "Ketika air terbatas, arahkan air pada kebutuhan yang "
                              "memberikan dampak paling penting — khususnya sensitivitas "
                              "tanaman dan produksi pangan."),
        ("ANTICIPATE", C_CYAN, "Ketika inflow tinggi akan datang, jangan otomatis membuang "
                               "air; keluarkan hanya bila benar-benar diperlukan untuk "
                               "keselamatan."),
        ("RECOVER", C_BLUE, "Setelah kekeringan, manfaatkan incoming water untuk memulihkan "
                            "tampungan sampai kembali pada target operasi secara aman."),
    ]
    for i, (nm, col, desc) in enumerate(phil):
        with fc[i]:
            st.markdown(
                f"""<div class="jr-card" style="border-color:{col}">
                <div class="jr-card-value" style="color:{col};font-size:1.15rem">{nm}</div>
                <div style="color:{C_MUTED};font-size:.80rem;margin-top:8px;line-height:1.5">
                {desc}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    section("Diagram Alur Novelty JALA RAKSA")
    flow = ["DROUGHT DETECTION", "EARLY CONSERVATION", "CROP-SENSITIVE ALLOCATION",
            "FORECAST-CONSTRAINED PRE-RELEASE", "RESERVOIR RECOVERY", "NORMAL OPERATION"]
    colors = [C_ORANGE, C_YELLOW, C_GREEN, C_CYAN, C_BLUE, "#22c55e"]
    html = ""
    for i, (f, c) in enumerate(zip(flow, colors)):
        html += (f'<div style="background:{c}1f;border:1px solid {c};border-left:6px solid {c};'
                 f'border-radius:12px;padding:11px 18px;margin:5px 0;color:{C_TEXT};'
                 f'font-weight:700;letter-spacing:.4px">{i+1}. {f}</div>')
        if i < len(flow) - 1:
            html += f'<div style="text-align:center;color:{C_CYAN};font-size:1.15rem">↓</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    section("Komponen Sistem & Posisinya")
    show_table(pd.DataFrame({
        "Komponen": ["IoT", "Database", "DEWS", "Fuzzy Mamdani", "Adaptive Hedging",
                     "Crop-Sensitive Allocation", "FEWS", "Reservoir Routing",
                     "Pre-Release Algorithm", "Recovery Mode", "Dashboard", "Operator"],
        "Fungsi": ["Akuisisi data", "Penyimpanan data", "Peringatan kekeringan",
                   "Penilaian reservoir stress", "Konservasi air",
                   "Alokasi air pertanian", "Prakiraan inflow tinggi",
                   "Evaluasi respons storage", "Penentuan release minimum",
                   "Pemulihan tampungan", "Antarmuka operator",
                   "Pengambil keputusan akhir"],
        "Posisi": ["Enabling technology", "Enabling technology", "Requirement lomba",
                   "Decision-support method", "Inovasi 1", "Inovasi 2",
                   "Requirement lomba", "Metode standar", "Inovasi 3", "Inovasi 4",
                   "Antarmuka", "Human-in-the-loop"],
    }))

    st.markdown("---")
    section("Empat Inovasi Utama")
    inov = [
        ("1. JALA RAKSA Early Conservation",
         "Menggunakan status kekeringan dan Reservoir Stress untuk memulai konservasi "
         "sebelum waduk memasuki kondisi kritis.",
         "Masalah yang diselesaikan: keterlambatan respons terhadap kekeringan."),
        ("2. JALA RAKSA Food-Smart Allocation",
         "Mengalokasikan air terbatas berdasarkan kebutuhan dan sensitivitas tanaman "
         "terhadap kekurangan air.",
         "Masalah yang diselesaikan: pengurangan air merata belum tentu menghasilkan "
         "dampak minimum terhadap produksi pangan."),
        ("3. JALA RAKSA Forecast-Constrained Pre-Release",
         "Menggunakan FEWS dan reservoir routing untuk menentukan apakah pre-release "
         "benar-benar diperlukan, lalu menghitung minimum necessary pre-release.",
         "Masalah yang diselesaikan: membuang terlalu banyak air setelah kekeringan, "
         "atau mempertahankan terlalu banyak air menjelang inflow tinggi."),
        ("4. JALA RAKSA Reservoir Recovery Mode",
         "Mengelola fase setelah kekeringan sampai tampungan kembali menuju target operasi.",
         "Masalah yang diselesaikan: operasi waduk tidak berhenti pada mitigasi drought "
         "atau flood, tetapi mengelola transisi menuju kondisi normal secara terencana."),
    ]
    for judul, isi, masalah in inov:
        with st.expander(judul, expanded=False):
            st.markdown(f"{isi}\n\n**{masalah}**")

    st.markdown("---")
    section("Posisi Novelty")
    alert(
        "<b>JALA RAKSA TIDAK mengklaim</b> bahwa IWRM, IoT, fuzzy logic, DEWS, FEWS, "
        "dashboard, early release, atau adaptive rule curve merupakan novelty tunggal — "
        "DEWS dan FEWS bahkan merupakan bagian dari requirement.<br><br>"
        "<b>Novelty JALA RAKSA berada pada INTEGRATED DROUGHT-TO-RECOVERY RESERVOIR "
        "OPERATION</b>, yaitu integrasi satu rangkaian keputusan operasi: deteksi "
        "kekeringan → konservasi dini → alokasi berbasis sensitivitas tanaman → "
        "pre-release berbasis prakiraan → pemulihan tampungan → operasi normal.", "info")

    st.markdown(f"""
### Human-in-the-loop & Fail-Safe

**Human-in-the-loop:** operator tetap menjadi pengambil keputusan akhir melalui
APPROVE / MODIFY / REJECT. Sistem tidak terhubung ke PLC/SCADA/gate actuator.

**Fail-safe:** apabila terjadi kegagalan sensor, komunikasi terputus, prakiraan tidak
tersedia, data tidak logis, atau kepercayaan model rendah, JALA RAKSA masuk
**FAIL-SAFE MODE** dan merekomendasikan kembali ke *Approved Baseline Operating Rule*
sampai kondisi data kembali valid.

**Keselamatan sebagai hard constraint:** seluruh optimasi tunduk pada batas keselamatan
bendungan. Prinsipnya *optimize benefits within safety limits*, bukan
*optimize benefits versus safety*.

### Kalimat Novelty Final

> Kebaruan JALA RAKSA terletak pada **integrasi operasi drought-to-recovery dalam satu
> Decision Support System bendungan multipurpose**. Sistem menghubungkan konservasi dini
> berdasarkan tekanan kekeringan, alokasi irigasi berbasis sensitivitas tanaman,
> evaluasi pre-release berbasis prakiraan dan reservoir routing, serta Reservoir Recovery
> Mode untuk mengembalikan tampungan menuju target operasi setelah kekeringan. Seluruh
> keputusan dibatasi oleh kondisi fisik, kebutuhan multi-purpose, dan persyaratan
> keselamatan bendungan.

### Selling Point

> *"Knowing when to conserve, where to allocate, when to release, and how to recover."*
>
> "Mengetahui kapan harus menghemat, ke mana air harus dialokasikan, kapan air perlu
> dilepas, dan bagaimana tampungan harus dipulihkan."
    """)

    st.markdown("---")
    section("Bagian yang WAJIB Dikalibrasi")
    show_table(pd.DataFrame({
        "Bagian Kode / Modul": [
            "DAM['CURVE_A'], DAM['CURVE_B']", "DAM elevasi LWL/NWL/FCL/HWL",
            "DAM['OUTLET_CAPACITY'], SPILLWAY_CAPACITY", "DAM['DOWNSTREAM_SAFE_FLOW']",
            "mf_storage / mf_inflow / mf_forecast / mf_dsr", "STRESS_MF & SEV_WEIGHT",
            "calculate_dews_status() — threshold guard", "calculate_hedging_level() — faktor release",
            "adaptive_operating_band()", "CROP_LIBRARY (Kc, Ky, sensitivitas fase)",
            "DAM['IRRIGATION_EFFICIENCY']", "calculate_crop_priority() — bobot 45/35/20",
            "forecast_inflow_scenarios() — respons hujan-aliran",
            "calculate_minimum_prerelease() — safety_factor", "route_forecast_storage() — koefisien pelimpah",
            "DAM['TARGET_SEASONAL_PCT']", "compute_performance_demo()",
        ],
        "Isi Saat Ini": [
            "Regresi kurva kapasitas (DEMO)", "Elevasi desain (DEMO)",
            "Kapasitas hidraulik (DEMO)", "Batas aman hilir (DEMO)",
            "Membership function fuzzy (DEMO)", "Term output & bobot keparahan (DEMO)",
            "Ambang 25/40/55%, tren −0,55%/hari (DEMO)", "0,90 / 0,70 / 0,45 (DEMO)",
            "Band 72–100 s.d. 35–70% (DEMO)", "Kc & Ky bergaya FAO-56/33 (DEMO)",
            "0,65 (DEMO)", "Bobot skor prioritas (DEMO)",
            "0,055 m³/dtk per mm & bentuk hidrograf (DEMO)",
            "1,15 (DEMO)", "38 × H^1,5 (DEMO)",
            "80% tampungan efektif (DEMO)", "SELURUHNYA nilai demonstrasi",
        ],
        "Sumber Kalibrasi yang Benar": [
            "Pengukuran topografi/bathimetri waduk",
            "Dokumen desain & sertifikasi bendungan",
            "Desain hidraulik outlet & pelimpah",
            "Kajian kapasitas sungai & keamanan hilir",
            "Data hidrologi historis + expert judgement",
            "Simulasi operasi waduk multi-tahun",
            "Analisis frekuensi kekeringan (SPI/SDI)",
            "Simulasi neraca air multi-tahun (reliability terbaik)",
            "Rule curve resmi & studi optimasi",
            "Kajian agronomi daerah irigasi setempat",
            "Audit efisiensi jaringan irigasi",
            "Fungsi kehilangan hasil (yield response) aktual",
            "Model hujan-aliran terkalibrasi (HEC-HMS/NRECA/Tank)",
            "Analisis ketidakpastian prakiraan",
            "Rating curve pelimpah hasil uji model fisik",
            "Studi optimasi operasi musiman",
            "Simulasi operasi waduk baseline vs JALA RAKSA",
        ],
    }))


# =============================================================================
# 35. SIDEBAR — NAVIGASI, MODE AKSES, DAN SUMBER DATA
# =============================================================================
PAGES = [
    "01 — Command Center",
    "02 — Reservoir Overview",
    "03 — DEWS & Conservation",
    "04 — Food-Smart Allocation",
    "05 — IWRM Multi-Purpose",
    "06 — FEWS & Flood Anticipation",
    "07 — Reservoir Recovery",
    "08 — Performance",
    "09 — IoT & Sensor Health",
    "10 — Decision Center",
    "11 — About JALA RAKSA",
]


def render_sidebar():
    """
    Sidebar: mode akses (public/operator), sumber data, navigasi, dan
    control panel mode simulasi interaktif.
    Mengembalikan dict pengaturan.
    """
    with st.sidebar:
        st.markdown(
            f"""<div style="text-align:center;padding:6px 0 12px">
            <div style="font-size:1.55rem;font-weight:900;color:{C_CYAN}">💧 JALA RAKSA</div>
            <div style="font-size:.72rem;color:{C_MUTED};margin-top:2px">
            Integrated Drought-to-Recovery<br>Reservoir Operation DSS</div></div>""",
            unsafe_allow_html=True)
        st.markdown("---")

        # ---- Mode akses ----
        st.markdown("**🔐 MODE AKSES**")
        access = st.radio(
            "Mode akses", ["PUBLIC MONITORING MODE", "OPERATOR DECISION SUPPORT MODE"],
            index=0, label_visibility="collapsed", key="access_mode")
        if access.startswith("PUBLIC"):
            st.caption("👥 Masyarakat umum dapat memantau kondisi waduk. "
                       "Keputusan operasi **tidak** tersedia pada mode ini.")
        else:
            st.caption("🛠️ Operator dapat memberikan APPROVE / MODIFY / REJECT "
                       "terhadap rekomendasi (simulasi decision support).")

        st.markdown("---")

        # ---- Sumber data ----
        st.markdown("**📡 DATA SOURCE / DATA MODE**")
        data_mode = st.selectbox(
            "Mode data",
            ["DEMO AUTO SIMULATION", "INTERACTIVE SIMULATION", "CSV UPLOAD"],
            index=0, label_visibility="collapsed", key="data_mode")

        uploaded = None
        n_days = 90
        seed = 2026
        if data_mode == "DEMO AUTO SIMULATION":
            n_days = st.slider("Panjang data demo (hari)", 30, 180, 90, 10, key="n_days")
            seed = st.number_input("Seed pembangkit data", 1, 99999, 2026, 1, key="seed")
            st.caption("⚠️ DEMONSTRATION DATA — dibangkitkan otomatis oleh Python "
                       "dengan pola: normal → kekeringan → kekeringan berat → "
                       "inflow naik → recovery → normal.")
        elif data_mode == "CSV UPLOAD":
            uploaded = st.file_uploader(
                "Unggah CSV data waduk", type=["csv"], key="csv_upload",
                help="Kolom minimum: datetime, storage atau reservoir_level, "
                     "inflow, outflow, rainfall. Kolom lain akan dilengkapi otomatis.")
            st.caption("Kolom yang dikenali: datetime, reservoir_level, storage, "
                       "inflow, outflow, rainfall, forecast_rainfall, forecast_inflow, "
                       "evaporation, domestic_demand, irrigation_demand, "
                       "environmental_flow, soil_moisture.")
        else:
            st.caption("🎛️ Gunakan panel kontrol di bawah untuk mengubah kondisi "
                       "hidrologi secara manual. Seluruh perhitungan diperbarui "
                       "**secara realtime**.")

        st.markdown("---")

        # ---- Navigasi ----
        st.markdown("**🧭 NAVIGASI**")
        page = st.radio("Halaman", PAGES, index=0,
                        label_visibility="collapsed", key="nav_page")

        sim_inputs = None
        if data_mode == "INTERACTIVE SIMULATION":
            sim_inputs = render_simulation_panel()

        st.markdown("---")
        st.markdown(
            f"""<div style="font-size:.70rem;color:{C_MUTED};line-height:1.5">
            <b style="color:{C_TEXT}">JALA RAKSA v{APP_VERSION}</b><br>
            Decision Support System — bukan sistem kendali otomatis.<br>
            Sistem tidak terhubung ke PLC/SCADA/gate actuator.<br><br>
            <span style="color:{C_ORANGE}">⚠️ Seluruh nilai adalah
            DEMONSTRATION / SIMULATION DATA.</span></div>""",
            unsafe_allow_html=True)

    return {"access": access, "data_mode": data_mode, "page": page,
            "uploaded": uploaded, "n_days": n_days, "seed": int(seed),
            "sim_inputs": sim_inputs}


# =============================================================================
# 36. CONTROL PANEL — INTERACTIVE SIMULATION MODE
# =============================================================================
def render_simulation_panel():
    """
    Panel kontrol simulasi interaktif.

    Setiap perubahan slider LANGSUNG memicu perhitungan ulang seluruh rantai:
      DEWS → Reservoir Stress → Hedging → Available Water → Crop Allocation
      → Multi-Purpose → FEWS → Flood Storage → Pre-Release → Recovery
      → Operating Mode → Rekomendasi
    (Streamlit menjalankan ulang skrip setiap kali widget berubah, dan seluruh
    halaman membaca hasil dari compute_system_state() yang sama.)
    """
    sim = st.session_state.sim

    st.markdown("---")
    st.markdown("**🎛️ INTERACTIVE SIMULATION**")
    st.caption("Ubah kondisi hidrologi — seluruh KPI, gauge, grafik, status, dan "
               "rekomendasi berubah seketika.")

    # ---- Tombol skenario cepat ----
    st.markdown("**Skenario Cepat**")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🔄 RESET SCENARIO", key="sc_reset"):
            apply_scenario("RESET"); st.rerun()
        if st.button("🌵 SIMULATE DROUGHT", key="sc_drought"):
            apply_scenario("DROUGHT"); st.rerun()
        if st.button("🔥 EXTREME DROUGHT", key="sc_extreme"):
            apply_scenario("EXTREME"); st.rerun()
    with b2:
        if st.button("🌊 SIMULATE HIGH INFLOW", key="sc_flood"):
            apply_scenario("HIGH_INFLOW"); st.rerun()
        if st.button("🔵 DROUGHT → RECOVERY", key="sc_recovery"):
            apply_scenario("RECOVERY"); st.rerun()
        if st.button("🚨 SENSOR FAILURE", key="sc_failsafe"):
            apply_scenario("FAILSAFE"); st.rerun()

    st.caption(f"Skenario aktif: **{st.session_state.scenario_name}** — "
               f"{SCENARIO_PRESETS[st.session_state.scenario_name]['label']}")

    st.markdown("---")

    # ---- Kondisi waduk ----
    with st.expander("💧 Kondisi Waduk", expanded=True):
        storage_pct = st.slider("Reservoir Storage (%)", 0.0, 110.0,
                                float(sim["storage_percent"]), 0.5, key="w_storage")
        elev_auto = storage_to_elevation(percentage_to_storage(storage_pct))
        st.caption(f"Elevasi terhitung dari kurva kapasitas: **{elev_auto:.2f} m dpl**")
        use_manual_elev = st.toggle("Atur elevasi secara manual", value=False,
                                    key="w_manual_elev")
        if use_manual_elev:
            elev = st.slider("Reservoir Elevation (m dpl)",
                             float(DAM["LWL"]) - 5.0, float(DAM["CREST"]),
                             float(elev_auto), 0.05, key="w_elev")
            storage_pct = calculate_storage_percentage(elevation_to_storage(elev))
            st.caption(f"Storage terhitung dari elevasi: **{storage_pct:.1f}%**")
        trend = st.slider("Storage Trend (%/hari)", -2.0, 2.0,
                          float(sim["storage_trend"]), 0.05, key="w_trend",
                          help="Negatif = tampungan menyusut, positif = mengisi.")

    # ---- Hidrologi ----
    with st.expander("🌧️ Hidrologi & Prakiraan", expanded=True):
        inflow = st.slider("Inflow (m³/detik)", 0.0, 60.0,
                           float(sim["inflow"]), 0.1, key="w_inflow")
        outflow = st.slider("Outflow terukur (m³/detik)", 0.0,
                            float(DAM["OUTLET_CAPACITY"]),
                            float(min(sim["outflow"], DAM["OUTLET_CAPACITY"])),
                            0.1, key="w_outflow")
        rainfall = st.slider("Rainfall (mm/hari)", 0.0, 250.0,
                             float(sim["rainfall"]), 1.0, key="w_rain")
        fc_rain = st.slider("Forecast Rainfall (mm / 7 hari)", 0.0, 400.0,
                            float(sim["forecast_rainfall"]), 1.0, key="w_fcrain")
        fc_inflow = st.slider("Forecast Inflow (m³/detik)", 0.0, 60.0,
                              float(sim["forecast_inflow"]), 0.1, key="w_fcinflow")
        scenario = st.selectbox("Forecast Scenario", ["LOW", "MOST LIKELY", "HIGH"],
                                index=["LOW", "MOST LIKELY", "HIGH"].index(sim["scenario"]),
                                key="w_scenario")
        et0 = st.slider("ET0 acuan (mm/hari)", 1.0, 10.0,
                        float(sim["et0"]), 0.1, key="w_et0")

    # ---- Kebutuhan air ----
    with st.expander("🚰 Kebutuhan Air", expanded=False):
        dom = st.number_input("Domestic Water Demand (m³/detik)", 0.0, 2.0,
                              float(sim["domestic_demand"]), 0.001,
                              format="%.3f", key="w_dom")
        irr = st.slider("Irrigation Demand (m³/detik)", 0.0,
                        float(DAM["Q_IRRIGATION_MAX"]) * 1.2,
                        float(sim["irrigation_demand"]), 0.05, key="w_irr")
        env = st.slider("Environmental Flow Requirement (m³/detik)", 0.0, 2.0,
                        float(sim["environmental_flow"]), 0.01, key="w_env")
        dsr_manual = st.toggle("Atur Demand-Supply Ratio secara manual",
                               value=False, key="w_dsr_toggle")
        dsr_override = None
        if dsr_manual:
            dsr_override = st.slider("Demand-Supply Ratio (-)", 0.0, 3.0, 1.0, 0.05,
                                     key="w_dsr",
                                     help="> 1,00 berarti kebutuhan melebihi pasokan.")

    # ---- Kondisi pertanian ----
    with st.expander("🌾 Kondisi Pertanian", expanded=False):
        sm = st.slider("Soil Moisture rata-rata (%)", 5.0, 100.0,
                       float(sim["soil_moisture"]), 1.0, key="w_sm")
        stage = st.selectbox("Crop Stage dominan", CROP_STAGES,
                             index=CROP_STAGES.index("Pembungaan"), key="w_stage")
        crop = st.selectbox("Crop Type dominan (Zone C)", list(CROP_LIBRARY.keys()),
                            index=0, key="w_crop")
        sens_mult = st.slider("Pengali Crop Sensitivity", 0.5, 1.5, 1.0, 0.05,
                              key="w_sens",
                              help="Menggeser kelembapan tanah seluruh zona untuk "
                                   "menguji sensitivitas alokasi.")

    # ---- Kondisi sistem ----
    with st.expander("🛠️ Kondisi Sistem", expanded=False):
        force_fs = st.toggle("Paksa FAIL-SAFE MODE (simulasi gangguan data)",
                             value=bool(sim.get("force_failsafe", False)),
                             key="w_failsafe")
        st.caption("Mensimulasikan sensor offline / data tidak valid sehingga sistem "
                   "kembali ke Approved Baseline Operating Rule.")

    # Simpan kembali ke session_state
    st.session_state.sim.update({
        "storage_percent": storage_pct, "inflow": inflow, "outflow": outflow,
        "rainfall": rainfall, "forecast_rainfall": fc_rain,
        "forecast_inflow": fc_inflow, "soil_moisture": sm,
        "irrigation_demand": irr, "domestic_demand": dom,
        "environmental_flow": env, "et0": et0, "scenario": scenario,
        "storage_trend": trend, "force_failsafe": force_fs,
    })

    return {
        "storage_percent": storage_pct, "inflow": inflow, "outflow": outflow,
        "rainfall": rainfall, "forecast_rainfall": fc_rain,
        "forecast_inflow": fc_inflow, "domestic_demand": dom,
        "irrigation_demand": irr, "environmental_flow": env,
        "soil_moisture": sm, "et0": et0, "scenario": scenario,
        "storage_trend": trend, "dsr_override": dsr_override,
        "force_failsafe": force_fs, "crop": crop, "stage": stage,
        "sens_mult": sens_mult, "evaporation": DAM["EVAP_MEAN_MM"],
    }


# =============================================================================
# 37. MAIN
# =============================================================================
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    cfg = render_sidebar()

    # ------------------------------------------------------------------
    # 1. Memuat data sesuai mode
    # ------------------------------------------------------------------
    data_mode = cfg["data_mode"]
    csv_error = None

    if data_mode == "CSV UPLOAD" and cfg["uploaded"] is not None:
        try:
            df = load_csv_data(cfg["uploaded"])
            if len(df) == 0:
                raise ValueError("File CSV tidak berisi data.")
        except Exception as e:      # noqa: BLE001 - tampilkan pesan ramah pengguna
            csv_error = str(e)
            df = generate_demo_data(cfg["n_days"], cfg["seed"])
    else:
        df = generate_demo_data(cfg["n_days"], cfg["seed"])

    latest = df.iloc[-1].to_dict()

    # ------------------------------------------------------------------
    # 2. Menyusun input kondisi saat ini
    # ------------------------------------------------------------------
    zones_base = generate_irrigation_zones(7, float(latest.get("storage_percent", 70.0)))

    if data_mode == "INTERACTIVE SIMULATION" and cfg["sim_inputs"]:
        s = cfg["sim_inputs"]
        inputs = dict(s)
        # Menggeser kelembapan tanah seluruh zona mengikuti slider simulasi
        zones_base = zones_base.copy()
        base_mean = float(zones_base["SM"].mean())
        shift = float(s["soil_moisture"]) - base_mean
        zones_base["SM"] = np.clip(
            (zones_base["SM"] + shift) * (2.0 - float(s["sens_mult"])), 5.0, 98.0)
        # Fase tanaman dominan diterapkan pada Zone C (zona uji)
        zones_base.loc[zones_base["Zone"] == "Zone C", "Crop"] = s["crop"]
        zones_base.loc[zones_base["Zone"] == "Zone C", "Stage"] = s["stage"]
        # Menyusun data historis sintetis agar grafik ikut mengikuti skenario
        df = _blend_history_with_simulation(df, inputs)
        latest = df.iloc[-1].to_dict()
    else:
        inputs = {
            "storage_percent": float(latest.get("storage_percent", 70.0)),
            "inflow": float(latest.get("inflow", DAM["MEAN_ANNUAL_INFLOW"])),
            "outflow": float(latest.get("outflow", 3.0)),
            "rainfall": float(latest.get("rainfall", 0.0)),
            "forecast_rainfall": float(latest.get("forecast_rainfall", 0.0)),
            "forecast_inflow": float(latest.get("forecast_inflow",
                                                DAM["MEAN_ANNUAL_INFLOW"])),
            "domestic_demand": float(latest.get("domestic_demand", DAM["Q_DOMESTIC"])),
            "irrigation_demand": float(latest.get("irrigation_demand",
                                                  DAM["Q_IRRIGATION_MAX"] * 0.7)),
            "environmental_flow": float(latest.get("environmental_flow",
                                                   DAM["Q_ENV_FLOW"])),
            "soil_moisture": float(latest.get("soil_moisture", 55.0)),
            "evaporation": float(latest.get("evaporation", DAM["EVAP_MEAN_MM"])),
            "et0": float(latest.get("evaporation", DAM["EVAP_MEAN_MM"])) * 1.05,
            "scenario": "MOST LIKELY",
            "dsr_override": None,
            "force_failsafe": False,
        }

    # ------------------------------------------------------------------
    # 3. Validasi data & sensor
    # ------------------------------------------------------------------
    validation = validate_data(df, latest)
    sensors = generate_sensor_data(
        st.session_state.sensor_seed, latest,
        failsafe_demo=bool(inputs.get("force_failsafe", False)))
    n_off = int((sensors["Status"] == "OFFLINE").sum())
    if n_off >= 3:
        validation = dict(validation)
        validation["issues"] = list(validation["issues"]) + [
            ("CRITICAL", f"{n_off} sensor OFFLINE termasuk sensor utama")]
        validation["status"] = "CRITICAL"
        validation["quality"] = "CRITICAL"
        validation["failsafe"] = True

    # ------------------------------------------------------------------
    # 4. Menjalankan seluruh rantai keputusan JALA RAKSA
    # ------------------------------------------------------------------
    ctx = compute_system_state(inputs, df, zones_base, validation)

    # ------------------------------------------------------------------
    # 5. Header
    # ------------------------------------------------------------------
    last_update = pd.to_datetime(latest["datetime"]).strftime("%d-%m-%Y %H:%M")
    render_header(ctx, data_mode, last_update)

    if csv_error:
        alert(f"<b>⚠️ Gagal membaca CSV:</b> {csv_error}<br>"
              f"Sistem kembali menggunakan DEMO DATA.", "warn")
    if data_mode == "CSV UPLOAD" and cfg["uploaded"] is None:
        alert("📄 Belum ada file CSV yang diunggah. Dashboard menampilkan "
              "<b>DEMO DATA</b> sementara. Unggah CSV melalui sidebar untuk "
              "menggunakan data Anda sendiri.", "info")
    if data_mode == "INTERACTIVE SIMULATION":
        alert(f"🎛️ <b>INTERACTIVE SIMULATION MODE aktif</b> — skenario "
              f"<b>{st.session_state.scenario_name}</b>. Setiap perubahan input pada "
              f"sidebar langsung menghitung ulang DEWS → Reservoir Stress → Hedging → "
              f"Alokasi → FEWS → Pre-Release → Recovery → Operating Mode → Rekomendasi.",
              "info")

    # ------------------------------------------------------------------
    # 6. Routing halaman
    # ------------------------------------------------------------------
    page = cfg["page"]
    is_operator = cfg["access"].startswith("OPERATOR")

    if page == PAGES[0]:
        page_command_center(ctx, df)
    elif page == PAGES[1]:
        page_reservoir_overview(ctx, df)
    elif page == PAGES[2]:
        page_dews(ctx, df)
        st.markdown("---")
        render_fuzzy_detail(ctx)
    elif page == PAGES[3]:
        page_food_allocation(ctx, df)
    elif page == PAGES[4]:
        page_iwrm(ctx, df)
    elif page == PAGES[5]:
        page_fews(ctx, df)
    elif page == PAGES[6]:
        page_recovery(ctx, df)
    elif page == PAGES[7]:
        page_performance(ctx, df)
    elif page == PAGES[8]:
        page_iot(ctx, df, sensors)
    elif page == PAGES[9]:
        if is_operator:
            page_decision_center(ctx, df)
        else:
            section("10 — Decision Center")
            alert(
                "<b>🔒 Halaman ini hanya tersedia pada OPERATOR DECISION SUPPORT MODE.</b>"
                "<br><br>Pengguna publik dapat memantau seluruh kondisi waduk, peringatan "
                "dini, alokasi air, dan rekomendasi umum JALA RAKSA, namun "
                "<b>tidak dapat memberikan keputusan operasi</b>.<br><br>"
                "Ubah <b>MODE AKSES</b> pada sidebar menjadi "
                "<b>OPERATOR DECISION SUPPORT MODE</b> untuk membuka panel "
                "APPROVE / MODIFY / REJECT.", "warn")
            st.markdown("---")
            section("Rekomendasi Umum JALA RAKSA (Tampilan Publik)")
            body = "".join(f"<li>{ln}</li>" for ln in ctx["recommendation"]["narrative"])
            alert(f"<b>Operating Mode: {ctx['mode']}</b>"
                  f"<ul style='margin:8px 0 0 0'>{body}</ul>", "info")
    else:
        page_about(ctx)

    render_footer()


def _blend_history_with_simulation(df, inputs):
    """
    Menyusun deret historis agar konsisten dengan kondisi simulasi interaktif.

    30 hari terakhir ditarik secara halus menuju nilai yang ditetapkan pengguna
    sehingga grafik riwayat, KPI, dan status tetap saling konsisten ketika
    skenario simulasi diubah. Ini adalah DEMONSTRATION DATA.
    """
    d = df.copy()
    n = len(d)
    k = min(30, n)
    w = np.linspace(0.0, 1.0, k) ** 1.6      # bobot transisi halus

    def blend(col, target):
        vals = d[col].to_numpy(dtype=float).copy()
        vals[-k:] = vals[-k:] * (1 - w) + float(target) * w
        d[col] = vals

    blend("storage_percent", inputs["storage_percent"])
    blend("inflow", inputs["inflow"])
    blend("outflow", inputs["outflow"])
    blend("rainfall", inputs["rainfall"])
    blend("forecast_rainfall", inputs["forecast_rainfall"])
    blend("forecast_inflow", inputs["forecast_inflow"])
    blend("soil_moisture", inputs["soil_moisture"])
    blend("irrigation_demand", inputs["irrigation_demand"])

    # Menerapkan tren tampungan pada beberapa hari terakhir
    trend = float(inputs.get("storage_trend", 0.0))
    if abs(trend) > 1e-6:
        sp = d["storage_percent"].to_numpy(dtype=float).copy()
        m = min(10, n)
        for i in range(1, m + 1):
            sp[-i] = float(np.clip(sp[-1] - trend * (i - 1), 0.0, 115.0))
        d["storage_percent"] = sp

    d["storage"] = d["storage_percent"].apply(percentage_to_storage)
    d["reservoir_level"] = d["storage"].apply(storage_to_elevation)
    return d


if __name__ == "__main__":
    main()
