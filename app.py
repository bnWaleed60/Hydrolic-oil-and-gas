"""Copperfield Control Room — interactive hydraulic pressure-drop calculator.

Design philosophy: an industrial editorial workspace with an asymmetric command
rail, graphite surfaces, machined-copper results, muted teal flow accents, and
a traceable calculation story. Every engineering value is paired with a unit
and the UI exposes the frictional and hydrostatic contributions separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from math import log10, pi, sqrt
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HERO_URL = "/manus-storage/hydraulic-hero_4c8b95de.jpg"
FLOW_TEXTURE_URL = "/manus-storage/hydraulic-flow-texture_70ff10fb.jpg"
WELLBORE_URL = "/manus-storage/wellbore-flow-illustration_340ad89e.jpg"
MARK_URL = "/manus-storage/hydraulic-mark_14fe0081.png"


@dataclass(frozen=True)
class HydraulicInputs:
    phase: str
    flow_rate: float
    diameter: float
    length: float
    viscosity_cp: float
    density: float
    roughness: float
    elevation_change: float
    unit_system: str


@dataclass(frozen=True)
class HydraulicResults:
    q_m3_s: float
    diameter_m: float
    length_m: float
    viscosity_pa_s: float
    density_kg_m3: float
    roughness_m: float
    elevation_change_m: float
    area_m2: float
    velocity_m_s: float
    reynolds: float
    friction_factor: float
    friction_head_m: float
    hydrostatic_head_m: float
    total_head_m: float
    friction_dp_pa: float
    hydrostatic_dp_pa: float
    total_dp_pa: float
    flow_regime: str
    regime_detail: str


def _convert_inputs(inputs: HydraulicInputs) -> dict[str, float | str]:
    """Convert the user-facing unit system to SI base units."""
    if inputs.unit_system == "Oilfield (bbl/day, in, ft, lb/ft³)":
        return {
            "phase": inputs.phase,
            "q_m3_s": inputs.flow_rate * 0.1589872949 / 86400.0,
            "diameter_m": inputs.diameter * 0.0254,
            "length_m": inputs.length * 0.3048,
            "viscosity_pa_s": inputs.viscosity_cp * 1e-3,
            "density_kg_m3": inputs.density * 16.018463,
            "roughness_m": inputs.roughness * 0.0254,
            "elevation_change_m": inputs.elevation_change * 0.3048,
        }

    return {
        "phase": inputs.phase,
        "q_m3_s": inputs.flow_rate / 86400.0,
        "diameter_m": inputs.diameter / 1000.0,
        "length_m": inputs.length,
        "viscosity_pa_s": inputs.viscosity_cp * 1e-3,
        "density_kg_m3": inputs.density,
        "roughness_m": inputs.roughness / 1000.0,
        "elevation_change_m": inputs.elevation_change,
    }


def _friction_factor(reynolds: float, roughness_m: float, diameter_m: float) -> float:
    """Return Darcy friction factor using laminar or Swamee–Jain approximation."""
    if reynolds <= 0:
        return 0.0
    if reynolds < 2300:
        return 64.0 / reynolds
    relative_roughness = max(roughness_m / diameter_m, 0.0)
    denominator = log10(relative_roughness / 3.7 + 5.74 / (reynolds**0.9))
    return 0.25 / (denominator**2)


def calculate_hydraulics(inputs: HydraulicInputs) -> HydraulicResults:
    """Calculate velocity, Reynolds number, friction, hydrostatic, and total ΔP."""
    si = _convert_inputs(inputs)
    q_m3_s = float(si["q_m3_s"])
    diameter_m = float(si["diameter_m"])
    length_m = float(si["length_m"])
    viscosity_pa_s = float(si["viscosity_pa_s"])
    density_kg_m3 = float(si["density_kg_m3"])
    roughness_m = float(si["roughness_m"])
    elevation_change_m = float(si["elevation_change_m"])

    if diameter_m <= 0 or length_m <= 0 or viscosity_pa_s <= 0 or density_kg_m3 <= 0:
        raise ValueError("Diameter, length, viscosity, and density must be greater than zero.")
    if q_m3_s < 0 or roughness_m < 0:
        raise ValueError("Flow rate and roughness cannot be negative.")

    area_m2 = pi * diameter_m**2 / 4.0
    velocity_m_s = q_m3_s / area_m2
    reynolds = density_kg_m3 * velocity_m_s * diameter_m / viscosity_pa_s
    friction_factor = _friction_factor(reynolds, roughness_m, diameter_m)
    velocity_head_m = velocity_m_s**2 / (2.0 * 9.80665)
    friction_head_m = friction_factor * (length_m / diameter_m) * velocity_head_m
    hydrostatic_head_m = elevation_change_m
    total_head_m = friction_head_m + hydrostatic_head_m
    friction_dp_pa = density_kg_m3 * 9.80665 * friction_head_m
    hydrostatic_dp_pa = density_kg_m3 * 9.80665 * hydrostatic_head_m
    total_dp_pa = friction_dp_pa + hydrostatic_dp_pa

    if reynolds < 2300:
        flow_regime = "Laminar"
        regime_detail = "Viscous-dominated flow"
    elif reynolds < 4000:
        flow_regime = "Turbulent"
        regime_detail = "Transitional band"
    else:
        flow_regime = "Turbulent"
        regime_detail = "Inertial-dominated flow"

    return HydraulicResults(
        q_m3_s=q_m3_s,
        diameter_m=diameter_m,
        length_m=length_m,
        viscosity_pa_s=viscosity_pa_s,
        density_kg_m3=density_kg_m3,
        roughness_m=roughness_m,
        elevation_change_m=elevation_change_m,
        area_m2=area_m2,
        velocity_m_s=velocity_m_s,
        reynolds=reynolds,
        friction_factor=friction_factor,
        friction_head_m=friction_head_m,
        hydrostatic_head_m=hydrostatic_head_m,
        total_head_m=total_head_m,
        friction_dp_pa=friction_dp_pa,
        hydrostatic_dp_pa=hydrostatic_dp_pa,
        total_dp_pa=total_dp_pa,
        flow_regime=flow_regime,
        regime_detail=regime_detail,
    )


def _pressure_unit_values(pressure_pa: float) -> tuple[float, float]:
    return pressure_pa / 6894.757293168, pressure_pa / 100000.0


def _length_label(unit_system: str) -> str:
    return "ft" if unit_system.startswith("Oilfield") else "m"


def _diameter_label(unit_system: str) -> str:
    return "in" if unit_system.startswith("Oilfield") else "mm"


def _flow_label(unit_system: str) -> str:
    return "bbl/day" if unit_system.startswith("Oilfield") else "m³/day"


def _display_metrics(results: HydraulicResults, unit_system: str) -> dict[str, float | str]:
    velocity = results.velocity_m_s * 3.280839895 if unit_system.startswith("Oilfield") else results.velocity_m_s
    velocity_unit = "ft/s" if unit_system.startswith("Oilfield") else "m/s"
    total_psi, total_bar = _pressure_unit_values(results.total_dp_pa)
    friction_psi, friction_bar = _pressure_unit_values(results.friction_dp_pa)
    hydro_psi, hydro_bar = _pressure_unit_values(results.hydrostatic_dp_pa)
    return {
        "velocity": velocity,
        "velocity_unit": velocity_unit,
        "total_psi": total_psi,
        "total_bar": total_bar,
        "friction_psi": friction_psi,
        "friction_bar": friction_bar,
        "hydro_psi": hydro_psi,
        "hydro_bar": hydro_bar,
    }


def _format_number(value: float, digits: int = 2) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.{digits}f}"


def _metric_card(label: str, value: str, unit: str, detail: str, accent: str) -> str:
    return f'<div class="metric-card" style="--accent:{accent}"><div class="metric-kicker">{label}</div><div class="metric-value">{value}<span>{unit}</span></div><div class="metric-detail">{detail}</div></div>'


def _chart_layout(fig: go.Figure, y_title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=12, r=14, t=22, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7, 18, 30, 0.72)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#E6E0D7"),
        hoverlabel=dict(bgcolor="#0B1A2B", font_color="#F4EFE7"),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="rgba(230,224,215,0.18)"),
        yaxis=dict(
            title=y_title,
            gridcolor="rgba(230,224,215,0.10)",
            zeroline=False,
            linecolor="rgba(230,224,215,0.18)",
        ),
    )
    return fig


def _sweep_results(inputs: HydraulicInputs, field: str, values: Iterable[float]) -> pd.DataFrame:
    rows = []
    for value in values:
        updated = HydraulicInputs(
            phase=inputs.phase,
            flow_rate=value if field == "flow_rate" else inputs.flow_rate,
            diameter=value if field == "diameter" else inputs.diameter,
            length=inputs.length,
            viscosity_cp=inputs.viscosity_cp,
            density=inputs.density,
            roughness=inputs.roughness,
            elevation_change=inputs.elevation_change,
            unit_system=inputs.unit_system,
        )
        calculated = calculate_hydraulics(updated)
        psi, bar = _pressure_unit_values(calculated.total_dp_pa)
        rows.append({"input": value, "psi": psi, "bar": bar})
    return pd.DataFrame(rows)


def build_pdf_report(inputs: HydraulicInputs, results: HydraulicResults) -> bytes:
    """Create a compact engineering record as a PDF byte stream."""
    display = _display_metrics(results, inputs.unit_system)
    input_unit = "Oilfield" if inputs.unit_system.startswith("Oilfield") else "SI"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Hydraulic Pressure Drop Engineering Report",
        author="Copperfield Control Room",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#102438"),
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#566575"),
        spaceAfter=14,
    )
    section = ParagraphStyle(
        "ReportSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#102438"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#263747"),
    )
    right = ParagraphStyle("ReportRight", parent=body, alignment=TA_RIGHT)

    story = [
        Paragraph("Hydraulic Pressure Drop Engineering Report", title),
        Paragraph(
            f"Single-phase {inputs.phase.lower()} pipe-flow assessment · {input_unit} unit system · generated {timestamp}",
            subtitle,
        ),
    ]

    input_rows = [
        ["Input parameter", "Value", "Unit"],
        ["Fluid phase", inputs.phase, "—"],
        ["Flow rate", _format_number(inputs.flow_rate), _flow_label(inputs.unit_system)],
        ["Pipe inner diameter", _format_number(inputs.diameter), _diameter_label(inputs.unit_system)],
        ["Pipe length", _format_number(inputs.length), _length_label(inputs.unit_system)],
        ["Dynamic viscosity", _format_number(inputs.viscosity_cp), "cP"],
        ["Fluid density", _format_number(inputs.density), "lb/ft³" if input_unit == "Oilfield" else "kg/m³"],
        ["Absolute roughness", _format_number(inputs.roughness, 4), _diameter_label(inputs.unit_system)],
        ["Elevation change", _format_number(inputs.elevation_change), _length_label(inputs.unit_system)],
    ]
    story += [Paragraph("1. Input parameters", section), _styled_table(input_rows, [2.7 * inch, 1.5 * inch, 1.4 * inch])]

    metric_rows = [
        ["Calculated metric", "Result", "Unit"],
        ["Total pressure drop", _format_number(display["total_psi"]), "psi"],
        ["Total pressure drop", _format_number(display["total_bar"], 3), "bar"],
        ["Frictional pressure drop", _format_number(display["friction_psi"]), "psi"],
        ["Hydrostatic pressure contribution", _format_number(display["hydro_psi"]), "psi"],
        ["Flow velocity", _format_number(display["velocity"]), str(display["velocity_unit"])],
        ["Reynolds number", _format_number(results.reynolds, 0), "—"],
        ["Darcy friction factor", _format_number(results.friction_factor, 5), "—"],
        ["Flow regime", results.flow_regime, results.regime_detail],
    ]
    story += [Paragraph("2. Hydraulic results", section), _styled_table(metric_rows, [2.7 * inch, 1.5 * inch, 1.4 * inch])]

    summary = (
        f"The calculated flow is classified as <b>{results.flow_regime}</b> ({results.regime_detail.lower()}). "
        f"The Darcy-Weisbach friction contribution is <b>{display['friction_psi']:.2f} psi</b>, while the signed "
        f"hydrostatic contribution is <b>{display['hydro_psi']:.2f} psi</b>. The combined pressure change is "
        f"<b>{display['total_psi']:.2f} psi</b> ({display['total_bar']:.3f} bar)."
    )
    story += [Paragraph("3. Summary analysis", section), Paragraph(summary, body)]
    story += [Spacer(1, 8), Paragraph("Method note", section)]
    story += [
        Paragraph(
            "The calculator uses the Darcy-Weisbach equation. Laminar friction factor is 64/Re; turbulent friction factor uses the explicit Swamee-Jain approximation. Gas calculations treat density and viscosity as local average properties, so compressibility and temperature variation are not solved in this simplified single-phase model.",
            body,
        )
    ]
    document.build(story)
    buffer.seek(0)
    return buffer.read()


def _styled_table(rows: list[list[str]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102438")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#263747")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F6")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5D8")),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root {{ --ink:#07121E; --navy:#0B1A2B; --navy-2:#102438; --sand:#E6E0D7; --muted:#9BA7AE; --copper:#C9784A; --teal:#5FB3A5; }}
        .stApp {{ background: #07121E; color: #E6E0D7; font-family: 'IBM Plex Sans', sans-serif; }}
        .block-container {{ padding-top: 2.3rem; padding-bottom: 3rem; max-width: 1500px; }}
        [data-testid="stSidebar"] {{ background: #0B1A2B; border-right: 1px solid rgba(230,224,215,0.12); }}
        [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.8rem; }}
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{ color: #C8D0D1 !important; }}
        [data-testid="stSidebar"] .stMarkdown h3 {{ color:#F4EFE7; font-family:'Space Grotesk', sans-serif; margin-top: 1.2rem; }}
        h1,h2,h3,h4 {{ font-family:'Space Grotesk', sans-serif !important; letter-spacing:-0.025em; color:#F4EFE7; }}
        h1 {{ font-size: 2.55rem !important; line-height: 1.04 !important; }}
        .hero {{ position:relative; overflow:hidden; min-height: 250px; border: 1px solid rgba(230,224,215,0.14); border-radius: 18px; padding: 2rem 2.1rem; margin-bottom: 1.2rem; background: linear-gradient(90deg, rgba(7,18,30,.98) 0%, rgba(7,18,30,.88) 47%, rgba(7,18,30,.35) 100%), url('{HERO_URL}') center/cover; box-shadow: 0 20px 50px rgba(0,0,0,.22); }}
        .hero:after {{ content:''; position:absolute; inset:0; pointer-events:none; opacity:.25; background: url('{FLOW_TEXTURE_URL}') center/cover; mix-blend-mode:screen; }}
        .hero-content {{ position:relative; z-index:1; max-width: 720px; }}
        .eyebrow {{ font: 500 .72rem/1 'IBM Plex Mono', monospace; letter-spacing:.16em; text-transform:uppercase; color:#C9784A; margin-bottom:1rem; }}
        .hero h1 {{ margin:0 0 .8rem; }}
        .hero p {{ color:#C8D0D1; font-size:1.03rem; max-width: 625px; line-height:1.6; margin-bottom:0; }}
        .topline {{ display:flex; align-items:center; gap:12px; margin-bottom:1.3rem; }}
        .brand-mark {{ width:42px; height:42px; display:grid; place-items:center; border:1px solid rgba(201,120,74,.7); border-radius:12px; background:linear-gradient(145deg,#D98A5E,#8E4D35) url('{MARK_URL}') center/contain no-repeat; box-shadow:0 4px 10px rgba(201,120,74,.2); }}
        .mark-fallback span {{ display:block; width:15px; height:24px; border:2px solid #07121E; border-top-width:5px; transform:skewY(-24deg); position:relative; }}
        .mark-fallback span:after {{ content:''; position:absolute; width:8px; height:2px; background:#07121E; right:-5px; bottom:4px; transform:rotate(-24deg); }}
        .brand-name {{ font:600 1rem/1 'Space Grotesk',sans-serif; letter-spacing:.02em; color:#F4EFE7; }}
        .brand-sub {{ font: 500 .67rem/1.4 'IBM Plex Mono',monospace; letter-spacing:.08em; text-transform:uppercase; color:#9BA7AE; margin-top:5px; }}
        .section-rail {{ font:500 .72rem/1 'IBM Plex Mono',monospace; letter-spacing:.14em; text-transform:uppercase; color:#9BA7AE; margin: 1.6rem 0 .65rem; }}
        .metric-grid {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: .75rem; margin: .6rem 0 1.2rem; }}
        .metric-card {{ position:relative; overflow:hidden; min-height: 122px; background: linear-gradient(145deg, rgba(16,36,56,.96), rgba(11,26,43,.94)); border: 1px solid rgba(230,224,215,.13); border-radius: 14px; padding: 1.05rem 1.1rem; box-shadow: 0 10px 26px rgba(0,0,0,.12); }}
        .metric-card:before {{ content:''; position:absolute; top:0; left:0; right:0; height:3px; background:var(--accent); }}
        .metric-kicker {{ font:500 .68rem/1.2 'IBM Plex Mono',monospace; color:#9BA7AE; letter-spacing:.08em; text-transform:uppercase; }}
        .metric-value {{ font:700 1.64rem/1.2 'Space Grotesk',sans-serif; color:#F4EFE7; margin-top:.62rem; }}
        .metric-value span {{ font:500 .78rem/1 'IBM Plex Mono',monospace; color:#AEB9BC; margin-left:.35rem; }}
        .metric-detail {{ color:#C8D0D1; font-size:.78rem; margin-top:.45rem; }}
        .panel {{ background:rgba(16,36,56,.62); border:1px solid rgba(230,224,215,.12); border-radius:14px; padding:1.05rem 1.15rem .8rem; margin-bottom:1rem; }}
        .panel-title {{ display:flex; align-items:baseline; justify-content:space-between; gap:1rem; margin-bottom:.45rem; }}
        .panel-title h3 {{ font-size:1.05rem; margin:0; }}
        .panel-title span {{ font:400 .68rem/1 'IBM Plex Mono',monospace; color:#9BA7AE; text-transform:uppercase; letter-spacing:.08em; }}
        .status-chip {{ display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .65rem; border-radius:999px; background:rgba(95,179,165,.12); color:#7CC9BC; border:1px solid rgba(95,179,165,.35); font:500 .72rem/1 'IBM Plex Mono',monospace; text-transform:uppercase; letter-spacing:.06em; }}
        .status-chip:before {{ content:''; width:7px; height:7px; border-radius:50%; background:#5FB3A5; box-shadow:0 0 0 4px rgba(95,179,165,.12); }}
        .readout {{ border-left: 3px solid #C9784A; padding:.4rem 0 .4rem 1rem; margin:.75rem 0 1.2rem; }}
        .readout strong {{ color:#F4EFE7; font-family:'Space Grotesk',sans-serif; }}
        .readout p {{ color:#AEB9BC; margin:.18rem 0 0; font-size:.86rem; line-height:1.5; }}
        .equation {{ background:rgba(7,18,30,.55); border:1px solid rgba(230,224,215,.1); border-radius:10px; padding:.8rem 1rem; color:#C8D0D1; font:400 .77rem/1.6 'IBM Plex Mono',monospace; }}
        .small-note {{ color:#9BA7AE; font-size:.78rem; line-height:1.5; }}
        .stButton > button, .stDownloadButton > button {{ border-radius:9px; border:1px solid rgba(201,120,74,.65); background:#C9784A; color:#07121E; font-family:'Space Grotesk',sans-serif; font-weight:700; min-height:2.55rem; transition:transform .16s ease, box-shadow .16s ease, background .16s ease; }}
        .stButton > button:hover, .stDownloadButton > button:hover {{ background:#E09A6A; box-shadow:0 8px 24px rgba(201,120,74,.22); transform:translateY(-1px); }}
        .stButton > button:active, .stDownloadButton > button:active {{ transform:scale(.98); }}
        [data-testid="stMetricValue"] {{ font-family:'Space Grotesk',sans-serif; }}
        [data-testid="stDataFrame"] {{ border:1px solid rgba(230,224,215,.12); }}
        footer {{ visibility:hidden; }}
        @media (max-width: 1050px) {{ .metric-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
        @media (max-width: 650px) {{ .block-container {{ padding:1rem .8rem 2rem; }} .hero {{ padding:1.4rem; min-height:260px; }} h1 {{font-size:2rem !important;}} .metric-grid {{ grid-template-columns:1fr; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> HydraulicInputs:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="topline">
              <div class="brand-mark" role="img" aria-label="Hydraulic calculator mark"><span></span></div>
              <div><div class="brand-name">Copperfield</div><div class="brand-sub">Hydraulic analysis</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Test condition")
        unit_system = st.selectbox(
            "Unit system",
            ["Oilfield (bbl/day, in, ft, lb/ft³)", "SI (m³/day, mm, m, kg/m³)"],
            index=0,
        )
        phase = st.selectbox("Fluid phase", ["Liquid", "Gas"], index=0)
        st.markdown("### Geometry & flow")
        if unit_system.startswith("Oilfield"):
            flow_rate = st.number_input("Flow rate · bbl/day", min_value=0.0, value=850.0, step=25.0)
            diameter = st.number_input("Pipe inner diameter · in", min_value=0.01, value=4.0, step=0.25)
            length = st.number_input("Pipe length · ft", min_value=0.1, value=5000.0, step=100.0)
            roughness = st.number_input("Absolute roughness · in", min_value=0.0, value=0.0006, step=0.0001, format="%.4f")
            elevation_change = st.number_input("Elevation change · ft", value=0.0, step=50.0)
            density = st.number_input("Fluid density · lb/ft³", min_value=0.01, value=55.0, step=1.0)
        else:
            flow_rate = st.number_input("Flow rate · m³/day", min_value=0.0, value=135.0, step=5.0)
            diameter = st.number_input("Pipe inner diameter · mm", min_value=0.1, value=101.6, step=5.0)
            length = st.number_input("Pipe length · m", min_value=0.1, value=1524.0, step=25.0)
            roughness = st.number_input("Absolute roughness · mm", min_value=0.0, value=0.015, step=0.005, format="%.4f")
            elevation_change = st.number_input("Elevation change · m", value=0.0, step=10.0)
            density = st.number_input("Fluid density · kg/m³", min_value=0.01, value=881.0, step=10.0)
        st.markdown("### Fluid properties")
        viscosity_cp = st.number_input("Dynamic viscosity · cP", min_value=0.001, value=4.5, step=0.5)
        st.caption("Assumption: density and viscosity are local average properties. Gas cases use an incompressible approximation.")
        return HydraulicInputs(
            phase=phase,
            flow_rate=flow_rate,
            diameter=diameter,
            length=length,
            viscosity_cp=viscosity_cp,
            density=density,
            roughness=roughness,
            elevation_change=elevation_change,
            unit_system=unit_system,
        )


def _render_dashboard(inputs: HydraulicInputs, results: HydraulicResults) -> None:
    display = _display_metrics(results, inputs.unit_system)
    pressure_unit = "psi" if inputs.unit_system.startswith("Oilfield") else "bar"
    pressure_value = display["total_psi"] if pressure_unit == "psi" else display["total_bar"]

    st.markdown(
        f"""
        <section class="hero">
          <div class="hero-content">
            <div class="eyebrow">Wellbore simulator / hydraulic run</div>
            <h1>Map the pressure story.</h1>
            <p>Single-phase pipe-flow analysis with a visible calculation chain: velocity, Reynolds number, friction factor, hydrostatic contribution, and total pressure drop.</p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-rail">Live readout · current test condition</div>', unsafe_allow_html=True)
    status_color = "#5FB3A5" if results.flow_regime == "Laminar" else "#C9784A"
    cards = "".join(
        [
            _metric_card("Total pressure drop", _format_number(float(pressure_value)), pressure_unit, f"{display['total_psi']:.2f} psi / {display['total_bar']:.3f} bar", "#C9784A"),
            _metric_card("Flow velocity", _format_number(float(display["velocity"])), str(display["velocity_unit"]), "Bulk mean velocity", "#5FB3A5"),
            _metric_card("Flow regime", results.flow_regime, "", f"Re = {_format_number(results.reynolds, 0)}", status_color),
            _metric_card("Darcy friction factor", _format_number(results.friction_factor, 5), "", results.regime_detail, "#C8A36A"),
        ]
    )
    st.html(f'<div class="metric-grid">{cards}</div>')

    if inputs.phase == "Gas":
        st.info("Gas mode is available for screening calculations. This implementation treats density and viscosity as local average properties and does not solve compressible pressure-volume behavior.")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><h3>Pressure drop sensitivity</h3><span>Darcy-Weisbach sweep</span></div>', unsafe_allow_html=True)
        diameter_values = [inputs.diameter * factor for factor in [0.55 + i * (1.35 / 29) for i in range(30)]]
        diameter_sweep = _sweep_results(inputs, "diameter", diameter_values)
        diameter_unit = _diameter_label(inputs.unit_system)
        fig_diameter = go.Figure()
        fig_diameter.add_trace(go.Scatter(x=diameter_sweep["input"], y=diameter_sweep["psi"], mode="lines", name="Pressure drop", line=dict(color="#C9784A", width=3), fill="tozeroy", fillcolor="rgba(201,120,74,.12)"))
        fig_diameter.add_vline(x=inputs.diameter, line_dash="dot", line_color="#E6E0D7", opacity=.65, annotation_text="Current ID", annotation_position="top right")
        fig_diameter.update_xaxes(title=f"Pipe inner diameter ({diameter_unit})")
        _chart_layout(fig_diameter, "Total pressure drop (psi)")
        st.plotly_chart(fig_diameter, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title"><h3>Pressure drop vs flow</h3><span>Current pipe geometry</span></div>', unsafe_allow_html=True)
        flow_values = [max(inputs.flow_rate * factor, 0.0001) for factor in [0.25 + i * (1.5 / 29) for i in range(30)]]
        flow_sweep = _sweep_results(inputs, "flow_rate", flow_values)
        flow_unit = _flow_label(inputs.unit_system)
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Scatter(x=flow_sweep["input"], y=flow_sweep["psi"], mode="lines", name="Pressure drop", line=dict(color="#5FB3A5", width=3)))
        fig_flow.add_vline(x=inputs.flow_rate, line_dash="dot", line_color="#E6E0D7", opacity=.65, annotation_text="Current rate", annotation_position="top right")
        fig_flow.update_xaxes(title=f"Flow rate ({flow_unit})")
        _chart_layout(fig_flow, "Total pressure drop (psi)")
        st.plotly_chart(fig_flow, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-rail">Pressure story · what is driving the result</div>', unsafe_allow_html=True)
    story_left, story_mid, story_right = st.columns([1, 1, 1], gap="medium")
    with story_left:
        st.markdown(
            f'<div class="readout"><strong>Frictional loss</strong><p>{display["friction_psi"]:.2f} psi · {results.friction_head_m:.2f} m of head<br/>Driven by velocity, ID, roughness, and flow regime.</p></div>',
            unsafe_allow_html=True,
        )
    with story_mid:
        st.markdown(
            f'<div class="readout"><strong>Hydrostatic contribution</strong><p>{display["hydro_psi"]:.2f} psi · {results.hydrostatic_head_m:.2f} m of head<br/>Signed elevation term; positive means net rise.</p></div>',
            unsafe_allow_html=True,
        )
    with story_right:
        st.markdown(
            f'<div class="readout"><strong>Total pressure change</strong><p>{display["total_psi"]:.2f} psi · {display["total_bar"]:.3f} bar<br/>Friction and elevation combined.</p></div>',
            unsafe_allow_html=True,
        )

    profile_tab, method_tab = st.tabs(["Wellbore pressure profile", "Calculation method"])
    with profile_tab:
        profile_length = results.length_m
        distances = [profile_length * i / 40.0 for i in range(41)]
        profile_rows = []
        for distance in distances:
            friction_head = results.friction_head_m * distance / profile_length
            hydro_head = results.hydrostatic_head_m * distance / profile_length
            pressure_pa = results.density_kg_m3 * 9.80665 * (friction_head + hydro_head)
            psi, bar = _pressure_unit_values(pressure_pa)
            profile_rows.append({"distance_m": distance, "psi": psi, "bar": bar})
        profile = pd.DataFrame(profile_rows)
        fig_profile = go.Figure()
        fig_profile.add_trace(go.Scatter(x=profile["distance_m"], y=profile["psi"], mode="lines", line=dict(color="#C9784A", width=3), fill="tozeroy", fillcolor="rgba(201,120,74,.10)", name="Cumulative ΔP"))
        fig_profile.update_xaxes(title="Distance along pipe (m)")
        _chart_layout(fig_profile, "Cumulative pressure change (psi)")
        st.plotly_chart(fig_profile, use_container_width=True, config={"displayModeBar": False})
        st.caption("The profile is a linear engineering visualization of cumulative friction and hydrostatic terms along the entered pipe length.")
    with method_tab:
        st.markdown(
            f"""
            <div class="equation">
            Velocity: v = Q / A<br/>
            Reynolds number: Re = ρvD / μ<br/>
            Darcy-Weisbach head loss: h<sub>f</sub> = f · (L/D) · v²/(2g)<br/>
            Hydrostatic head: h<sub>z</sub> = Δz<br/>
            Total pressure change: ΔP = ρg(h<sub>f</sub> + h<sub>z</sub>)<br/><br/>
            Friction factor: f = 64/Re for Re &lt; 2300; otherwise Swamee-Jain explicit approximation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<p class='small-note'>Engineering scope: steady, single-phase flow through a constant-diameter pipe. Minor losses, multiphase effects, compressibility, temperature variation, and fluid-property gradients are outside this screening model.</p>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Hydraulic Pressure Drop Calculator", page_icon=MARK_URL, layout="wide", initial_sidebar_state="expanded")
    _inject_styles()
    inputs = _render_sidebar()
    try:
        results = calculate_hydraulics(inputs)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    _render_dashboard(inputs, results)
    st.markdown('<div class="section-rail">Engineering record</div>', unsafe_allow_html=True)
    report_col, note_col = st.columns([1, 2], gap="large")
    with report_col:
        pdf_bytes = build_pdf_report(inputs, results)
        st.download_button(
            "Download Engineering PDF Report",
            data=pdf_bytes,
            file_name=f"hydraulic_pressure_drop_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with note_col:
        st.markdown(
            '<div class="status-chip">Calculation complete</div><p class="small-note" style="margin-top:.65rem">The report includes the active inputs, calculated hydraulic metrics, regime classification, pressure-loss breakdown, engineering summary, and UTC generation timestamp.</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
