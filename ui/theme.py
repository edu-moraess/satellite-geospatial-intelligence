"""
Tema e identidade visual do Satellite Geospatial Intelligence.
Paleta, estilos CSS e configurações de fonte.
"""

import streamlit as st

# =============================================================================
# PALETA DE CORES
# =============================================================================
COLORS = {
    "bg_primary": "#071014",
    "bg_secondary": "#0B171C",
    "bg_panel": "#0F2026",
    "border": "rgba(255,255,255,0.08)",
    "text_primary": "#E8EEF2",
    "text_secondary": "#8FA3AD",
    "accent": "#2BB0D9",          # azul/ciano tecnológico
    "accent_dim": "rgba(43, 176, 217, 0.15)",
    "success": "#2ECC71",          # verde discreto
    "warning": "#F1C40F",          # âmbar discreto
    "danger": "#E74C3C",
}

# =============================================================================
# CSS GLOBAL
# =============================================================================
def load_theme():
    """Retorna o CSS global para injetar via st.markdown."""
    css = f"""
    <style>
        /* Reset e base */
        .stApp {{
            background: {COLORS["bg_primary"]};
            color: {COLORS["text_primary"]};
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        .stApp > header {{
            background: transparent !important;
        }}
        .block-container {{
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 1400px !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {COLORS["bg_secondary"]};
            border-right: 1px solid {COLORS["border"]};
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding: 1.5rem 1.2rem !important;
        }}
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {{
            color: {COLORS["text_primary"]};
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stDateInput label {{
            color: {COLORS["text_secondary"]};
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Títulos */
        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS["text_primary"]};
            font-weight: 500;
            letter-spacing: -0.01em;
        }}
        .section-title {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {COLORS["text_secondary"]};
            margin-bottom: 0.25rem;
            border-bottom: 1px solid {COLORS["border"]};
            padding-bottom: 0.5rem;
        }}
        .section-title span {{
            color: {COLORS["accent"]};
        }}

        /* Cards e painéis */
        .panel {{
            background: {COLORS["bg_panel"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }}
        .panel-accent {{
            border-left: 3px solid {COLORS["accent"]};
        }}
        .metric-value {{
            font-size: 1.6rem;
            font-weight: 600;
            color: {COLORS["text_primary"]};
            line-height: 1.2;
        }}
        .metric-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["text_secondary"]};
        }}
        .metric-delta {{
            font-size: 0.8rem;
            color: {COLORS["success"]};
        }}

        /* Tabelas */
        .catalog-table {{
            width: 100%;
            font-size: 0.85rem;
            border-collapse: collapse;
        }}
        .catalog-table th {{
            text-align: left;
            color: {COLORS["text_secondary"]};
            font-weight: 400;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 0.4rem 0.5rem;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        .catalog-table td {{
            padding: 0.4rem 0.5rem;
            border-bottom: 1px solid {COLORS["border"]};
            color: {COLORS["text_primary"]};
        }}
        .catalog-table tr:hover td {{
            background: rgba(255,255,255,0.03);
        }}
        .status-badge {{
            display: inline-block;
            padding: 0.1rem 0.6rem;
            border-radius: 12px;
            font-size: 0.65rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .status-ready {{
            background: rgba(46, 204, 113, 0.2);
            color: {COLORS["success"]};
        }}
        .status-pending {{
            background: rgba(241, 196, 15, 0.2);
            color: {COLORS["warning"]};
        }}
        .status-error {{
            background: rgba(231, 76, 60, 0.2);
            color: {COLORS["danger"]};
        }}

        /* Pipeline */
        .pipeline {{
            display: flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.75rem 0;
        }}
        .pipeline-stage {{
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
            text-align: center;
        }}
        .pipeline-stage .label {{
            font-size: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["text_secondary"]};
            margin-top: 0.2rem;
        }}
        .pipeline-stage .icon {{
            font-size: 1.2rem;
            line-height: 1;
        }}
        .pipeline-stage.done .icon {{ color: {COLORS["success"]}; }}
        .pipeline-stage.active .icon {{ color: {COLORS["accent"]}; }}
        .pipeline-stage.pending .icon {{ color: {COLORS["text_secondary"]}; }}
        .pipeline-connector {{
            flex: 0.3;
            height: 2px;
            background: {COLORS["border"]};
        }}
        .pipeline-connector.done {{
            background: {COLORS["success"]};
        }}

        /* Imagens e mapas */
        .image-container {{
            border: 1px solid {COLORS["border"]};
            border-radius: 6px;
            overflow: hidden;
            background: {COLORS["bg_secondary"]};
        }}
        .image-container img {{
            width: 100%;
            display: block;
        }}

        /* Botões */
        .stButton button {{
            background: transparent;
            border: 1px solid {COLORS["border"]};
            color: {COLORS["text_primary"]};
            border-radius: 4px;
            font-size: 0.8rem;
            padding: 0.3rem 1rem;
            transition: all 0.15s;
        }}
        .stButton button:hover {{
            background: {COLORS["accent_dim"]};
            border-color: {COLORS["accent"]};
            color: {COLORS["accent"]};
        }}
        .stButton button[data-baseweb="button"]:focus {{
            box-shadow: none;
        }}

        /* Selectbox, inputs */
        .stSelectbox div[data-baseweb="select"] > div,
        .stDateInput input,
        .stNumberInput input {{
            background: {COLORS["bg_secondary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            color: {COLORS["text_primary"]};
        }}
        .stSelectbox div[data-baseweb="select"]:focus-within,
        .stDateInput input:focus,
        .stNumberInput input:focus {{
            border-color: {COLORS["accent"]};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {COLORS["text_secondary"]};
            font-size: 0.8rem;
            padding: 0.3rem 0.8rem;
            border-radius: 4px 4px 0 0;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLORS["text_primary"]};
            background: {COLORS["bg_panel"]};
            border-bottom: 2px solid {COLORS["accent"]};
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            color: {COLORS["text_secondary"]};
            font-size: 0.8rem;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        .streamlit-expanderHeader:hover {{
            color: {COLORS["text_primary"]};
        }}

        /* Scrollbars */
        ::-webkit-scrollbar {{
            width: 4px;
            height: 4px;
        }}
        ::-webkit-scrollbar-track {{
            background: {COLORS["bg_secondary"]};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {COLORS["border"]};
            border-radius: 2px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS["text_secondary"]};
        }}

        /* Responsividade */
        @media (max-width: 768px) {{
            .block-container {{
                padding-left: 0.8rem !important;
                padding-right: 0.8rem !important;
            }}
            .metric-value {{
                font-size: 1.2rem;
            }}
        }}
    </style>
    """
    return css 