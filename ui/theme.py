"""
ui/theme.py – Tema e estilos CSS para o Satellite Geospatial Intelligence.
Estilo dark, aerospace, minimalista e com alta densidade de informação.
"""

import streamlit as st

def apply_theme():
    """
    Aplica o CSS global ao aplicativo.
    """
    st.markdown("""
    <style>
    /* Reset e base */
    .main > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    .stApp {
        background-color: #0b0e14;
    }
    .stSidebar {
        background-color: #0f131a;
        border-right: 1px solid #1e2630;
    }
    .stSidebar .stMarkdown, .stSidebar .stTextInput, .stSidebar .stSelectbox, .stSidebar .stDateInput {
        font-size: 0.85rem;
    }
    .stSidebar .stButton button {
        width: 100%;
        background: #1a73e8;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: background 0.2s;
    }
    .stSidebar .stButton button:hover {
        background: #2b7ff0;
    }

    /* Cabeçalho global */
    .sgi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.3rem 0 0.2rem 0;
        border-bottom: 1px solid #1e2630;
        margin-bottom: 0.5rem;
    }
    .sgi-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .sgi-logo {
        font-size: 1.3rem;
        font-weight: 700;
        color: #c8d0dc;
        letter-spacing: 1px;
    }
    .sgi-logo span {
        color: #4caf50;
    }
    .sgi-title {
        font-size: 1.0rem;
        font-weight: 500;
        color: #e8edf5;
        letter-spacing: 0.5px;
    }
    .sgi-subtitle {
        font-size: 0.7rem;
        color: #7a869a;
        margin-top: -0.1rem;
    }
    .sgi-status {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.65rem;
        font-weight: 500;
        color: #4caf50;
        background: #1a2a1a;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        border: 1px solid #2a4a2a;
        letter-spacing: 0.5px;
    }
    .sgi-status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        background: #4caf50;
        border-radius: 50%;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Seções */
    .sgi-section {
        margin-bottom: 0.4rem;
        padding: 0.2rem 0;
    }
    .sgi-section-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #7a869a;
        margin-bottom: 0.1rem;
        padding-bottom: 0.1rem;
        border-bottom: 1px solid #1e2630;
    }
    .sgi-section-description {
        font-size: 0.7rem;
        color: #5a667a;
        margin-bottom: 0.3rem;
    }

    /* Métricas compactas */
    .sgi-metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 0.3rem;
        margin: 0.2rem 0;
    }
    .sgi-metric-card {
        background: #131a22;
        border-radius: 3px;
        padding: 0.3rem 0.5rem;
        border-left: 2px solid #1a73e8;
        box-shadow: 0 1px 2px rgba(0,0,0,0.2);
    }
    .sgi-metric-label {
        font-size: 0.55rem;
        text-transform: uppercase;
        color: #7a869a;
        letter-spacing: 0.3px;
    }
    .sgi-metric-value {
        font-size: 1.0rem;
        font-weight: 600;
        color: #e8edf5;
        line-height: 1.2;
    }

    /* Pipeline */
    .sgi-pipeline {
        display: flex;
        align-items: center;
        gap: 0.2rem;
        flex-wrap: wrap;
        background: #0f131a;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        border: 1px solid #1e2630;
        margin: 0.2rem 0;
    }
    .sgi-pipeline-stage {
        display: flex;
        align-items: center;
        gap: 0.2rem;
        font-size: 0.65rem;
        color: #7a869a;
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
        background: transparent;
    }
    .sgi-pipeline-stage.active {
        color: #e8edf5;
        background: #1a2630;
    }
    .sgi-pipeline-stage.done {
        color: #4caf50;
    }
    .sgi-pipeline-stage .status-icon {
        font-size: 0.55rem;
    }
    .sgi-pipeline-arrow {
        color: #3a4a5a;
        font-size: 0.6rem;
        margin: 0 0.05rem;
    }

    /* Catálogo de cenas (tabela) */
    .sgi-scene-catalog {
        max-height: 180px;
        overflow-y: auto;
        border: 1px solid #1e2630;
        border-radius: 3px;
        background: #0b0e14;
        padding: 0.1rem;
    }
    .sgi-scene-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.1rem 0.3rem;
        border-bottom: 1px solid #1a222a;
        font-size: 0.7rem;
        color: #c8d0dc;
    }
    .sgi-scene-row:hover {
        background: #131a22;
    }
    .sgi-scene-row .date-col { width: 90px; }
    .sgi-scene-row .cloud-col { width: 60px; text-align: right; color: #7a869a; }
    .sgi-scene-row .quality-col { width: 70px; text-align: center; }
    .sgi-scene-row .action-col { width: 70px; text-align: right; }
    .sgi-scene-row .action-col button {
        background: transparent;
        border: 1px solid #2a3a4a;
        color: #c8d0dc;
        padding: 0.05rem 0.4rem;
        font-size: 0.6rem;
        border-radius: 2px;
        cursor: pointer;
        transition: all 0.15s;
    }
    .sgi-scene-row .action-col button:hover {
        background: #1a73e8;
        border-color: #1a73e8;
        color: white;
    }

    /* Imagens lado a lado */
    .sgi-image-pair {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.4rem;
        margin: 0.2rem 0;
    }
    .sgi-image-pair .image-container {
        background: #0b0e14;
        border-radius: 3px;
        overflow: hidden;
        border: 1px solid #1e2630;
    }
    .sgi-image-pair .image-container img {
        width: 100%;
        height: auto;
        display: block;
    }

    /* Mapa */
    .sgi-map-wrapper {
        border-radius: 3px;
        overflow: hidden;
        border: 1px solid #1e2630;
        margin: 0.2rem 0;
        background: #0b0e14;
    }

    /* Rodapé */
    .sgi-footer {
        border-top: 1px solid #1e2630;
        padding: 0.3rem 0 0.1rem 0;
        margin-top: 0.4rem;
        font-size: 0.55rem;
        color: #5a667a;
        text-align: center;
        letter-spacing: 0.3px;
    }

    /* Ajustes responsivos */
    @media (max-width: 768px) {
        .sgi-image-pair {
            grid-template-columns: 1fr;
        }
        .sgi-metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .sgi-header {
            flex-wrap: wrap;
        }
        .sgi-status {
            margin-top: 0.2rem;
        }
    }

    /* Ocultar elementos padrão do Streamlit */
    .stAlert {
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        margin: 0.15rem 0;
        border-radius: 3px;
    }
    .stAlert .stMarkdown {
        margin: 0;
    }
    .stSubheader {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #c8d0dc !important;
        margin: 0.15rem 0 0.1rem 0 !important;
        padding: 0 !important;
        border-bottom: none !important;
    }
    .stHeader { display: none; }
    .stButton button {
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 3px;
    }
    .streamlit-expanderHeader {
        font-size: 0.75rem;
        padding: 0.15rem 0.4rem;
        background: #0f131a;
        border-radius: 3px;
        border: 1px solid #1e2630;
    }
    .streamlit-expanderContent {
        padding: 0.15rem 0.4rem;
    }
    .dataframe {
        font-size: 0.65rem !important;
    }
    .dataframe td, .dataframe th {
        padding: 0.05rem 0.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)