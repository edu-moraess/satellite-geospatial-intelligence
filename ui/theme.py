from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """
    Apply the global visual identity of the platform.

    The design follows a dark geospatial / aerospace
    analytics interface.
    """

    st.markdown(
        """
        <style>

        /* =====================================================
           GLOBAL
        ===================================================== */

        .stApp {
            background:
                radial-gradient(
                    circle at 80% 0%,
                    rgba(30, 90, 120, 0.12),
                    transparent 35%
                ),
                #071014;
            color: #E8EEF2;
        }

        .main {
            background: transparent;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }


        /* =====================================================
           SIDEBAR
        ===================================================== */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #081216 0%,
                    #050A0D 100%
                );

            border-right:
                1px solid rgba(255,255,255,0.07);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.5rem;
        }


        /* =====================================================
           TYPOGRAPHY
        ===================================================== */

        h1 {
            font-size: 2.1rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em;
        }

        h2 {
            font-size: 1.45rem !important;
            font-weight: 650 !important;
        }

        h3 {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }

        p {
            color: #AAB8C0;
        }


        /* =====================================================
           HEADER
        ===================================================== */

        .sgi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;

            padding:
                1rem 1.25rem;

            margin-bottom: 1.5rem;

            border:
                1px solid rgba(255,255,255,0.07);

            border-radius: 14px;

            background:
                rgba(10, 20, 25, 0.78);

            backdrop-filter: blur(14px);
        }

        .sgi-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .sgi-logo {
            width: 42px;
            height: 42px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 11px;

            background:
                rgba(70, 160, 190, 0.12);

            border:
                1px solid rgba(90, 180, 210, 0.25);

            font-size: 1.25rem;
        }

        .sgi-title {
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }

        .sgi-subtitle {
            font-size: 0.72rem;
            color: #7F929B;
            margin-top: 0.15rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }


        /* =====================================================
           STATUS
        ===================================================== */

        .sgi-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;

            padding:
                0.45rem 0.75rem;

            border-radius: 999px;

            background:
                rgba(70, 170, 120, 0.08);

            border:
                1px solid rgba(70, 170, 120, 0.18);

            color: #8BD4AD;

            font-size: 0.72rem;
            font-weight: 600;
        }

        .sgi-status-dot {
            width: 7px;
            height: 7px;

            border-radius: 50%;

            background: #6FD39A;

            box-shadow:
                0 0 10px rgba(111,211,154,0.65);
        }


        /* =====================================================
           CARDS
        ===================================================== */

        .sgi-card {
            background:
                rgba(13, 24, 29, 0.82);

            border:
                1px solid rgba(255,255,255,0.065);

            border-radius: 14px;

            padding: 1rem;

            min-height: 100px;

            box-shadow:
                0 12px 30px rgba(0,0,0,0.14);
        }

        .sgi-card-label {
            color: #7F929B;

            font-size: 0.72rem;

            text-transform: uppercase;

            letter-spacing: 0.08em;

            margin-bottom: 0.55rem;
        }

        .sgi-card-value {
            color: #EDF4F7;

            font-size: 1.45rem;

            font-weight: 700;

            line-height: 1.1;
        }

        .sgi-card-description {
            color: #73858E;

            font-size: 0.72rem;

            margin-top: 0.45rem;
        }


        /* =====================================================
           SECTION
        ===================================================== */

        .sgi-section {
            display: flex;
            align-items: center;
            gap: 0.65rem;

            margin-top: 1.8rem;
            margin-bottom: 0.8rem;

            padding-bottom: 0.55rem;

            border-bottom:
                1px solid rgba(255,255,255,0.055);
        }

        .sgi-section-title {
            font-size: 1rem;
            font-weight: 650;
        }

        .sgi-section-description {
            font-size: 0.72rem;
            color: #71838C;
        }


        /* =====================================================
           MAP CONTAINER
        ===================================================== */

        .sgi-map-wrapper {
            border:
                1px solid rgba(255,255,255,0.07);

            border-radius: 14px;

            overflow: hidden;

            background: #081014;
        }


        /* =====================================================
           BUTTONS
        ===================================================== */

        .stButton > button {
            border-radius: 9px;

            border:
                1px solid rgba(255,255,255,0.09);

            background:
                rgba(255,255,255,0.035);

            color: #E7EFF2;

            font-weight: 600;

            transition:
                all 0.15s ease;
        }

        .stButton > button:hover {
            border-color:
                rgba(100,180,210,0.35);

            background:
                rgba(100,180,210,0.08);
        }


        /* =====================================================
           INPUTS
        ===================================================== */

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background:
                rgba(255,255,255,0.035);

            border-color:
                rgba(255,255,255,0.08);
        }


        /* =====================================================
           DATAFRAME
        ===================================================== */

        div[data-testid="stDataFrame"] {
            border:
                1px solid rgba(255,255,255,0.07);

            border-radius: 12px;

            overflow: hidden;
        }


        /* =====================================================
           ALERTS
        ===================================================== */

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }


        /* =====================================================
           FOOTER
        ===================================================== */

        .sgi-footer {
            margin-top: 3rem;

            padding-top: 1rem;

            border-top:
                1px solid rgba(255,255,255,0.06);

            color: #596B73;

            font-size: 0.7rem;

            text-align: center;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )