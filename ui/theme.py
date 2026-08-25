from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    """
    Apply the global visual identity of the platform.

    Dark geospatial / aerospace analytics interface.
    Focused on compact information hierarchy and
    operational dashboard presentation.
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
                    circle at 85% 0%,
                    rgba(30, 90, 120, 0.13),
                    transparent 32%
                ),
                linear-gradient(
                    180deg,
                    #071014 0%,
                    #081216 100%
                );

            color: #E8EEF2;
        }

        .main {
            background: transparent;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.15rem;
            padding-bottom: 2.5rem;
        }

        hr {
            border-color: rgba(255,255,255,0.055) !important;
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
            padding-top: 1.2rem;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            margin-top: 0.7rem;
        }


        /* =====================================================
           TYPOGRAPHY
        ===================================================== */

        h1 {
            font-size: 2rem !important;
            font-weight: 720 !important;
            letter-spacing: -0.035em;
        }

        h2 {
            font-size: 1.35rem !important;
            font-weight: 680 !important;
            letter-spacing: -0.02em;
        }

        h3 {
            font-size: 1rem !important;
            font-weight: 620 !important;
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

            padding: 0.9rem 1.15rem;
            margin-bottom: 0.9rem;

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
            gap: 0.8rem;
        }

        .sgi-logo {
            width: 40px;
            height: 40px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 10px;

            background:
                rgba(70, 160, 190, 0.12);

            border:
                1px solid rgba(90, 180, 210, 0.25);

            font-size: 1.2rem;
        }

        .sgi-title {
            font-size: 1.02rem;
            font-weight: 720;
            letter-spacing: 0.02em;
        }

        .sgi-subtitle {
            font-size: 0.69rem;
            color: #7F929B;
            margin-top: 0.12rem;
            letter-spacing: 0.075em;
            text-transform: uppercase;
        }

        .sgi-status {
            display: flex;
            align-items: center;
            gap: 0.45rem;

            padding: 0.4rem 0.7rem;

            border-radius: 999px;

            background:
                rgba(70, 170, 120, 0.08);

            border:
                1px solid rgba(70, 170, 120, 0.18);

            color: #8BD4AD;

            font-size: 0.68rem;
            font-weight: 650;
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
           KPI GRID
        ===================================================== */

        .sgi-kpi {
            background:
                rgba(13, 24, 29, 0.76);

            border:
                1px solid rgba(255,255,255,0.06);

            border-radius: 12px;

            padding: 0.75rem 0.9rem;

            min-height: 78px;

            box-shadow:
                0 8px 24px rgba(0,0,0,0.11);
        }

        .sgi-kpi-label {
            color: #71838C;
            font-size: 0.64rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }

        .sgi-kpi-value {
            color: #EDF4F7;
            font-size: 1.18rem;
            font-weight: 720;
            line-height: 1.05;
        }

        .sgi-kpi-sub {
            color: #687A82;
            font-size: 0.65rem;
            margin-top: 0.28rem;
        }


        /* =====================================================
           CARDS
        ===================================================== */

        .sgi-card {
            background:
                rgba(13, 24, 29, 0.82);

            border:
                1px solid rgba(255,255,255,0.065);

            border-radius: 13px;

            padding: 0.9rem;

            min-height: 88px;

            box-shadow:
                0 10px 26px rgba(0,0,0,0.12);
        }

        .sgi-card-label {
            color: #7F929B;
            font-size: 0.69rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.48rem;
        }

        .sgi-card-value {
            color: #EDF4F7;
            font-size: 1.32rem;
            font-weight: 720;
            line-height: 1.1;
        }

        .sgi-card-description {
            color: #73858E;
            font-size: 0.68rem;
            margin-top: 0.4rem;
        }


        /* =====================================================
           SECTION
        ===================================================== */

        .sgi-section {
            display: flex;
            align-items: baseline;
            gap: 0.55rem;

            margin-top: 1.15rem;
            margin-bottom: 0.65rem;

            padding-bottom: 0.45rem;

            border-bottom:
                1px solid rgba(255,255,255,0.055);
        }

        .sgi-section-title {
            font-size: 0.94rem;
            font-weight: 680;
            color: #E7EFF2;
        }

        .sgi-section-description {
            font-size: 0.67rem;
            color: #71838C;
        }


        /* =====================================================
           OPERATIONS CENTER
        ===================================================== */

        .sgi-operations {
            background:
                rgba(8, 16, 20, 0.72);

            border:
                1px solid rgba(255,255,255,0.07);

            border-radius: 15px;

            padding: 0.75rem;

            margin-bottom: 0.75rem;

            box-shadow:
                0 12px 35px rgba(0,0,0,0.15);
        }

        .sgi-operations-title {
            display: flex;
            justify-content: space-between;
            align-items: center;

            padding: 0.15rem 0.35rem 0.65rem;
        }

        .sgi-operations-name {
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .sgi-operations-meta {
            color: #657780;
            font-size: 0.66rem;
        }


        /* =====================================================
           MAP
        ===================================================== */

        .sgi-map-wrapper {
            border:
                1px solid rgba(255,255,255,0.07);

            border-radius: 13px;

            overflow: hidden;

            background: #081014;
        }


        /* =====================================================
           TABS
        ===================================================== */

        button[data-baseweb="tab"] {
            color: #81939B !important;
            font-weight: 650 !important;
            font-size: 0.76rem !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #DCE9ED !important;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
        }


        /* =====================================================
           BUTTONS
        ===================================================== */

        .stButton > button {
            border-radius: 8px;

            border:
                1px solid rgba(255,255,255,0.09);

            background:
                rgba(255,255,255,0.035);

            color: #E7EFF2;

            font-weight: 620;

            transition: all 0.15s ease;
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

            border-radius: 11px;

            overflow: hidden;
        }


        /* =====================================================
           EXPANDERS
        ===================================================== */

        div[data-testid="stExpander"] {
            border:
                1px solid rgba(255,255,255,0.055);

            border-radius: 10px;

            background:
                rgba(255,255,255,0.018);
        }


        /* =====================================================
           ALERTS
        ===================================================== */

        div[data-testid="stAlert"] {
            border-radius: 9px;
        }


        /* =====================================================
           PIPELINE
        ===================================================== */

        .sgi-pipeline {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            width: 100%;
            overflow-x: auto;
            padding: 0.25rem 0;
        }

        .sgi-pipeline-stage {
            display: flex;
            align-items: center;
            gap: 0.35rem;

            padding: 0.45rem 0.65rem;

            border-radius: 8px;

            background:
                rgba(255,255,255,0.025);

            border:
                1px solid rgba(255,255,255,0.055);

            color: #778991;

            font-size: 0.65rem;
            white-space: nowrap;
        }

        .sgi-pipeline-stage.active {
            color: #B9E2CE;
            border-color:
                rgba(111,211,154,0.18);

            background:
                rgba(70,170,120,0.055);
        }

        .sgi-pipeline-arrow {
            color: #44545B;
            font-size: 0.65rem;
        }


        /* =====================================================
           FOOTER
        ===================================================== */

        .sgi-footer {
            margin-top: 2rem;
            padding-top: 0.9rem;

            border-top:
                1px solid rgba(255,255,255,0.06);

            color: #596B73;

            font-size: 0.66rem;
            text-align: center;
        }


        /* =====================================================
           MOBILE
        ===================================================== */

        @media (max-width: 900px) {

            .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }

            .sgi-header {
                padding: 0.75rem;
            }

            .sgi-status {
                display: none;
            }

            h1 {
                font-size: 1.65rem !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )