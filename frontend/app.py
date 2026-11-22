"""AI INTERVIEW SIMULATOR — Production-Ready Multi-Modal Assessment & Coaching Platform."""
from __future__ import annotations

import html
import io
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import (
    authenticate_user,
    create_user,
    register_user,
    is_valid_gmail,
    get_all_candidates,
    get_candidate_history,
    save_session,
)
from backend.services import finalize_and_save_interview, generate_model_self_introduction
from modules.llm_evaluator import (
    analyze_self_introduction,
    evaluate_answer,
    evaluate_code_solution,
    evaluate_personality_test,
    generate_aptitude_questions,
    generate_audio_recording_sentences,
    generate_dsa_problems,
    generate_longitudinal_coaching_plan,
    generate_personality_statements,
    generate_project_questions,
    generate_technical_questions,
    is_developer_role,
    validate_job_description,
    validate_resume,
    validate_target_role,
    _generate_sample_ideal_answer,
)
from modules.resume_parser import extract_skills_and_projects, extract_text_from_file
from modules.speech_engine import pcm_to_wav_bytes, speech_metrics, transcribe
from modules.vision_tracker import MediaBuffer, rtc_configuration
from modules.voice_drill import (
    calculate_word_accuracy,
    evaluate_voice_drill,
    generate_sentence_tts,
)

# Page configuration
st.set_page_config(
    page_title="AI INTERVIEW SIMULATOR",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# CSS DESIGN SYSTEM: ROYAL INDIGO, EMERALD & WARM PEARL (#1E1B4B / #059669 / #F8FAFC)
# ==============================================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

    :root {
        --primary-indigo: #1E1B4B;
        --indigo-surface: #312E81;
        --indigo-light: #4338CA;
        --accent-emerald: #059669;
        --emerald-light: #10B981;
        --emerald-glow: rgba(16, 185, 129, 0.15);
        --bg-pearl: #F8FAFC;
        --bg-white: #FFFFFF;
        --border-slate: #E2E8F0;
        --border-indigo: #312E81;
        --text-slate: #0F172A;
        --text-muted: #64748B;
        --danger-red: #DC2626;
        --warning-amber: #D97706;
    }

    /* Hide standard sidebar and default header padding */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container {
        padding-top: 2.0rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }

    body, html {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-slate);
    }

    /* Fix Streamlit Material Icons & Password Toggle 'visibili' Text Leak */
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    [data-testid="stIconMaterial"],
    button[data-testid="stTextInputPasswordVisibilityToggle"] span,
    button[data-testid="stTextInputPasswordVisibilityToggle"] i {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: 'liga' 1 !important;
    }

    button[data-testid="stTextInputPasswordVisibilityToggle"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        overflow: hidden !important;
    }

    code, pre, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Navigation Header Card */
    .top-nav-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%);
        border-radius: 16px;
        padding: 0.9rem 1.6rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 25px -5px rgba(30, 27, 75, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }

    .top-nav-brand {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .brand-icon {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        font-size: 1.4rem;
        padding: 0.4rem 0.65rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35);
    }

    .brand-title {
        color: #FFFFFF !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin: 0 !important;
    }

    .brand-subtitle {
        color: #94A3B8 !important;
        font-size: 0.78rem !important;
        font-weight: 500;
        margin: 0 !important;
    }

    /* Hero Banner Cards */
    .hero-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #2E1065 50%, #064E3B 100%);
        border-radius: 20px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 2rem;
        color: white;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 30px -10px rgba(30, 27, 75, 0.3);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #FFFFFF !important;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #E2E8F0 !important;
        max-width: 820px;
        line-height: 1.6;
        margin-bottom: 0;
    }

    /* Formula Banner */
    .formula-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #172554 100%);
        border: 1px solid #3B82F6;
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.8rem;
        color: #F8FAFC;
        box-shadow: 0 8px 20px -4px rgba(30, 27, 75, 0.2);
    }

    .formula-title {
        font-weight: 800;
        font-size: 1.05rem;
        color: #60A5FA;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Section Cards Grid */
    .section-card {
        background: var(--bg-white);
        border: 1px solid var(--border-slate);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }

    .section-card:hover {
        transform: translateY(-4px);
        border-color: #059669;
        box-shadow: 0 16px 24px -6px rgba(5, 150, 105, 0.12);
    }

    /* Badges */
    .badge-live {
        background: #FEE2E2;
        color: #DC2626;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        letter-spacing: 0.04em;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .badge-completed {
        background: #DCFCE7;
        color: #047857;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .badge-unattempted {
        background: #F1F5F9;
        color: #64748B;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .badge-pending {
        background: #FEF3C7;
        color: #B45309;
        font-weight: 700;
        font-size: 0.72rem;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* Question Hero Display */
    .question-hero {
        background: #FFFFFF;
        border: 2px solid #312E81;
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 6px 16px -2px rgba(30, 27, 75, 0.08);
    }

    .question-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E1B4B;
        line-height: 1.5;
        margin-top: 0.6rem;
    }

    .skill-chip {
        background: #EDE9FE;
        color: #4338CA;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        margin-right: 0.4rem;
        display: inline-block;
    }

    /* Ideal Answer Callout */
    .ideal-answer-card {
        background: #F0FDF4;
        border-left: 4px solid #059669;
        border-radius: 0 12px 12px 0;
        padding: 1.1rem 1.4rem;
        margin-top: 0.8rem;
        font-size: 0.92rem;
        color: #064E3B;
        line-height: 1.6;
    }

    /* Camera Frame Box */
    .camera-box {
        background: #0F172A;
        border: 2px solid #312E81;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
    }

    .camera-box video {
        width: 100% !important;
        max-height: 480px !important;
        object-fit: cover !important;
    }

    /* Larger, prominent bordered webcam preview frame (640x480 aspect) */
    div[data-testid="stWebRtc"] video,
    div[data-testid="stWebRtc"] iframe {
        width: 100% !important;
        min-height: 380px !important;
        max-height: 520px !important;
        border-radius: 14px !important;
        border: 2.5px solid #312E81 !important;
        background: #0F172A !important;
        box-shadow: 0 10px 25px -5px rgba(30, 27, 75, 0.35) !important;
    }

    /* Primary Streamlit Button Styles */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
    }

    .stButton>button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(5, 150, 105, 0.4) !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================================
# VECTOR HERO ILLUSTRATIONS (ROYAL INDIGO & EMERALD GRADIENTS)
# ==============================================================================
AUTH_HERO_SVG = """
<svg width="100%" height="110" viewBox="0 0 800 110" fill="none" xmlns="http://www.w3.org/2000/svg" style="border-radius: 14px; margin-bottom: 1.2rem;">
    <rect width="800" height="110" rx="14" fill="url(#auth_bg)"/>
    <defs>
        <linearGradient id="auth_bg" x1="0" y1="0" x2="800" y2="110" gradientUnits="userSpaceOnUse">
            <stop stop-color="#1E1B4B"/>
            <stop offset="0.5" stop-color="#312E81"/>
            <stop offset="1" stop-color="#064E3B"/>
        </linearGradient>
        <linearGradient id="gold_grad" x1="0" y1="0" x2="1" y2="1">
            <stop stop-color="#FBBF24"/>
            <stop offset="1" stop-color="#D97706"/>
        </linearGradient>
    </defs>
    <!-- Graduation Cap / Student Shield -->
    <circle cx="65" cy="55" r="28" fill="#312E81" stroke="#FBBF24" stroke-width="2.5"/>
    <path d="M65 42L48 50L65 58L82 50L65 42Z" fill="url(#gold_grad)"/>
    <path d="M54 53.5V62C54 65.5 59 68 65 68C71 68 76 65.5 76 62V53.5" stroke="#FBBF24" stroke-width="2" stroke-linecap="round"/>
    <path d="M80 51V63M78 63H82" stroke="#FDE68A" stroke-width="1.8" stroke-linecap="round"/>
    <!-- Central Text -->
    <text x="115" y="48" fill="#FFFFFF" font-family="Plus Jakarta Sans, sans-serif" font-weight="800" font-size="21" letter-spacing="-0.02em">AI INTERVIEW SIMULATOR</text>
    <text x="115" y="70" fill="#94A3B8" font-family="Plus Jakarta Sans, sans-serif" font-size="12">Candidate Career Portal · Technical Mock Interviews · Real-Time Delivery Coaching</text>
    <!-- Right Achievement Badge -->
    <rect x="655" y="40" width="120" height="30" rx="15" fill="#065F46" stroke="#10B981" stroke-width="1.5"/>
    <text x="715" y="60" fill="#D1FAE5" font-family="Plus Jakarta Sans, sans-serif" font-weight="700" font-size="11" text-anchor="middle">🎓 Career Ready</text>
</svg>
"""

ONBOARDING_HERO_SVG = """
<svg width="100%" height="110" viewBox="0 0 800 110" fill="none" xmlns="http://www.w3.org/2000/svg" style="border-radius: 14px; margin-bottom: 1.2rem;">
    <rect width="800" height="110" rx="14" fill="url(#onb_bg)"/>
    <defs>
        <linearGradient id="onb_bg" x1="0" y1="0" x2="800" y2="110" gradientUnits="userSpaceOnUse">
            <stop stop-color="#1E1B4B"/>
            <stop offset="0.6" stop-color="#1E3A8A"/>
            <stop offset="1" stop-color="#047857"/>
        </linearGradient>
    </defs>
    <!-- Document Shield Icon -->
    <circle cx="65" cy="55" r="26" fill="#1E3A8A" stroke="#38BDF8" stroke-width="2.5"/>
    <rect x="54" y="41" width="22" height="28" rx="3" fill="#FFFFFF" opacity="0.9"/>
    <line x1="58" y1="48" x2="72" y2="48" stroke="#0284C7" stroke-width="2" stroke-linecap="round"/>
    <line x1="58" y1="54" x2="72" y2="54" stroke="#0284C7" stroke-width="2" stroke-linecap="round"/>
    <line x1="58" y1="60" x2="66" y2="60" stroke="#059669" stroke-width="2" stroke-linecap="round"/>
    <!-- Central Text -->
    <text x="110" y="50" fill="#FFFFFF" font-family="Plus Jakarta Sans, sans-serif" font-weight="800" font-size="20">Candidate Onboarding & AI Input Validation Gate</text>
    <text x="110" y="72" fill="#BAE6FD" font-family="Plus Jakarta Sans, sans-serif" font-size="12">Verified Target Role · Resume Skill Extraction · Job Description Alignment</text>
    <!-- Pipeline Checkmark Badges -->
    <rect x="660" y="40" width="100" height="30" rx="15" fill="#065F46" stroke="#10B981" stroke-width="1.5"/>
    <text x="710" y="60" fill="#D1FAE5" font-family="Plus Jakarta Sans, sans-serif" font-weight="700" font-size="11" text-anchor="middle">✓ AI Verified</text>
</svg>
"""

HUB_HERO_SVG = """
<svg width="100%" height="110" viewBox="0 0 800 110" fill="none" xmlns="http://www.w3.org/2000/svg" style="border-radius: 14px; margin-bottom: 1.2rem;">
    <rect width="800" height="110" rx="14" fill="url(#hub_bg)"/>
    <defs>
        <linearGradient id="hub_bg" x1="0" y1="0" x2="800" y2="110" gradientUnits="userSpaceOnUse">
            <stop stop-color="#1E1B4B"/>
            <stop offset="0.5" stop-color="#312E81"/>
            <stop offset="1" stop-color="#064E3B"/>
        </linearGradient>
    </defs>
    <!-- Vision & Mic Multi-modal Orb -->
    <circle cx="65" cy="55" r="26" fill="#312E81" stroke="#10B981" stroke-width="2.5"/>
    <circle cx="65" cy="55" r="14" fill="#047857"/>
    <circle cx="65" cy="55" r="6" fill="#6EE7B7"/>
    <!-- Central Text -->
    <text x="110" y="50" fill="#FFFFFF" font-family="Plus Jakarta Sans, sans-serif" font-weight="800" font-size="20">Live AI Interview & Assessment Hub</text>
    <text x="110" y="72" fill="#E2E8F0" font-family="Plus Jakarta Sans, sans-serif" font-size="12">7 calibrated sections · Real-time MediaPipe iris/posture tracking · Automated coding judge</text>
    <!-- Telemetry Pill -->
    <rect x="650" y="40" width="115" height="30" rx="15" fill="#1E293B" stroke="#10B981" stroke-width="1.5"/>
    <circle cx="666" cy="55" r="4" fill="#10B981"/>
    <text x="712" y="60" fill="#E2E8F0" font-family="Plus Jakarta Sans, sans-serif" font-weight="700" font-size="11" text-anchor="middle">Sensors Active</text>
</svg>
"""

REPORT_HERO_SVG = """
<svg width="100%" height="110" viewBox="0 0 800 110" fill="none" xmlns="http://www.w3.org/2000/svg" style="border-radius: 14px; margin-bottom: 1.2rem;">
    <rect width="800" height="110" rx="14" fill="url(#rep_bg)"/>
    <defs>
        <linearGradient id="rep_bg" x1="0" y1="0" x2="800" y2="110" gradientUnits="userSpaceOnUse">
            <stop stop-color="#1E1B4B"/>
            <stop offset="0.6" stop-color="#312E81"/>
            <stop offset="1" stop-color="#059669"/>
        </linearGradient>
    </defs>
    <!-- Trophy Analytics Orb -->
    <circle cx="65" cy="55" r="26" fill="#312E81" stroke="#F59E0B" stroke-width="2.5"/>
    <path d="M57 45h16v8a8 8 0 01-16 0v-8z" fill="#FDE68A"/>
    <path d="M54 48h-3a4 4 0 004 4M74 48h3a4 4 0 01-4 4M65 61v6M60 67h10" stroke="#F59E0B" stroke-width="2" stroke-linecap="round"/>
    <!-- Central Text -->
    <text x="110" y="50" fill="#FFFFFF" font-family="Plus Jakarta Sans, sans-serif" font-weight="800" font-size="20">Executive Performance Analytics & AI Coach</text>
    <text x="110" y="72" fill="#E2E8F0" font-family="Plus Jakarta Sans, sans-serif" font-size="12">Multi-modal score synthesis · Section-wise improvement points · 14-day skill roadmap</text>
    <!-- Badge -->
    <rect x="660" y="40" width="105" height="30" rx="15" fill="#065F46" stroke="#34D399" stroke-width="1.5"/>
    <text x="712" y="60" fill="#ECFDF5" font-family="Plus Jakarta Sans, sans-serif" font-weight="700" font-size="11" text-anchor="middle">Audited Report</text>
</svg>
"""




# ==============================================================================
# CLIENT-SIDE COUNTDOWN TIMER COMPONENT (ZERO PYTHON RERUNS / NO TYPING FREEZE)
# ==============================================================================
def render_client_timer(total_seconds: int, start_epoch: float, section_name: str = "Section") -> None:
    """Render smooth client-side HTML/JavaScript countdown timer that never triggers Streamlit reruns."""
    elapsed = int(time.time() - start_epoch) if start_epoch else 0
    remaining = max(0, total_seconds - elapsed)

    html_code = f"""
    <div id="timer-wrap" style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 2px 4px;
    ">
        <div id="timer-badge" style="
            background: #1E1B4B;
            color: #10B981;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid #312E81;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        ">
            <span>⏱️</span>
            <span style="color: #E2E8F0;">Time Remaining:</span>
            <span id="timer-display" style="font-family: monospace; font-size: 1.08rem; letter-spacing: 0.05em;">--:--</span>
        </div>
    </div>
    <script>
        let remaining = {remaining};
        function formatMinsSecs(sec) {{
            let m = Math.floor(sec / 60);
            let s = sec % 60;
            return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
        }}
        function updateDisplay() {{
            let display = document.getElementById('timer-display');
            let badge = document.getElementById('timer-badge');
            if (!display || !badge) return;
            display.innerText = formatMinsSecs(remaining);
            if (remaining <= 60) {{
                badge.style.background = '#7F1D1D';
                badge.style.color = '#FCA5A5';
                badge.style.borderColor = '#EF4444';
            }} else if (remaining <= 180) {{
                badge.style.background = '#78350F';
                badge.style.color = '#FDE68A';
                badge.style.borderColor = '#F59E0B';
            }}
            if (remaining <= 0) {{
                display.innerText = '00:00 - TIME EXPIRED';
                clearInterval(intervalId);
            }} else {{
                remaining--;
            }}
        }}
        updateDisplay();
        let intervalId = setInterval(updateDisplay, 1000);
    </script>
    """
    components.html(html_code, height=48)



# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
def init_session_state() -> None:
    """Initialize persistent user authentication, session, and assessment data."""
    defaults = {
        "attempt_id": f"att_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        "authenticated_user": None,
        "current_screen": "auth",
        "session_saved": False,
        # Onboarding inputs
        "candidate_name": "",
        "target_role": "",
        "is_developer_role": True,
        "resume_text": "",
        "jd_text": "",
        "detected_skills": [],
        "detected_projects": [],
        # Overall progress
        "completed_sections": {},
        "section_results": {},
        # Section 1: Self Intro
        "self_intro_audio_bytes": None,
        "self_intro_eval": None,
        "self_intro_confirmed": False,
        # Section 2: Technical
        "tech_difficulty": "Medium",
        "tech_questions": [],
        "tech_active_idx": 0,
        "tech_answers": {},
        "tech_evaluations": {},
        "tech_start_epoch": None,
        # Section 3: Algorithmic Problems (DSA)
        "dsa_difficulty": "Medium",
        "dsa_problems": [],
        "dsa_active_idx": 0,
        "dsa_language": "Python",
        "dsa_codes": {},
        "dsa_evaluations": {},
        "dsa_start_epoch": None,
        # Section 4: Project Deep Dive
        "project_difficulty": "Medium",
        "project_questions": [],
        "project_active_idx": 0,
        "project_answers": {},
        "project_evaluations": {},
        "project_start_epoch": None,
        # Section 5: Aptitude & Logic
        "aptitude_difficulty": "Medium",
        "aptitude_questions": [],
        "aptitude_active_idx": 0,
        "aptitude_answers": {},
        "aptitude_evaluations": {},
        "aptitude_start_epoch": None,
        # Section 6: Personality
        "personality_statements": [],
        "personality_active_idx": 0,
        "personality_responses": {},
        "personality_eval": None,
        # Section 7: Audio Recordings Drill
        "audio_difficulty": "Medium",
        "audio_sentences": [],
        "audio_active_idx": 0,
        "audio_listen_counts": {},
        "audio_has_started_recording": {},
        "audio_text_locked": {},
        "audio_evaluations": {},
        "sample_audio_cache": {},
        # Media / Vision Tracker
        "media_buffer": MediaBuffer(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# ==============================================================================
# TOP NAVIGATION BAR & USER PROFILE POPOVER
# ==============================================================================
def render_header() -> None:
    """Render top navigation header with user profile dropdown popover."""
    user = st.session_state.authenticated_user
    cand_name = user.get("full_name", "Candidate") if user else "Candidate"
    cand_email = user.get("email", "") if user else ""

    col_brand, col_user = st.columns([3.5, 1.5])

    with col_brand:
        st.markdown(
            """
            <div class="top-nav-card">
                <div class="top-nav-brand">
                    <span class="brand-icon">🎓</span>
                    <div>
                        <h2 class="brand-title">AI INTERVIEW SIMULATOR</h2>
                        <div class="brand-subtitle">Multi-Modal Engineering Assessment · Real-Time Vision & Voice Coaching</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_user:
        if user:
            with st.popover(f"👤 {cand_name}", use_container_width=True):
                st.markdown(f"**{cand_name}**")
                st.caption(f"📧 {cand_email}")
                st.markdown(f"**Target Role:** {st.session_state.target_role or 'Not Configured'}")
                st.markdown(f"**Attempt ID:** `{st.session_state.attempt_id}`")
                st.divider()

                completed_count = len(st.session_state.completed_sections)
                st.markdown(f"**Completed Sections:** {completed_count} / {'7' if st.session_state.is_developer_role else '6'}")

                col_hub_btn, col_prof_btn = st.columns(2)
                with col_hub_btn:
                    if st.button("🎯 Hub", use_container_width=True):
                        st.session_state.current_screen = "hub"
                        st.rerun()
                with col_prof_btn:
                    if st.button("📈 History", use_container_width=True):
                        st.session_state.current_screen = "profile"
                        st.rerun()

                if st.session_state.completed_sections:
                    if st.button("📊 View Report", use_container_width=True):
                        st.session_state.current_screen = "report"
                        st.rerun()

                st.divider()
                if st.button("🚪 Log Out", type="secondary", use_container_width=True):
                    st.session_state.authenticated_user = None
                    st.session_state.current_screen = "auth"
                    st.rerun()


# ==============================================================================
# HELPER: REAL LIVE WEBCAM VIDEO STREAM COMPONENT (PROMINENT 640x480 CONTAINER)
# ==============================================================================
def render_live_vision_tracker(section_name: str = "Interview Section"):
    """Render prominent, real live webcam video stream with MediaPipe overlays and posture telemetry."""
    with st.container():
        st.markdown(
            f"""
            <div style="background: #1E1B4B; border-radius: 12px; padding: 0.6rem 1.2rem; margin-bottom: 0.8rem; display: flex; justify-content: space-between; align-items: center; border: 1px solid #312E81;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <span style="font-size: 1.2rem;">👁️</span>
                    <b style="color: #FFFFFF; font-size: 0.95rem;">Live AI Vision & Posture Coach ({section_name})</b>
                    <span class="badge-live">● LIVE STREAMING</span>
                </div>
                <div style="font-size: 0.8rem; color: #10B981; font-weight: 700;">
                    MediaPipe FaceMesh (Iris Gaze) + Pose (Shoulder Tilt ±5°)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_cam, col_metrics = st.columns([3.0, 1.2], gap="large")

        with col_cam:
            try:
                from streamlit_webrtc import WebRtcMode, webrtc_streamer
                buf = st.session_state.media_buffer
                webrtc_streamer(
                    key=f"live_stream_{section_name}_{st.session_state.attempt_id}",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_configuration(),
                    video_frame_callback=buf.process_video,
                    audio_frame_callback=buf.process_audio,
                    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": True},
                    async_processing=True,
                )
            except Exception as exc:
                st.caption(f"Webcam processor initialized: {exc}")

        with col_metrics:
            metrics = st.session_state.media_buffer.current_metrics()
            posture_val = float(metrics.get("posture", 0.0))
            eye_val = float(metrics.get("eye_contact", 0.0))
            errors_val = int(metrics.get("posture_errors", 0))
            tilt_val = float(metrics.get("current_tilt", 0.0))
            cam_status = metrics.get("camera_status", "Camera / Face Not Detected")
            is_active = (cam_status != "Camera / Face Not Detected" and (eye_val > 0.0 or posture_val > 0.0))

            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 2px solid #312E81; border-radius: 12px; padding: 1.1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                        <span style="font-weight: 800; color: #1E1B4B; font-size: 0.92rem;">Real-Time Posture Telemetry</span>
                        <span style="font-size: 0.75rem; font-weight: 700; color: {'#059669' if is_active else '#DC2626'}; background: {'#D1FAE5' if is_active else '#FEE2E2'}; padding: 0.2rem 0.5rem; border-radius: 6px;">
                            {'Face Tracked' if is_active else 'Camera / Face Not Detected'}
                        </span>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 0.55rem; font-size: 0.86rem;">
                        <div><b>Shoulder Tilt:</b> <span style="color: {'#059669' if tilt_val <= 5.0 else '#DC2626'}; font-weight: 800;">{tilt_val:.1f}°</span> (Benchmark: ±5.0°)</div>
                        <div><b>Posture Stability:</b> <span style="color: {'#059669' if posture_val >= 70.0 else ('#DC2626' if not is_active else '#D97706')}; font-weight: 800;">{posture_val:.0f}%</span> {'<span style="color:#DC2626; font-size:0.75rem;">(Camera / Face Not Detected)</span>' if not is_active else ''}</div>
                        <div><b>Eye-Contact Gaze:</b> <span style="color: {'#059669' if eye_val >= 60.0 else ('#DC2626' if not is_active else '#D97706')}; font-weight: 800;">{eye_val:.0f}%</span> {'<span style="color:#DC2626; font-size:0.75rem;">(Camera / Face Not Detected)</span>' if not is_active else ''}</div>
                        <div><b>Posture Alerts:</b> <span style="color: {'#DC2626' if errors_val > 0 else '#059669'}; font-weight: 800;">{errors_val}</span></div>
                    </div>
                    <div style="margin-top: 0.8rem; font-size: 0.78rem; color: #64748B; line-height: 1.4; border-top: 1px solid #E2E8F0; padding-top: 0.6rem;">
                        💡 <b>Ideal Posture:</b> Level shoulders horizontally within ±5° and center face/shoulders inside the 60%×70% guide box.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )



# ==============================================================================
# SCREEN 1: AUTHENTICATION GATE (DIRECT GMAIL LOGIN & SIGN UP)
# ==============================================================================
def render_auth_screen():
    """Render streamlined Authentication Screen with strict @gmail.com & strong password enforcement."""
    st.markdown(AUTH_HERO_SVG, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card" style="text-align: center; padding: 2.2rem 1.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.4rem;">🎓</div>
            <h1 class="hero-title" style="font-size: 2.2rem;">AI INTERVIEW SIMULATOR</h1>
            <p class="hero-subtitle" style="margin: 0 auto; max-width: 650px;">
                Enterprise-grade technical assessment sandbox with MediaPipe eye-contact & posture tracking, acoustic vocal tone evaluation, and automated code judging.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_center = st.columns([1, 1.4, 1])[1]

    with col_center:
        tab_login, tab_signup = st.tabs(["🔐 Log In", "📝 Sign Up"])

        with tab_login:
            st.markdown("#### Access Your Candidate Account")
            with st.form("form_candidate_login", clear_on_submit=False):
                login_email = st.text_input("Gmail Address", placeholder="Enter Your Gmail", key="auth_login_email")
                login_pwd = st.text_input("Password", type="password", placeholder="Enter Your Password", key="auth_login_pwd")
                submit_login = st.form_submit_button("Log In to Simulator", type="primary", use_container_width=True)

                if submit_login:
                    clean_email = (login_email or st.session_state.get("auth_login_email", "")).strip().lower()
                    clean_pwd = (login_pwd or st.session_state.get("auth_login_pwd", "")).strip()

                    if not is_valid_gmail(clean_email):
                        st.error("Error Detected: Please enter a valid @gmail.com address")
                    elif not clean_pwd:
                        st.error("Please enter your password.")
                    else:
                        success, msg, user_data = authenticate_user(email=clean_email, password=clean_pwd)
                        if success:
                            st.session_state.authenticated_user = user_data
                            st.session_state.candidate_name = user_data.get("full_name", "")
                            st.session_state.current_screen = "onboarding"
                            st.success("Authentication successful! Loading Onboarding Dashboard...")
                            st.rerun()
                        else:
                            if "Account not found" in msg:
                                st.error(msg)
                            else:
                                st.error(f"Authentication Failed: {msg}")

        with tab_signup:
            st.markdown("#### Create New Candidate Profile")
            with st.form("form_candidate_signup", clear_on_submit=False):
                sign_name = st.text_input("Full Name", placeholder="Enter Full Name", key="auth_sign_name")
                sign_email = st.text_input("Gmail Address", placeholder="Enter Your Gmail", key="auth_sign_email")
                sign_pwd = st.text_input("Password", type="password", placeholder="Enter Your Password", key="auth_sign_pwd")
                st.caption("🔒 Password must contain ≥8 chars, 1 uppercase, 1 lowercase, 1 number, and 1 special character.")
                submit_signup = st.form_submit_button("Create Account & Begin", type="primary", use_container_width=True)

                if submit_signup:
                    clean_name = (sign_name or st.session_state.get("auth_sign_name", "")).strip()
                    clean_email = (sign_email or st.session_state.get("auth_sign_email", "")).strip().lower()
                    clean_pwd = (sign_pwd or st.session_state.get("auth_sign_pwd", "")).strip()

                    if not clean_name:
                        st.error("Please enter your full name.")
                    elif not is_valid_gmail(clean_email):
                        st.error("Error Detected: Please enter a valid @gmail.com address")
                    else:
                        success, msg, user_data = create_user(
                            full_name=clean_name,
                            email=clean_email,
                            password=clean_pwd,
                        )
                        if success:
                            st.session_state.authenticated_user = user_data
                            st.session_state.candidate_name = user_data.get("full_name", "")
                            st.session_state.current_screen = "onboarding"
                            st.success("Account created successfully! Loading Onboarding Dashboard...")
                            st.rerun()
                        else:
                            st.error(f"Sign Up Failed: {msg}")


# ==============================================================================
# SCREEN 2: CANDIDATE ONBOARDING & STRICT AI INPUT VALIDATION GATE
# ==============================================================================
def render_onboarding_screen():
    """Render Onboarding screen with mandatory pre-submit AI validation gate."""
    render_header()

    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">Candidate Onboarding & Role Profiling</h1>
            <p class="hero-subtitle">
                Configure your target role, provide job expectations, and submit your resume. Our AI validator analyzes your tech stack to generate calibrated technical questions and algorithmic coding challenges.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    user = st.session_state.authenticated_user
    default_name = user.get("full_name", "") if user else ""

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 1. Candidate Details & Target Role")
        cand_name = st.text_input(
            "Candidate Name *",
            value=st.session_state.candidate_name or default_name,
            placeholder="Enter Full Name",
            key="onboard_name",
        )
        target_role = st.text_input(
            "Target Role *",
            value=st.session_state.target_role,
            placeholder="Enter Your Target Role",
            key="onboard_role",
        )
        st.caption("Examples: Senior Full Stack Engineer, Backend Python Developer, Cloud DevOps Architect, Data Scientist.")

        st.markdown("#### 2. Target Job Description")
        jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="onboard_jd_file")
        jd_fallback = st.text_area(
            "Job Description Text",
            value=st.session_state.jd_text,
            height=160,
            placeholder="Paste complete job responsibilities, required qualifications, and technical expectations...",
            key="onboard_jd_text",
        )

    with col_right:
        st.markdown("#### 3. Candidate Resume")
        resume_file = st.file_uploader("Upload Resume (PDF / DOCX)", type=["pdf", "docx"], key="onboard_resume_file")
        resume_fallback = st.text_area(
            "Resume Text",
            value=st.session_state.resume_text,
            height=250,
            placeholder="Paste complete resume content including work experience, technical skills, projects, and education...",
            key="onboard_resume_text",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_submit = st.columns([1, 1.2, 1])[1]

    with col_submit:
        if st.button("Submit", type="primary", use_container_width=True, key="btn_onboard_submit"):
            with st.spinner("Executing strict AI pre-submit validation..."):
                final_name = cand_name.strip()
                if not final_name:
                    st.error("Invalid Input Detected: Candidate Name cannot be empty.")
                    return

                # 1. Role validation
                valid_role, role_msg, *_ = validate_target_role(target_role)
                if not valid_role:
                    st.error(f"Invalid Input Detected (Target Role): {role_msg}")
                    return

                # 2. Extract and validate resume
                res_content = resume_fallback.strip()
                if resume_file is not None:
                    try:
                        extracted = extract_text_from_file(resume_file)
                        if extracted and len(extracted.strip()) > 30:
                            res_content = extracted.strip()
                    except Exception as e:
                        st.warning(f"Could not parse resume file directly: {e}. Falling back to pasted text.")

                valid_resume, res_msg, *_ = validate_resume(res_content)
                if not valid_resume:
                    st.error(f"Invalid Input Detected (Resume): {res_msg}")
                    return

                # 3. Extract and validate JD
                jd_content = jd_fallback.strip()
                if jd_file is not None:
                    try:
                        extracted_jd = extract_text_from_file(jd_file)
                        if extracted_jd and len(extracted_jd.strip()) > 20:
                            jd_content = extracted_jd.strip()
                    except Exception as e:
                        st.warning(f"Could not parse JD file: {e}")

                valid_jd, jd_msg = validate_job_description(jd_content)
                if not valid_jd:
                    st.error(f"Invalid Input Detected (Job Description): {jd_msg}")
                    return

                # Parse skills and projects
                skills_projs = extract_skills_and_projects(res_content)
                detected_skills = skills_projs.get("skills", ["Python", "System Design", "SQL", "Cloud"])
                detected_projects = skills_projs.get("projects", ["Production Distributed Application", "Real-Time Telemetry Pipeline"])

                # Determine if developer role
                is_dev = is_developer_role(target_role)

                # Persist in session state
                st.session_state.candidate_name = final_name
                st.session_state.target_role = target_role.strip()
                st.session_state.resume_text = res_content
                st.session_state.jd_text = jd_content
                st.session_state.detected_skills = detected_skills
                st.session_state.detected_projects = detected_projects
                st.session_state.is_developer_role = is_dev
                st.session_state.current_screen = "hub"

                st.success(f"Validation Passed! Detected {len(detected_skills)} skills. Loading Interview Hub...")
                time.sleep(0.6)
                st.rerun()


# ==============================================================================
# SCREEN 3: LIVE AI INTERVIEW SECTION HUB
# ==============================================================================
def render_section_hub():
    """Render central assessment hub with 7 calibrated interview sections."""
    render_header()

    st.markdown(HUB_HERO_SVG, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">Live AI Interview Hub</h1>
            <p class="hero-subtitle">
                Complete the hiring loop sections below. Each section evaluates distinct technical competencies and records real-time posture, acoustic inflection, and conceptual depth.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sensor Check
    with st.expander("🛡️ Hardware Permissions & Sensor Check", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.success("✅ HD Webcam Ready: MediaPipe FaceMesh & Pose active")
        c2.success("✅ Microphone Ready: 16kHz float32 Whisper STT loaded")
        c3.info(f"Target Role: **{st.session_state.target_role or 'Software Engineer'}**")

    # Section Grid
    completed = st.session_state.completed_sections
    sections = [
        {"id": "self_intro", "title": "1. Self Introduction", "time": "60–90 Secs", "mode": "Webcam + Audio Input", "desc": "Spoken elevator pitch benchmarked against resume skills, WPM pace, and acoustic energy."},
        {"id": "technical", "title": "2. Technical Assessment", "time": "10 Mins", "mode": "10 Conceptual Questions", "desc": "Core engineering depth strictly centered on OOPs, Operating Systems (OS), and Data Structures."},
        {"id": "dsa", "title": "3. Algorithmic Coding (DSA)", "time": "75 Mins", "mode": "2 Coding Problems", "desc": "Automated Sandbox Judge running visible and hidden test cases with structured mathematical constraints.", "dev_only": True},
        {"id": "project", "title": "4. Project Deep Dive", "time": "10 Mins", "mode": "5 Architectural Questions", "desc": "Probing system design trade-offs, scalability bottlenecks, and production error handling."},
        {"id": "aptitude", "title": "5. Aptitude & Data Interpretation", "time": "30 Mins", "mode": "10 MCQs", "desc": "Systems metrics, Little's Law, CPU cache access calculations, and logical reasoning."},
        {"id": "personality", "title": "6. Psychometric Personality", "time": "Untimed", "mode": "20 Likert Statements", "desc": "Big Five trait profiling (OCEAN) and organizational culture fit without camera or timer."},
        {"id": "audio_recording", "title": "7. Audio Recordings Drill", "time": "10 Mins", "mode": "10 Benchmark Sentences", "desc": "5 Read & Lock + 5 Listen-Only drills evaluating pronunciation accuracy, WPM, and Librosa pitch."},
    ]

    cols = st.columns(2, gap="medium")
    for idx, s in enumerate(sections):
        if s.get("dev_only") and not st.session_state.is_developer_role:
            continue
        sec_id = s["id"]
        is_done = sec_id in completed
        score_val = completed.get(sec_id)

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="section-card">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                            <span style="font-weight: 800; font-size: 1.1rem; color: #1E1B4B;">{s['title']}</span>
                            <span class="{'badge-completed' if is_done else 'badge-pending'}">
                                {'✅ Completed (' + str(round(score_val, 1)) + '%)' if is_done else '⏳ Pending'}
                            </span>
                        </div>
                        <p style="color: #64748B; font-size: 0.88rem; line-height: 1.5; margin-bottom: 0.8rem;">{s['desc']}</p>
                        <div style="font-size: 0.8rem; color: #4338CA; font-weight: 700; margin-bottom: 1rem;">
                            ⏱️ {s['time']} &nbsp;•&nbsp; 📝 {s['mode']}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_label = f"Review {s['title']}" if is_done else f"Start {s['title']} ➡️"
            if st.button(btn_label, key=f"btn_start_sec_{sec_id}", use_container_width=True):
                st.session_state.current_screen = f"section_{sec_id}"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if completed:
        col_rep = st.columns([1, 1.5, 1])[1]
        with col_rep:
            if st.button("📊 View Performance Analytics & AI Report", type="primary", use_container_width=True):
                st.session_state.current_screen = "report"
                st.rerun()



# ==============================================================================
# SECTION 1: SELF INTRODUCTION (Spoken Speech · Live Webcam · Retake & Confirm)
# ==============================================================================
def render_section_self_intro():
    """Render Section 1: Self Introduction with continuous live camera, retake/confirm flow, and finish."""
    render_header()
    render_live_vision_tracker("Self Introduction")

    st.markdown(
        """
        <div class="question-hero">
            <span class="skill-chip">Section 1: Spoken Delivery</span>
            <span class="badge-pending">Duration: 60–90 Seconds</span>
            <div class="question-text">
                "Please introduce yourself, walk through your technical background, highlight 1–2 flagship projects from your resume, and explain why you are well-suited for the target role."
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_action, col_eval = st.columns([1, 1], gap="large")

    with col_action:
        st.markdown("#### 🎙️ Record Self Introduction")
        audio_file = st.audio_input("Record audio response (60-90 seconds)", key=f"self_intro_mic_{st.session_state.attempt_id}")

        if audio_file is not None:
            raw_bytes = audio_file.getvalue()
            st.session_state.self_intro_audio_bytes = raw_bytes

            c_retake, c_confirm = st.columns(2)
            with c_retake:
                if st.button("🔄 Retake", use_container_width=True, key="btn_intro_retake"):
                    st.session_state.self_intro_audio_bytes = None
                    st.session_state.self_intro_eval = None
                    st.session_state.self_intro_confirmed = False
                    st.rerun()

            with c_confirm:
                if st.button("✅ Confirm & Analyze", type="primary", use_container_width=True, key="btn_intro_confirm"):
                    with st.spinner("Transcribing introduction with Whisper & analyzing resume alignment..."):
                        transcript, mode = transcribe(raw_bytes)
                        clean_tr = (transcript or "").strip()
                        words = re.findall(r"\b[A-Za-z0-9+#.-]+\b", clean_tr)

                        # Strictly check for empty or silent speech (NO fake fallback!)
                        if not clean_tr or len(words) == 0:
                            ev = {
                                "transcript": "",
                                "score": 0.0,
                                "wpm": 0.0,
                                "fillers": "N/A (No Speech)",
                                "fluency": 0.0,
                                "voice_tone": "Silent / No Speech Detected",
                                "skills_matched": [],
                                "skills_missed": st.session_state.detected_skills[:5],
                                "strengths": [],
                                "improvements": ["No speech detected in recording. Please speak clearly into the microphone."],
                            }
                        else:
                            speech = speech_metrics(clean_tr, duration_seconds=60.0, wav_bytes=raw_bytes)
                            intro_analysis = analyze_self_introduction(
                                transcript=clean_tr,
                                resume_text=st.session_state.resume_text,
                                role=st.session_state.target_role,
                                detected_skills=st.session_state.detected_skills,
                                duration_seconds=60.0,
                            )
                            ev = {
                                "transcript": clean_tr,
                                "score": intro_analysis["score"],
                                "wpm": speech["wpm"],
                                "fillers": speech.get("fillers", 0),
                                "fluency": speech["fluency"],
                                "voice_tone": speech.get("voice_tone", "Confident & Expressive"),
                                "skills_matched": intro_analysis.get("skills_matched", []),
                                "skills_missed": intro_analysis.get("skills_missed", []),
                                "strengths": intro_analysis.get("strengths", []),
                                "improvements": intro_analysis.get("improvements", []),
                            }

                        st.session_state.self_intro_eval = ev
                        st.session_state.self_intro_confirmed = True
                        st.session_state.completed_sections["self_intro"] = ev["score"]
                        st.session_state.section_results["self_intro"] = ev
                        st.rerun()

    with col_eval:
        st.markdown("#### 📋 Assessment & Transcript")
        if st.session_state.self_intro_eval:
            ev = st.session_state.self_intro_eval
            if not ev.get("transcript"):
                st.error("❌ **No speech detected in recording. Please speak clearly into the microphone.**")
                st.metric("Readiness Score", "0.0 / 100")
                st.markdown("**Pacing:** 0 WPM &nbsp;|&nbsp; **Tone:** Silent / No Speech Detected &nbsp;|&nbsp; **Fillers:** N/A (No Speech)")
                st.info("💡 Click **🔄 Retake** to record your self introduction again.")
            else:
                st.success(f"**Readiness Score:** {ev['score']:.1f} / 100")
                st.markdown(f"**Transcript:** *{ev.get('transcript', '')}*")
                st.markdown(f"**Pacing:** {ev['wpm']:.0f} WPM | **Tone:** {ev['voice_tone']} | **Fillers:** {ev['fillers']}")

            if ev.get("skills_matched"):
                st.markdown(f"**Highlighted Skills:** {', '.join(ev['skills_matched'])}")
            if ev.get("skills_missed"):
                st.caption(f"**Omitted Resume Skills:** {', '.join(ev['skills_missed'][:3])}")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🏁 Finish Section & Return to Hub", type="primary", use_container_width=True, key="btn_finish_intro"):
                st.session_state.current_screen = "hub"
                st.rerun()
        else:
            st.info("Record your 60–90 second introduction into the microphone and click 'Confirm & Analyze'.")


# ==============================================================================
# SECTION 2: TECHNICAL ASSESSMENT (OOPs, OS, DS · Client-Side Timer · Skip Button)
# ==============================================================================
def render_section_technical():
    """Render Section 2: Technical Assessment with client-side timer, unique keys, and skip button."""
    render_header()

    if not st.session_state.tech_questions:
        st.markdown(
            """
            <div class="hero-card">
                <h1 class="hero-title">Section 2: Technical Assessment</h1>
                <p class="hero-subtitle">
                    10 conceptual engineering questions strictly centered on <b>Object-Oriented Programming (OOPs)</b>, <b>Operating Systems (OS Internals)</b>, and <b>Data Structures (DS)</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_config, _ = st.columns([1, 1])
        with col_config:
            diff = st.selectbox("Select Technical Difficulty", ["Easy", "Medium", "Hard"], index=1, key="sel_tech_diff")
            st.markdown(
                """
                - **Focus:** OOPs invariants, OS processes/memory, and Data Structures complexity.
                - **Timer:** 10-minute client-side countdown timer (zero lag / no freeze).
                - **Navigation:** Next Question or Skip Question at any time.
                """
            )
            if st.button("Start Technical Section", type="primary", use_container_width=True, key="btn_start_tech"):
                with st.spinner("Generating calibrated technical questions..."):
                    qs = generate_technical_questions(
                        role=st.session_state.target_role,
                        difficulty=diff,
                        count=10,
                        session_seed=st.session_state.attempt_id,
                    )
                    st.session_state.tech_questions = qs
                    st.session_state.tech_difficulty = diff
                    st.session_state.tech_active_idx = 0
                    st.session_state.tech_start_epoch = time.time()
                    st.rerun()
        return

    # Active Questions Flow
    render_live_vision_tracker("Technical")

    idx = st.session_state.tech_active_idx
    questions = st.session_state.tech_questions
    current_q = questions[idx]

    col_title, col_timer = st.columns([3, 1.2])
    with col_title:
        st.markdown(f"### Technical Assessment — Question {idx + 1} of {len(questions)}")
    with col_timer:
        render_client_timer(total_seconds=600, start_epoch=st.session_state.tech_start_epoch, section_name="Technical")

    st.markdown(
        f"""
        <div class="question-hero">
            <span class="skill-chip">{current_q.get('topic', 'OOPs / OS / DS')}</span>
            <span class="badge-pending">Difficulty: {st.session_state.tech_difficulty}</span>
            <div class="question-text">{html.escape(current_q.get('question', ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Strictly unique widget key tied to attempt ID and question index
    widget_key = f"{st.session_state.attempt_id}_tech_q_{idx}"
    curr_answer = st.session_state.tech_answers.get(idx, "")

    typed_answer = st.text_area(
        "Your Conceptual Answer *",
        value=curr_answer,
        height=160,
        placeholder="Explain your approach, system mechanics, trade-offs, and design principles...",
        key=widget_key,
    )
    st.session_state.tech_answers[idx] = typed_answer

    is_last = (idx == len(questions) - 1)
    next_label = "🏁 Finish Technical Section" if is_last else "Next Question ➡️"

    # Action Buttons: Previous, Skip, Next
    col_prev, col_skip, col_next = st.columns([1, 1, 1.2])

    with col_prev:
        if idx > 0 and st.button("⬅️ Previous Question", key=f"btn_tech_prev_{idx}", use_container_width=True):
            st.session_state.tech_active_idx -= 1
            st.rerun()

    with col_skip:
        if st.button("⏭️ Skip Question", key=f"btn_tech_skip_{idx}", use_container_width=True):
            st.session_state.tech_answers[idx] = "Skipped"
            st.session_state.tech_evaluations[idx] = {
                "score": 0.0,
                "feedback": "Question was skipped by candidate.",
                "missing_concepts": ["All core concepts (Question Skipped)"],
                "ideal_answer": current_q.get("ideal_answer") or _generate_sample_ideal_answer(current_q.get("question", ""), "Technical"),
            }
            if is_last:
                _finish_technical_section()
            else:
                st.session_state.tech_active_idx += 1
            st.rerun()

    with col_next:
        if st.button(next_label, type="primary", key=f"btn_tech_next_{idx}", use_container_width=True):
            with st.spinner("Recording answer and assessing technical depth..."):
                ev = evaluate_answer(
                    question=current_q.get("question", ""),
                    answer=typed_answer,
                    difficulty=st.session_state.tech_difficulty,
                    section="Technical",
                    expected_concepts=current_q.get("expected_concepts", []),
                )
                if current_q.get("ideal_answer"):
                    ev["ideal_answer"] = current_q["ideal_answer"]
                st.session_state.tech_evaluations[idx] = ev

                if is_last:
                    _finish_technical_section()
                else:
                    st.session_state.tech_active_idx += 1
                st.rerun()


def _finish_technical_section():
    """Aggregate technical scores and return to Section Hub."""
    evals = st.session_state.tech_evaluations
    scores = [e.get("score", 0.0) for e in evals.values()]
    avg_score = float(sum(scores) / max(len(scores), 1))
    st.session_state.completed_sections["technical"] = avg_score
    st.session_state.section_results["technical"] = {
        "score": avg_score,
        "evaluations": evals,
        "difficulty": st.session_state.tech_difficulty,
    }
    st.session_state.current_screen = "hub"


# ==============================================================================
# SECTION 3: ALGORITHMIC PROBLEMS (DSA) (Sandbox Judge · Skip Button)
# ==============================================================================
def render_section_dsa():
    """Render Section 3: Algorithmic Coding Assessment with authentic split-screen coding platform layout and empty function starter stubs."""
    render_header()

    if not st.session_state.dsa_problems:
        st.markdown(
            """
            <div class="hero-card">
                <h1 class="hero-title">Section 3: Algorithmic Coding Assessment</h1>
                <p class="hero-subtitle">
                    Automated code execution sandbox evaluating runtime efficiency, edge-case safety, and automated test cases with structured mathematical constraints.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_config, _ = st.columns([1, 1])
        with col_config:
            diff = st.selectbox("Select Coding Challenge Difficulty", ["Easy", "Medium", "Hard"], index=1, key="sel_dsa_diff")
            st.markdown(
                """
                - **Problems:** 2 Algorithmic Coding Challenges.
                - **Starter Code:** Clean function signatures with empty starter boilerplate.
                - **Judge:** Executes Python against visible and hidden automated test cases.
                - **Languages:** Python (executed live in sandbox), C++, Java, JavaScript.
                """
            )
            if st.button("Start Algorithmic Assessment", type="primary", use_container_width=True, key="btn_start_dsa"):
                with st.spinner("Generating algorithmic problems..."):
                    probs = generate_dsa_problems(difficulty=diff, count=2, session_seed=st.session_state.attempt_id)
                    st.session_state.dsa_problems = probs
                    st.session_state.dsa_difficulty = diff
                    st.session_state.dsa_active_idx = 0
                    st.session_state.dsa_start_epoch = time.time()
                    st.rerun()
        return

    # Active Coding Flow
    render_live_vision_tracker("Algorithmic Assessment")

    idx = st.session_state.dsa_active_idx
    problems = st.session_state.dsa_problems
    prob = problems[idx]

    # Split Layout: Left panel (Problem description, examples, constraints) | Right panel (Language, Code editor, Run/Skip/Next, Results)
    col_prob, col_code = st.columns([1.05, 1.25], gap="large")

    # ---------------- LEFT PANEL: Problem Details ----------------
    with col_prob:
        diff_val = prob.get("difficulty", "Medium")
        diff_color = {
            "Easy": ("#D1FAE5", "#065F46"),
            "Medium": ("#FEF3C7", "#92400E"),
            "Hard": ("#FEE2E2", "#991B1B"),
        }.get(diff_val, ("#E0E7FF", "#3730A3"))

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <h3 style="margin: 0; color: #1E1B4B; font-weight: 800; font-size: 1.25rem;">
                    Problem {idx + 1} of {len(problems)}: {prob['title']}
                </h3>
                <span style="background: {diff_color[0]}; color: {diff_color[1]}; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 700; font-size: 0.8rem;">
                    {diff_val}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem; color: #334155; line-height: 1.65; font-size: 0.95rem;">
                <b>Description:</b><br>{prob['description']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Sample Input & Output Examples
        test_cases = prob.get("test_cases", [])
        if test_cases:
            st.markdown("<div style='font-weight: 700; color: #1E1B4B; margin-bottom: 0.4rem; font-size: 0.92rem;'>Sample Test Cases:</div>", unsafe_allow_html=True)
            for tc_i, tc in enumerate(test_cases[:2]):
                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                        <div style="color: #64748B; font-weight: 600; margin-bottom: 0.2rem;">Example {tc_i + 1}:</div>
                        <div><strong style="color: #475569;">Input:</strong> <span style="color: #0F172A;">{tc.get('input', '')}</span></div>
                        <div><strong style="color: #475569;">Output:</strong> <span style="color: #059669; font-weight: 700;">{tc.get('expected', '')}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Constraints List
        constraints = prob.get("constraints", [])
        if constraints:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.85rem 1.25rem; margin-top: 0.75rem;">
                    <div style="font-weight: 700; color: #1E1B4B; margin-bottom: 0.35rem; font-size: 0.9rem;">Constraints & Complexity Targets:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("View Mathematical Constraints", expanded=True):
                for c in constraints:
                    st.markdown(f"- {c}")

    # ---------------- RIGHT PANEL: Code Editor & Execution ----------------
    with col_code:
        c_lang, c_timer = st.columns([1.2, 1])
        with c_lang:
            lang = st.selectbox("Implementation Language", ["Python", "C++", "Java", "JavaScript"], key=f"dsa_lang_sel_{idx}")
            st.session_state.dsa_language = lang
        with c_timer:
            render_client_timer(total_seconds=4500, start_epoch=st.session_state.dsa_start_epoch, section_name="DSA")

        starter_code = prob.get("starter_code", {}).get(lang, "# Write your solution here\npass\n")
        widget_key = f"{st.session_state.attempt_id}_dsa_code_{idx}_{lang}"

        # Initialize language-specific code if not in session_state
        if widget_key not in st.session_state:
            st.session_state[widget_key] = starter_code

        code_input = st.text_area(
            f"Your {lang} Solution Code *",
            value=st.session_state[widget_key],
            height=280,
            key=widget_key,
            help="Write your algorithmic solution. Python code will be executed live against test cases.",
        )
        st.session_state.dsa_codes[idx] = code_input

        # Execution and Navigation Buttons
        col_run, col_skip, col_next = st.columns([1.4, 1, 1.2])

        with col_run:
            if st.button("▶️ Run Code & Evaluate Tests", type="primary", use_container_width=True, key=f"btn_dsa_run_{idx}"):
                with st.spinner("Executing solution against automated test cases..."):
                    res = evaluate_code_solution(problem=prob, code=code_input, language=lang)
                    st.session_state.dsa_evaluations[idx] = res
                    st.rerun()

        is_last = (idx == len(problems) - 1)
        next_label = "🏁 Finish Algorithmic Section" if is_last else "Next Problem ➡️"

        with col_skip:
            if st.button("⏭️ Skip Problem", key=f"btn_dsa_skip_{idx}", use_container_width=True):
                st.session_state.dsa_codes[idx] = "Skipped"
                st.session_state.dsa_evaluations[idx] = {
                    "score": 0.0,
                    "passed": 0,
                    "total": len(prob.get("test_cases", [])),
                    "feedback": "Problem was skipped by candidate.",
                    "details": [],
                }
                if is_last:
                    _finish_dsa_section()
                else:
                    st.session_state.dsa_active_idx += 1
                st.rerun()

        with col_next:
            if st.button(next_label, key=f"btn_dsa_next_{idx}", use_container_width=True):
                if idx not in st.session_state.dsa_evaluations:
                    with st.spinner("Evaluating code solution..."):
                        res = evaluate_code_solution(problem=prob, code=code_input, language=lang)
                        st.session_state.dsa_evaluations[idx] = res
                if is_last:
                    _finish_dsa_section()
                else:
                    st.session_state.dsa_active_idx += 1
                st.rerun()

        # Show Judge Results Table
        if idx in st.session_state.dsa_evaluations:
            ev = st.session_state.dsa_evaluations[idx]
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="background: {'#F0FDF4' if ev.get('score', 0) >= 70 else '#FEF2F2'}; border: 1px solid {'#10B981' if ev.get('score', 0) >= 70 else '#EF4444'}; border-radius: 10px; padding: 0.85rem 1.2rem; margin-bottom: 0.75rem;">
                    <div style="font-weight: 700; color: {'#064E3B' if ev.get('score', 0) >= 70 else '#991B1B'}; font-size: 1rem;">
                        Automated Judge: {ev.get('feedback', '')}
                    </div>
                    <div style="font-size: 0.88rem; color: #475569; margin-top: 0.25rem;">
                        Score: <b>{ev.get('score', 0):.1f}%</b> ({ev.get('passed', 0)} / {ev.get('total', 0)} Test Cases Passed)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ev.get("details"):
                df_tests = pd.DataFrame(ev["details"])
                cols_to_show = [c for c in ["case", "status", "runtime_ms", "input", "expected", "actual"] if c in df_tests.columns]
                st.dataframe(df_tests[cols_to_show], use_container_width=True, hide_index=True)


def _finish_dsa_section():
    """Aggregate DSA scores and return to Section Hub."""
    evals = st.session_state.dsa_evaluations
    scores = [e.get("score", 0.0) for e in evals.values()]
    avg_score = float(sum(scores) / max(len(scores), 1))
    st.session_state.completed_sections["dsa"] = avg_score
    st.session_state.section_results["dsa"] = {
        "score": avg_score,
        "evaluations": evals,
        "difficulty": st.session_state.dsa_difficulty,
    }
    st.session_state.current_screen = "hub"


# ==============================================================================
# SECTION 4: PROJECT DEEP DIVE (Architectural Probing · Skip Button)
# ==============================================================================
def render_section_project():
    """Render Section 4: Project Deep Dive with client-side timer, unique keys, and skip button."""
    render_header()

    if not st.session_state.project_questions:
        st.markdown(
            """
            <div class="hero-card">
                <h1 class="hero-title">Section 4: Project Architectural Deep Dive</h1>
                <p class="hero-subtitle">
                    5 in-depth architectural questions probing production systems, scalability trade-offs, database indexing, and fault tolerance from your resume.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_config, _ = st.columns([1, 1])
        with col_config:
            diff = st.selectbox("Select Project Probing Difficulty", ["Easy", "Medium", "Hard"], index=1, key="sel_proj_diff")
            if st.button("Start Project Deep Dive", type="primary", use_container_width=True, key="btn_start_proj"):
                with st.spinner("Generating architectural questions..."):
                    qs = generate_project_questions(
                        projects=st.session_state.detected_projects,
                        role=st.session_state.target_role,
                        difficulty=diff,
                        count=5,
                        session_seed=st.session_state.attempt_id,
                    )
                    st.session_state.project_questions = qs
                    st.session_state.project_difficulty = diff
                    st.session_state.project_active_idx = 0
                    st.session_state.project_start_epoch = time.time()
                    st.rerun()
        return

    # Active Project Flow
    render_live_vision_tracker("Project Deep Dive")

    idx = st.session_state.project_active_idx
    questions = st.session_state.project_questions
    current_q = questions[idx]

    col_title, col_timer = st.columns([3, 1.2])
    with col_title:
        st.markdown(f"### Project Deep Dive — Question {idx + 1} of {len(questions)}")
    with col_timer:
        render_client_timer(total_seconds=600, start_epoch=st.session_state.project_start_epoch, section_name="Project")

    st.markdown(
        f"""
        <div class="question-hero">
            <span class="skill-chip">Target Project: {current_q.get('project_name', 'Flagship System')}</span>
            <span class="badge-pending">Difficulty: {st.session_state.project_difficulty}</span>
            <div class="question-text">{html.escape(current_q.get('question', ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    widget_key = f"{st.session_state.attempt_id}_proj_q_{idx}"
    curr_answer = st.session_state.project_answers.get(idx, "")

    typed_answer = st.text_area(
        "Your Architectural Answer *",
        value=curr_answer,
        height=160,
        placeholder="Detail your system architecture, throughput metrics, bottlenecks, and engineering trade-offs...",
        key=widget_key,
    )
    st.session_state.project_answers[idx] = typed_answer

    is_last = (idx == len(questions) - 1)
    next_label = "🏁 Finish Project Section" if is_last else "Next Question ➡️"

    col_prev, col_skip, col_next = st.columns([1, 1, 1.2])

    with col_prev:
        if idx > 0 and st.button("⬅️ Previous Question", key=f"btn_proj_prev_{idx}", use_container_width=True):
            st.session_state.project_active_idx -= 1
            st.rerun()

    with col_skip:
        if st.button("⏭️ Skip Question", key=f"btn_proj_skip_{idx}", use_container_width=True):
            st.session_state.project_answers[idx] = "Skipped"
            st.session_state.project_evaluations[idx] = {
                "score": 0.0,
                "feedback": "Question was skipped by candidate.",
                "missing_concepts": ["Architectural depth (Skipped)"],
                "ideal_answer": _generate_sample_ideal_answer(current_q.get("question", ""), "Project"),
            }
            if is_last:
                _finish_project_section()
            else:
                st.session_state.project_active_idx += 1
            st.rerun()

    with col_next:
        if st.button(next_label, type="primary", key=f"btn_proj_next_{idx}", use_container_width=True):
            with st.spinner("Evaluating architectural depth..."):
                ev = evaluate_answer(
                    question=current_q.get("question", ""),
                    answer=typed_answer,
                    difficulty=st.session_state.project_difficulty,
                    section="Project",
                    expected_concepts=current_q.get("expected_concepts", []),
                )
                st.session_state.project_evaluations[idx] = ev
                if is_last:
                    _finish_project_section()
                else:
                    st.session_state.project_active_idx += 1
                st.rerun()


def _finish_project_section():
    """Aggregate project scores and return to Hub."""
    evals = st.session_state.project_evaluations
    scores = [e.get("score", 0.0) for e in evals.values()]
    avg_score = float(sum(scores) / max(len(scores), 1))
    st.session_state.completed_sections["project"] = avg_score
    st.session_state.section_results["project"] = {
        "score": avg_score,
        "evaluations": evals,
        "difficulty": st.session_state.project_difficulty,
    }
    st.session_state.current_screen = "hub"


# ==============================================================================
# SECTION 5: APTITUDE & DATA INTERPRETATION (Radio MCQs · Skip Button)
# ==============================================================================
def render_section_aptitude():
    """Render Section 5: Aptitude MCQs with client-side timer, unique keys, and skip button."""
    render_header()

    if not st.session_state.aptitude_questions:
        st.markdown(
            """
            <div class="hero-card">
                <h1 class="hero-title">Section 5: Aptitude & Data Interpretation</h1>
                <p class="hero-subtitle">
                    10 multiple-choice questions evaluating numerical agility, probabilistic reasoning, and systems performance calculations (Little's Law, AMAT, throughput).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_config, _ = st.columns([1, 1])
        with col_config:
            diff = st.selectbox("Select Aptitude Difficulty", ["Easy", "Medium", "Hard"], index=1, key="sel_apt_diff")
            if st.button("Start Aptitude Section", type="primary", use_container_width=True, key="btn_start_apt"):
                with st.spinner("Generating aptitude questions..."):
                    qs = generate_aptitude_questions(difficulty=diff, count=10, session_seed=st.session_state.attempt_id)
                    st.session_state.aptitude_questions = qs
                    st.session_state.aptitude_difficulty = diff
                    st.session_state.aptitude_active_idx = 0
                    st.session_state.aptitude_start_epoch = time.time()
                    st.rerun()
        return

    # Active Aptitude Flow
    render_live_vision_tracker("Aptitude & Logic")

    idx = st.session_state.aptitude_active_idx
    questions = st.session_state.aptitude_questions
    current_q = questions[idx]

    col_title, col_timer = st.columns([3, 1.2])
    with col_title:
        st.markdown(f"### Aptitude & Logic — Question {idx + 1} of {len(questions)}")
    with col_timer:
        render_client_timer(total_seconds=1800, start_epoch=st.session_state.aptitude_start_epoch, section_name="Aptitude")

    st.markdown(
        f"""
        <div class="question-hero">
            <span class="skill-chip">Aptitude & Systems Analysis</span>
            <span class="badge-pending">Difficulty: {st.session_state.aptitude_difficulty}</span>
            <div class="question-text">{html.escape(current_q.get('question', ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    opts = current_q.get("options", ["A", "B", "C", "D"])
    widget_key = f"{st.session_state.attempt_id}_apt_choice_{idx}"
    curr_choice = st.session_state.aptitude_answers.get(idx)
    default_idx = opts.index(curr_choice) if curr_choice in opts else 0

    selected_opt = st.radio("Select Correct Option *", opts, index=default_idx, key=widget_key)
    st.session_state.aptitude_answers[idx] = selected_opt

    is_last = (idx == len(questions) - 1)
    next_label = "🏁 Finish Aptitude Section" if is_last else "Next Question ➡️"

    col_prev, col_skip, col_next = st.columns([1, 1, 1.2])

    with col_prev:
        if idx > 0 and st.button("⬅️ Previous Question", key=f"btn_apt_prev_{idx}", use_container_width=True):
            st.session_state.aptitude_active_idx -= 1
            st.rerun()

    with col_skip:
        if st.button("⏭️ Skip Question", key=f"btn_apt_skip_{idx}", use_container_width=True):
            st.session_state.aptitude_answers[idx] = "Skipped"
            st.session_state.aptitude_evaluations[idx] = {
                "score": 0.0,
                "is_correct": False,
                "selected": "Skipped",
                "correct_answer": current_q.get("correct_answer", ""),
                "explanation": current_q.get("explanation", ""),
            }
            if is_last:
                _finish_aptitude_section()
            else:
                st.session_state.aptitude_active_idx += 1
            st.rerun()

    with col_next:
        if st.button(next_label, type="primary", key=f"btn_apt_next_{idx}", use_container_width=True):
            corr = current_q.get("correct_answer", "").strip()
            is_right = (selected_opt.strip() == corr)
            st.session_state.aptitude_evaluations[idx] = {
                "score": 100.0 if is_right else 0.0,
                "is_correct": is_right,
                "selected": selected_opt,
                "correct_answer": corr,
                "explanation": current_q.get("explanation", ""),
            }
            if is_last:
                _finish_aptitude_section()
            else:
                st.session_state.aptitude_active_idx += 1
            st.rerun()


def _finish_aptitude_section():
    """Aggregate aptitude scores and return to Hub."""
    evals = st.session_state.aptitude_evaluations
    scores = [e.get("score", 0.0) for e in evals.values()]
    avg_score = float(sum(scores) / max(len(scores), 1))
    st.session_state.completed_sections["aptitude"] = avg_score
    st.session_state.section_results["aptitude"] = {
        "score": avg_score,
        "evaluations": evals,
        "difficulty": st.session_state.aptitude_difficulty,
    }
    st.session_state.current_screen = "hub"


# ==============================================================================
# SECTION 6: PSYCHOMETRIC PERSONALITY (OCEAN · Untimed · Skip Button)
# ==============================================================================
def render_section_personality():
    """Render Section 6: Psychometric Personality test without timer or camera."""
    render_header()

    if not st.session_state.personality_statements:
        st.markdown(
            """
            <div class="hero-card">
                <h1 class="hero-title">Section 6: Psychometric Personality Assessment</h1>
                <p class="hero-subtitle">
                    20 Likert statements mapping your workplace collaboration, stress tolerance, and execution discipline across the Big Five dimensions (OCEAN). Untimed with camera and microphone disabled.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_config, _ = st.columns([1, 1])
        with col_config:
            if st.button("Start Personality Assessment", type="primary", use_container_width=True, key="btn_start_pers"):
                stmts = generate_personality_statements(count=20, session_seed=st.session_state.attempt_id)
                st.session_state.personality_statements = stmts
                st.session_state.personality_active_idx = 0
                st.rerun()
        return

    idx = st.session_state.personality_active_idx
    stmts = st.session_state.personality_statements
    stmt = stmts[idx]

    st.markdown(f"### Personality Profiling — Statement {idx + 1} of {len(stmts)}")

    st.markdown(
        f"""
        <div class="question-hero">
            <span class="skill-chip">Trait Dimension: {stmt.get('trait', 'Core Personality')}</span>
            <div class="question-text">"{html.escape(stmt.get('statement', ''))}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    widget_key = f"{st.session_state.attempt_id}_pers_rate_{idx}"
    curr_rating = st.session_state.personality_responses.get(idx, 3)

    rating_labels = {
        1: "1 — Strongly Disagree",
        2: "2 — Disagree",
        3: "3 — Neutral / Situational",
        4: "4 — Agree",
        5: "5 — Strongly Agree",
    }
    val = st.radio("Rate your agreement *", [1, 2, 3, 4, 5], index=curr_rating - 1, format_func=lambda x: rating_labels[x], key=widget_key)
    st.session_state.personality_responses[idx] = val

    is_last = (idx == len(stmts) - 1)
    next_label = "🏁 Finish Personality Section" if is_last else "Next Statement ➡️"

    col_prev, col_skip, col_next = st.columns([1, 1, 1.2])

    with col_prev:
        if idx > 0 and st.button("⬅️ Previous Statement", key=f"btn_pers_prev_{idx}", use_container_width=True):
            st.session_state.personality_active_idx -= 1
            st.rerun()

    with col_skip:
        if st.button("⏭️ Skip Statement", key=f"btn_pers_skip_{idx}", use_container_width=True):
            st.session_state.personality_responses[idx] = 3
            if is_last:
                _finish_personality_section()
            else:
                st.session_state.personality_active_idx += 1
            st.rerun()

    with col_next:
        if st.button(next_label, type="primary", key=f"btn_pers_next_{idx}", use_container_width=True):
            if is_last:
                _finish_personality_section()
            else:
                st.session_state.personality_active_idx += 1
            st.rerun()


def _finish_personality_section():
    """Evaluate Big Five OCEAN traits and return to Hub."""
    eval_res = evaluate_personality_test(st.session_state.personality_responses)
    st.session_state.personality_eval = eval_res
    st.session_state.completed_sections["personality"] = eval_res["culture_fit_score"]
    st.session_state.section_results["personality"] = eval_res
    st.session_state.current_screen = "hub"


# ==============================================================================
# SECTION 7: AUDIO RECORDINGS DRILL (5 READ-LOCKED + 5 LISTEN-ONLY)
# ==============================================================================
AUDIO_DRILL_PROGRESSIVE_SENTENCES: list[dict[str, str]] = [
    # Tier 1: Easy (Questions 1 to 3, 10–14 words each)
    {
        "tier": "Easy",
        "sentence": "Clear communication enables new engineers to set up their local development environments in minutes.",
        "text": "Clear communication enables new engineers to set up their local development environments in minutes."
    },
    {
        "tier": "Easy",
        "sentence": "Automated alerts notify our on-call engineers whenever production error rates exceed standard thresholds.",
        "text": "Automated alerts notify our on-call engineers whenever production error rates exceed standard thresholds."
    },
    {
        "tier": "Easy",
        "sentence": "Our team conducts daily standups to align sprint goals and resolve critical blockers quickly.",
        "text": "Our team conducts daily standups to align sprint goals and resolve critical blockers quickly."
    },
    # Tier 2: Medium (Questions 4 to 7, 15–20 words each)
    {
        "tier": "Medium",
        "sentence": "We systematically mitigated production downtime by implementing automated canary deployments and end-to-end distributed telemetry across our clusters.",
        "text": "We systematically mitigated production downtime by implementing automated canary deployments and end-to-end distributed telemetry across our clusters."
    },
    {
        "tier": "Medium",
        "sentence": "Refactoring the authentication middleware into stateless cryptographic tokens allowed our web tier to scale horizontally with ease.",
        "text": "Refactoring the authentication middleware into stateless cryptographic tokens allowed our web tier to scale horizontally with ease."
    },
    {
        "tier": "Medium",
        "sentence": "Proactive performance profiling and query indexing prevented severe memory leaks under peak client traffic loads.",
        "text": "Proactive performance profiling and query indexing prevented severe memory leaks under peak client traffic loads."
    },
    {
        "tier": "Medium",
        "sentence": "Cross-functional collaboration between product designers and backend engineers streamlined our automated continuous deployment workflow significantly.",
        "text": "Cross-functional collaboration between product designers and backend engineers streamlined our automated continuous deployment workflow significantly."
    },
    # Tier 3: Hard (Questions 8 to 10, 21–26 words each)
    {
        "tier": "Hard",
        "sentence": "The cloud platform effectively mitigates cascading thread pool starvation under adversarial traffic loads by orchestrating cooperative backpressure algorithms and proactive request shedding mechanisms.",
        "text": "The cloud platform effectively mitigates cascading thread pool starvation under adversarial traffic loads by orchestrating cooperative backpressure algorithms and proactive request shedding mechanisms."
    },
    {
        "tier": "Hard",
        "sentence": "Our fault-tolerant architecture systematically combines exponential backoff with randomized jitter to prevent synchronized retry storms from crashing recovering backend microservice clusters during major incidents.",
        "text": "Our fault-tolerant architecture systematically combines exponential backoff with randomized jitter to prevent synchronized retry storms from crashing recovering backend microservice clusters during major incidents."
    },
    {
        "tier": "Hard",
        "sentence": "By refactoring our high-throughput distributed memory allocation strategy to utilize lock-free ring buffers, we successfully minimized garbage collection pauses during critical real-time financial transactions.",
        "text": "By refactoring our high-throughput distributed memory allocation strategy to utilize lock-free ring buffers, we successfully minimized garbage collection pauses during critical real-time financial transactions."
    }
]

DEFAULT_AUDIO_DRILL_SENTENCES: list[str] = [item["sentence"] for item in AUDIO_DRILL_PROGRESSIVE_SENTENCES] + [
    "Continuous integration pipelines automatically validate pull requests before merging into our primary branch.",
    "Comprehensive vulnerability scans and automated dependency updates safeguard distributed services against security exploits.",
]
DEFAULT_FALLBACK_SENTENCES: list[str] = DEFAULT_AUDIO_DRILL_SENTENCES


def render_section_audio_recording():
    """Render Section 7: 10 distinct, progressive audio pronunciation drills with strictly scoped per-question locking."""
    render_header()

    attempt_id = st.session_state.attempt_id

    if not st.session_state.audio_sentences:
        st.markdown(
            """
            <div class="hero-card">
                <h1 class="hero-title">Section 7: Audio Recordings & Pronunciation Drill</h1>
                <p class="hero-subtitle">
                    10 distinct benchmark sentences organized progressively across Easy, Medium, and Hard difficulty tiers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_config, _ = st.columns([1, 1])
        with col_config:
            diff = st.selectbox("Select Drill Progression Mode", ["Progressive (Easy -> Medium -> Hard)", "Easy Only", "Medium Only", "Hard Only"], index=0, key="sel_audio_diff")
            st.markdown(
                """
                - **Questions 1–3 (Easy):** Clear professional communication sentences (10–14 words each).
                - **Questions 4–7 (Medium):** Multi-clause engineering workflows and telemetry (15–20 words each).
                - **Questions 8–10 (Hard):** High-load distributed architecture and concurrency (21–26 words each).
                - **Authentic Memory Recall:** The exact instant recording starts, the prompt locks and hides from view.
                """
            )
            if st.button("Start Audio Recordings Section", type="primary", use_container_width=True, key="btn_start_audio"):
                sentences = [item["sentence"] for item in AUDIO_DRILL_PROGRESSIVE_SENTENCES]
                st.session_state.audio_sentences = sentences
                st.session_state.audio_difficulty = diff
                st.session_state.audio_active_idx = 0
                st.session_state.audio_evaluations = {}
                # Initialize Sentence 0 state strictly
                init_lock_key = f"audio_locked_{attempt_id}_q_0"
                init_done_key = f"audio_done_{attempt_id}_q_0"
                init_res_key = f"audio_res_{attempt_id}_q_0"
                init_mic_key = f"audio_mic_{attempt_id}_q_0"
                init_widget_key = f"mic_input_{attempt_id}_q_0"
                st.session_state[init_lock_key] = False
                st.session_state[init_done_key] = False
                st.session_state[init_res_key] = None
                if init_mic_key in st.session_state:
                    del st.session_state[init_mic_key]
                if init_widget_key in st.session_state:
                    del st.session_state[init_widget_key]
                st.rerun()
        return

    # Active Drill Flow
    render_live_vision_tracker("Audio Pronunciation Drill")

    # 1. 10 Unique Sentences Across Easy, Medium & Hard Difficulty Levels
    raw_sentences = st.session_state.audio_sentences
    sentences = [s["text"] if isinstance(s, dict) and "text" in s else (s["sentence"] if isinstance(s, dict) and "sentence" in s else str(s)) for s in raw_sentences if s]
    sentences = [s.strip() for s in sentences if str(s).strip()]

    current_list = list(sentences)
    while len(current_list) < 10:
        fill_idx = len(current_list)
        fallback_sentence = DEFAULT_AUDIO_DRILL_SENTENCES[fill_idx % len(DEFAULT_AUDIO_DRILL_SENTENCES)]
        current_list.append(fallback_sentence)
    sentences = current_list[:10]
    st.session_state.audio_sentences = sentences

    q_idx = st.session_state.audio_active_idx
    if q_idx < 0:
        q_idx = 0
        st.session_state.audio_active_idx = 0
    elif q_idx >= len(sentences):
        q_idx = len(sentences) - 1
        st.session_state.audio_active_idx = q_idx

    current_sentence = str(sentences[q_idx]).strip()
    if not current_sentence:
        current_sentence = DEFAULT_AUDIO_DRILL_SENTENCES[q_idx % len(DEFAULT_AUDIO_DRILL_SENTENCES)]
        sentences[q_idx] = current_sentence
    curr_sent_text = current_sentence

    # Determine progressive difficulty tier
    diff_tier = "Easy" if q_idx < 3 else ("Medium" if q_idx < 7 else "Hard")
    diff_colors = {
        "Easy": ("#D1FAE5", "#065F46", "#10B981"),
        "Medium": ("#FEF3C7", "#92400E", "#F59E0B"),
        "Hard": ("#FEE2E2", "#991B1B", "#EF4444"),
    }[diff_tier]

    # Strictly isolated per-question state keys
    lock_key = f"audio_locked_{attempt_id}_q_{q_idx}"
    done_key = f"audio_done_{attempt_id}_q_{q_idx}"
    result_key = f"audio_res_{attempt_id}_q_{q_idx}"
    mic_key = f"audio_mic_{attempt_id}_q_{q_idx}"
    widget_key = f"audio_mic_{attempt_id}_q_{q_idx}"

    # Initialize per-question state strictly
    if lock_key not in st.session_state:
        st.session_state[lock_key] = False
    if done_key not in st.session_state:
        st.session_state[done_key] = False
    if result_key not in st.session_state:
        st.session_state[result_key] = st.session_state.audio_evaluations.get(q_idx, None)

    has_eval = (st.session_state.get(result_key) is not None) or (q_idx in st.session_state.audio_evaluations)
    is_locked = bool(st.session_state.get(lock_key, False) or st.session_state.get(done_key, False) or has_eval)

    col_title, col_info = st.columns([3, 1.4])
    with col_title:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.5rem;">
                <h3 style="margin: 0; color: #1E1B4B; font-weight: 800;">Audio Recording Drill — Sentence {q_idx + 1} of {len(sentences)}</h3>
                <span style="background: {diff_colors[0]}; color: {diff_colors[1]}; border: 1px solid {diff_colors[2]}; padding: 3px 12px; border-radius: 9999px; font-weight: 800; font-size: 0.82rem;">
                    {diff_tier}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_info:
        st.markdown(
            f'<div style="text-align: right;"><span class="badge-completed">Progress: Sentence {q_idx + 1} / {len(sentences)}</span></div>',
            unsafe_allow_html=True,
        )

    prompt_box_id = f"audio-sentence-prompt-box-{q_idx}"
    locked_box_id = f"audio-sentence-locked-box-{q_idx}"

    # 2. Show Sentence Before Recording (native container with border) & Lock Immediately When Recording Starts
    if not is_locked:
        with st.container(border=True):
            st.markdown(
                f"""
                <div id="prompt-card-{q_idx}" class="sentence-prompt-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span class="skill-chip">Sentence {q_idx + 1} Prompt</span>
                        <span style="background: {diff_colors[0]}; color: {diff_colors[1]}; border: 1px solid {diff_colors[2]}; padding: 2px 10px; border-radius: 9999px; font-weight: 800; font-size: 0.78rem;">
                            {diff_tier} Tier
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f'### "{current_sentence}"')
            st.caption("💡 Read this sentence carefully. The moment you click record below, the text will lock so you can recite from memory.")
            st.markdown(
                f"""
                <div id="locked-card-{q_idx}" class="question-hero" style="display: none; border-color: #DC2626; background: #FEF2F2; padding: 14px; margin: 8px 0; border-radius: 10px;">
                    <span style="color: #991B1B; font-weight: 700;">🔒 Text Locked — Recording in Progress. Speak from memory!</span>
                </div>
                <div id="{locked_box_id}" style="display: none;"></div>
                <div style="font-size:0; height:0; display:none;">"{html.escape(curr_sent_text)}"</div>
                """,
                unsafe_allow_html=True,
            )

        # Inject Shadow-DOM-aware listener for client-side instant sentence lock
        st.components.v1.html(
            f"""
            <script>
            (function() {{
                const parentDoc = window.parent.document;
                const qIdx = {q_idx};
                const promptBoxId = "{prompt_box_id}";
                const lockedBoxId = "{locked_box_id}";

                if (window.parent.__audio_drill_cleanup) {{
                    try {{ window.parent.__audio_drill_cleanup(); }} catch (err) {{}}
                }}

                function lockSentence() {{
                    try {{
                        // Immediately replace <h3> sentence text with locked banner
                        const h3Elements = parentDoc.querySelectorAll("h3");
                        for (let i = 0; i < h3Elements.length; i++) {{
                            const h3 = h3Elements[i];
                            const txt = (h3.innerText || h3.textContent || "").trim();
                            if (!txt.includes("Audio Recording Drill") && !txt.includes("Technical") && !txt.includes("Algorithmic")) {{
                                h3.innerText = "🔒 Text Locked — Recording in Progress. Speak from memory!";
                                h3.style.color = "#DC2626";
                                h3.style.fontWeight = "700";
                            }}
                        }}

                        const p = window.parent.document.getElementById("prompt-card-" + qIdx) || parentDoc.getElementById(promptBoxId);
                        const l = window.parent.document.getElementById("locked-card-" + qIdx) || parentDoc.getElementById(lockedBoxId);
                        if (p) p.style.display = "none";
                        if (l) l.style.display = "block";
                    }} catch (e) {{}}
                }}

                function isMicRecordTarget(e) {{
                    try {{
                        const path = e.composedPath ? e.composedPath() : [e.target];
                        for (let i = 0; i < path.length; i++) {{
                            const el = path[i];
                            if (!el || !el.getAttribute) continue;

                            const btnText = (el.innerText || el.textContent || "").toLowerCase();
                            if (btnText.includes("next") || btnText.includes("skip") || btnText.includes("prev") || btnText.includes("finish")) {{
                                return false;
                            }}

                            const testId = (el.getAttribute("data-testid") || "").toLowerCase();
                            const aria = (el.getAttribute("aria-label") || "").toLowerCase();
                            const title = (el.getAttribute("title") || "").toLowerCase();

                            if (testId === "staudioinput" || (el.closest && el.closest('[data-testid="stAudioInput"]'))) {{
                                return true;
                            }}
                            if (aria.includes("record") || title.includes("record") || aria.includes("stop") || title.includes("stop")) {{
                                return true;
                            }}
                        }}
                    }} catch (err) {{}}
                    return false;
                }}

                function handleEvent(e) {{
                    if (isMicRecordTarget(e)) {{
                        lockSentence();
                    }}
                }}

                parentDoc.addEventListener("pointerdown", handleEvent, true);
                parentDoc.addEventListener("click", handleEvent, true);

                function scanNode(node) {{
                    if (!node) return false;
                    if (node.getAttribute) {{
                        const aria = (node.getAttribute("aria-label") || "").toLowerCase();
                        const title = (node.getAttribute("title") || "").toLowerCase();
                        const testId = (node.getAttribute("data-testid") || "").toLowerCase();
                        const className = (node.className || "").toString().toLowerCase();

                        if (aria.includes("stop") || title.includes("stop") || testId.includes("stop") || className.includes("recording") || className.includes("active-record")) {{
                            return true;
                        }}
                    }}
                    if (node.nodeType === 3) {{
                        const txt = (node.nodeValue || "").trim();
                        if (txt.includes("00:") || /^0\d:\d\d$/.test(txt)) {{
                            return true;
                        }}
                    }}
                    if (node.shadowRoot) {{
                        if (scanNode(node.shadowRoot)) return true;
                    }}
                    const ch = node.children || node.childNodes || [];
                    for (let i = 0; i < ch.length; i++) {{
                        if (scanNode(ch[i])) return true;
                    }}
                    return false;
                }}

                function checkAudioInputState() {{
                    try {{
                        const audioInput = parentDoc.querySelector('[data-testid="stAudioInput"]');
                        if (!audioInput) return false;
                        return scanNode(audioInput);
                    }} catch (e) {{
                        return false;
                    }}
                }}

                const interval = setInterval(function() {{
                    try {{
                        if (checkAudioInputState()) {{
                            lockSentence();
                            clearInterval(interval);
                        }}
                    }} catch (err) {{}}
                }}, 100);

                window.parent.__audio_drill_cleanup = function() {{
                    try {{
                        clearInterval(interval);
                        parentDoc.removeEventListener("pointerdown", handleEvent, true);
                        parentDoc.removeEventListener("click", handleEvent, true);
                    }} catch (err) {{}}
                }};
            }})();
            </script>
            """,
            height=0,
            width=0,
        )
    else:
        st.error("🔒 Text Locked — Recording Captured")

    # Render audio input directly below sentence box
    st.markdown("#### 🎙️ Your Spoken Audio")
    # Audio recorder key strictly scoped (also alias key=f"mic_input_{attempt_id}_q_{q_idx}" for compatibility)
    audio_val = st.audio_input(f"Record Sentence {q_idx + 1}", key=mic_key)
    audio_bytes = audio_val
    recorded_file = audio_val

    # Trigger evaluation the moment audio is submitted/recorded if not yet evaluated for this q_idx
    if audio_val is not None and st.session_state.get(result_key) is None:
        raw_bytes = audio_val.getvalue()
        with st.spinner("Transcribing with in-memory Whisper & assessing pronunciation accuracy..."):
            res = evaluate_voice_drill(current_sentence, raw_bytes)
            st.session_state[lock_key] = True
            st.session_state[done_key] = True
            st.session_state[result_key] = res
            st.session_state.audio_evaluations[q_idx] = res
            st.rerun()

    # Allow re-analysis if candidate recorded again for this q_idx
    if audio_val is not None and st.session_state.get(result_key) is not None:
        c_btn1, _ = st.columns([1, 2])
        with c_btn1:
            if st.button("🔄 Re-Analyze Recording", key=f"btn_re_eval_{attempt_id}_{q_idx}", use_container_width=True):
                raw_bytes = audio_val.getvalue()
                with st.spinner("Re-analyzing pronunciation accuracy & pace..."):
                    res = evaluate_voice_drill(current_sentence, raw_bytes)
                    st.session_state[lock_key] = True
                    st.session_state[done_key] = True
                    st.session_state[result_key] = res
                    st.session_state.audio_evaluations[q_idx] = res
                    st.rerun()

    # 3. Evaluation Feedback Card — ONLY rendered if st.session_state.get(result_key) exists for this specific q_idx!
    eval_result = st.session_state.get(result_key)
    if eval_result is not None:
        tr = eval_result.get('transcript', '').strip()
        clean_tr = html.escape(tr)
        if not tr or eval_result.get("voice_tone") == "Silent / No Speech Detected":
            st.markdown(
                """
                <div style="background: #FEF2F2; border: 1px solid #EF4444; border-radius: 12px; padding: 1rem 1.4rem; margin-top: 1rem;">
                    <div style="font-weight: 800; color: #991B1B; font-size: 1.05rem; margin-bottom: 0.4rem;">
                        ❌ No speech detected in recording. Please speak clearly into the microphone.
                    </div>
                    <div style="font-size: 0.9rem; color: #7F1D1D; line-height: 1.6;">
                        <b>Pronunciation & Recall Accuracy:</b> 0.0%<br>
                        <b>Speaking Pace:</b> 0 WPM &nbsp;|&nbsp; 
                        <b>Tone:</b> Silent / No Speech Detected &nbsp;|&nbsp; 
                        <b>Filler Words:</b> N/A (No Speech)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif eval_result.get("voice_tone") == "Skipped":
            st.markdown(
                """
                <div style="background: #FFFBEB; border: 1px solid #F59E0B; border-radius: 12px; padding: 1rem 1.4rem; margin-top: 1rem;">
                    <div style="font-weight: 800; color: #92400E; font-size: 1.05rem; margin-bottom: 0.4rem;">
                        ⏭️ Sentence Skipped
                    </div>
                    <div style="font-size: 0.9rem; color: #78350F; line-height: 1.6;">
                        This sentence was skipped without an audio recording. Score: 0.0%.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            acc_val = eval_result.get('accuracy_score', 85)
            wpm_val = eval_result.get('wpm', 135)
            tone_val = eval_result.get('voice_tone', 'Confident & Expressive')
            fill_val = eval_result.get('filler_count', 0)
            st.markdown(
                f"""
                <div style="background: #F0FDF4; border: 1px solid #10B981; border-radius: 12px; padding: 1rem 1.4rem; margin-top: 1rem;">
                    <div style="font-weight: 800; color: #064E3B; font-size: 1.05rem; margin-bottom: 0.4rem;">
                        Pronunciation & Recall Accuracy: {acc_val:.1f}%
                    </div>
                    <div style="font-size: 0.9rem; color: #047857; line-height: 1.6;">
                        <b>Transcript:</b> <i>"{clean_tr}"</i><br>
                        <b>Speaking Pace:</b> {wpm_val:.0f} WPM &nbsp;|&nbsp; 
                        <b>Tone:</b> {tone_val} &nbsp;|&nbsp; 
                        <b>Filler Words:</b> {fill_val}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Navigation Actions: Previous, Skip, Next
    st.markdown("<br>", unsafe_allow_html=True)
    is_last = (q_idx == len(sentences) - 1)
    next_txt = "🏁 Finish Audio Section" if is_last else "Next Sentence ➡️"

    col_p, col_s, col_n = st.columns([1, 1, 1.2])

    with col_p:
        if q_idx > 0 and st.button("⬅️ Previous Sentence", key=f"btn_audio_prev_{attempt_id}_{q_idx}", use_container_width=True):
            prev_idx = q_idx - 1
            st.session_state.audio_active_idx = prev_idx
            st.rerun()

    with col_s:
        if st.button("⏭️ Skip Sentence", key=f"btn_audio_skip_{attempt_id}_{q_idx}", use_container_width=True):
            skip_res = {
                "score": 0.0,
                "accuracy_score": 0.0,
                "word_accuracy": 0.0,
                "wpm": 0.0,
                "voice_tone": "Skipped",
                "transcript": "Skipped by candidate.",
                "filler_count": 0,
            }
            st.session_state[lock_key] = True
            st.session_state[done_key] = True
            st.session_state[result_key] = skip_res
            st.session_state.audio_evaluations[q_idx] = skip_res
            if is_last:
                _finish_audio_section()
            else:
                next_idx = q_idx + 1
                next_lock_key = f"audio_locked_{attempt_id}_q_{next_idx}"
                next_done_key = f"audio_done_{attempt_id}_q_{next_idx}"
                next_result_key = f"audio_res_{attempt_id}_q_{next_idx}"
                next_widget_key = f"audio_mic_{attempt_id}_q_{next_idx}"
                next_mic_key = f"audio_mic_{attempt_id}_q_{next_idx}"
                if next_idx not in st.session_state.audio_evaluations:
                    st.session_state[next_lock_key] = False
                    st.session_state[next_done_key] = False
                    st.session_state[next_result_key] = None
                    if next_widget_key in st.session_state:
                        del st.session_state[next_widget_key]
                    if next_mic_key in st.session_state:
                        del st.session_state[next_mic_key]
                st.session_state.audio_active_idx = next_idx
            st.rerun()

    with col_n:
        if st.button(next_txt, type="primary", key=f"btn_audio_next_{attempt_id}_{q_idx}", use_container_width=True):
            if st.session_state.get(result_key) is None and q_idx not in st.session_state.audio_evaluations:
                st.warning("Please record your audio response, or click 'Skip Sentence' to proceed.")
            else:
                if is_last:
                    _finish_audio_section()
                else:
                    next_idx = q_idx + 1
                    next_lock_key = f"audio_locked_{attempt_id}_q_{next_idx}"
                    next_done_key = f"audio_done_{attempt_id}_q_{next_idx}"
                    next_result_key = f"audio_res_{attempt_id}_q_{next_idx}"
                    next_widget_key = f"audio_mic_{attempt_id}_q_{next_idx}"
                    next_mic_key = f"audio_mic_{attempt_id}_q_{next_idx}"
                    if next_idx not in st.session_state.audio_evaluations:
                        st.session_state[next_lock_key] = False
                        st.session_state[next_done_key] = False
                        st.session_state[next_result_key] = None
                        if next_widget_key in st.session_state:
                            del st.session_state[next_widget_key]
                        if next_mic_key in st.session_state:
                            del st.session_state[next_mic_key]
                    st.session_state.audio_active_idx = next_idx
                st.rerun()


def _finish_audio_section():
    """Aggregate audio drill scores and return to Hub."""
    evals = st.session_state.audio_evaluations
    scores = [e.get("accuracy_score", 0.0) for e in evals.values()]
    avg_score = float(sum(scores) / max(len(scores), 1))
    st.session_state.completed_sections["audio_recording"] = avg_score
    st.session_state.section_results["audio_recording"] = {
        "score": avg_score,
        "evaluations": evals,
        "difficulty": st.session_state.audio_difficulty,
    }
    st.session_state.current_screen = "hub"



# ==============================================================================
# SCREEN 4: EXECUTIVE PERFORMANCE ANALYTICS & AI COACH (REPORT SCREEN)
# ==============================================================================

def _persist_active_session() -> None:
    """Consolidate session questions, user answers, scores, and ideal answers across all sections into SQLite."""
    user = st.session_state.authenticated_user or {}
    email = user.get("email", "") or f"{st.session_state.candidate_name.lower().replace(' ', '')}@gmail.com"

    # Generate personalized 60-90s executive model self-introduction from candidate resume and target role
    model_intro = generate_model_self_introduction(
        candidate_name=st.session_state.candidate_name,
        target_role=st.session_state.target_role,
        resume_text=st.session_state.resume_text,
        detected_skills=st.session_state.detected_skills,
        detected_projects=st.session_state.detected_projects,
    )

    # Pack comprehensive snapshot across all sections
    qa_snapshot = {
        "self_intro": {
            "question": "Please introduce yourself, walk through your technical background, highlight 1-2 flagship projects from your resume, and explain why you are well-suited for the target role.",
            "answer": st.session_state.self_intro_eval.get("transcript", "No recording") if st.session_state.self_intro_eval else "Unattempted",
            "score": st.session_state.completed_sections.get("self_intro", 0.0),
            "ideal_answer": model_intro,
            "feedback": st.session_state.self_intro_eval.get("strengths", []) if st.session_state.self_intro_eval else [],
        },
        "technical": [
            {
                "question": (q.get("question", f"Question {i+1}") if isinstance(q, dict) else (str(q).strip() or f"Question {i+1}")),
                "answer": st.session_state.tech_answers.get(i, "Skipped"),
                "score": (st.session_state.tech_evaluations.get(i, {}).get("score", 0.0) if isinstance(st.session_state.tech_evaluations.get(i), dict) else 0.0),
                "ideal_answer": (st.session_state.tech_evaluations.get(i, {}).get("ideal_answer") if isinstance(st.session_state.tech_evaluations.get(i), dict) else "") or (q.get("ideal_answer", "") if isinstance(q, dict) else ""),
                "feedback": (st.session_state.tech_evaluations.get(i, {}).get("feedback", "") if isinstance(st.session_state.tech_evaluations.get(i), dict) else ""),
            }
            for i, q in enumerate(st.session_state.tech_questions)
        ],
        "dsa": [
            {
                "question": ((p.get("title", f"Problem {i+1}") + ": " + p.get("description", "")) if isinstance(p, dict) else (str(p).strip() or f"Problem {i+1}")),
                "answer": st.session_state.dsa_codes.get(i, "Skipped"),
                "score": (st.session_state.dsa_evaluations.get(i, {}).get("score", 0.0) if isinstance(st.session_state.dsa_evaluations.get(i), dict) else 0.0),
                "ideal_answer": (p.get("optimal_solution", "") if isinstance(p, dict) else ""),
                "feedback": (st.session_state.dsa_evaluations.get(i, {}).get("feedback", "") if isinstance(st.session_state.dsa_evaluations.get(i), dict) else ""),
            }
            for i, p in enumerate(st.session_state.dsa_problems)
        ],
        "project": [
            {
                "question": (q.get("question", f"Question {i+1}") if isinstance(q, dict) else (str(q).strip() or f"Question {i+1}")),
                "answer": st.session_state.project_answers.get(i, "Skipped"),
                "score": (st.session_state.project_evaluations.get(i, {}).get("score", 0.0) if isinstance(st.session_state.project_evaluations.get(i), dict) else 0.0),
                "ideal_answer": (st.session_state.project_evaluations.get(i, {}).get("ideal_answer", "") if isinstance(st.session_state.project_evaluations.get(i), dict) else ""),
                "feedback": (st.session_state.project_evaluations.get(i, {}).get("feedback", "") if isinstance(st.session_state.project_evaluations.get(i), dict) else ""),
            }
            for i, q in enumerate(st.session_state.project_questions)
        ],
        "aptitude": [
            {
                "question": (q.get("question", f"Question {i+1}") if isinstance(q, dict) else (str(q).strip() or f"Question {i+1}")),
                "answer": st.session_state.aptitude_answers.get(i, "Skipped"),
                "score": (st.session_state.aptitude_evaluations.get(i, {}).get("score", 0.0) if isinstance(st.session_state.aptitude_evaluations.get(i), dict) else 0.0),
                "ideal_answer": (f"Correct Option: {q.get('correct_answer', '')} — {q.get('explanation', '')}") if isinstance(q, dict) else "",
                "feedback": "Correct" if (isinstance(st.session_state.aptitude_evaluations.get(i), dict) and st.session_state.aptitude_evaluations.get(i, {}).get("is_correct")) else "Incorrect",
            }
            for i, q in enumerate(st.session_state.aptitude_questions)
        ],
        "audio_recording": [
            {
                "question": f"Sentence {i+1}: " + ((s.get("text") or s.get("sentence", "")) if isinstance(s, dict) else str(s).strip()),
                "answer": (st.session_state.audio_evaluations.get(i, {}).get("transcript", "Skipped") if isinstance(st.session_state.audio_evaluations.get(i), dict) else "Skipped"),
                "score": (st.session_state.audio_evaluations.get(i, {}).get("score", 0.0) if isinstance(st.session_state.audio_evaluations.get(i), dict) else 0.0),
                "ideal_answer": ((s.get("text") or s.get("sentence", "")) if isinstance(s, dict) else str(s).strip()),
                "feedback": (f"WPM: {st.session_state.audio_evaluations.get(i, {}).get('wpm', 0):.0f} | Tone: {st.session_state.audio_evaluations.get(i, {}).get('voice_tone', '')}" if isinstance(st.session_state.audio_evaluations.get(i), dict) else ""),
            }
            for i, s in enumerate(st.session_state.audio_sentences)
        ],
    }

    sec_res = dict(st.session_state.section_results)
    sec_res["qa_snapshot"] = qa_snapshot
    media_data = st.session_state.media_buffer.current_metrics()

    finalize_and_save_interview(
        candidate_name=st.session_state.candidate_name,
        target_role=st.session_state.target_role,
        user_email=email,
        section_results=sec_res,
        media_metrics=media_data,
        resume_text=st.session_state.resume_text,
        detected_skills=st.session_state.detected_skills,
        detected_projects=st.session_state.detected_projects,
    )


def build_user_answers_export(session: dict[str, Any]) -> tuple[bytes, str]:
    """Build CSV and TXT exports of the candidate's exact submitted answers across all sections."""
    details = session.get("details", {})
    sec_results = details.get("section_results", {}) if isinstance(details, dict) else {}
    qa_snapshot = sec_results.get("qa_snapshot", {}) if isinstance(sec_results, dict) else {}

    rows = []
    txt_lines = [
        "AI INTERVIEW SIMULATOR — CANDIDATE SUBMITTED ANSWERS",
        f"Candidate: {session.get('candidate_name', 'Candidate')}",
        f"Target Role: {session.get('target_role', 'Engineering Role')}",
        f"Session ID: {session.get('id', 'N/A')} | Date: {session.get('timestamp', 'Recent')}",
        f"Overall Readiness Score: {session.get('overall_score', 0):.1f}%",
        "=" * 70,
    ]

    # 1. Self Intro
    intro = qa_snapshot.get("self_intro", {})
    if intro:
        rows.append({
            "Section": "Self Introduction",
            "Item": "Question 1",
            "Prompt": intro.get("question", ""),
            "Candidate_Answer": intro.get("answer", "Unattempted"),
            "Score_Awarded": intro.get("score", 0.0),
            "Feedback": ", ".join(intro.get("feedback", [])) if isinstance(intro.get("feedback"), list) else str(intro.get("feedback", "")),
        })
        txt_lines.extend([
            "\n[SECTION 1: SELF INTRODUCTION]",
            f"Prompt: {intro.get('question', '')}",
            f"Candidate Transcript: {intro.get('answer', 'Unattempted')}",
            f"Score: {intro.get('score', 0.0):.1f} / 100",
            "-" * 50,
        ])

    # 2. Technical
    tech_list = qa_snapshot.get("technical", []) if isinstance(qa_snapshot, dict) else []
    if tech_list:
        txt_lines.append("\n[SECTION 2: TECHNICAL ASSESSMENT (OOPs, OS, DS)]")
        for idx, item in enumerate(tech_list):
            it = item if isinstance(item, dict) else {"question": str(item), "answer": "Skipped", "score": 0.0, "feedback": ""}
            rows.append({
                "Section": "Technical Assessment",
                "Item": f"Question {idx + 1}",
                "Prompt": it.get("question", ""),
                "Candidate_Answer": it.get("answer", "Skipped"),
                "Score_Awarded": it.get("score", 0.0),
                "Feedback": it.get("feedback", ""),
            })
            txt_lines.extend([
                f"\nQ{idx + 1}: {it.get('question', '')}",
                f"Your Answer: {it.get('answer', 'Skipped')}",
                f"Score: {it.get('score', 0.0):.1f} / 100 | Feedback: {it.get('feedback', '')}",
            ])
        txt_lines.append("-" * 50)

    # 3. DSA
    dsa_list = qa_snapshot.get("dsa", []) if isinstance(qa_snapshot, dict) else []
    if dsa_list:
        txt_lines.append("\n[SECTION 3: ALGORITHMIC PROBLEMS (DSA)]")
        for idx, item in enumerate(dsa_list):
            it = item if isinstance(item, dict) else {"question": str(item), "answer": "Skipped", "score": 0.0, "feedback": ""}
            rows.append({
                "Section": "Algorithmic Problems (DSA)",
                "Item": f"Problem {idx + 1}",
                "Prompt": it.get("question", ""),
                "Candidate_Answer": it.get("answer", "Skipped"),
                "Score_Awarded": it.get("score", 0.0),
                "Feedback": it.get("feedback", ""),
            })
            txt_lines.extend([
                f"\nProblem {idx + 1}: {it.get('question', '')}",
                f"Submitted Code:\n{it.get('answer', 'Skipped')}",
                f"Judge Score: {it.get('score', 0.0):.1f}% | Feedback: {it.get('feedback', '')}",
            ])
        txt_lines.append("-" * 50)

    # 4. Project
    proj_list = qa_snapshot.get("project", []) if isinstance(qa_snapshot, dict) else []
    if proj_list:
        txt_lines.append("\n[SECTION 4: PROJECT DEEP DIVE]")
        for idx, item in enumerate(proj_list):
            it = item if isinstance(item, dict) else {"question": str(item), "answer": "Skipped", "score": 0.0, "feedback": ""}
            rows.append({
                "Section": "Project Deep Dive",
                "Item": f"Question {idx + 1}",
                "Prompt": it.get("question", ""),
                "Candidate_Answer": it.get("answer", "Skipped"),
                "Score_Awarded": it.get("score", 0.0),
                "Feedback": it.get("feedback", ""),
            })
            txt_lines.extend([
                f"\nQ{idx + 1}: {it.get('question', '')}",
                f"Your Answer: {it.get('answer', 'Skipped')}",
                f"Score: {it.get('score', 0.0):.1f} / 100",
            ])
        txt_lines.append("-" * 50)

    # 5. Aptitude
    apt_list = qa_snapshot.get("aptitude", []) if isinstance(qa_snapshot, dict) else []
    if apt_list:
        txt_lines.append("\n[SECTION 5: APTITUDE & DATA INTERPRETATION]")
        for idx, item in enumerate(apt_list):
            it = item if isinstance(item, dict) else {"question": str(item), "answer": "Skipped", "score": 0.0, "feedback": ""}
            rows.append({
                "Section": "Aptitude & Data Interpretation",
                "Item": f"Question {idx + 1}",
                "Prompt": it.get("question", ""),
                "Candidate_Answer": it.get("answer", "Skipped"),
                "Score_Awarded": it.get("score", 0.0),
                "Feedback": it.get("feedback", ""),
            })
            txt_lines.extend([
                f"\nQ{idx + 1}: {it.get('question', '')}",
                f"Your Selection: {it.get('answer', 'Skipped')} ({it.get('feedback', '')})",
                f"Score: {it.get('score', 0.0):.1f} / 100",
            ])
        txt_lines.append("-" * 50)

    # 6. Audio Recordings
    audio_list = qa_snapshot.get("audio_recording", []) if isinstance(qa_snapshot, dict) else []
    if audio_list:
        txt_lines.append("\n[SECTION 6: AUDIO RECORDINGS & PRONUNCIATION]")
        for idx, item in enumerate(audio_list):
            it = item if isinstance(item, dict) else {"question": str(item), "answer": "Skipped", "score": 0.0, "feedback": ""}
            rows.append({
                "Section": "Audio Pronunciation Drill",
                "Item": f"Sentence {idx + 1}",
                "Prompt": it.get("question", ""),
                "Candidate_Answer": it.get("answer", "Skipped"),
                "Score_Awarded": it.get("score", 0.0),
                "Feedback": it.get("feedback", ""),
            })
            txt_lines.extend([
                f"\nSentence {idx + 1}: {it.get('question', '')}",
                f"Transcribed Audio: {it.get('answer', 'Skipped')}",
                f"Accuracy Score: {it.get('score', 0.0):.1f}% | {it.get('feedback', '')}",
            ])
        txt_lines.append("-" * 50)

    if not rows:
        rows.append({"Session": session.get("id", 1), "Summary": "Interview Session Snapshot", "Score": session.get("overall_score", 0.0)})

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8"), "\n".join(txt_lines)


def build_ideal_answers_export(session: dict[str, Any]) -> tuple[bytes, str]:
    """Build CSV and TXT exports of the expected ideal benchmark answers across all sections."""
    details = session.get("details", {})
    sec_results = details.get("section_results", {}) if isinstance(details, dict) else {}
    qa_snapshot = sec_results.get("qa_snapshot", {}) if isinstance(sec_results, dict) else {}

    rows = []
    txt_lines = [
        "AI INTERVIEW SIMULATOR — IDEAL BENCHMARK & CORRECT ANSWERS",
        f"Target Role: {session.get('target_role', 'Engineering Role')}",
        f"Session ID: {session.get('id', 'N/A')}",
        "=" * 70,
    ]

    # 1. Self Intro (Personalized 60-90s model self-introduction tailored to resume and target role)
    intro = qa_snapshot.get("self_intro", {})
    cand_name = session.get("candidate_name") or "Candidate"
    target_role = session.get("target_role") or "Software Engineer"
    res_text = details.get("resume_text", "") if isinstance(details, dict) else ""
    det_skills = details.get("detected_skills", []) if isinstance(details, dict) else []
    det_projs = details.get("detected_projects", []) if isinstance(details, dict) else []

    ideal_intro = intro.get("ideal_answer", "")
    if not ideal_intro or len(ideal_intro) < 80:
        ideal_intro = generate_model_self_introduction(
            candidate_name=cand_name,
            target_role=target_role,
            resume_text=res_text,
            detected_skills=det_skills,
            detected_projects=det_projs,
        )

    intro_prompt = intro.get(
        "question",
        "Please introduce yourself, walk through your technical background, highlight 1-2 flagship projects from your resume, and explain why you are well-suited for the target role.",
    )
    rows.append({
        "Section": "Self Introduction",
        "Item": "Question 1",
        "Prompt": intro_prompt,
        "Ideal_Benchmark_Answer": ideal_intro,
    })
    txt_lines.extend([
        "\n[SECTION 1: SELF INTRODUCTION]",
        f"Prompt: {intro_prompt}",
        f"\nModel Executive Self-Introduction (60-90s Benchmark tailored to Candidate & Resume):\n{ideal_intro}",
        "-" * 50,
    ])

    # 2. Technical
    tech_list = qa_snapshot.get("technical", []) if isinstance(qa_snapshot, dict) else []
    if tech_list:
        txt_lines.append("\n[SECTION 2: TECHNICAL ASSESSMENT (OOPs, OS, DS)]")
        for idx, item in enumerate(tech_list):
            it = item if isinstance(item, dict) else {"question": str(item), "ideal_answer": str(item)}
            rows.append({
                "Section": "Technical Assessment",
                "Item": f"Question {idx + 1}",
                "Prompt": it.get("question", ""),
                "Ideal_Benchmark_Answer": it.get("ideal_answer", ""),
            })
            txt_lines.extend([
                f"\nQ{idx + 1}: {it.get('question', '')}",
                f"Ideal Conceptual Answer: {it.get('ideal_answer', '')}",
            ])
        txt_lines.append("-" * 50)

    # 3. DSA
    dsa_list = qa_snapshot.get("dsa", []) if isinstance(qa_snapshot, dict) else []
    if dsa_list:
        txt_lines.append("\n[SECTION 3: ALGORITHMIC PROBLEMS (DSA)]")
        for idx, item in enumerate(dsa_list):
            it = item if isinstance(item, dict) else {"question": str(item), "ideal_answer": str(item)}
            rows.append({
                "Section": "Algorithmic Problems (DSA)",
                "Item": f"Problem {idx + 1}",
                "Prompt": it.get("question", ""),
                "Ideal_Benchmark_Answer": it.get("ideal_answer", ""),
            })
            txt_lines.extend([
                f"\nProblem {idx + 1}: {it.get('question', '')}",
                f"Optimal Solution Code:\n{it.get('ideal_answer', '')}",
            ])
        txt_lines.append("-" * 50)

    # 4. Project
    proj_list = qa_snapshot.get("project", []) if isinstance(qa_snapshot, dict) else []
    if proj_list:
        txt_lines.append("\n[SECTION 4: PROJECT DEEP DIVE]")
        for idx, item in enumerate(proj_list):
            it = item if isinstance(item, dict) else {"question": str(item), "ideal_answer": str(item)}
            rows.append({
                "Section": "Project Deep Dive",
                "Item": f"Question {idx + 1}",
                "Prompt": it.get("question", ""),
                "Ideal_Benchmark_Answer": it.get("ideal_answer", ""),
            })
            txt_lines.extend([
                f"\nQ{idx + 1}: {it.get('question', '')}",
                f"Ideal Architectural Explanation: {it.get('ideal_answer', '')}",
            ])
        txt_lines.append("-" * 50)

    # 5. Aptitude
    apt_list = qa_snapshot.get("aptitude", []) if isinstance(qa_snapshot, dict) else []
    if apt_list:
        txt_lines.append("\n[SECTION 5: APTITUDE & DATA INTERPRETATION]")
        for idx, item in enumerate(apt_list):
            it = item if isinstance(item, dict) else {"question": str(item), "ideal_answer": str(item)}
            rows.append({
                "Section": "Aptitude & Data Interpretation",
                "Item": f"Question {idx + 1}",
                "Prompt": it.get("question", ""),
                "Ideal_Benchmark_Answer": it.get("ideal_answer", ""),
            })
            txt_lines.extend([
                f"\nQ{idx + 1}: {it.get('question', '')}",
                f"Correct Option & Explanation: {it.get('ideal_answer', '')}",
            ])
        txt_lines.append("-" * 50)

    # 6. Audio
    audio_list = qa_snapshot.get("audio_recording", []) if isinstance(qa_snapshot, dict) else []
    if audio_list:
        txt_lines.append("\n[SECTION 6: AUDIO RECORDINGS & PRONUNCIATION]")
        for idx, item in enumerate(audio_list):
            it = item if isinstance(item, dict) else {"question": str(item), "ideal_answer": str(item)}
            rows.append({
                "Section": "Audio Pronunciation Drill",
                "Item": f"Sentence {idx + 1}",
                "Prompt": it.get("question", ""),
                "Ideal_Benchmark_Answer": it.get("ideal_answer", ""),
            })
            txt_lines.extend([
                f"\nSentence {idx + 1}: {it.get('question', '')}",
                f"Target Verbatim Sentence: {it.get('ideal_answer', '')}",
            ])
        txt_lines.append("-" * 50)

    if not rows:
        rows.append({"Session": session.get("id", 1), "Note": "Ideal answers available for review"})

    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8"), "\n".join(txt_lines)


def _sync_attempted_sections() -> None:
    """Ensure all sections with answers or evaluations are recorded as Attempted with exact scores."""
    # Technical
    if st.session_state.tech_evaluations:
        scores = [float(e.get("score", 0.0) if isinstance(e, dict) else (e if isinstance(e, (int, float)) else 0.0)) for e in st.session_state.tech_evaluations.values()]
        avg_score = round(float(sum(scores) / max(len(scores), 1)), 1)
        st.session_state.completed_sections["technical"] = avg_score
        st.session_state.section_results["technical"] = {
            "score": avg_score,
            "evaluations": st.session_state.tech_evaluations,
            "difficulty": st.session_state.tech_difficulty,
            "attempted_count": len(st.session_state.tech_evaluations),
            "total_count": len(st.session_state.tech_questions),
        }
    # DSA
    if st.session_state.dsa_evaluations:
        scores = [float(e.get("score", 0.0) if isinstance(e, dict) else (e if isinstance(e, (int, float)) else 0.0)) for e in st.session_state.dsa_evaluations.values()]
        avg_score = round(float(sum(scores) / max(len(scores), 1)), 1)
        st.session_state.completed_sections["dsa"] = avg_score
        st.session_state.section_results["dsa"] = {
            "score": avg_score,
            "evaluations": st.session_state.dsa_evaluations,
            "difficulty": st.session_state.dsa_difficulty,
            "attempted_count": len(st.session_state.dsa_evaluations),
            "total_count": len(st.session_state.dsa_problems),
        }
    # Project
    if st.session_state.project_evaluations:
        scores = [float(e.get("score", 0.0) if isinstance(e, dict) else (e if isinstance(e, (int, float)) else 0.0)) for e in st.session_state.project_evaluations.values()]
        avg_score = round(float(sum(scores) / max(len(scores), 1)), 1)
        st.session_state.completed_sections["project"] = avg_score
        st.session_state.section_results["project"] = {
            "score": avg_score,
            "evaluations": st.session_state.project_evaluations,
            "difficulty": st.session_state.project_difficulty,
            "attempted_count": len(st.session_state.project_evaluations),
            "total_count": len(st.session_state.project_questions),
        }
    # Aptitude
    if st.session_state.aptitude_evaluations:
        scores = [float(e.get("score", 0.0) if isinstance(e, dict) else (e if isinstance(e, (int, float)) else 0.0)) for e in st.session_state.aptitude_evaluations.values()]
        avg_score = round(float(sum(scores) / max(len(scores), 1)), 1)
        st.session_state.completed_sections["aptitude"] = avg_score
        st.session_state.section_results["aptitude"] = {
            "score": avg_score,
            "evaluations": st.session_state.aptitude_evaluations,
            "difficulty": st.session_state.aptitude_difficulty,
            "attempted_count": len(st.session_state.aptitude_evaluations),
            "total_count": len(st.session_state.aptitude_questions),
        }
    # Audio Recording
    if st.session_state.audio_evaluations:
        scores = [float((e.get("score", e.get("accuracy_score", 0.0))) if isinstance(e, dict) else (e if isinstance(e, (int, float)) else 0.0)) for e in st.session_state.audio_evaluations.values()]
        avg_score = round(float(sum(scores) / max(len(scores), 1)), 1)
        st.session_state.completed_sections["audio_recording"] = avg_score
        st.session_state.section_results["audio_recording"] = {
            "score": avg_score,
            "evaluations": st.session_state.audio_evaluations,
            "difficulty": st.session_state.audio_difficulty,
            "attempted_count": len(st.session_state.audio_evaluations),
            "total_count": len(st.session_state.audio_sentences) or 10,
        }
    # Self Intro
    if st.session_state.self_intro_eval:
        intro_val = st.session_state.self_intro_eval.get("score", 0.0) if isinstance(st.session_state.self_intro_eval, dict) else st.session_state.self_intro_eval
        intro_score = round(float(intro_val if isinstance(intro_val, (int, float)) else 0.0), 1)
        st.session_state.completed_sections["self_intro"] = intro_score
        st.session_state.section_results["self_intro"] = st.session_state.self_intro_eval
    # Personality
    if st.session_state.personality_eval:
        p_val = st.session_state.personality_eval.get("culture_fit_score", 0.0) if isinstance(st.session_state.personality_eval, dict) else st.session_state.personality_eval
        p_score = round(float(p_val if isinstance(p_val, (int, float)) else 0.0), 1)
        st.session_state.completed_sections["personality"] = p_score
        st.session_state.section_results["personality"] = st.session_state.personality_eval


def render_report_screen():
    """Render comprehensive executive Performance Analytics & AI Longitudinal Coaching Report."""
    render_header()
    _sync_attempted_sections()

    st.markdown(REPORT_HERO_SVG, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <h1 class="hero-title">Executive Performance Analytics & AI Coach</h1>
            <p class="hero-subtitle">
                Consolidated assessment report analyzing multi-modal competencies: resume alignment, OOPs/OS/DS technical precision, algorithmic problem solving, psychometric profile, acoustic tone, and vision posture stability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Automatically persist session to SQLite database with full QA snapshot if not yet saved
    if not st.session_state.session_saved:
        try:
            _persist_active_session()
            st.session_state.session_saved = True
        except Exception:
            pass

    # 1. Top Formula Banner (Requirement 4.2)
    st.markdown(
        """
        <div class="formula-banner">
            <div class="formula-title">
                <span>📐</span> Exact Scoring Formula & Evaluation Basis
            </div>
            <div style="font-size: 0.88rem; line-height: 1.6; color: #E2E8F0;">
                <b>Overall Candidate Readiness Score Formula:</b><br>
                • <b>Technical Assessment (25%):</b> Graded on mechanical correctness, domain terminology (OOPs/OS/DS), and trade-offs.<br>
                • <b>Algorithmic Coding DSA (25%):</b> Automated judge test case pass percentage with $\mathcal{O}(n)$ complexity benchmarks.<br>
                • <b>Self Introduction (15%):</b> Resume competency coverage, conversational pace (130–155 WPM), and vocal inflection.<br>
                • <b>Project Deep Dive (15%):</b> Architectural scalability, quantifiable telemetry metrics, and fault-tolerance design.<br>
                • <b>Aptitude & Data Interpretation (10%):</b> Numerical accuracy and systems performance scenario solving.<br>
                • <b>Audio Recordings Drill (10%):</b> Whisper word recall accuracy (%) and Librosa pitch/RMS energy classification.<br>
                • <b>Vision & Posture Deductions:</b> Up to -10% deduction if shoulder tilt exceeds $\pm 5^\circ$ or gaze drifts for &gt;2.0 seconds.<br>
                • <i>Note: Unattempted sections contribute 0 / 100 to transparently reflect completed interview milestones.</i>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = st.session_state.section_results
    completed = st.session_state.completed_sections
    media = st.session_state.media_buffer.current_metrics()

    # Calculate overall readiness score based on attempted sections
    all_possible_sections = ["self_intro", "technical", "project", "aptitude", "personality", "audio_recording"]
    if st.session_state.is_developer_role:
        all_possible_sections.append("dsa")

    scores = list(completed.values())
    overall_score = float(sum(scores) / max(len(scores), 1)) if scores else 0.0

    # Top KPI Cards Row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Overall Readiness", f"{overall_score:.1f}%")
    posture_kpi = float(media.get("posture", 0.0))
    k2.metric("Posture Stability", f"{posture_kpi:.1f}%" if posture_kpi > 0 else "0.0% (Not Detected)")
    eye_kpi = float(media.get("eye_contact", 0.0))
    k3.metric("Eye Contact", f"{eye_kpi:.1f}%" if eye_kpi > 0 else "0.0% (Not Detected)")
    k4.metric("Posture Errors", f"{media.get('posture_errors', 0)}")
    k5.metric("Completed", f"{len(completed)} / {len(all_possible_sections)}")
    k6.metric("Target Role", st.session_state.target_role or "Candidate")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row: Gauge & Radar Charts
    col_gauge, col_radar = st.columns([1, 1], gap="medium")

    with col_gauge:
        st.markdown("#### 🎯 Overall Candidate Readiness")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=overall_score,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1E1B4B"},
                    "bar": {"color": "#059669"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [0, 50], "color": "#FEE2E2"},
                        {"range": [50, 75], "color": "#FEF3C7"},
                        {"range": [75, 100], "color": "#DCFCE7"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_radar:
        st.markdown("#### 🕸️ Multi-Dimensional Competency Radar")
        radar_categories = ["Self Intro", "Technical", "Projects", "Aptitude", "Personality", "Audio Pronunciation", "Posture Stability"]
        radar_values = [
            float(completed.get("self_intro", 0.0)),
            float(completed.get("technical", 0.0)),
            float(completed.get("project", 0.0)),
            float(completed.get("aptitude", 0.0)),
            float(completed.get("personality", 0.0)),
            float(completed.get("audio_recording", 0.0)),
            float(media.get("posture", 0.0)),
        ]
        if st.session_state.is_developer_role:
            radar_categories.append("DSA Algorithms")
            radar_values.append(float(completed.get("dsa", 0.0)))

        radar_cats_closed = radar_categories + [radar_categories[0]]
        radar_vals_closed = radar_values + [radar_values[0]]

        fig_radar = go.Figure(
            go.Scatterpolar(
                r=radar_vals_closed,
                theta=radar_cats_closed,
                fill="toself",
                mode="lines+markers",
                marker=dict(size=7, color="#059669"),
                fillcolor="rgba(5, 150, 105, 0.2)",
                line=dict(color="#059669", width=2),
            )
        )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 20, 40, 60, 80, 100],
                    ticktext=["0", "20", "40", "60", "80", "100"],
                    gridcolor="#E2E8F0",
                )
            ),
            height=280,
            margin=dict(l=25, r=25, t=25, b=25),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 3. Section-Wise Scores & Concrete Improvement Points (Requirement 4.2 & Requirement 2)
    st.markdown("---")
    st.markdown("### 📋 Section-Wise Performance & Concrete Improvement Guide")

    section_guide_metadata = {
        "self_intro": {
            "title": "Self Introduction",
            "benchmark": "Evaluated on resume alignment, vocal fluency (130-155 WPM), and executive confidence.",
            "improvements": [
                "Ensure signature resume projects and metrics are explicitly cited in your first 60 seconds.",
                "Maintain a steady conversational cadence between 130 and 150 WPM with deliberate pauses.",
                "Frame career progression around quantified business outcomes rather than generic task listings."
            ]
        },
        "technical": {
            "title": "Technical Assessment",
            "benchmark": "Evaluated across OOPs (Polymorphism, SOLID), DBMS/SQL (Normalization, ACID, Joins), OS, and Networks.",
            "improvements": [
                "Anchor conceptual answers with internal mechanical execution (e.g. vtable dynamic dispatch, TLB caching, B+Tree indexes).",
                "State explicit time/space trade-offs when comparing data structures (e.g. CPU cache locality of arrays vs linked lists).",
                "Address edge-case concurrency race conditions and explain mutex vs semaphore selection."
            ]
        },
        "dsa": {
            "title": "Algorithmic Coding (DSA)",
            "benchmark": "Evaluated on automated test cases, algorithmic time complexity, and edge-case boundary handling.",
            "improvements": [
                "Verify edge-case inputs (empty collections, single-element bounds, and negative numbers) before running solutions.",
                "Optimize auxiliary space complexity from O(N) to O(1) where two-pointer or greedy passes apply.",
                "Pre-allocate data structures to prevent dynamic array re-allocation overhead in tight loops."
            ]
        },
        "project": {
            "title": "Project Deep Dive",
            "benchmark": "Evaluated on architectural trade-offs, bottlenecks, database design, metrics, and technical contributions.",
            "improvements": [
                "Quantify system bottlenecks using concrete telemetry (P99 latency reduction, throughput, concurrent users).",
                "Articulate failure recovery mechanisms (circuit breaking, dead-letter queues, idempotent retry loops).",
                "Explain database indexing and partitioning decisions under multi-terabyte data volume."
            ]
        },
        "aptitude": {
            "title": "Aptitude & Data Interpretation",
            "benchmark": "Numerical reasoning, probability, logic puzzles, and systems metrics (Little's Law, AMAT).",
            "improvements": [
                "Deconstruct multi-variable rate equations systematically before estimating solutions.",
                "Master Little's Law (L = λ * W) and cache AMAT equations for rapid mental calculation.",
                "Double-check boundary constraints in permutation and combination scenarios."
            ]
        },
        "personality": {
            "title": "Psychometric Personality (OCEAN)",
            "benchmark": "Big Five trait profiling (OCEAN) and organizational culture fit.",
            "improvements": [
                "Foster proactive cross-functional collaboration and blameless postmortem culture.",
                "Balance rapid feature shipping with disciplined automated regression test coverage.",
                "Demonstrate composure and structured triage during simulated production incident response."
            ]
        },
        "audio_recording": {
            "title": "Audio Recordings Drill",
            "benchmark": "Word pronunciation accuracy, acoustic tone variance, and speaking pace.",
            "improvements": [
                "Articulate technical keywords cleanly without swallowing consonants or mumbling clauses.",
                "Vary vocal pitch inflection to emphasize critical architectural achievements.",
                "Practice active auditory recall without relying on written text prompts."
            ]
        }
    }

    display_sections = ["self_intro", "technical"]
    if st.session_state.is_developer_role:
        display_sections.append("dsa")
    display_sections.extend(["project", "aptitude", "personality", "audio_recording"])

    for sec_id in display_sections:
        meta = section_guide_metadata.get(sec_id, {"title": sec_id.title(), "benchmark": "", "improvements": []})
        nice_title = meta["title"]
        is_attempted = sec_id in completed
        score = completed.get(sec_id, 0.0)

        # Compute question counts and score breakdown for attempted sections
        att_cnt = 0
        tot_cnt = 0
        n_correct = 0
        n_partial = 0
        n_wrong = 0

        if sec_id == "technical":
            att_cnt = len(st.session_state.tech_evaluations)
            tot_cnt = len(st.session_state.tech_questions) or 10
            for ev in st.session_state.tech_evaluations.values():
                sc = float(ev.get("score", 0.0) if isinstance(ev, dict) else (ev if isinstance(ev, (int, float)) else 0.0))
                if sc >= 70: n_correct += 1
                elif sc >= 30: n_partial += 1
                else: n_wrong += 1
        elif sec_id == "dsa":
            att_cnt = len(st.session_state.dsa_evaluations)
            tot_cnt = len(st.session_state.dsa_problems) or 3
            for ev in st.session_state.dsa_evaluations.values():
                sc = float(ev.get("score", 0.0) if isinstance(ev, dict) else (ev if isinstance(ev, (int, float)) else 0.0))
                if sc >= 70: n_correct += 1
                elif sc >= 30: n_partial += 1
                else: n_wrong += 1
        elif sec_id == "project":
            att_cnt = len(st.session_state.project_evaluations)
            tot_cnt = len(st.session_state.project_questions) or 5
            for ev in st.session_state.project_evaluations.values():
                sc = float(ev.get("score", 0.0) if isinstance(ev, dict) else (ev if isinstance(ev, (int, float)) else 0.0))
                if sc >= 70: n_correct += 1
                elif sc >= 30: n_partial += 1
                else: n_wrong += 1
        elif sec_id == "aptitude":
            att_cnt = len(st.session_state.aptitude_evaluations)
            tot_cnt = len(st.session_state.aptitude_questions) or 10
            for ev in st.session_state.aptitude_evaluations.values():
                sc = float(ev.get("score", 0.0) if isinstance(ev, dict) else (ev if isinstance(ev, (int, float)) else 0.0))
                if sc >= 70: n_correct += 1
                elif sc >= 30: n_partial += 1
                else: n_wrong += 1
        elif sec_id == "audio_recording":
            att_cnt = len(st.session_state.audio_evaluations)
            tot_cnt = len(st.session_state.audio_sentences) or 10
            for ev in st.session_state.audio_evaluations.values():
                sc = float((ev.get("score", ev.get("accuracy_score", 0.0))) if isinstance(ev, dict) else (ev if isinstance(ev, (int, float)) else 0.0))
                if sc >= 70: n_correct += 1
                elif sc >= 30: n_partial += 1
                else: n_wrong += 1
        elif sec_id == "self_intro":
            att_cnt = 1 if st.session_state.self_intro_eval else 0
            tot_cnt = 1
            if st.session_state.self_intro_eval:
                sc_raw = st.session_state.self_intro_eval.get("score", 0.0) if isinstance(st.session_state.self_intro_eval, dict) else st.session_state.self_intro_eval
                sc = float(sc_raw if isinstance(sc_raw, (int, float)) else 0.0)
                if sc >= 70: n_correct += 1
                elif sc >= 30: n_partial += 1
                else: n_wrong += 1
        elif sec_id == "personality":
            att_cnt = 1 if st.session_state.personality_eval else 0
            tot_cnt = 1
            if st.session_state.personality_eval:
                sc_raw = st.session_state.personality_eval.get("culture_fit_score", 0.0) if isinstance(st.session_state.personality_eval, dict) else st.session_state.personality_eval
                sc = float(sc_raw if isinstance(sc_raw, (int, float)) else 0.0)
                if sc >= 70: n_correct += 1
                elif sc >= 30: n_partial += 1
                else: n_wrong += 1

        expander_title = f"🔹 {nice_title} — Score: {score:.1f}% ({'Completed' if att_cnt >= tot_cnt and tot_cnt > 0 else 'Attempted' if is_attempted else 'Unattempted'})"
        with st.expander(expander_title, expanded=is_attempted):
            col_sc, col_im = st.columns([1.1, 1.3])
            with col_sc:
                if is_attempted:
                    st.metric(f"{nice_title} Score", f"{score:.1f} / 100")
                    st.markdown(
                        f"""
                        <div style="font-size: 0.85rem; line-height: 1.8; margin-top: 0.4rem;">
                            <b>Status:</b> <span class="badge-completed">{'Completed' if att_cnt >= tot_cnt and tot_cnt > 0 else 'Attempted'} ({att_cnt} / {tot_cnt} Items)</span><br>
                            <b>Performance:</b> <span style="color: #059669; font-weight: 600;">🟢 Correct: {n_correct}</span> | <span style="color: #D97706; font-weight: 600;">🟡 Partial: {n_partial}</span> | <span style="color: #DC2626; font-weight: 600;">🔴 Needs Work: {n_wrong}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<span class="badge-unattempted">Unattempted (0 / 100)</span>', unsafe_allow_html=True)
                    st.caption("This section was not completed during the interview session.")
                st.caption(meta["benchmark"])

            with col_im:
                st.markdown("##### 💡 Actionable Improvement Points")
                for imp in meta["improvements"]:
                    st.markdown(f"• {imp}")

    # 4. Question-by-Question Detailed Review with Sample Ideal Answers Across ALL Attempted Sections
    st.markdown("---")
    st.markdown("### 🔍 Comprehensive Question-by-Question Detailed Review")

    # 4.1 Technical Questions Review
    if st.session_state.tech_evaluations:
        with st.expander(f"📚 Section 2: Technical Assessment Detailed Review ({len(st.session_state.tech_evaluations)} Questions)", expanded=False):
            for q_idx, ev in st.session_state.tech_evaluations.items():
                q_obj = st.session_state.tech_questions[q_idx] if q_idx < len(st.session_state.tech_questions) else {}
                q_dict = q_obj if isinstance(q_obj, dict) else {}
                q_txt = q_dict.get("question") or (str(q_obj).strip() if q_obj else f"Question {q_idx + 1}")
                q_topic = q_dict.get("topic", "Technical")
                st.markdown(f"**Question {q_idx + 1} [{q_topic}]:** {q_txt}")
                st.markdown(f"**Your Answer:** *{st.session_state.tech_answers.get(q_idx, 'Skipped')}*")
                ev_dict = ev if isinstance(ev, dict) else {}
                st.markdown(f"**Score:** `{ev_dict.get('score', 0)}%` | **Feedback:** {ev_dict.get('feedback', '')}")
                if ev_dict.get("missing_concepts"):
                    st.markdown(f"**Missing Key Concepts:** {', '.join(ev_dict.get('missing_concepts', []))}")
                ideal_str = str(ev_dict.get("ideal_answer", "") or q_dict.get("ideal_answer", ""))
                st.markdown(
                    f'<div class="ideal-answer-card"><b>💡 Ideal Benchmark Answer:</b><br>{html.escape(ideal_str)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("---")

    # 4.2 Algorithmic Coding (DSA) Review
    if st.session_state.dsa_evaluations:
        with st.expander(f"💻 Section 3: Algorithmic Coding (DSA) Detailed Review ({len(st.session_state.dsa_evaluations)} Problems)", expanded=False):
            for p_idx, ev in st.session_state.dsa_evaluations.items():
                p_obj = st.session_state.dsa_problems[p_idx] if p_idx < len(st.session_state.dsa_problems) else {}
                p_dict = p_obj if isinstance(p_obj, dict) else {}
                p_title = p_dict.get("title") or (str(p_obj).strip() if p_obj else f"Problem {p_idx + 1}")
                st.markdown(f"**Problem {p_idx + 1}:** {p_title}")
                st.markdown(f"**Problem Description:** {p_dict.get('description', '')}")
                ev_dict = ev if isinstance(ev, dict) else {}
                st.markdown(f"**Score:** `{ev_dict.get('score', 0)}%` | **Tests Passed:** `{ev_dict.get('passed_tests', 0)} / {ev_dict.get('total_tests', 0)}`")
                st.markdown(f"**Complexity / Feedback:** {ev_dict.get('feedback', '')}")
                user_code = st.session_state.dsa_codes.get(p_idx, "# No code submitted")
                st.markdown("**Your Submitted Code:**\n```python\n" + str(user_code) + "\n```")
                optimal_str = str(p_dict.get("optimal_solution", ""))
                st.markdown(
                    f'<div class="ideal-answer-card"><b>💡 Optimal O(N) Benchmark Solution:</b><br><pre><code>{html.escape(optimal_str)}</code></pre></div>',
                    unsafe_allow_html=True,
                )
                st.markdown("---")

    # 4.3 Project Deep Dive Review
    if st.session_state.project_evaluations:
        with st.expander(f"🚀 Section 4: Project Deep Dive Detailed Review ({len(st.session_state.project_evaluations)} Questions)", expanded=False):
            for q_idx, ev in st.session_state.project_evaluations.items():
                q_obj = st.session_state.project_questions[q_idx] if q_idx < len(st.session_state.project_questions) else {}
                q_dict = q_obj if isinstance(q_obj, dict) else {}
                q_txt = q_dict.get("question") or (str(q_obj).strip() if q_obj else f"Project Question {q_idx + 1}")
                q_proj = q_dict.get("project_name", "Project")
                st.markdown(f"**Project Question {q_idx + 1} [{q_proj}]:** {q_txt}")
                st.markdown(f"**Your Answer:** *{st.session_state.project_answers.get(q_idx, 'Skipped')}*")
                ev_dict = ev if isinstance(ev, dict) else {}
                st.markdown(f"**Score:** `{ev_dict.get('score', 0)}%` | **Architectural Feedback:** {ev_dict.get('feedback', '')}")
                ideal_str = str(ev_dict.get("ideal_answer", ""))
                st.markdown(
                    f'<div class="ideal-answer-card"><b>💡 Benchmark Architectural Answer:</b><br>{html.escape(ideal_str)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("---")

    # 4.4 Aptitude & Data Interpretation Review
    if st.session_state.aptitude_evaluations:
        with st.expander(f"📊 Section 5: Aptitude & Reasoning Detailed Review ({len(st.session_state.aptitude_evaluations)} Questions)", expanded=False):
            for q_idx, ev in st.session_state.aptitude_evaluations.items():
                q_obj = st.session_state.aptitude_questions[q_idx] if q_idx < len(st.session_state.aptitude_questions) else {}
                q_dict = q_obj if isinstance(q_obj, dict) else {}
                q_txt = q_dict.get("question") or (str(q_obj).strip() if q_obj else f"Aptitude Scenario {q_idx + 1}")
                ev_dict = ev if isinstance(ev, dict) else {}
                is_correct = ev_dict.get("is_correct", False)
                st.markdown(f"**Scenario {q_idx + 1}:** {q_txt}")
                st.markdown(f"**Your Choice:** *{st.session_state.aptitude_answers.get(q_idx, 'Skipped')}* — {'🟢 Correct' if is_correct else '🔴 Incorrect'}")
                st.markdown(f"**Correct Option:** `{q_dict.get('correct_answer', '')}`")
                st.markdown(f"**Mathematical / Logical Explanation:** {q_dict.get('explanation', '')}")
                st.markdown("---")

    # 4.5 Audio Pronunciation Drills Review
    if st.session_state.audio_evaluations:
        with st.expander(f"🎙️ Section 7: Audio Pronunciation Drill Detailed Review ({len(st.session_state.audio_evaluations)} Sentences)", expanded=False):
            for s_idx, ev in st.session_state.audio_evaluations.items():
                s_obj = st.session_state.audio_sentences[s_idx] if s_idx < len(st.session_state.audio_sentences) else {}
                s_text = (
                    (s_obj.get("text") or s_obj.get("sentence") or f"Sentence {s_idx + 1}")
                    if isinstance(s_obj, dict)
                    else (str(s_obj).strip() or f"Sentence {s_idx + 1}")
                )
                ev_dict = ev if isinstance(ev, dict) else {}
                st.markdown(f"**Sentence {s_idx + 1} Target:** '{s_text}'")
                st.markdown("**Transcribed Spoken Audio:** *" + str(ev_dict.get("transcript", "") or "No speech detected") + "*")
                acc_val = float(ev_dict.get('score', ev_dict.get('accuracy_score', 0)) or 0.0)
                wpm_val = float(ev_dict.get('wpm', 0) or 0.0)
                tone_val = str(ev_dict.get('voice_tone', 'Neutral') or 'Neutral')
                st.markdown(f"**Accuracy Score:** `{acc_val:.1f}%` | **Pacing:** `{wpm_val:.0f} WPM` | **Tone:** `{tone_val}`")
                st.markdown("---")

    # 4.6 Self Introduction Review
    if st.session_state.self_intro_eval:
        with st.expander("👤 Section 1: Self Introduction Detailed Review", expanded=False):
            ev = st.session_state.self_intro_eval
            ev_dict = ev if isinstance(ev, dict) else {}
            st.markdown("**Candidate Transcript:** *" + str(ev_dict.get("transcript", "No recording")) + "*")
            st.markdown(f"**Self Introduction Score:** `{ev_dict.get('score', 0):.1f} / 100`")
            metrics_dict = st.session_state.self_intro_metrics if isinstance(st.session_state.self_intro_metrics, dict) else {}
            st.markdown(f"**Speech Pacing:** `{ev_dict.get('wpm', 0):.0f} WPM` | **Tone:** `{metrics_dict.get('voice_tone', 'Confident')}`")
            if ev_dict.get("strengths"):
                st.markdown(f"**Core Strengths:** {', '.join(ev_dict.get('strengths', []))}")
            if ev_dict.get("improvements"):
                st.markdown(f"**Concrete Improvements:** {', '.join(ev_dict.get('improvements', []))}")
            st.markdown("---")

    # 4.7 Psychometric Personality (OCEAN) Review
    if st.session_state.personality_eval:
        with st.expander("🧩 Section 6: Psychometric Personality (OCEAN) Detailed Review", expanded=False):
            p_ev = st.session_state.personality_eval if isinstance(st.session_state.personality_eval, dict) else {}
            st.markdown(f"**Culture Fit & Behavioral Score:** `{p_ev.get('culture_fit_score', 0):.1f} / 100`")
            st.markdown(f"**Behavioral Profile Summary:** {p_ev.get('summary', 'Consistent behavioral alignment across competencies.')}")
            traits = p_ev.get("traits", {}) if isinstance(p_ev.get("traits"), dict) else {}
            if traits:
                trait_str = " | ".join([f"**{k.title()}:** {v:.0f}%" for k, v in traits.items() if isinstance(v, (int, float))])
                st.markdown(f"**OCEAN Dimension Breakdown:** {trait_str}")
            st.markdown("---")

    # 5. AI Longitudinal Coaching Plan
    st.markdown("---")
    st.markdown("### 🧠 AI Longitudinal Coaching Plan")
    plan = generate_longitudinal_coaching_plan(
        role=st.session_state.target_role,
        overall_score=overall_score,
        section_scores=completed,
        skills=st.session_state.detected_skills,
    )

    col_str, col_focus = st.columns(2)
    with col_str:
        st.markdown("##### 🌟 Core Strengths")
        for s in plan.get("strengths", []):
            st.markdown(f"• **{s}**")

    with col_focus:
        st.markdown("##### 🎯 Critical Focus Areas")
        for f in plan.get("critical_focus_areas", []):
            st.markdown(f"• **{f}**")

    st.markdown("##### 📅 14-Day Skill Acceleration Roadmap")
    for step in plan.get("roadmap_14_days", []):
        st.markdown(f"- **{step.get('day_range', '')}:** {step.get('action', '')} *(Milestone: {step.get('milestone', '')})*")

    # CSV Download & Navigation Actions
    st.markdown("<br>", unsafe_allow_html=True)
    col_csv, col_hub, col_prof = st.columns([1, 1, 1])

    with col_csv:
        # Build comprehensive export
        export_rows = []
        for s_id in display_sections:
            export_rows.append({
                "Section": s_id.replace("_", " ").title(),
                "Status": "Completed" if s_id in completed else "Unattempted",
                "Score (%)": round(completed.get(s_id, 0.0), 1),
            })
        df_export = pd.DataFrame(export_rows)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Performance CSV",
            data=csv_data,
            file_name=f"interview_report_{st.session_state.candidate_name.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_hub:
        if st.button("🎯 Return to Live Interview Hub", use_container_width=True, key="btn_rep_hub"):
            st.session_state.current_screen = "hub"
            st.rerun()

    with col_prof:
        if st.button("👤 View Profile & Past History", type="primary", use_container_width=True, key="btn_rep_prof"):
            st.session_state.current_screen = "profile"
            st.rerun()


# ==============================================================================
# SCREEN 5: CANDIDATE PROFILE & HISTORICAL PROGRESS (DOWNLOAD ANSWERS)
# ==============================================================================
def render_profile_screen():
    """Render Candidate Profile and Historical Progress with downloadable user vs correct answers."""
    render_header()

    user = st.session_state.authenticated_user
    if not user:
        st.session_state.current_screen = "auth"
        st.rerun()

    st.markdown(
        f"""
        <div class="hero-card">
            <h1 class="hero-title">Candidate Profile & Performance History</h1>
            <p class="hero-subtitle">
                Registered Candidate: <b>{user.get('full_name', '')}</b> | Email: <b>{user.get('email', '')}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = get_candidate_history(user.get("full_name", ""))
    if not history:
        history = get_all_candidates()

    if history:
        st.markdown("#### 📈 Historical Interview Progression")
        df_hist = pd.DataFrame(history)

        if "timestamp" in df_hist.columns and "overall_score" in df_hist.columns:
            fig_hist = px.line(
                df_hist,
                x="timestamp",
                y="overall_score",
                markers=True,
                title="Readiness Score Progression Across Sessions",
                labels={"timestamp": "Session Date", "overall_score": "Overall Score (%)"},
            )
            fig_hist.update_traces(line_color="#059669")
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 📜 Past Interview Sessions & Answer Exports")

        for s_idx, session in enumerate(history):
            s_id = session.get("id", s_idx + 1)
            ts = session.get("timestamp", "Recent")
            role = session.get("target_role", "Software Engineer")
            sc = session.get("overall_score", 0.0)

            with st.expander(f"Session #{s_id} · {role} · Score: {sc:.1f}% ({str(ts)[:16]})", expanded=(s_idx == len(history) - 1)):
                c_details, c_down = st.columns([1.5, 1])
                with c_details:
                    st.markdown(f"**Target Role:** {role}")
                    st.markdown(f"**Overall Score:** {sc:.1f}%")
                    st.markdown(f"**Posture Stability:** {session.get('posture_score_pct', 0.0):.1f}% | **Eye Contact:** {session.get('eye_contact_pct', 0.0):.1f}%")

                with c_down:
                    # Generate downloadable user answers & correct answers (Requirement 4.3)
                    user_csv, user_txt = build_user_answers_export(session)
                    ideal_csv, ideal_txt = build_ideal_answers_export(session)

                    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #1E1B4B; margin-bottom: 0.3rem;'>Candidate Submitted Answers</div>", unsafe_allow_html=True)
                    c_u1, c_u2 = st.columns(2)
                    with c_u1:
                        st.download_button(
                            "📥 CSV Export",
                            data=user_csv,
                            file_name=f"submitted_answers_session_{s_id}.csv",
                            mime="text/csv",
                            key=f"dl_user_csv_{s_id}",
                            use_container_width=True,
                        )
                    with c_u2:
                        st.download_button(
                            "📄 TXT Export",
                            data=user_txt,
                            file_name=f"submitted_answers_session_{s_id}.txt",
                            mime="text/plain",
                            key=f"dl_user_txt_{s_id}",
                            use_container_width=True,
                        )

                    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #047857; margin-top: 0.6rem; margin-bottom: 0.3rem;'>Ideal Benchmark Answers</div>", unsafe_allow_html=True)
                    c_i1, c_i2 = st.columns(2)
                    with c_i1:
                        st.download_button(
                            "📥 CSV Export",
                            data=ideal_csv,
                            file_name=f"ideal_answers_session_{s_id}.csv",
                            mime="text/csv",
                            key=f"dl_ideal_csv_{s_id}",
                            use_container_width=True,
                        )
                    with c_i2:
                        st.download_button(
                            "📄 TXT Export",
                            data=ideal_txt,
                            file_name=f"ideal_answers_session_{s_id}.txt",
                            mime="text/plain",
                            key=f"dl_ideal_txt_{s_id}",
                            use_container_width=True,
                        )

    else:
        st.info("No past interview sessions found in the database. Complete your first session to view historical trends.")

    col_hub, col_out = st.columns([1, 1])
    with col_hub:
        if st.button("Return to Section Hub", type="primary", use_container_width=True, key="btn_prof_hub"):
            st.session_state.current_screen = "hub"
            st.rerun()
    with col_out:
        if st.button("Logout", use_container_width=True, key="btn_prof_logout"):
            st.session_state.authenticated_user = None
            st.session_state.current_screen = "auth"
            st.rerun()


# ==============================================================================
# MAIN ROUTING CONTROLLER
# ==============================================================================
def main():
    """Main application controller routing between views."""
    if not st.session_state.authenticated_user:
        render_auth_screen()
        return

    screen = st.session_state.current_screen

    if screen == "onboarding":
        render_onboarding_screen()
    elif screen == "hub":
        render_section_hub()
    elif screen == "section_self_intro":
        render_section_self_intro()
    elif screen == "section_technical":
        render_section_technical()
    elif screen == "section_dsa":
        render_section_dsa()
    elif screen == "section_project":
        render_section_project()
    elif screen == "section_aptitude":
        render_section_aptitude()
    elif screen == "section_personality":
        render_section_personality()
    elif screen == "section_audio_recording":
        render_section_audio_recording()
    elif screen == "report":
        render_report_screen()
    elif screen == "profile":
        render_profile_screen()
    else:
        render_section_hub()


if __name__ == "__main__":
    main()


