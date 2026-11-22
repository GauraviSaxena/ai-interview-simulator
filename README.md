# 🎓 AI INTERVIEW SIMULATOR

An enterprise-grade, multi-modal **AI Interview Simulator, Automated Coding Sandbox & Executive Delivery Coach** built with **Python**, **Streamlit**, **MediaPipe**, **Whisper**, **Librosa**, **gTTS**, **pyttsx3**, **rapidfuzz**, and **SQLAlchemy**, featuring a **Royal Indigo, Emerald & Warm Pearl (`#1E1B4B` / `#059669` / `#F8FAFC`)** visual design system with custom vector SVG hero illustrations.

The platform simulates realistic technical, algorithmic, and behavioral hiring loops by generating tailored assessments across **7 calibrated sections**, conducting live vision posture & eye-contact monitoring with real-time feedback banners, assessing vocal delivery with acoustic tone classification, evaluating live code against automated test cases, and maintaining persistent longitudinal performance records in SQLite.

---

## 🌟 Key Features & Architectural Capabilities 

### 1. Zero-Prompt Startup & Smooth Client-Side Countdown Timers
- **Zero Startup Prompts**: Configured with `gatherUsageStats = false` and `headless = true` in `.streamlit/config.toml` and `run.py` to completely eliminate first-run email prompts in any environment.
- **Client-Side HTML/JS Countdown Timer**: Countdown timers run purely in browser JavaScript (`streamlit.components.v1.html`), providing smooth live countdowns without Python page reruns, lag, or input freezing.
- **Strict Widget Scoping & No-Leak State**: Every input (`st.text_area`, `st.radio`, `st.audio_input`) uses a unique widget key (`f"{attempt_id}_{section}_q_{idx}"`), guaranteeing that previous answers never persist when navigating between questions.
- **Universal "Skip Question" Buttons**: Placed across every section (Technical, DSA, Project, Aptitude, Personality, Audio), recording `"Skipped"` with a `0` score and immediately advancing to the next question.

### 2. Dual-Mode Audio Recordings Drill (5 Read-Locked + 5 Listen-Only)
- **Questions 1 to 5 (Read & Lock)**: Candidate reads the sentence on screen. The moment recording starts or the candidate clicks `"Lock Sentence"`, the text is immediately hidden so they recite from memory.
- **Questions 6 to 10 (Listen-Only — No Text Shown)**: Strictly hides all written sentence text. Only provides a spoken reference audio player (max 2 plays allowed), which is automatically locked once recording begins.
- **Real Sound & Multi-Tier Synthesizer**: Generates clear spoken audio using high-fidelity `gTTS`, backed by automatic offline `pyttsx3` and `win32com` SAPI5 fallbacks, verifying audio byte length $> 1000$ bytes before playback.
- **Accurate Speech-to-Text Scoring**: Defensively processes `st.audio_input` WAV bytes, transcribes via local Whisper (with SpeechRecognition fallback), and evaluates pronunciation using token-level and fuzzy similarity (`rapidfuzz` / `difflib`), reliably awarding **80%–100%** for correct speech and **0%** for silent audio or gibberish.

### 3. Strict Content Grading & Dynamic Non-Repeating Questions
- **Zero Marks for Gibberish / Irrelevant Input**: Evaluates answers with strict rubric gates: keyboard smashing (`asdf`, `jvjg`), copy-pasting the question, or irrelevant off-topic content is awarded strictly **0 to 10 marks** (no artificial free baseline points).
- **Unattempted Sections = 0 / Unattempted**: Performance report explicitly marks unattempted sections with an `Unattempted (0 / 100)` badge and zero score contribution.
- **Fresh Questions Every Attempt**: Expands offline banks to 30+ items per category, shuffled with session-based seeds and nonces so every attempt produces unique questions.

### 4. Larger Camera Frame, Top Formula Banner & Downloadable Answers
- **Prominent 640x480 Live Camera Frame**: Large, clear live webcam preview container with royal indigo borders and MediaPipe FaceMesh (iris gaze) and Pose (shoulder tilt $\pm 5^\circ$) tracking overlays.
- **Top Formula Banner**: Exact mathematical readiness formula displayed at the top of the Performance Report dashboard.
- **Section-Wise Improvement Guide**: Detailed, bulleted improvements needed for every attempted section and overall.
- **Downloadable Historical Answers**: For every historical attempt in the Profile dashboard, provides:
  1. `"Download My Submitted Answers"` (both CSV and TXT formats) containing all questions, candidate answers, and scores.
  2. `"Download Ideal Correct Answers"` (both CSV and TXT formats) containing expected benchmark answers for self-study.
   - Clean, streamlined flow: candidate types their answer and clicks `"Next Question"`, automatically evaluating and advancing without separate manual evaluation steps.
   - Comprehensive AI grading storing score, feedback, improvement points, and **Sample Ideal Answers**.

3. **Algorithmic Coding Assessment (DSA) (2 Challenges · 75 Mins · Live Code Judge)**:
   - **CONDITIONAL**: Rendered exclusively for Developer / Software Engineering roles; automatically hidden for non-developer roles.
   - Configurable difficulty: **Easy**, **Medium**, **Hard**.
   - Completely agnostic problem definitions with structured mathematical constraint bullet points (e.g., $1 \le n \le 10^5$, $\mathcal{O}(n)$ time complexity, $\mathcal{O}(1)$ auxiliary space).
   - Language selector: **Python**, **C++**, **Java**, **JavaScript**.
   - Live code editor prefilled with starter function stubs.
   - **Automated Sandbox Judge**: Executes Python code against visible and hidden test cases, returning runtime, output, and pass/fail status.

4. **Project Deep Dive (5 Questions · 10 Mins · Text Only)**:
   - Configurable difficulty: **Easy**, **Medium**, **Hard**.
   - 5 architectural questions probing candidate projects detected from the resume.
   - Evaluates system trade-offs, bottlenecks, metrics, and technical contributions.

5. **Aptitude & Data Interpretation (10 MCQs · 30 Mins · Radio Buttons Only)**:
   - Configurable difficulty: **Easy**, **Medium**, **Hard**.
   - Numerical reasoning, probability, logic puzzles, and systems metrics scenarios (e.g. Little's Law, AMAT).
   - 4 options per question via `st.radio`.

6. **Personality Test (20 Likert Statements · Untimed · No Camera/Mic)**:
   - 20 psychometric statements evaluated on a 5-point Likert scale (1 = Strongly Disagree to 5 = Strongly Agree).
   - Derives Big Five personality traits (OCEAN: Openness, Conscientiousness, Extraversion, Agreeableness, Emotional Stability) and Culture Fit score (0–100).
   - Relaxed, untimed environment with camera and microphone disabled.

7. **Audio Recordings (10 Benchmark Sentences · Max 2 Listens)**:
   - Configurable difficulty: **Easy**, **Medium**, **Hard** (10 pre-loaded sentences per difficulty).
   - High-fidelity text-to-speech audio via `gTTS`.
   - **Strict Rule: Maximum 2 plays per sentence!** Listen counter disables playback once 2 plays are used.
   - Audio player is locked/hidden once recording begins to prevent simultaneous listening and recording.
   - Candidate repeats the sentence into the microphone (`st.audio_input`).
   - Whisper STT computes word & pronunciation accuracy (%).
   - Librosa analyzes vocal pitch variation, WPM, and classifies tone (**Confident & Expressive**, **Monotonous**, or **Low Energy / Hesitant**).

### 6. Executive Performance Analytics & AI Longitudinal Coach
- **Readiness Gauge**: Overall candidate score (0–100) with color-coded readiness zones.
- **Multi-Dimensional Radar Chart**: 7–8 dimensions covering Self Intro, Technical, DSA, Projects, Aptitude, Personality, Voice, and Posture.
- **Section-Wise Breakdown Cards**: Displays individual scores and concrete improvement points for every attempted section.
- **Summary Telemetry**: Overall score, posture stability %, eye contact %, posture errors count, speaking pace (WPM), dominant tone.
- **Question-by-Question Review**: Expandable cards displaying question prompts, candidate answers, missing concepts, and AI Ideal Answers.
- **14-Day AI Coaching Roadmap**: Personalized improvement plan with milestones.
- **CSV & Report Export**: One-click download of session summary.

---

## 📁 Project Structure

```text
ai_interview_simulator_refactored/
├── run.py                          # Clean application entry point launcher
├── requirements.txt                # Complete production dependencies
├── .gitignore                      # Git ignore rules for data, env, cache
├── README.md                       # Documentation & architecture guide
├── .streamlit/
│   ├── config.toml                 # Modern light theme (#2563EB accent)
│   └── secrets.toml.example        # Template for API keys & SMTP settings
├── data/
│   └── interview_history.db        # SQLite database (auto-generated)
├── backend/
│   ├── __init__.py
│   ├── database.py                 # SQLite + SQLAlchemy ORM database & User auth
│   ├── email_service.py            # SMTP OTP mailer with demo fallback
│   └── services.py                 # Central evaluation orchestrator
├── modules/
│   ├── __init__.py
│   ├── resume_parser.py            # PDF/DOCX text & skill/project discovery
│   ├── llm_evaluator.py            # 7-section question generators, code judge & AI grader
│   ├── vision_tracker.py           # MediaPipe FaceMesh & Pose posture coach
│   ├── speech_engine.py            # Whisper STT & Librosa acoustic analyzer
│   └── voice_drill.py              # Audio recording gTTS & pronunciation drill
└── frontend/
    ├── __init__.py
    └── app.py                      # Modern card-based Streamlit UI
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or 3.11 installed.

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate virtual environment (macOS/Linux)
# source venv/bin/activate

# Install required packages
python -m pip install -r requirements.txt
```

### 3. Configure Optional Secrets (Optional)
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
```toml
# Optional: LLM API Keys (Offline fallback is built-in!)
OPENAI_API_KEY = "sk-..."
GEMINI_API_KEY = "AIzaSy..."

# Optional: SMTP Email Server (for live OTP emails to @gmail.com)
# If unconfigured, the app runs in Demo Mode and displays the OTP directly on-screen.
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"
SMTP_FROM_EMAIL = "your_email@gmail.com"
```

### 4. Launch the Application
Run the launcher script:
```bash
python run.py
```
Or directly via Streamlit:
```bash
python -m streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔒 Security & Privacy Notes
- **Local Data Storage**: All interview sessions and user profiles are stored locally in `data/interview_history.db`.
- **Zero In-Memory Leakage**: Audio chunks and video frames are processed in-memory and cleared between questions.
- **Sandboxed Code Execution**: Python code solutions are validated inside a restricted `exec` environment with timeouts.
