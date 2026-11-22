"""Specialized Voice, Tone, and Audio Recall drill module using gTTS, Whisper, and Librosa."""
from __future__ import annotations

import difflib
import io
import re
from typing import Any, Optional

from modules.speech_engine import analyze_audio_tone, speech_metrics, transcribe

ROUND1_BENCHMARK_SENTENCES = [
    "Our cloud architecture decouples data ingestion from real-time analytics to guarantee fault tolerance and sub-second query latency.",
    "We systematically mitigated production downtime by implementing automated canary deployments and end-to-end distributed telemetry.",
    "The engineering team refactored the persistent caching tier to handle fifty thousand concurrent requests during peak product launches.",
]

ROUND2_BENCHMARK_SENTENCES = [
    "Leadership requires aligning cross-functional teams around shared measurable outcomes while proactively managing technical trade-offs.",
    "To maintain system throughput under sudden load spikes, we partitioned database queries across isolated read replicas.",
    "Clear communication transforms complex architectural constraints into actionable roadmaps that non-technical stakeholders readily endorse.",
]


def _clean_sentence_str(val: Any) -> str:
    """Extract clean string if val is dict or object."""
    if isinstance(val, dict):
        return str(val.get("sentence") or val.get("text") or val.get("content") or "").strip()
    return str(val or "").strip()


def generate_sentence_tts(text: Any) -> bytes:
    """Generate in-memory real spoken English audio (MP3/WAV) using gTTS with offline pyttsx3/SAPI fallback."""
    clean_text = _clean_sentence_str(text)
    if not clean_text:
        return b""

    # 1. Primary: High-fidelity gTTS (Google Text-to-Speech)
    try:
        from gtts import gTTS
        tts = gTTS(text=clean_text, lang="en", slow=False)
        bio = io.BytesIO()
        tts.write_to_fp(bio)
        val = bio.getvalue()
        if len(val) > 1000:
            return val
    except Exception:
        pass

    # 2. Offline Fallback 1: pyttsx3 local system synthesizer
    try:
        import os
        import tempfile
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name
        engine.save_to_file(clean_text, tmp_wav)
        engine.runAndWait()
        if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 1000:
            with open(tmp_wav, "rb") as f:
                val = f.read()
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass
            return val
    except Exception:
        pass

    # 3. Offline Fallback 2: Windows native SAPI (win32com)
    try:
        import os
        import tempfile
        import win32com.client

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(tmp_wav, 3, False)
        speaker.AudioOutputStream = stream
        speaker.Speak(clean_text)
        stream.Close()
        if os.path.exists(tmp_wav) and os.path.getsize(tmp_wav) > 1000:
            with open(tmp_wav, "rb") as f:
                val = f.read()
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass
            return val
    except Exception:
        pass

    return b""


def calculate_word_accuracy(target_text: Any, spoken_text: Any) -> float:
    """Calculate normalized word-level pronunciation and recall accuracy percentage using token + fuzzy similarity."""
    def normalize_str(s: Any) -> str:
        raw = _clean_sentence_str(s)
        return re.sub(r"[^\w\s]", "", raw.lower()).strip()

    t_clean = normalize_str(target_text)
    s_clean = normalize_str(spoken_text)

    if not t_clean:
        return 100.0 if not s_clean else 50.0
    if not s_clean:
        return 0.0

    t_words = t_clean.split()
    s_words = s_clean.split()

    if not t_words:
        return 0.0

    # 1. RapidFuzz token matching
    try:
        from rapidfuzz import fuzz
        token_set_ratio = float(fuzz.token_set_ratio(t_clean, s_clean))
        token_sort_ratio = float(fuzz.token_sort_ratio(t_clean, s_clean))

        # Word-level fuzzy recall
        matched_words = 0
        for tw in t_words:
            best_match = max([fuzz.ratio(tw, sw) for sw in s_words], default=0)
            if best_match >= 75:
                matched_words += 1
        word_recall = (matched_words / len(t_words)) * 100.0

        # Weighted composite score
        score = 0.40 * token_set_ratio + 0.30 * token_sort_ratio + 0.30 * word_recall
        return round(float(max(0.0, min(100.0, score))), 1)

    except ImportError:
        # Fallback to standard difflib sequence matching
        matcher = difflib.SequenceMatcher(None, t_words, s_words)
        char_matcher = difflib.SequenceMatcher(None, t_clean, s_clean)
        ratio = (matcher.ratio() * 0.5 + char_matcher.ratio() * 0.5) * 100.0
        return round(float(max(0.0, min(100.0, ratio))), 1)


def evaluate_voice_drill(
    target_sentence: Any,
    wav_bytes: bytes,
    duration_seconds: float = 0.0,
    round_name: str = "Round 1 — Read Once & Speak",
) -> dict[str, Any]:
    """Transcribe user audio, assess word recall accuracy, analyze Librosa tone, and grade delivery."""
    clean_target = _clean_sentence_str(target_sentence)

    # 1. Transcribe spoken audio
    transcript, mode = transcribe(wav_bytes)

    if not transcript or not transcript.strip():
        return {
            "round_name": round_name,
            "target_sentence": clean_target,
            "transcript": "",
            "word_accuracy": 0.0,
            "accuracy_score": 0.0,
            "wpm": 0.0,
            "fillers": 0,
            "filler_count": 0,
            "fluency": 0.0,
            "voice_tone": "Silent / No Speech Detected",
            "pitch_std_hz": 0.0,
            "mean_rms_energy": 0.0,
            "drill_score": 0.0,
            "score": 0.0,
            "feedback": ["No speech detected in recording. Please speak clearly into the microphone."],
            "transcription_mode": mode,
        }
    
    # Estimate duration if not provided
    if duration_seconds <= 0.0 and wav_bytes:
        try:
            import soundfile as sf
            with io.BytesIO(wav_bytes) as bio:
                info = sf.info(bio)
                duration_seconds = float(info.duration)
        except Exception:
            duration_seconds = max(3.0, len(transcript.split()) / 2.2)

    # 2. Word accuracy
    word_accuracy = calculate_word_accuracy(clean_target, transcript)

    # 3. Speech metrics (WPM, Fillers, Fluency)
    speech = speech_metrics(transcript, duration_seconds, wav_bytes=wav_bytes)

    # 4. Acoustic tone analysis via Librosa
    tone_data = analyze_audio_tone(wav_bytes)

    # 5. Tone score bonus
    tone_label = tone_data.get("voice_tone", "Confident & Expressive")
    if tone_label == "Confident & Expressive":
        tone_score = 95.0
    elif tone_label == "Monotonous":
        tone_score = 65.0
    else:
        tone_score = 55.0

    # Composite drill score
    composite = (
        0.45 * word_accuracy +
        0.35 * speech["fluency"] +
        0.20 * tone_score
    )
    composite = round(max(10.0, min(100.0, composite)), 1)

    feedback_items = []
    if word_accuracy < 75.0:
        feedback_items.append(f"Recall accuracy was {word_accuracy:.0f}%. Focus on articulating each phrase cleanly without omitting technical keywords.")
    else:
        feedback_items.append(f"Excellent recall and pronunciation ({word_accuracy:.0f}% accuracy).")

    if speech["wpm"] < 110:
        feedback_items.append("Your speaking pace was slightly deliberate. Aim for a confident 130–150 WPM conversational speed.")
    elif speech["wpm"] > 165:
        feedback_items.append("You spoke rapidly. Introduce deliberate pauses between major clauses for executive presence.")

    if tone_data.get("tone_feedback"):
        feedback_items.append(tone_data["tone_feedback"])

    return {
        "round_name": round_name,
        "target_sentence": clean_target,
        "transcript": transcript,
        "word_accuracy": word_accuracy,
        "accuracy_score": word_accuracy,  # Alias for callers
        "wpm": speech["wpm"],
        "fillers": int(speech["fillers"]) if isinstance(speech.get("fillers"), (int, float)) else 0,
        "filler_count": int(speech["fillers"]) if isinstance(speech.get("fillers"), (int, float)) else 0,  # Alias for callers
        "fluency": speech["fluency"],
        "voice_tone": tone_label,
        "pitch_std_hz": tone_data.get("pitch_std_hz", 22.0),
        "mean_rms_energy": tone_data.get("mean_rms_energy", 0.04),
        "drill_score": composite,
        "score": composite,  # Alias
        "feedback": feedback_items,
        "transcription_mode": mode,
    }
