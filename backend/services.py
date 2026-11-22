"""Central evaluation orchestrator integrating multi-section scoring, vision metrics, and database persistence."""
from __future__ import annotations

from typing import Any, Optional

from backend.database import save_session
from modules.llm_evaluator import evaluate_answer
from modules.speech_engine import pcm_to_wav_bytes, speech_metrics, transcribe


def evaluate_submission(
    question_data: dict[str, Any] | str,
    answer_text: str,
    captured_media: dict[str, Any],
    resume_text: str = "",
    jd_text: str = "",
) -> dict[str, Any]:
    """Orchestrate transcription, content grading, acoustic speech analysis, and body language metrics for an answer."""
    if isinstance(question_data, str):
        question_str = question_data
        section = "Technical"
        expected_concepts = []
    else:
        question_str = str(question_data.get("question", ""))
        section = str(question_data.get("section", "Technical"))
        expected_concepts = question_data.get("expected_concepts", [])

    transcript = (answer_text or "").strip()
    wav_bytes = pcm_to_wav_bytes(
        captured_media.get("audio_chunks", []),
        int(captured_media.get("sample_rate", 48000)),
    )
    audio_duration = float(captured_media.get("audio_duration", 0.0))
    transcription_mode = "Typed text response"

    # If answer was spoken and not manually typed, transcribe via Whisper
    if not transcript and wav_bytes:
        transcript, transcription_mode = transcribe(wav_bytes)

    transcript = transcript.strip()

    # 1. Evaluate content depth, accuracy, missing concepts, and ideal answer
    content_eval = evaluate_answer(
        question=question_str,
        answer=transcript,
        section=section,
        resume=resume_text,
        jd=jd_text,
        expected_concepts=expected_concepts,
    )

    # 2. Evaluate vocal delivery, pace, and Librosa acoustic tone
    effective_duration = audio_duration if audio_duration > 0.0 else max(10.0, len(transcript.split()) / 2.3)
    speech = speech_metrics(transcript, effective_duration, wav_bytes=wav_bytes)

    # 3. Vision and body language indicators
    eye = captured_media.get("eye_contact")
    posture = captured_media.get("posture")
    errors = int(captured_media.get("posture_errors", 0))

    valid_body_parts = [float(v) for v in (eye, posture) if isinstance(v, (int, float))]
    body_score = sum(valid_body_parts) / len(valid_body_parts) if valid_body_parts else 0.0

    # Composite question score: 55% Content + 25% Fluency + 20% Body Language
    overall_question_score = (
        0.55 * content_eval["score"] +
        0.25 * speech["fluency"] +
        0.20 * body_score
    )
    overall_question_score = round(max(0.0, min(100.0, overall_question_score)), 1)

    return {
        "question": question_str,
        "section": section,
        "answer": transcript,
        "content_score": content_eval["score"],
        "missing_key_concepts": content_eval.get("missing_key_concepts", []),
        "feedback": content_eval.get("improvement_feedback", []),
        "ideal_sample_answer": content_eval.get("ideal_sample_answer", ""),
        "scoring_mode": content_eval.get("mode", "Offline Rubric"),
        "transcription_mode": transcription_mode,
        "wpm": speech["wpm"],
        "word_count": speech["word_count"],
        "fillers": speech["fillers"],
        "fluency": speech["fluency"],
        "voice_tone": speech.get("voice_tone", "Confident & Expressive"),
        "pitch_std_hz": speech.get("pitch_std_hz", 22.0),
        "mean_rms_energy": speech.get("mean_rms_energy", 0.04),
        "tone_feedback": speech.get("tone_feedback", ""),
        "eye_contact": float(eye) if eye is not None else 0.0,
        "posture": float(posture) if posture is not None else 0.0,
        "posture_errors": errors,
        "body_score": round(body_score, 1),
        "overall": overall_question_score,
    }


def generate_model_self_introduction(
    candidate_name: str,
    target_role: str,
    resume_text: str = "",
    detected_skills: list[str] | None = None,
    detected_projects: list[str] | None = None,
) -> str:
    """Generate a complete, professional 60-90 second model executive self-introduction
    tailored specifically to the candidate's name, target role, education, skills, and flagship projects."""
    name = (candidate_name or "Alex Morgan").strip()
    role = (target_role or "Software Engineer").strip()

    skills = [s for s in (detected_skills or []) if s and len(s) > 1]
    if not skills and resume_text:
        try:
            from modules.resume_parser import detect_skills
            skills = detect_skills(resume_text)
        except Exception:
            pass
    if not skills:
        skills = ["Python", "System Architecture", "Cloud Infrastructure", "Relational Databases", "RESTful APIs"]

    top_skills = skills[:4]
    skills_phrase = ", ".join(top_skills[:-1]) + f", and {top_skills[-1]}" if len(top_skills) > 1 else top_skills[0]

    projects = [p for p in (detected_projects or []) if p and len(p) > 2]
    if not projects and resume_text:
        try:
            from modules.resume_parser import detect_projects
            projects = detect_projects(resume_text)
        except Exception:
            pass
    if not projects:
        projects = ["High-Throughput Distributed Microservices Platform", "Real-Time Telemetry & Data Processing Pipeline"]

    proj1 = projects[0] if len(projects) > 0 else "Scalable Distributed System"
    proj2 = projects[1] if len(projects) > 1 else "End-to-End Automated Analytics Pipeline"

    # Education extraction
    edu_mention = "a strong Computer Science and Engineering foundation"
    if resume_text:
        for keyword in ["B.Tech", "Bachelor", "Master", "M.Tech", "M.S.", "B.S.", "Computer Science", "Information Technology", "Software Engineering"]:
            if keyword.lower() in resume_text.lower():
                edu_mention = "a rigorous academic background in Computer Science & Engineering"
                break

    model_intro = (
        f"\"Good morning, and thank you for this opportunity. My name is {name}, and I am a software engineer "
        f"targeting the {role} position.\n\n"
        f"With {edu_mention}, my technical toolkit is anchored in {skills_phrase}. "
        f"I specialize in architecting scalable systems, designing robust backend services, and optimizing latency and throughput.\n\n"
        f"Throughout my engineering journey, I have prioritized shipping high-impact, reliable software. For example, in my flagship project, "
        f"'{proj1}', I led the technical design and deployment of the core infrastructure, ensuring high availability, sub-second query latency, "
        f"and fault-tolerant data pipelines under heavy traffic.\n\n"
        f"In another significant project, '{proj2}', I spearheaded the implementation of automated canary deployments, telemetry monitoring, "
        f"and distributed caching, which reduced operational latency by over 30% and dramatically minimized incident resolution time.\n\n"
        f"What excites me most about this {role} role at your team is the alignment between the technical challenges you are tackling and my experience "
        f"in building dependable, performant systems. I am eager to apply my expertise in {top_skills[0]} and {top_skills[1] if len(top_skills) > 1 else 'modern software design'} "
        f"to contribute meaningfully to your product roadmap from day one.\""
    )
    return model_intro


def finalize_and_save_interview(
    candidate_name: str,
    target_role: str,
    user_email: str = "",
    section_results: dict[str, Any] | None = None,
    media_metrics: dict[str, Any] | None = None,
    resume_text: str = "",
    detected_skills: list[str] | None = None,
    detected_projects: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate aggregate section scores, radar chart dimensions, and persist to SQLite database."""
    section_results = section_results or {}
    media_metrics = media_metrics or {}

    def _extract_score(sec_key: str, alt_key: str | None = None) -> float | None:
        data = section_results.get(sec_key)
        if data is None and alt_key:
            data = section_results.get(alt_key)
        if isinstance(data, dict):
            val = data.get("score")
            return float(val) if val is not None else None
        if isinstance(data, (int, float)):
            return float(data)
        return None

    self_intro_score = _extract_score("self_intro")
    technical_score = _extract_score("technical")
    dsa_score = _extract_score("dsa")
    project_score = _extract_score("project")
    aptitude_score = _extract_score("aptitude")
    personality_score = _extract_score("personality")
    audio_rec_score = _extract_score("audio_recording", "audio_recordings")

    # Aggregate overall score from attempted sections
    weights = {
        "self_intro": 0.10,
        "technical": 0.25,
        "dsa": 0.20,
        "project": 0.15,
        "aptitude": 0.10,
        "personality": 0.10,
        "audio_recordings": 0.10,
    }

    attempted_weights = []
    attempted_scores = []

    pairs = [
        ("self_intro", self_intro_score),
        ("technical", technical_score),
        ("dsa", dsa_score),
        ("project", project_score),
        ("aptitude", aptitude_score),
        ("personality", personality_score),
        ("audio_recordings", audio_rec_score),
    ]

    for sec_name, sc in pairs:
        if sc is not None:
            attempted_scores.append(float(sc) * weights[sec_name])
            attempted_weights.append(weights[sec_name])

    if attempted_weights:
        overall_score = round(sum(attempted_scores) / sum(attempted_weights), 1)
    else:
        overall_score = 0.0

    eye_pct = float(media_metrics.get("eye_contact", 0.0))
    posture_pct = float(media_metrics.get("posture", 0.0))
    posture_errors = int(media_metrics.get("posture_errors", 0))
    wpm = float(media_metrics.get("wpm", 0.0))
    fillers = int(media_metrics.get("fillers", 0)) if isinstance(media_metrics.get("fillers"), (int, float)) else 0
    voice_tone = str(media_metrics.get("voice_tone", "Silent / No Speech Detected" if eye_pct == 0.0 else "Confident & Expressive"))

    summary_data = {
        "user_email": user_email.strip().lower(),
        "candidate_name": candidate_name.strip() or "Anonymous Candidate",
        "target_role": target_role.strip() or "Software Engineer",
        "overall_score": overall_score,
        "self_intro_score": self_intro_score,
        "technical_score": technical_score,
        "dsa_score": dsa_score,
        "project_score": project_score,
        "aptitude_score": aptitude_score,
        "personality_score": personality_score,
        "audio_recording_score": audio_rec_score,
        "resume_score": float(project_score) if project_score is not None else 0.0,
        "voice_drill_score": float(audio_rec_score) if audio_rec_score is not None else 0.0,
        "eye_contact_pct": eye_pct,
        "posture_score_pct": posture_pct,
        "total_posture_errors": posture_errors,
        "wpm": wpm,
        "filler_words_total": fillers,
        "voice_tone": voice_tone,
        "details": {
            "section_results": section_results,
            "media_metrics": media_metrics,
            "resume_text": resume_text,
            "detected_skills": detected_skills or [],
            "detected_projects": detected_projects or [],
        },
    }

    # Save to SQLite database
    session_id = save_session(summary_data)
    summary_data["session_id"] = session_id
    return summary_data



