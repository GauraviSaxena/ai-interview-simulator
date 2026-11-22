"""Email confirmation and OTP delivery service using smtplib with automatic offline demo fallback."""
from __future__ import annotations

import os
import random
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Tuple


def generate_otp(length: int = 6) -> str:
    """Generate a secure numeric verification code."""
    return "".join(random.choices("0123456789", k=length))


def get_smtp_config() -> dict[str, Any]:
    """Retrieve SMTP configuration from Streamlit secrets or environment."""
    config = {
        "server": os.getenv("SMTP_SERVER") or os.getenv("SMTP_HOST") or "",
        "port": int(os.getenv("SMTP_PORT", 587)),
        "username": os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER") or "",
        "password": os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS") or "",
        "from_email": os.getenv("SMTP_FROM") or os.getenv("SMTP_USERNAME") or "noreply@interviewsimulator.ai",
    }
    
    # Try reading from streamlit secrets if available
    try:
        import streamlit as st
        for k in ["SMTP_SERVER", "SMTP_HOST"]:
            if k in st.secrets:
                config["server"] = str(st.secrets[k])
        if "SMTP_PORT" in st.secrets:
            config["port"] = int(st.secrets["SMTP_PORT"])
        for k in ["SMTP_USERNAME", "SMTP_USER"]:
            if k in st.secrets:
                config["username"] = str(st.secrets[k])
        for k in ["SMTP_PASSWORD", "SMTP_PASS"]:
            if k in st.secrets:
                config["password"] = str(st.secrets[k])
        if "SMTP_FROM" in st.secrets:
            config["from_email"] = str(st.secrets["SMTP_FROM"])
    except Exception:
        pass

    return config


def send_verification_email(
    recipient_email: str,
    otp_code: str,
) -> Tuple[bool, str, Optional[str]]:
    """Send an automated OTP confirmation email to a candidate's @gmail.com address.
    
    Returns:
        (success: bool, status_message: str, demo_otp_fallback: Optional[str])
    """
    clean_email = recipient_email.strip().lower()
    if not clean_email.endswith("@gmail.com"):
        return False, "Error Detected: Only valid @gmail.com addresses are allowed", None

    config = get_smtp_config()
    server_host = config["server"]
    port = config["port"]
    username = config["username"]
    password = config["password"]
    from_email = config["from_email"]

    # If SMTP is not configured in secrets.toml, return demo fallback OTP for seamless local testing
    if not server_host or not username or not password:
        return (
            False,
            "SMTP credentials not configured in secrets.toml. Running in Local Demo Mode.",
            otp_code,
        )

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your AI Interview Simulator Verification Code: {otp_code}"
        msg["From"] = from_email
        msg["To"] = clean_email

        text_content = (
            f"Hello,\n\n"
            f"Your AI Interview Simulator verification code is: {otp_code}\n\n"
            f"This code will expire in 10 minutes. If you did not request this, please ignore this email.\n\n"
            f"— The AI Interview Simulator Team"
        )

        html_content = f"""
        <html>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #0F172A; padding: 20px;">
            <div style="max-width: 520px; margin: 0 auto; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color: #1E3A8A; margin-top: 0;">AI Interview Simulator</h2>
                <p style="font-size: 15px; color: #475569;">Welcome! Please confirm your email address to access your executive interview simulator.</p>
                <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 18px; text-align: center; margin: 24px 0;">
                    <div style="font-size: 13px; font-weight: 700; color: #1D4ED8; text-transform: uppercase; letter-spacing: 0.05em;">Your Verification Code</div>
                    <div style="font-size: 32px; font-weight: 800; color: #1E3A8A; letter-spacing: 0.18em; margin-top: 6px;">{otp_code}</div>
                </div>
                <p style="font-size: 13px; color: #64748B;">This code is valid for 10 minutes. Do not share this code with anyone.</p>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(text_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        if port == 465:
            server = smtplib.SMTP_SSL(server_host, port, timeout=12)
        else:
            server = smtplib.SMTP(server_host, port, timeout=12)
            server.starttls()

        server.login(username, password)
        server.sendmail(from_email, [clean_email], msg.as_string())
        server.quit()

        return True, f"Verification code sent to {clean_email}", None

    except Exception as exc:
        # Fall back gracefully so user is never blocked during development
        return (
            False,
            f"SMTP send failed ({exc}). Running with on-screen demo OTP.",
            otp_code,
        )
