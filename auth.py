import secrets
import smtplib
import asyncio
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import get_settings
from database import get_db, User

settings = get_settings()
templates = Jinja2Templates(directory="web/templates")
router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ─── Password helpers ───────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ─── Email helper ────────────────────────────────────────────────────────

def _send_email_sync(to: str, subject: str, body: str):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = settings.smtp_from
    msg["To"] = to
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as smtp:
        smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.sendmail(settings.smtp_from, [to], msg.as_string())


async def send_reset_email(email: str, link: str) -> None:
    subject = "Password reset — Frameshift AI"
    body = f"Click this link to reset your password (valid 1 hour):\n\n{link}\n\nIf you didn't request this, ignore this email."
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _send_email_sync, email, subject, body)


# ─── Google OAuth helpers ────────────────────────────────────────────────

def build_google_auth_url(state: str) -> str:
    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": settings.google_oauth_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "offline",
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


async def exchange_google_code(code: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "redirect_uri": settings.google_oauth_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_google_userinfo(access_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()


# ─── Routes ──────────────────────────────────────────────────────────────

def _google_enabled() -> bool:
    return bool(settings.google_oauth_client_id and settings.google_oauth_client_secret)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse("login.html", {"request": request, "error": error, "google_enabled": _google_enabled()})


@router.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    user = (await db.execute(select(User).where(User.email == email.lower()))).scalar_one_or_none()
    if not user or not user.password_hash or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password", "google_enabled": _google_enabled()},
            status_code=401,
        )
    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: str = ""):
    return templates.TemplateResponse("register.html", {"request": request, "error": error, "google_enabled": _google_enabled()})


@router.post("/register")
async def register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    ge = _google_enabled()
    if password != confirm_password:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Passwords do not match", "google_enabled": ge},
            status_code=400,
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Password must be at least 8 characters", "google_enabled": ge},
            status_code=400,
        )
    existing = (await db.execute(select(User).where(User.email == email.lower()))).scalar_one_or_none()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "An account with that email already exists", "google_enabled": ge},
            status_code=400,
        )
    user = User(
        email=email.lower(),
        name=name,
        password_hash=hash_password(password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@router.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request, sent: str = ""):
    return templates.TemplateResponse("forgot_password.html", {"request": request, "sent": sent, "dev_link": ""})


@router.post("/forgot-password")
async def forgot_password(
    request: Request,
    email: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    user = (await db.execute(select(User).where(User.email == email.lower()))).scalar_one_or_none()
    dev_link = ""
    if user:
        token = secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
        await db.commit()
        reset_link = f"{request.base_url}reset-password?token={token}"
        if settings.smtp_host:
            try:
                await send_reset_email(email, reset_link)
            except Exception:
                dev_link = reset_link
        else:
            dev_link = reset_link

    return templates.TemplateResponse(
        "forgot_password.html",
        {"request": request, "sent": "1", "dev_link": dev_link},
    )


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str = "", error: str = ""):
    return templates.TemplateResponse(
        "reset_password.html",
        {"request": request, "token": token, "error": error},
    )


@router.post("/reset-password")
async def reset_password(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    if password != confirm_password:
        return templates.TemplateResponse(
            "reset_password.html",
            {"request": request, "token": token, "error": "Passwords do not match"},
            status_code=400,
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            "reset_password.html",
            {"request": request, "token": token, "error": "Password must be at least 8 characters"},
            status_code=400,
        )
    user = (await db.execute(select(User).where(User.reset_token == token))).scalar_one_or_none()
    if not user or not user.reset_token_expires or user.reset_token_expires < datetime.utcnow():
        return templates.TemplateResponse(
            "reset_password.html",
            {"request": request, "token": token, "error": "Invalid or expired reset link"},
            status_code=400,
        )
    user.password_hash = hash_password(password)
    user.reset_token = None
    user.reset_token_expires = None
    await db.commit()
    return RedirectResponse("/login?reset=1", status_code=303)


@router.get("/auth/google")
async def google_login(request: Request):
    if not settings.google_oauth_client_id:
        return RedirectResponse("/login", status_code=303)
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    return RedirectResponse(build_google_auth_url(state), status_code=302)


@router.get("/auth/google/callback")
async def google_callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    db: AsyncSession = Depends(get_db),
):
    if error or not code:
        return RedirectResponse("/login?error=google_denied", status_code=303)
    if state != request.session.get("oauth_state"):
        return RedirectResponse("/login?error=invalid_state", status_code=303)
    request.session.pop("oauth_state", None)

    try:
        token_data = await exchange_google_code(code)
        userinfo = await fetch_google_userinfo(token_data["access_token"])
    except Exception:
        return RedirectResponse("/login?error=google_failed", status_code=303)

    google_id = userinfo.get("sub")
    email = userinfo.get("email", "").lower()
    name = userinfo.get("name", "")

    # Find by google_id first, then by email
    user = (await db.execute(select(User).where(User.google_id == google_id))).scalar_one_or_none()
    if not user and email:
        user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user:
            user.google_id = google_id

    if not user:
        user = User(email=email, name=name, google_id=google_id)
        db.add(user)

    await db.commit()
    await db.refresh(user)
    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)
