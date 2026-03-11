import asyncio
import json
import logging
import os
from typing import Dict, List, Optional

import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from fastapi.websockets import WebSocketDisconnect
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import Connect, VoiceResponse

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clinic-ai-bot")

# -----------------------------
# Environment variables
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

SMS_MODEL = os.getenv("SMS_MODEL", "gpt-4.1-mini")
VOICE_MODEL = os.getenv("VOICE_MODEL", "gpt-realtime")
VOICE_NAME = os.getenv("VOICE_NAME", "cedar")  # cedar or marin

CLINIC_NAME = os.getenv("CLINIC_NAME", "ID Eye & Aesthetics")
CLINIC_PHONE = os.getenv("CLINIC_PHONE", "+1 877-268-2880")
CLINIC_HOURS = os.getenv("CLINIC_HOURS", "Monday through Friday, 9 AM to 5 PM.")
CLINIC_ADDRESS = os.getenv("CLINIC_ADDRESS", "Please ask the office for the exact address.")
CLINIC_SERVICES = os.getenv(
    "CLINIC_SERVICES",
    "eye exams, consultations, follow-ups, dry eye care, and aesthetics appointments",
)
CLINIC_INSURANCE = os.getenv(
    "CLINIC_INSURANCE",
    "Please ask the office which insurance plans are currently accepted.",
)

AI_GREETS_FIRST = os.getenv("AI_GREETS_FIRST", "true").lower() == "true"

client = OpenAI(api_key=OPENAI_API_KEY)

# In-memory SMS history for testing
sms_conversations: Dict[str, List[Dict[str, str]]] = {}


# -----------------------------
# Prompts
# -----------------------------
def build_sms_system_prompt() -> str:
    return f"""
You are the front desk receptionist for {CLINIC_NAME}.

You help with:
- scheduling appointments
- rescheduling appointments
- answering basic clinic questions
- collecting callback information

Clinic info:
- Name: {CLINIC_NAME}
- Phone: {CLINIC_PHONE}
- Hours: {CLINIC_HOURS}
- Address: {CLINIC_ADDRESS}
- Services: {CLINIC_SERVICES}
- Insurance: {CLINIC_INSURANCE}

Rules:
- Sound warm, natural, and professional.
- Keep replies short and text-friendly.
- Ask one question at a time.
- Never give medical advice.
- If something sounds urgent, tell them to call {CLINIC_PHONE} immediately or seek urgent medical care.
- Do not say you are an AI unless directly asked.
- If booking, collect:
  1. full name
  2. appointment type
  3. preferred day
  4. preferred time
  5. callback number if needed
- If all booking details are collected, say the office will follow up to confirm.
""".strip()


def build_voice_system_prompt() -> str:
    return f"""
You are the live phone receptionist for {CLINIC_NAME}.

Your job:
- greet callers warmly
- help them schedule or reschedule appointments
- answer basic questions about hours, address, services, and insurance
- sound calm, natural, and professional
- keep the conversation flowing like a real receptionist

Clinic info:
- Name: {CLINIC_NAME}
- Phone: {CLINIC_PHONE}
- Hours: {CLINIC_HOURS}
- Address: {CLINIC_ADDRESS}
- Services: {CLINIC_SERVICES}
- Insurance: {CLINIC_INSURANCE}

Voice style rules:
- Speak in short sentences.
- Sound human, warm, and confident.
- Do not ramble.
- Ask one question at a time.
- If the caller already gave information, do not ask for it again.
- Never give medical advice.
- If something sounds urgent, say: "Please call the office right away at {CLINIC_PHONE}, or seek urgent medical care."
- If you are unsure, say the office can follow up.

Scheduling flow:
Collect:
1. full name
2. appointment type
3. preferred day
4. preferred time
5. callback number if needed

When the booking details are complete, say:
"Perfect. I have your request and the office will follow up to confirm."
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def trim_history(history: List[Dict[str, str]], keep_last: int = 12) -> List[Dict[str, str]]:
    if not history:
        return history
    system_msg = history[0]
    recent = history[1:][-keep_last:]
    return [system_msg] + recent


def get_sms_history(phone_number: str) -> List[Dict[str, str]]:
    key = phone_number or "unknown"
    if key not in sms_conversations:
        sms_conversations[key] = [{"role": "system", "content": build_sms_system_prompt()}]
    return sms_conversations[key]


def public_ws_url(request: Request, path: str) -> str:
    if PUBLIC_BASE_URL:
        base = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}{path}"

    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}://{host}{path}"


def safe_text(value: Optional[str]) -> str:
    return (value or "").strip()


# -----------------------------
# SMS
# -----------------------------
def ask_sms_ai(phone_number: str, user_message: str) -> str:
    history = get_sms_history(phone_number)

    lowered = user_message.strip().lower()
    if lowered in {"reset", "restart", "start over"}:
        sms_conversations[phone_number] = [{"role": "system", "content": build_sms_system_prompt()}]
        return "No problem — we can start over. How can I help you today?"

    history.append({"role": "user", "content": user_message})
    history = trim_history(history)
    sms_conversations[phone_number] = history

    try:
        completion = client.chat.completions.create(
            model=SMS_MODEL,
            messages=history,
            temperature=0.5,
            max_tokens=180,
        )
        reply = (completion.choices[0].message.content or "").strip()
        if not reply:
            reply = "Thanks for reaching out. How can I help you today?"
    except Exception:
        logger.exception("SMS OpenAI request failed")
        reply = (
            f"Thanks for contacting {CLINIC_NAME}. "
            f"I'm having trouble right now. Please call {CLINIC_PHONE} if you need immediate help."
        )

    sms_conversations[phone_number].append({"role": "assistant", "content": reply})
    sms_conversations[phone_number] = trim_history(sms_conversations[phone_number])
    return reply


# -----------------------------
# Basic routes
# -----------------------------
@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "clinic-ai-bot",
            "clinic": CLINIC_NAME,
            "voice_model": VOICE_MODEL,
            "voice_name": VOICE_NAME,
        }
    )


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


# -----------------------------
# SMS webhook
# -----------------------------
@app.post("/sms")
async def sms_reply(request: Request) -> Response:
    form = await request.form()
    incoming_msg = safe_text(form.get("Body")) or "Hello"
    from_number = safe_text(form.get("From")) or "unknown"

    logger.info("Incoming SMS from %s: %s", from_number, incoming_msg)

    ai_response = ask_sms_ai(from_number, incoming_msg)

    twilio_resp = MessagingResponse()
    twilio_resp.message(ai_response)
    return Response(content=str(twilio_resp), media_type="application/xml")


# -----------------------------
# Voice webhook
# -----------------------------
@app.api_route("/voice", methods=["GET", "POST"])
async def voice_entry(request: Request) -> Response:
    ws_url = public_ws_url(request, "/media-stream")

    vr = VoiceResponse()
    connect = Connect()
    connect.stream(url=ws_url)
    vr.append(connect)

    logger.info("Voice webhook hit. Streaming call to %s", ws_url)
    return Response(content=str(vr), media_type="application/xml")


# -----------------------------
# Realtime bridge helpers
# -----------------------------
async def initialize_realtime_session(openai_ws) -> None:
    session_update = {
        "type": "session.update",
        "session": {
            "instructions": build_voice_system_prompt(),
            "voice": VOICE_NAME,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "modalities": ["audio", "text"],
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            },
            "temperature": 0.7,
        },
    }

    await openai_ws.send(json.dumps(session_update))
    logger.info("Realtime session initialized")


async def send_initial_greeting(openai_ws) -> None:
    initial_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"Greet the caller warmly as the receptionist for {CLINIC_NAME}. "
                        f"Introduce the clinic once, then ask how you can help today. "
                        f"Keep it natural and under 18 words."
                    ),
                }
            ],
        },
    }
    await openai_ws.send(json.dumps(initial_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


# -----------------------------
# Media stream websocket
# -----------------------------
@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Twilio media stream connected")

    stream_sid: Optional[str] = None
    latest_media_timestamp = 0
    last_assistant_item: Optional[str] = None
    response_start_timestamp_twilio: Optional[int] = None
    mark_queue: List[str] = []

    openai_uri = f"wss://api.openai.com/v1/realtime?model={VOICE_MODEL}"

    try:
        async with websockets.connect(
            openai_uri,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as openai_ws:
            await initialize_realtime_session(openai_ws)

            if AI_GREETS_FIRST:
                await send_initial_greeting(openai_ws)

            async def send_mark() -> None:
                nonlocal stream_sid, mark_queue
                if stream_sid:
                    mark_event = {
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {"name": "responsePart"},
                    }
                    await websocket.send_json(mark_event)
                    mark_queue.append("responsePart")

            async def handle_speech_started_event() -> None:
                nonlocal response_start_timestamp_twilio, last_assistant_item, mark_queue

                if mark_queue and response_start_timestamp_twilio is not None and last_assistant_item:
                    elapsed_time = latest_media_timestamp - response_start_timestamp_twilio

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time,
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                    await websocket.send_json(
                        {
                            "event": "clear",
                            "streamSid": stream_sid,
                        }
                    )

                    mark_queue = []
                    last_assistant_item = None
                    response_start_timestamp_twilio = None

            async def receive_from_twilio() -> None:
                nonlocal stream_sid, latest_media_timestamp
                try:
                    while True:
                        message = await websocket.receive_text()
                        data = json.loads(message)

                        event_type = data.get("event")

                        if event_type == "start":
                            stream_sid = data["start"]["streamSid"]
                            logger.info("Incoming call stream started: %s", stream_sid)
                            latest_media_timestamp = 0

                        elif event_type == "media":
                            latest_media_timestamp = int(data["media"]["timestamp"])
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data["media"]["payload"],
                            }
                            await openai_ws.send(json.dumps(audio_append))

                        elif event_type == "mark":
                            if mark_queue:
                                mark_queue.pop(0)

                        elif event_type == "stop":
                            logger.info("Twilio stream stopped")
                            break
                except WebSocketDisconnect:
                    logger.info("Twilio websocket disconnected")
                except Exception:
                    logger.exception("Error receiving from Twilio")

            async def send_to_twilio() -> None:
                nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        event_type = response.get("type")

                        if event_type == "response.audio.delta" and response.get("delta"):
                            audio_delta = response["delta"]

                            media_event = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_delta,
                                },
                            }
                            await websocket.send_json(media_event)

                            if response_start_timestamp_twilio is None:
                                response_start_timestamp_twilio = latest_media_timestamp

                            if response.get("item_id"):
                                last_assistant_item = response["item_id"]

                            await send_mark()

                        elif event_type == "input_audio_buffer.speech_started":
                            await handle_speech_started_event()

                        elif event_type == "response.done":
                            logger.info("OpenAI response finished")

                        elif event_type == "error":
                            logger.error("OpenAI realtime error: %s", response)

                except Exception:
                    logger.exception("Error sending to Twilio")

            await asyncio.gather(receive_from_twilio(), send_to_twilio())

    except Exception:
        logger.exception("Realtime voice bridge failed")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Media stream closed")
