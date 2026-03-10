from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse
from twilio.twiml.messaging_response import MessagingResponse
from twilio.twiml.voice_response import VoiceResponse, Gather
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLINIC_NAME = os.getenv("CLINIC_NAME", "ID Eye & Aesthetics")
CLINIC_PHONE = os.getenv("CLINIC_PHONE", "+18303355399")
CLINIC_HOURS = os.getenv(
    "CLINIC_HOURS",
    "Monday through Friday, 9 AM to 5 PM."
)
CLINIC_ADDRESS = os.getenv(
    "CLINIC_ADDRESS",
    "Please ask the office for the exact address."
)
CLINIC_SERVICES = os.getenv(
    "CLINIC_SERVICES",
    "eye exams, consultations, follow-ups, and aesthetics appointments"
)

# Simple in-memory conversation history
# Good for testing. Later you can move this to a database.
conversations = {}


def build_system_prompt() -> str:
    return f"""
You are the front desk receptionist for {CLINIC_NAME}.

Your job:
- help patients schedule appointments
- help patients reschedule appointments
- answer basic clinic questions
- collect patient information when needed
- sound warm, natural, short, and professional

Clinic details:
- Clinic name: {CLINIC_NAME}
- Phone: {CLINIC_PHONE}
- Hours: {CLINIC_HOURS}
- Address: {CLINIC_ADDRESS}
- Services: {CLINIC_SERVICES}

Important rules:
- Sound like a real receptionist, not a robot.
- Keep replies short and conversational.
- Ask only one question at a time whenever possible.
- Never diagnose, prescribe, or give medical advice.
- If something sounds urgent or medical, tell them to call {CLINIC_PHONE} right away or seek immediate medical care.
- If you do not know something, say the office can follow up.
- Do not mention being an AI unless directly asked.
- Be polite and confident.

Scheduling flow:
When someone wants to book, collect these in order:
1. full name
2. appointment type
3. preferred day
4. preferred time
5. callback phone number if needed

If they already gave one of those, do not ask for it again.
Always ask for the next missing detail.

If the booking details are complete, reply with something like:
"Perfect — I have your request for [appointment type] on [day] at [time]. The office will follow up to confirm."

Style examples:
Patient: I need an appointment
Assistant: Absolutely — what kind of appointment would you like to schedule?

Patient: eye exam
Assistant: Great. What day works best for you?

Patient: Tuesday afternoon
Assistant: Perfect. Can I get your full name?

Patient: Nick Krupin
Assistant: Thanks, Nick. What’s the best callback number if the office needs to confirm anything?

Basic questions:
- If asked about hours, use the clinic hours above.
- If asked about location, use the clinic address above.
- If asked what services are offered, use the services list above.

Keep every reply helpful and natural.
""".strip()


def get_conversation_key(phone_number: str) -> str:
    return phone_number or "unknown"


def get_or_create_history(phone_number: str):
    key = get_conversation_key(phone_number)

    if key not in conversations:
        conversations[key] = [
            {"role": "system", "content": build_system_prompt()}
        ]

    return conversations[key]


def trim_history(history, keep_last: int = 12):
    system_msg = history[0]
    recent = history[1:][-keep_last:]
    return [system_msg] + recent


def ask_ai(phone_number: str, user_message: str) -> str:
    history = get_or_create_history(phone_number)

    if user_message.strip().lower() in {"reset", "restart", "start over"}:
        conversations[get_conversation_key(phone_number)] = [
            {"role": "system", "content": build_system_prompt()}
        ]
        return "No problem — we can start over. How can I help you today?"

    history.append({"role": "user", "content": user_message})
    history = trim_history(history)
    conversations[get_conversation_key(phone_number)] = history

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=history,
        temperature=0.5
    )

    reply = completion.choices[0].message.content or ""
    reply = reply.strip()

    conversations[get_conversation_key(phone_number)].append(
        {"role": "assistant", "content": reply}
    )
    conversations[get_conversation_key(phone_number)] = trim_history(
        conversations[get_conversation_key(phone_number)]
    )

    return reply


@app.get("/")
async def root():
    return JSONResponse(
        {
            "status": "ok",
            "service": "clinic-ai-bot",
            "clinic": CLINIC_NAME
        }
    )


@app.post("/sms")
async def sms_reply(request: Request):
    form = await request.form()
    incoming_msg = (form.get("Body") or "").strip()
    from_number = (form.get("From") or "").strip()

    if not incoming_msg:
        incoming_msg = "Hello"

    ai_response = ask_ai(from_number, incoming_msg)

    twilio_resp = MessagingResponse()
    twilio_resp.message(ai_response)

    return Response(content=str(twilio_resp), media_type="application/xml")


@app.post("/voice")
async def voice_reply(request: Request):
    response = VoiceResponse()

    gather = Gather(
        input="speech",
        action="/voice/process",
        method="POST",
        speech_timeout="auto"
    )
    gather.say(
        f"Hi, thank you for calling {CLINIC_NAME}. "
        "How can I help you today?"
    )
    response.append(gather)

    response.say("Sorry, I didn't catch that. Please call again.")
    return Response(content=str(response), media_type="application/xml")


@app.post("/voice/process")
async def voice_process(request: Request):
    form = await request.form()
    speech_result = (form.get("SpeechResult") or "").strip()
    from_number = (form.get("From") or "").strip()

    if not speech_result:
        speech_result = "Hello"

    ai_response = ask_ai(from_number, speech_result)

    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/voice/process",
        method="POST",
        speech_timeout="auto"
    )
    gather.say(ai_response)
    response.append(gather)

    return Response(content=str(response), media_type="application/xml")
