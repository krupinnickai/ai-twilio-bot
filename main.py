from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os
import re
from openai import OpenAI

# load environment variables
load_dotenv()

app = FastAPI()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLINIC_NAME = os.getenv("CLINIC_NAME", "Example Med Spa")
CLINIC_PHONE = os.getenv("CLINIC_PHONE", "+1XXXXXXXXXX")

# words that indicate emergency situations
URGENT_PATTERNS = [
    r"sudden vision loss",
    r"flashes",
    r"floaters",
    r"curtain",
    r"severe pain",
    r"trouble breathing",
    r"chest pain"
]

def is_urgent(text: str):
    text = text.lower()
    for pattern in URGENT_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


SYSTEM_PROMPT = f"""
You are an AI receptionist for {CLINIC_NAME}.

Your job is to:
- help schedule appointments
- answer service questions
- collect lead information (name, service, preferred appointment time)

Rules:
- Do NOT diagnose medical conditions
- If the message sounds urgent tell them to call {CLINIC_PHONE}
- Keep replies short (1-3 sentences)
- Ask one question at a time
"""


@app.post("/sms")
async def sms_reply(request: Request):

    form = await request.form()
    incoming_msg = form.get("Body")

    twiml = MessagingResponse()

    # emergency check
    if is_urgent(incoming_msg):
        twiml.message(
            f"This may be urgent. Please call us immediately at {CLINIC_PHONE}. "
            "If symptoms are severe seek emergency care."
        )
        return Response(content=str(twiml), media_type="text/xml")

    # AI response
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": incoming_msg}
        ],
        temperature=0.4
    )

    ai_reply = completion.choices[0].message.content

    twiml.message(ai_reply)

    return Response(content=str(twiml), media_type="text/xml")


@app.get("/")
def home():
    return {"status": "AI clinic bot running"}
