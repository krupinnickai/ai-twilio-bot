from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLINIC_NAME = "ID Eye & Aesthetics"
CLINIC_PHONE = "+18772682880"

conversations = {}

SYSTEM_PROMPT = f"""
You are the front desk receptionist for {CLINIC_NAME}.

Your job is to help patients:
- schedule appointments
- reschedule appointments
- answer basic clinic questions

Rules:
- Sound warm and natural like a real receptionist.
- Keep responses short.
- Ask one question at a time.
- Never give medical advice.
- If something sounds urgent tell them to call {CLINIC_PHONE}.

When scheduling collect:
1. patient name
2. appointment type
3. preferred day
4. preferred time
"""

def ask_ai(phone_number, message):

    if phone_number not in conversations:
        conversations[phone_number] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    conversations[phone_number].append(
        {"role": "user", "content": message}
    )

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=conversations[phone_number]
    )

    reply = completion.choices[0].message.content

    conversations[phone_number].append(
        {"role": "assistant", "content": reply}
    )

    return reply


@app.post("/sms")
async def sms_reply(request: Request):

    form = await request.form()
    incoming_msg = form.get("Body")
    from_number = form.get("From")

    ai_response = ask_ai(from_number, incoming_msg)

    twilio_resp = MessagingResponse()
    twilio_resp.message(ai_response)

    return Response(content=str(twilio_resp), media_type="application/xml")
