import os
import logging
from twilio.rest import Client
from env import load_dotenv
# Load Twilio configuration from environment
# TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
# TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
# TWILIO_WHATSAPP_NUMBER = os.environ.get("TWILIO_WHATSAPP_NUMBER")  # e.g., "whatsapp:+14155238886"

from config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_WHATSAPP_NUMBER
)

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_whatsapp_message(to: str, body: str):
    try:
        message = twilio_client.messages.create(
            body=body,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to
        )
        logging.info(f"Sent message SID: {message.sid} to {to}")
        return message.sid
    except Exception as e:
        logging.error(f"Error sending WhatsApp message: {e}")
        return None

#send_whatsapp_message("whatsapp:+96176626078", "Hello from Twilio!")