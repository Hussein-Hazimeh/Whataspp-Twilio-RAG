import asyncio
import logging
import aiohttp
import tempfile
from dotenv import load_dotenv
import os
import openai  # Make sure OPENAI_API_KEY is set in your environment
from aiohttp import BasicAuth
from fastapi import FastAPI, Form
from fastapi.responses import Response
from rag_agent import query_rag_agent
from twilio_logic import send_whatsapp_message

# Set up logging
logging.basicConfig(level=logging.INFO)
load_dotenv()
app = FastAPI()

@app.get("/hello")
async def say_hello():
    return "Hello World"

@app.post("/webhook")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(None),
    NumMedia: int = Form(0),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None)
):
    if NumMedia > 0 and MediaContentType0 and MediaContentType0.startswith("audio"):
        logging.info(f"Received voice message from {From}: {MediaUrl0}")
        asyncio.create_task(process_voice_message(From, MediaUrl0))
    if NumMedia > 0 and MediaContentType0 and MediaContentType0.startswith("image"):
        logging.info(f"Received image message from {From}: {MediaUrl0}")
        #asyncio.create_task(process_image_message(From, MediaUrl0))
    else:
        logging.info(f"Received text message from {From}: {Body}")
        asyncio.create_task(process_message(From, Body))
    
    # Return immediate TwiML response to Twilio
    response_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Response></Response>"""
    return Response(content=response_xml, media_type="application/xml")

async def process_message(from_number: str, message_body: str):
    try:
        response_text = await query_rag_agent(message_body)
        logging.info(f"AI Agent response: {response_text}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_whatsapp_message, from_number, response_text)
    except Exception as e:
        logging.error(f"Error processing message from {from_number}: {e}")

async def process_voice_message(from_number: str, media_url: str):
    #logging.info(f"Received voice message from {from_number}: {media_url}")
    try:
        
        transcription_text = await transcribe_audio(media_url)
        logging.info(f"Transcription: {transcription_text}")
        
        response_text = await query_rag_agent(transcription_text)
        logging.info(f"AI Agent response for voice: {response_text}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_whatsapp_message, from_number, response_text)
    except Exception as e:
        logging.error(f"Error processing voice message from {from_number}: {e}")



async def transcribe_audio(media_url: str) -> str:
    logging.info(f"Transcribing audio from {media_url}")
    # Get Twilio credentials from environment variables
    TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
    
    # Set up basic authentication for Twilio media URL
    logging.info(f"TWILIO_ACCOUNT_SID: {TWILIO_ACCOUNT_SID}, TWILIO_AUTH_TOKEN: {TWILIO_AUTH_TOKEN}")
    auth = BasicAuth(login=TWILIO_ACCOUNT_SID, password=TWILIO_AUTH_TOKEN)
    
    # Download the audio file using aiohttp with authentication
    #logging.info(f"Downloading audio from {media_url}")
    async with aiohttp.ClientSession() as session:
        #logging.info(f"Downloading audio from {media_url}")
        async with session.get(media_url, auth=auth) as response:
            logging.info(f"Downloading audio from {media_url}")
            if response.status != 200:
                raise Exception(f"Failed to download audio: HTTP {response.status}")
            logging.info(f"reading audio from {media_url}")
            audio_data = await response.read()

    # Save the downloaded audio to a temporary file
    logging.info(f"Saving audio to temp file")
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_file.write(audio_data)
        tmp_file_path = tmp_file.name

    # Synchronous transcription using OpenAI Whisper wrapped in an executor
    def transcribe_sync():
        with open(tmp_file_path, "rb") as audio_file:
            logging.info(f"Transcribing audio")
            transcript = openai.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
            logging.info(f"Transcription: {transcript.text}")
        return transcript.text

    loop = asyncio.get_event_loop()
    transcription = await loop.run_in_executor(None, transcribe_sync)
    
    # Clean up temporary file
    os.remove(tmp_file_path)
    return transcription

# To run:
#   uvicorn controller:app --host 0.0.0.0 --port 8000
