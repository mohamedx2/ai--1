from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import time
import os
import tempfile
import uuid
from datetime import datetime

# Import our modules
from transcription import transcribe_audio
from retrieval import retrieve_context
from llm_response import generate_response
from utils import detect_language
from logging_config import setup_logging
import os

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get Gemini API key from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBBQ-okicASaEm1CP9Q8rzvLfYZJMEHvIc')

# Initialize FastAPI app
app = FastAPI(
    title="Assistant Agricole Multilingue API",
    description="Multilingual agricultural assistant API with voice and text support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global conversation storage (in production, use Redis or database)
conversations: Dict[str, List[Dict]] = {}

# Pydantic models for API
class TextInput(BaseModel):
    text: str
    conversation_id: Optional[str] = None
    language: Optional[str] = None

class AudioInput(BaseModel):
    audio_data: str  # Base64 encoded audio
    conversation_id: Optional[str] = None
    language: Optional[str] = None

class Response(BaseModel):
    response: str
    transcript: Optional[str] = None
    language: str
    conversation_id: str
    timestamp: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]

# Utility functions
def get_or_create_conversation(conversation_id: Optional[str]) -> str:
    """Get existing conversation or create new one"""
    if conversation_id is None or conversation_id not in conversations:
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = []
    return conversation_id

def add_to_conversation(conversation_id: str, role: str, content: str, language: str):
    """Add message to conversation history"""
    if conversation_id in conversations:
        conversations[conversation_id].append({
            "role": role,
            "content": content,
            "language": language,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 messages to manage memory
        if len(conversations[conversation_id]) > 20:
            conversations[conversation_id] = conversations[conversation_id][-20:]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - redirect to web interface"""
    return FileResponse("static/index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/api/text", response_model=Response)
async def process_text_input(input_data: TextInput):
    """Process text input and generate agricultural advice"""
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    
    try:
        logger.info(f"[{request_id}] Processing text input: {input_data.text[:100]}...")
        
        # Get or create conversation
        conversation_id = get_or_create_conversation(input_data.conversation_id)
        
        # Detect language if not provided
        if input_data.language:
            detected_lang = input_data.language
        else:
            detected_lang = detect_language(input_data.text)
        
        # Add user input to conversation
        add_to_conversation(conversation_id, "user", input_data.text, detected_lang)
        
        # Retrieve context
        logger.info(f"[{request_id}] Retrieving context")
        context = retrieve_context(input_data.text, detected_lang)
        
        # Generate response
        logger.info(f"[{request_id}] Generating response")
        response_text = generate_response(input_data.text, context, detected_lang, gemini_api_key=GEMINI_API_KEY)
        
        # Add assistant response to conversation
        add_to_conversation(conversation_id, "assistant", response_text, detected_lang)
        
        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Request completed in {processing_time:.2f}s")
        
        return Response(
            response=response_text,
            transcript=input_data.text,
            language=detected_lang,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request_id}] Text processing failed after {processing_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/audio", response_model=Response)
async def process_audio_input(input_data: AudioInput):
    """Process audio input and generate agricultural advice"""
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    
    try:
        logger.info(f"[{request_id}] Processing audio input")
        
        # Get or create conversation
        conversation_id = get_or_create_conversation(input_data.conversation_id)
        
        # Decode base64 audio and save to temporary file
        import base64
        audio_data = base64.b64decode(input_data.audio_data)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe audio
            logger.info(f"[{request_id}] Transcribing audio")
            transcript, detected_lang = transcribe_audio(temp_file_path)
            
            if not transcript:
                raise HTTPException(status_code=400, detail="Could not transcribe audio")
            
            # Detect language if not provided
            if input_data.language:
                detected_lang = input_data.language
            
            # Add user input to conversation
            add_to_conversation(conversation_id, "user", transcript, detected_lang)
            
            # Retrieve context
            logger.info(f"[{request_id}] Retrieving context")
            context = retrieve_context(transcript, detected_lang)
            
            # Generate response
            logger.info(f"[{request_id}] Generating response")
            response_text = generate_response(transcript, context, detected_lang, gemini_api_key=GEMINI_API_KEY)
            
            # Add assistant response to conversation
            add_to_conversation(conversation_id, "assistant", response_text, detected_lang)
            
            processing_time = time.time() - start_time
            logger.info(f"[{request_id}] Request completed in {processing_time:.2f}s")
            
            return Response(
                response=response_text,
                transcript=transcript,
                language=detected_lang,
                conversation_id=conversation_id,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request_id}] Audio processing failed after {processing_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/conversation/{conversation_id}", response_model=ConversationHistory)
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationHistory(
        conversation_id=conversation_id,
        messages=conversations[conversation_id]
    )

@app.delete("/api/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversations[conversation_id] = []
    return {"message": "Conversation cleared", "conversation_id": conversation_id}

@app.get("/api/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    from utils import get_supported_languages
    return {"languages": get_supported_languages()}

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "active_conversations": len(conversations),
        "total_messages": sum(len(conv) for conv in conversations.values()),
        "timestamp": datetime.now().isoformat(),
        "system_status": "healthy"
    }

# Background task to clean up old conversations
async def cleanup_old_conversations():
    """Clean up conversations older than 24 hours"""
    current_time = datetime.now()
    conversations_to_remove = []
    
    for conv_id, messages in conversations.items():
        if messages:
            last_message_time = datetime.fromisoformat(messages[-1]["timestamp"])
            if (current_time - last_message_time).days > 1:
                conversations_to_remove.append(conv_id)
    
    for conv_id in conversations_to_remove:
        del conversations[conv_id]
        logger.info(f"Cleaned up old conversation: {conv_id}")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Assistant Agricole Multilingue FastAPI starting up...")
    logger.info("Web interface available at: /")
    logger.info("API Documentation available at: /docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Assistant Agricole Multilingue FastAPI shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
