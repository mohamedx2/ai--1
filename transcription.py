import whisper
import logging
from utils import detect_language

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model to avoid reloading
_model = None

def load_whisper_model(model_size="base"):
    """Load Whisper model once and reuse"""
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {model_size}")
        _model = whisper.load_model(model_size)
        logger.info("Whisper model loaded successfully")
    return _model

def transcribe_audio(audio_path, model_size="base"):
    """
    Transcribe audio file using Whisper with multilingual support
    
    Args:
        audio_path (str): Path to audio file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        tuple: (transcript_text, detected_language)
    """
    try:
        model = load_whisper_model(model_size)
        
        # Transcribe with automatic language detection
        result = model.transcribe(
            audio_path,
            language=None,  # Auto-detect language
            task="transcribe",
            fp16=False  # Better compatibility
        )
        
        transcript = result["text"].strip()
        detected_lang = result.get("language", "en")
        
        # Fallback to langdetect if Whisper fails to detect language
        if not detected_lang or detected_lang == "en" and len(transcript) > 10:
            detected_lang = detect_language(transcript)
        
        logger.info(f"Transcribed {len(transcript)} characters, language: {detected_lang}")
        return transcript, detected_lang
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return "", "en"
