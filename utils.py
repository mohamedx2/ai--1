from langdetect import detect, DetectorFactory
import logging
import re
from typing import Dict, List

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language mapping and configuration
LANGUAGE_CONFIG = {
    'en': {'name': 'English', 'code': 'en', 'direction': 'ltr'},
    'fr': {'name': 'Français', 'code': 'fr', 'direction': 'ltr'},
    'es': {'name': 'Español', 'code': 'es', 'direction': 'ltr'},
    'ar': {'name': 'العربية', 'code': 'ar', 'direction': 'rtl'},
    'de': {'name': 'Deutsch', 'code': 'de', 'direction': 'ltr'},
    'it': {'name': 'Italiano', 'code': 'it', 'direction': 'ltr'},
    'pt': {'name': 'Português', 'code': 'pt', 'direction': 'ltr'},
    'zh': {'name': '中文', 'code': 'zh', 'direction': 'ltr'},
    'hi': {'name': 'हिन्दी', 'code': 'hi', 'direction': 'ltr'},
    'ru': {'name': 'Русский', 'code': 'ru', 'direction': 'ltr'},
    'ja': {'name': '日本語', 'code': 'ja', 'direction': 'ltr'},
    'ko': {'name': '한국어', 'code': 'ko', 'direction': 'ltr'}
}

# Common agricultural keywords by language for better detection
AGRICULTURAL_KEYWORDS = {
    'en': ['wheat', 'corn', 'soil', 'water', 'irrigation', 'fertilizer', 'pest', 'harvest', 'planting', 'crop'],
    'fr': ['blé', 'maïs', 'sol', 'eau', 'irrigation', 'engrais', 'ravageur', 'récolte', 'plantation', 'culture'],
    'es': ['trigo', 'maíz', 'suelo', 'agua', 'riego', 'fertilizante', 'plaga', 'cosecha', 'siembra', 'cultivo'],
    'ar': ['قمح', 'ذرة', 'تربة', 'ماء', 'ري', 'سماد', 'آفة', 'حصاد', 'زراعة', 'محصول'],
    'de': ['weizen', 'mais', 'boden', 'wasser', 'bewässerung', 'dünger', 'schädling', 'ernte', 'pflanzung', 'frucht'],
    'it': ['grano', 'mais', 'terreno', 'acqua', 'irrigazione', 'fertilizzante', 'parassita', 'raccolto', 'piantagione', 'coltura'],
    'pt': ['trigo', 'milho', 'solo', 'água', 'irrigação', 'fertilizante', 'praga', 'colheita', 'plantação', 'cultura']
}

def detect_language(text: str) -> str:
    """
    Detect the language of the given text with enhanced accuracy
    
    Args:
        text (str): Text to detect language for
        
    Returns:
        str: Language code (e.g., 'en', 'fr', 'es', 'ar')
    """
    try:
        if not text or len(text.strip()) < 3:
            return "en"
        
        # Clean text for better detection
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # First attempt: keyword-based detection for agricultural content
        keyword_lang = _detect_by_keywords(clean_text)
        if keyword_lang:
            logger.info(f"Language detected by keywords: {keyword_lang}")
            return keyword_lang
        
        # Second attempt: langdetect library
        detected = detect(clean_text)
        
        # Validate detected language
        if detected in LANGUAGE_CONFIG:
            logger.info(f"Language detected by langdetect: {detected}")
            return detected
        
        # Fallback to English if unsupported language
        logger.warning(f"Unsupported language detected: {detected}, falling back to English")
        return "en"
        
    except Exception as e:
        logger.error(f"Language detection failed: {str(e)}")
        return "en"

def _detect_by_keywords(text: str) -> str:
    """
    Detect language based on agricultural keywords
    
    Args:
        text (str): Cleaned text
        
    Returns:
        str: Language code if detected, None otherwise
    """
    words = text.lower().split()
    language_scores = {}
    
    for lang_code, keywords in AGRICULTURAL_KEYWORDS.items():
        score = sum(1 for word in words if word in keywords)
        if score > 0:
            language_scores[lang_code] = score
    
    if language_scores:
        # Return language with highest keyword match
        best_lang = max(language_scores, key=language_scores.get)
        if language_scores[best_lang] >= 2:  # Require at least 2 keyword matches
            return best_lang
    
    return None

def get_language_config(lang_code: str) -> Dict:
    """
    Get language configuration
    
    Args:
        lang_code (str): Language code
        
    Returns:
        Dict: Language configuration
    """
    return LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG['en'])

def get_supported_languages() -> List[Dict]:
    """
    Get list of supported languages
    
    Returns:
        List[Dict]: List of supported language configurations
    """
    return [
        {'code': code, 'name': config['name'], 'direction': config['direction']}
        for code, config in LANGUAGE_CONFIG.items()
    ]

def is_rtl_language(lang_code: str) -> bool:
    """
    Check if language is right-to-left
    
    Args:
        lang_code (str): Language code
        
    Returns:
        bool: True if RTL, False otherwise
    """
    config = get_language_config(lang_code)
    return config.get('direction', 'ltr') == 'rtl'

def format_text_for_language(text: str, lang_code: str) -> str:
    """
    Format text according to language conventions
    
    Args:
        text (str): Text to format
        lang_code (str): Language code
        
    Returns:
        str: Formatted text
    """
    config = get_language_config(lang_code)
    
    # Basic formatting based on language
    if config['direction'] == 'rtl':
        # For RTL languages, you might want to add additional formatting
        return text
    
    # Add language-specific formatting if needed
    return text

def translate_language_name(lang_code: str, target_lang: str = 'en') -> str:
    """
    Translate language name to target language
    
    Args:
        lang_code (str): Source language code
        target_lang (str): Target language code
        
    Returns:
        str: Translated language name
    """
    source_config = LANGUAGE_CONFIG.get(lang_code, {})
    return source_config.get('name', lang_code)

def validate_multilingual_input(text: str) -> Dict:
    """
    Validate and analyze multilingual input
    
    Args:
        text (str): Input text
        
    Returns:
        Dict: Analysis results
    """
    result = {
        'text': text,
        'length': len(text),
        'word_count': len(text.split()),
        'detected_language': detect_language(text),
        'is_agricultural': False,
        'has_keywords': False,
        'confidence': 'medium'
    }
    
    # Check for agricultural keywords
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = clean_text.split()
    
    for keywords in AGRICULTURAL_KEYWORDS.values():
        if any(word in keywords for word in words):
            result['is_agricultural'] = True
            result['has_keywords'] = True
            break
    
    # Simple confidence assessment
    if result['word_count'] > 10 and result['is_agricultural']:
        result['confidence'] = 'high'
    elif result['word_count'] < 3:
        result['confidence'] = 'low'
    
    return result
