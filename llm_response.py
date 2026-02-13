import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import logging
from typing import Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriculturalLLM:
    """Enhanced LLM with Google Gemini integration for agricultural advice"""
    
    def __init__(self, model_name: str = None, gemini_api_key: str = None):
        """
        Initialize LLM with automatic model selection including Gemini
        
        Args:
            model_name: Specific model to use, or None for auto-selection
            gemini_api_key: Google Gemini API key
        """
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name or self._select_best_model()
        self.generator = None
        self.tokenizer = None
        self.model_type = None
        self.use_gemini = False
        self.gemini_model = None
        self._initialize_model()
    
    def _select_best_model(self) -> str:
        """Select the best available model for agricultural advice"""
        
        # Priority list of models for agricultural advice
        model_options = [
            # Check if Gemini API key is available first
            "gemini-2.5-flash" if self.gemini_api_key else None,
            # Instruction-tuned models (best for detailed responses)
            "google/flan-t5-large",  # Instruction-following model
            "facebook/bart-large-cnn",  # Summarization/generation
            "microsoft/DialoGPT-large",  # Larger conversational model
            "microsoft/DialoGPT-medium",  # Fallback conversational
            "gpt2",  # Final fallback
        ]
        
        # Filter out None values
        model_options = [model for model in model_options if model is not None]
        
        # Try models in order of preference
        for model_name in model_options:
            try:
                if model_name == "gemini-2.5-flash":
                    logger.info("Gemini API key available - selected Gemini 2.5 Flash")
                    return model_name
                    
                logger.info(f"Attempting to load model: {model_name}")
                
                if "flan-t5" in model_name:
                    # Test T5 model availability
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if tokenizer:
                        logger.info(f"Selected model: {model_name}")
                        return model_name
                        
                elif "bart" in model_name:
                    # Test BART model availability
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if tokenizer:
                        logger.info(f"Selected model: {model_name}")
                        return model_name
                        
                elif "DialoGPT" in model_name:
                    # Test DialoGPT model availability
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if tokenizer:
                        logger.info(f"Selected model: {model_name}")
                        return model_name
                        
                elif "gpt2" in model_name:
                    # Test GPT-2 model availability
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    if tokenizer:
                        logger.info(f"Selected model: {model_name}")
                        return model_name
                        
            except Exception as e:
                logger.warning(f"Model {model_name} not available: {e}")
                continue
        
        # If none work, use GPT-2 as final fallback
        logger.warning("Using GPT-2 as final fallback model")
        return "gpt2"
    
    def _initialize_model(self):
        """Initialize the selected model"""
        try:
            logger.info(f"Initializing model: {self.model_name}")
            
            if "gemini" in self.model_name:
                self._initialize_gemini()
            elif "flan-t5" in self.model_name:
                self._initialize_t5_model()
            elif "bart" in self.model_name:
                self._initialize_bart_model()
            elif "DialoGPT" in self.model_name:
                self._initialize_dialogpt_model()
            elif "gpt2" in self.model_name:
                self._initialize_gpt2_model()
            else:
                # Fallback to GPT-2
                self.model_name = "gpt2"
                self._initialize_gpt2_model()
                
            logger.info(f"Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {e}")
            self._load_fallback_model()
    
    def _initialize_gemini(self):
        """Initialize Google Gemini"""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel(self.model_name)
            self.use_gemini = True
            self.model_type = "gemini"
            logger.info(f"Gemini {self.model_name} initialized successfully")
        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            raise
    
    def _initialize_t5_model(self):
        """Initialize T5 model for instruction following"""
        self.model_type = "t5"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.generator = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
        except Exception as e:
            logger.error(f"T5 initialization failed: {e}")
            raise
    
    def _initialize_bart_model(self):
        """Initialize BART model for generation"""
        self.model_type = "bart"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.generator = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
        except Exception as e:
            logger.error(f"BART initialization failed: {e}")
            raise
    
    def _initialize_dialogpt_model(self):
        """Initialize DialoGPT model"""
        self.model_type = "dialogpt"
        
        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
        except Exception as e:
            logger.error(f"DialoGPT initialization failed: {e}")
            raise
    
    def _initialize_gpt2_model(self):
        """Initialize GPT-2 model"""
        self.model_type = "gpt2"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
        except Exception as e:
            logger.error(f"GPT-2 initialization failed: {e}")
            raise
    
    def _load_fallback_model(self):
        """Load fallback model if primary fails"""
        try:
            logger.info("Loading fallback model: gpt2")
            self.model_name = "gpt2"
            self._initialize_gpt2_model()
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")
            self.generator = None
    
    def _create_prompt(self, query: str, context: str, language: str) -> str:
        """Create optimized prompt based on model type"""
        
        # Language-specific instructions
        language_instructions = {
            "en": "Provide detailed, helpful agricultural advice in English",
            "fr": "Fournissez des conseils agricoles détaillés et utiles en français",
            "es": "Proporcione consejos agrícolas detallados y útiles en español",
            "ar": "قدم نصائح زراعية مفصلة ومفيدة باللغة العربية",
            "de": "Geben Sie detaillierte, hilfreiche landwirtschaftliche Ratschläge auf Deutsch",
            "it": "Fornire consigli agricoli dettagliati e utili in italiano",
            "pt": "Forneça conselhos agrícolas detalhados e úteis em português",
            "zh": "用中文提供详细有用的农业建议",
            "hi": "हिंदी में विस्तृत और सहायक कृषि सलाह प्रदान करें"
        }
        
        instruction = language_instructions.get(language, "Provide detailed, helpful agricultural advice in English")
        
        if self.use_gemini:
            # Gemini works well with structured prompts
            prompt = f"""You are an expert agricultural assistant with years of experience in farming and crop management. {instruction}.

Using the provided agricultural context, give practical, detailed, and science-based advice to farmers. Be comprehensive and specific.

AGRICULTURAL CONTEXT:
{context}

FARMER'S QUESTION: {query}

Please provide a thorough, expert response:"""
            
        elif self.model_type == "t5":
            # T5 works best with clear task instructions
            prompt = f"""{instruction}: 

Context: {context}

Question: {query}

Detailed Answer:"""
            
        elif self.model_type == "bart":
            # BART for summarization and generation
            prompt = f"""{instruction}

Based on the following agricultural information:
{context}

Answer this question: {query}

Provide a comprehensive response:"""
            
        else:
            # GPT-style models
            prompt = f"""You are an expert agricultural assistant with years of experience in farming and crop management. {instruction}.

Using the provided agricultural context, give practical, detailed, and science-based advice to farmers.

Context:
{context}

Farmer's Question: {query}

Expert Agricultural Answer:"""
        
        return prompt
    
    def generate_response(self, query: str, context: str, language: str = "en") -> str:
        """
        Generate agricultural response using the best available model
        
        Args:
            query: User's question
            context: Retrieved knowledge base context
            language: Target language for response
            
        Returns:
            Generated agricultural advice
        """
        if not self.generator and not self.gemini_model:
            return "I apologize, but I'm unable to generate responses at the moment. Please try again later."
        
        try:
            prompt = self._create_prompt(query, context, language)
            
            if self.use_gemini:
                logger.info(f"Generating response with Gemini {self.model_name}")
                return self._generate_gemini_response(prompt, language)
            else:
                logger.info(f"Generating response with {self.model_name} ({self.model_type})")
                return self._generate_local_response(prompt, language)
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_fallback_response(query, context, language)
    
    def _generate_gemini_response(self, prompt: str, language: str) -> str:
        """Generate response using Gemini"""
        try:
            response = self.gemini_model.generate_content(prompt)
            generated_text = response.text
            
            # Clean up the response
            response = self._clean_response(generated_text)
            
            # Ensure we have a meaningful response
            if not response or len(response) < 10:
                response = self._generate_fallback_response("", "", language)
            
            logger.info(f"Generated {len(response)} characters response with Gemini in {language}")
            return response
            
        except Exception as e:
            logger.error(f"Gemini response generation failed: {e}")
            raise
    
    def _generate_local_response(self, prompt: str, language: str) -> str:
        """Generate response using local models"""
        
        # Adjust generation parameters based on model type
        if self.model_type in ["t5", "bart"]:
            # Seq2Seq models
            result = self.generator(
                prompt,
                max_length=300,
                min_length=50,
                temperature=0.7,
                do_sample=True,
                num_beams=3,
                early_stopping=True
            )
            response = result[0]['generated_text']
            
        else:
            # Causal LM models (GPT-style)
            if self.model_type == "dialogpt":
                result = self.generator(
                    prompt,
                    max_length=500,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.generator.tokenizer.eos_token_id
                )
            else:
                result = self.generator(
                    prompt,
                    max_new_tokens=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None
                )
            
            generated_text = result[0]['generated_text']
            
            # Clean up the response
            if prompt in generated_text:
                response = generated_text.replace(prompt, "").strip()
            else:
                response = generated_text.strip()
            
            # Remove any remaining prompt artifacts
            response = self._clean_response(response)
        
        # Ensure we have a meaningful response
        if not response or len(response) < 10:
            response = self._generate_fallback_response("", "", language)
        
        logger.info(f"Generated {len(response)} characters response in {language}")
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean response from prompt artifacts"""
        
        # Remove common prompt artifacts
        artifacts = [
            "Expert Agricultural Answer:",
            "Detailed Answer:",
            "Answer:",
            "Response:",
            "Helpful Answer:",
            "Agricultural Advice:",
            "Farmer's Question:",
            "Context:",
            "Question:",
            "Based on the following agricultural information:",
            "Using the provided agricultural context,",
            "You are an expert agricultural assistant",
            "Please provide a thorough, expert response:",
            "AGRICULTURAL CONTEXT:",
            "FARMER'S QUESTION:"
        ]
        
        for artifact in artifacts:
            if artifact in response:
                response = response.split(artifact, 1)[-1].strip()
        
        # Remove any remaining incomplete sentences
        if response.endswith(':') or response.endswith('\n'):
            response = response.rstrip(':').strip()
        
        return response
    
    def _generate_fallback_response(self, query: str, context: str, language: str) -> str:
        """Generate fallback response from context when model fails"""
        
        # Extract key information from context
        if "water" in query.lower() and "water" in context.lower():
            if language == "fr":
                return "D'après le contexte agricole, l'irrigation modérée est recommandée. Arrosez aux stades clés comme le semis, le tallage et le remplissage des grains pour obtenir de meilleurs rendements."
            elif language == "es":
                return "Según el contexto agrícola, se recomienda riego moderado. Riegue en etapas clave como la siembra, el ahijamiento y el llenado de granos para obtener mejores rendimientos."
            else:
                return "Based on agricultural context, moderate irrigation is recommended. Water at key stages like sowing, tillering, and grain filling for better yields."
        
        elif "plant" in query.lower():
            if language == "fr":
                return "Le moment de plantation dépend du climat local et de la variété. Consultez le calendrier de plantation agricole pour votre région spécifique."
            elif language == "es":
                return "El momento de la siembra depende del clima local y la variedad. Consulte el calendario de siembra agrícola para su región específica."
            else:
                return "Planting time depends on local climate and variety. Consult agricultural planting calendars for your specific region."
        
        else:
            if language == "fr":
                return "Pour des conseils agricoles spécifiques, veuillez consulter les services de vulgarisation agricole locaux ou les experts en culture."
            elif language == "es":
                return "Para consejos agrícolas específicos, consulte los servicios de extensión agrícola locales o expertos en cultivos."
            else:
                return "For specific agricultural advice, please consult local agricultural extension services or crop experts."

# Global LLM instance
_llm_instance = None

def get_llm(model_name: str = None, gemini_api_key: str = None) -> AgriculturalLLM:
    """Get or create LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = AgriculturalLLM(model_name, gemini_api_key)
    return _llm_instance

def generate_response(query: str, context: str, language: str = "en", gemini_api_key: str = None) -> str:
    """Generate response using global LLM instance"""
    llm = get_llm(gemini_api_key=gemini_api_key)
    return llm.generate_response(query, context, language)

if __name__ == "__main__":
    # Test the enhanced LLM with Gemini
    gemini_key = "AIzaSyBBQ-okicASaEm1CP9Q8rzvLfYZJMEHvIc"
    llm = AgriculturalLLM(gemini_api_key=gemini_key)
    
    test_query = "How much water does wheat need and when should I irrigate it?"
    test_context = "Wheat requires moderate irrigation. Water at sowing, tillering, and grain filling stages for best yield. Typical water requirement: 450-600mm per season."
    
    response = llm.generate_response(test_query, test_context, "en")
    print(f"Query: {test_query}")
    print(f"Model: {llm.model_name}")
    print(f"Response: {response}")
