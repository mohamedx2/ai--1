import gradio as gr
import logging
import time
from typing import Tuple, Optional
from transcription import transcribe_audio
from retrieval import retrieve_context
from llm_response import generate_response
from utils import detect_language
from logging_config import setup_logging

# Setup enhanced logging
setup_logging()

# Configure module logger
logger = logging.getLogger(__name__)

# Conversation history for context
conversation_history = []

def process_input(audio: Optional[str], text: Optional[str]) -> Tuple[str, str, str]:
    """
    Process user input (audio or text) and generate response
    
    Args:
        audio: Path to audio file or None
        text: Text input or None
        
    Returns:
        Tuple of (transcript, response, processing_status)
    """
    start_time = time.time()
    request_id = f"req_{int(start_time)}"
    
    try:
        processing_status = "🔄 Processing..."
        logger.info(f"[{request_id}] Starting request processing")
        
        # Get user input from audio or text
        if audio:
            processing_status = "🎤 Transcribing audio..."
            logger.info(f"[{request_id}] Transcribing audio from: {audio}")
            transcript, detected_lang = transcribe_audio(audio)
            user_input = transcript
            logger.info(f"[{request_id}] Audio transcribed: {transcript[:100]}...")
        elif text and text.strip():
            user_input = text.strip()
            detected_lang = detect_language(user_input)
            transcript = user_input
            logger.info(f"[{request_id}] Text input received: {user_input[:100]}...")
        else:
            logger.warning(f"[{request_id}] No input provided")
            return "", "Please speak or type your question.", "❌ No input provided"
        
        if not user_input:
            logger.warning(f"[{request_id}] Empty input after processing")
            return "", "I couldn't understand your input. Please try again.", "❌ Input not understood"
        
        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_input, "language": detected_lang})
        logger.info(f"[{request_id}] Language detected: {detected_lang}")
        
        processing_status = "🔍 Searching knowledge base..."
        logger.info(f"[{request_id}] Retrieving context for query")
        context = retrieve_context(user_input, detected_lang)
        logger.info(f"[{request_id}] Context retrieved: {len(context)} characters")
        
        processing_status = "🤖 Generating response..."
        logger.info(f"[{request_id}] Generating response")
        answer = generate_response(user_input, context, detected_lang)
        logger.info(f"[{request_id}] Response generated: {len(answer)} characters")
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": answer, "language": detected_lang})
        
        # Keep only last 10 exchanges to manage memory
        if len(conversation_history) > 20:
            conversation_history[:] = conversation_history[-20:]
        
        processing_status = "✅ Complete"
        total_time = time.time() - start_time
        logger.info(f"[{request_id}] Request completed in {total_time:.2f}s")
        
        return transcript, answer, processing_status
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"[{request_id}] Processing failed after {total_time:.2f}s: {str(e)}", exc_info=True)
        error_msg = "An error occurred while processing your request. Please try again."
        return transcript if 'transcript' in locals() else "", error_msg, "❌ Error occurred"

def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    logger.info("Conversation history cleared")
    return "", "", "🗑️ Conversation cleared"

def get_language_flag(lang_code: str) -> str:
    """Get language flag emoji"""
    flags = {
        "en": "🇺🇸",
        "fr": "🇫🇷", 
        "es": "🇪🇸",
        "ar": "🇸🇦",
        "de": "🇩🇪",
        "it": "🇮🇹",
        "pt": "🇵🇹",
        "zh": "🇨🇳",
        "hi": "🇮🇳"
    }
    return flags.get(lang_code, "🌐")

# Create custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #2ecc71, #27ae60);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.input-section {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
.output-section {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
.status-indicator {
    padding: 10px;
    border-radius: 5px;
    font-weight: bold;
}
"""

# Build the Gradio interface
with gr.Blocks(css=custom_css, title="Assistant Agricole Multilingue") as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🌱 Assistant Agricole Multilingue</h1>
        <p>Your multilingual agricultural advisor - Ask questions in voice or text!</p>
        <p>🌍 Supports: English 🇺🇸 | Français 🇫🇷 | Español 🇪🇸 | العربية 🇸🇦</p>
    </div>
    """)
    
    # Status indicator
    status_display = gr.Markdown("🟢 Ready to help!", elem_classes=["status-indicator"])
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")
            
            # Input section
            with gr.Group(elem_classes=["input-section"]):
                audio_input = gr.Audio(
                    source="microphone", 
                    type="filepath", 
                    label="🎤 Click to speak (or wait for auto-stop)"
                )
                
                gr.Markdown("--- OR ---")
                
                text_input = gr.Textbox(
                    label="⌨️ Type your question here",
                    placeholder="e.g., How much water does wheat need?",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Get Advice", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            
            # Output section
            with gr.Group(elem_classes=["output-section"]):
                transcript_output = gr.Textbox(
                    label="🎤 What you said (transcript)",
                    interactive=False,
                    lines=2,
                    max_lines=4
                )
                
                answer_output = gr.Textbox(
                    label="🤖 Agricultural Advice",
                    interactive=False,
                    lines=8,
                    max_lines=15
                )
    
    # Examples section
    gr.Markdown("### 💡 Example Questions")
    with gr.Row():
        gr.Examples(
            examples=[
                ["How much water does wheat need?"],
                ["What's the best time to plant corn?"],
                ["How do I control aphids naturally?"],
                ["What soil pH is best for most crops?"],
                ["Comment irriguer le blé?"],
                ["¿Cuándo plantar maíz?"],
                ["كيفية ري القمح؟"]
            ],
            inputs=[text_input],
            label="Click to try these questions"
        )
    
    # Footer
    gr.Markdown("""
    ---
    🔊 **Tips**: 
    - Speak clearly and close to your microphone
    - The system automatically detects your language
    - Responses are based on agricultural best practices
    - Works offline once models are downloaded
    """)
    
    # Event handlers
    submit_btn.click(
        process_input,
        inputs=[audio_input, text_input],
        outputs=[transcript_output, answer_output, status_display],
        show_progress=True
    )
    
    clear_btn.click(
        clear_conversation,
        outputs=[transcript_output, answer_output, status_display]
    )
    
    # Auto-submit on audio upload
    audio_input.change(
        process_input,
        inputs=[audio_input, text_input],
        outputs=[transcript_output, answer_output, status_display],
        show_progress=True
    )
    
    # Auto-submit on text input (with small delay)
    text_input.submit(
        process_input,
        inputs=[audio_input, text_input],
        outputs=[transcript_output, answer_output, status_display],
        show_progress=True
    )

if __name__ == "__main__":
    logger.info("Starting Assistant Agricole Multilingue")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {e}", exc_info=True)
        raise
