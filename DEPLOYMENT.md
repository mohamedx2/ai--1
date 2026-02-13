# Deployment Instructions

## 1. Install Python (3.10+ recommended)

## 2. Install dependencies
```
pip install -r requirements.txt
```

## 3. Download Whisper and LLM models (first run will auto-download, or download manually for offline use)

## 4. Run the application
```
python app.py
```

## 5. Access the interface
Open your browser to the local Gradio link (usually http://127.0.0.1:7860)

## Notes
- All data and models are local; no internet required after initial setup.
- To update the knowledge base, edit `knowledge_base/agri_knowledge.json` or add new data.
- For more languages, add more entries to the knowledge base and ensure Whisper/LLM support.
