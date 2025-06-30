# Whisper Video Analysis Service

This Python Flask service provides video transcription and AI-based topic extraction using OpenAI's Whisper model with advanced NLP analysis.

## Requirements

- Python 3.9+ (tested and compatible with Python 3.13.5)
- Sufficient disk space for PyTorch and Whisper models (~2GB)

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run compatibility test (optional):
```bash
python test_python313_compatibility.py
```

3. Run the service:
```bash
python app.py
```

The service will start on `http://localhost:5000`

## Features

- **Video Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **AI-based Topic Extraction**: Advanced NLP analysis using NLTK, TF-IDF, and domain pattern matching
- **Multiple Video Formats**: Supports mp4, avi, mov, mkv, webm, flv, m4v, wmv
- **Dynamic Topic Generation**: Extracts contextually relevant topics from actual video content

## Python 3.13.5 Compatibility

All dependencies have been updated to support Python 3.13.5:
- Flask 3.1.0+
- OpenAI Whisper 20250625+
- PyTorch 2.7.0+
- NumPy 2.3.0+
- NLTK 3.9.1+
- scikit-learn 1.7.0+
- spaCy 3.8.0+

## API Endpoints

### Health Check
- **GET** `/health` - Returns service status

### Video Analysis
- **POST** `/analyze-video` - Analyzes video file and extracts topics
  - Accepts multipart/form-data with 'file' field
  - Supported formats: mp4, avi, mov, mkv, webm, flv, m4v, wmv
  - Returns JSON with topics, duration, summary, and transcription

## Response Format

```json
{
  "topics": ["Technology", "Business"],
  "durationSeconds": 120,
  "summary": "Video transcription completed...",
  "transcription": "Full transcribed text...",
  "analysisStatus": "completed"
}
```

## Notes

- Uses Whisper 'base' model for balance of speed and accuracy
- Automatically extracts audio from video files
- Topic extraction based on keyword analysis of transcribed text
- Temporary files are automatically cleaned up after processing
