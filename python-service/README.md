# Whisper Video Analysis Service

This Python Flask service provides video transcription and topic extraction using OpenAI's Whisper model.

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
python app.py
```

The service will start on `http://localhost:5000`

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
