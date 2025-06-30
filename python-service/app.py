from flask import Flask, request, jsonify
import whisper
import tempfile
import os
import logging
from werkzeug.utils import secure_filename
import re
from collections import Counter

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = whisper.load_model("base")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_topics_from_text(text):
    """Extract relevant topics from transcribed text using keyword analysis"""
    if not text:
        return []
    
    topic_keywords = {
        'Technology': ['software', 'computer', 'digital', 'technology', 'programming', 'coding', 'development', 'app', 'website', 'data', 'algorithm', 'artificial intelligence', 'machine learning', 'cloud'],
        'Business': ['business', 'company', 'market', 'sales', 'revenue', 'profit', 'strategy', 'management', 'leadership', 'team', 'organization', 'corporate', 'enterprise'],
        'Education': ['education', 'learning', 'teaching', 'student', 'course', 'training', 'knowledge', 'skill', 'university', 'school', 'academic', 'study'],
        'Finance': ['finance', 'money', 'investment', 'budget', 'cost', 'financial', 'accounting', 'banking', 'economy', 'economic', 'capital', 'funding'],
        'Marketing': ['marketing', 'advertising', 'brand', 'customer', 'promotion', 'campaign', 'social media', 'content', 'engagement', 'audience'],
        'Project Management': ['project', 'planning', 'timeline', 'deadline', 'milestone', 'task', 'workflow', 'process', 'methodology', 'agile', 'scrum'],
        'Communication': ['communication', 'meeting', 'presentation', 'discussion', 'collaboration', 'feedback', 'report', 'update', 'announcement'],
        'Health': ['health', 'medical', 'healthcare', 'wellness', 'fitness', 'nutrition', 'therapy', 'treatment', 'patient', 'doctor'],
        'Science': ['science', 'research', 'experiment', 'analysis', 'hypothesis', 'theory', 'discovery', 'innovation', 'scientific'],
        'Entertainment': ['entertainment', 'movie', 'music', 'game', 'sport', 'fun', 'leisure', 'hobby', 'creative', 'art']
    }
    
    text_lower = text.lower()
    topic_scores = {}
    
    for topic, keywords in topic_keywords.items():
        score = 0
        for keyword in keywords:
            score += text_lower.count(keyword)
        if score > 0:
            topic_scores[topic] = score
    
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    return [topic for topic, score in sorted_topics[:3]]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'whisper-analyzer'})

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 400
        
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, filename)
        file.save(temp_file_path)
        
        logger.info(f"Processing video file: {filename}")
        
        try:
            result = model.transcribe(temp_file_path)
            transcribed_text = result["text"]
            
            topics = extract_topics_from_text(transcribed_text)
            
            duration_seconds = int(result.get("duration", 0))
            
            summary = f"Video transcription completed. Extracted {len(topics)} relevant topics from {duration_seconds} seconds of content."
            if transcribed_text:
                text_preview = transcribed_text[:200] + "..." if len(transcribed_text) > 200 else transcribed_text
                summary += f" Content preview: {text_preview}"
            
            response = {
                'topics': topics if topics else ['General Content'],
                'durationSeconds': duration_seconds,
                'summary': summary,
                'transcription': transcribed_text,
                'analysisStatus': 'completed'
            }
            
            logger.info(f"Analysis completed for {filename}: {len(topics)} topics, {duration_seconds}s duration")
            return jsonify(response)
            
        finally:
            try:
                os.remove(temp_file_path)
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
    
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
