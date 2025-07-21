from flask import Flask, request, jsonify
import whisper
import tempfile
import os
import logging
from werkzeug.utils import secure_filename
import re
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from moviepy import VideoFileClip

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

model = whisper.load_model("base")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_topics_from_text(text):
    """Extract relevant topics from transcribed text using AI-based NLP analysis"""
    if not text or len(text.strip()) < 10:
        return []
    
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk import pos_tag
        
        stop_words = set(stopwords.words('english'))
        
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return extract_keywords_from_short_text(text)
        
        words = word_tokenize(text.lower())
        
        important_words = []
        pos_tags = pos_tag(words)
        
        for word, pos in pos_tags:
            if (word not in stop_words and 
                len(word) > 2 and 
                word.isalpha() and
                pos.startswith(('NN', 'JJ', 'VB'))):  # Nouns, adjectives, verbs
                important_words.append(word)
        
        if len(important_words) < 3:
            return extract_keywords_from_short_text(text)
        
        chunk_size = max(50, len(words) // 5)  # Adaptive chunk size
        text_chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 10:
                text_chunks.append(chunk)
        
        if len(text_chunks) < 2:
            text_chunks = sentences[:min(5, len(sentences))]
        
        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words='english',
            ngram_range=(1, 2),  # Include both single words and bigrams
            min_df=1,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(text_chunks)
            feature_names = vectorizer.get_feature_names_out()
            
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            top_indices = np.argsort(mean_scores)[::-1][:10]
            top_terms = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            
            topics = generate_topics_from_terms(top_terms, text)
            
            return topics[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.warning(f"TF-IDF analysis failed: {e}, falling back to keyword extraction")
            return extract_keywords_from_short_text(text)
            
    except Exception as e:
        logger.error(f"Topic extraction failed: {e}")
        return extract_keywords_from_short_text(text)

def extract_keywords_from_short_text(text):
    """Fallback method for short texts or when NLP fails"""
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        
        keywords = [word for word in words 
                   if word not in stop_words 
                   and len(word) > 3 
                   and word.isalpha()]
        
        word_freq = Counter(keywords)
        common_words = [word for word, count in word_freq.most_common(10)]
        
        topics = []
        for word in common_words[:5]:
            topics.append(word.capitalize())
        
        return topics if topics else ['General Content']
        
    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return ['General Content']

def generate_topics_from_terms(terms, original_text):
    """Convert extracted terms into meaningful topic labels"""
    topics = []
    
    domain_patterns = {
        'Technology & Software': ['software', 'technology', 'digital', 'computer', 'programming', 'code', 'development', 'app', 'system', 'data', 'algorithm', 'ai', 'machine learning', 'cloud', 'platform'],
        'Business & Management': ['business', 'company', 'management', 'strategy', 'market', 'sales', 'revenue', 'team', 'organization', 'leadership', 'corporate', 'enterprise', 'customer'],
        'Education & Learning': ['education', 'learning', 'teaching', 'training', 'course', 'student', 'knowledge', 'skill', 'university', 'school', 'academic', 'study', 'research'],
        'Finance & Economics': ['finance', 'financial', 'money', 'investment', 'budget', 'cost', 'economic', 'banking', 'capital', 'funding', 'profit', 'revenue'],
        'Health & Wellness': ['health', 'medical', 'healthcare', 'wellness', 'fitness', 'nutrition', 'treatment', 'patient', 'doctor', 'therapy', 'medicine'],
        'Communication & Media': ['communication', 'media', 'presentation', 'discussion', 'meeting', 'social', 'content', 'message', 'information', 'news'],
        'Project & Process': ['project', 'process', 'planning', 'workflow', 'methodology', 'task', 'timeline', 'milestone', 'agile', 'management'],
        'Science & Research': ['science', 'research', 'experiment', 'analysis', 'study', 'discovery', 'innovation', 'scientific', 'theory', 'hypothesis']
    }
    
    text_lower = original_text.lower()
    
    for domain, keywords in domain_patterns.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score >= 2:  # At least 2 domain keywords found
            topics.append(domain)
    
    for term in terms[:3]:
        if len(term) > 2 and term not in [t.lower() for t in topics]:
            clean_term = ' '.join(word.capitalize() for word in term.split())
            topics.append(clean_term)
    
    seen = set()
    unique_topics = []
    for topic in topics:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            unique_topics.append(topic)
    
    return unique_topics[:5] if unique_topics else ['General Content']

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
            logger.info("Extracting audio from video using moviepy...")
            video_clip = VideoFileClip(temp_file_path)
            
            if video_clip.audio is None:
                video_clip.close()
                return jsonify({'error': 'Video file does not contain an audio track'}), 400
            
            audio_temp_path = os.path.join(temp_dir, f"audio_{filename.rsplit('.', 1)[0]}.wav")
            video_clip.audio.write_audiofile(audio_temp_path, verbose=False, logger=None)
            video_clip.close()
            
            logger.info("Audio extraction completed, starting transcription with whisper...")
            
            result = model.transcribe(audio_temp_path)
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
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                
                audio_temp_path = os.path.join(temp_dir, f"audio_{filename.rsplit('.', 1)[0]}.wav")
                if os.path.exists(audio_temp_path):
                    os.remove(audio_temp_path)
                
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")
    
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
