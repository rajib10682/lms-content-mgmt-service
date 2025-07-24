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
import ssl
import urllib.request
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_ssl_context():
    """Configure SSL context to handle SSL issues in Python 3.13 and multiple HTTP libraries"""
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        ssl._create_default_https_context = lambda: ssl_context
        
        try:
            import requests
            requests.packages.urllib3.disable_warnings()
        except ImportError:
            pass
        
        try:
            import huggingface_hub
            huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY = True
            import huggingface_hub.utils
            if hasattr(huggingface_hub.utils, '_http_backoff'):
                original_session = huggingface_hub.utils._http_backoff
                def patched_session(*args, **kwargs):
                    session = original_session(*args, **kwargs)
                    session.verify = False
                    return session
                huggingface_hub.utils._http_backoff = patched_session
        except (ImportError, AttributeError):
            pass
        
        try:
            import sentence_transformers.util
            if hasattr(sentence_transformers.util, 'http_get'):
                original_http_get = sentence_transformers.util.http_get
                def patched_http_get(url, *args, **kwargs):
                    kwargs['verify'] = False
                    return original_http_get(url, *args, **kwargs)
                sentence_transformers.util.http_get = patched_http_get
        except (ImportError, AttributeError):
            pass
        
        import os
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
        
        logger.info("SSL context configured for Python 3.13 compatibility with multiple HTTP libraries including huggingface_hub")
    except Exception as e:
        logger.warning(f"Could not configure SSL context: {e}")

def safe_nltk_download(resource_name, max_retries=3):
    """Safely download NLTK resources with retry logic and SSL handling"""
    for attempt in range(max_retries):
        try:
            nltk.download(resource_name, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource_name}")
            return True
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to download {resource_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to download {resource_name} after {max_retries} attempts")
                return False

def safe_whisper_load(model_name="base", max_retries=3):
    """Safely load Whisper model with retry logic"""
    for attempt in range(max_retries):
        try:
            model = whisper.load_model(model_name)
            logger.info(f"Successfully loaded Whisper model: {model_name}")
            return model
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to load Whisper model: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to load Whisper model after {max_retries} attempts")
                raise

def safe_embedding_models_load():
    """Safely load embedding models with error handling"""
    models = {}
    
    try:
        from sentence_transformers import SentenceTransformer
        models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded SentenceTransformer model")
    except Exception as e:
        logger.warning(f"Failed to load SentenceTransformer: {e}")
        models['sentence_transformer'] = None
    
    try:
        from keybert import KeyBERT
        models['keybert'] = KeyBERT()
        logger.info("Successfully loaded KeyBERT model")
    except Exception as e:
        logger.warning(f"Failed to load KeyBERT: {e}")
        models['keybert'] = None
    
    try:
        from bertopic import BERTopic
        models['bertopic'] = BERTopic()
        logger.info("Successfully loaded BERTopic model")
    except Exception as e:
        logger.warning(f"Failed to load BERTopic (this is optional): {e}")
        logger.info("BERTopic requires hdbscan which needs build tools. See INSTALLATION.md for alternatives.")
        models['bertopic'] = None
    
    return models

configure_ssl_context()

nltk_resources = [
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
    ('corpora/stopwords', 'stopwords'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
]

for resource_path, resource_name in nltk_resources:
    try:
        nltk.data.find(resource_path)
        logger.info(f"NLTK resource {resource_name} already available")
    except LookupError:
        logger.info(f"Downloading NLTK resource: {resource_name}")
        safe_nltk_download(resource_name)

try:
    model = safe_whisper_load("base")
except Exception as e:
    logger.error(f"Critical error: Could not load Whisper model: {e}")
    model = None

embedding_models = safe_embedding_models_load()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_topics_with_keybert(text, model):
    """Extract topics using KeyBERT"""
    try:
        keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), 
                                        stop_words='english')
        topics = [keyword.title() for keyword, score in keywords[:5]]
        return topics if topics else []
    except Exception as e:
        logger.warning(f"KeyBERT topic extraction failed: {e}")
        return []

def extract_topics_with_bertopic(text, model):
    """Extract topics using BERTopic"""
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return []
        
        topics, probs = model.fit_transform(sentences)
        topic_info = model.get_topic_info()
        
        valid_topics = topic_info[topic_info.Topic != -1].head(5)
        topic_names = []
        for _, row in valid_topics.iterrows():
            topic_words = model.get_topic(row.Topic)[:3]  # Top 3 words
            topic_name = ' '.join([word for word, _ in topic_words]).title()
            topic_names.append(topic_name)
        
        return topic_names if topic_names else []
    except Exception as e:
        logger.warning(f"BERTopic topic extraction failed: {e}")
        return []

def extract_topics_with_openai_embeddings(text):
    """Extract topics using OpenAI embeddings (if API key available)"""
    try:
        import openai
        import os
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.info("OpenAI API key not found, skipping OpenAI embeddings")
            return []
        
        try:
            import httpx
            client = openai.OpenAI(
                api_key=api_key,
                http_client=httpx.Client(verify=False)
            )
        except ImportError:
            client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract 3-5 main topics from the following text. Return only topic names separated by commas, no explanations."},
                {"role": "user", "content": text[:2000]}  # Limit text length
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        topics_text = response.choices[0].message.content.strip()
        topics = [topic.strip().title() for topic in topics_text.split(',')]
        return topics[:5] if topics else []
        
    except Exception as e:
        logger.warning(f"OpenAI embeddings topic extraction failed: {e}")
        return []

def extract_topics_from_text(text):
    """Extract relevant topics from transcribed text using advanced embedding methods with fallbacks"""
    if not text or len(text.strip()) < 10:
        return []
    
    logger.info("Starting topic extraction with embedding methods")
    
    
    topics = extract_topics_with_openai_embeddings(text)
    if topics:
        logger.info(f"Successfully extracted topics using OpenAI: {topics}")
        return topics
    
    if embedding_models.get('keybert'):
        topics = extract_topics_with_keybert(text, embedding_models['keybert'])
        if topics:
            logger.info(f"Successfully extracted topics using KeyBERT: {topics}")
            return topics
    
    if embedding_models.get('bertopic') and len(text.split()) > 50:
        topics = extract_topics_with_bertopic(text, embedding_models['bertopic'])
        if topics:
            logger.info(f"Successfully extracted topics using BERTopic: {topics}")
            return topics
    
    logger.info("Falling back to NLTK/TF-IDF topic extraction")
    return extract_topics_from_text_nltk(text)

def extract_topics_from_text_nltk(text):
    """Original NLTK/TF-IDF topic extraction method (renamed for clarity)"""
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
    global model
    status = {
        'status': 'healthy' if model is not None else 'degraded',
        'service': 'whisper-analyzer',
        'whisper_model': 'loaded' if model is not None else 'failed_to_load',
        'ssl_configured': True,
        'ssl_verification_disabled': True
    }
    
    nltk_status = {}
    for resource_path, resource_name in [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'pos_tagger')
    ]:
        try:
            nltk.data.find(resource_path)
            nltk_status[resource_name] = 'available'
        except LookupError:
            nltk_status[resource_name] = 'missing'
    
    status['nltk_resources'] = nltk_status
    
    embedding_status = {}
    for model_name, embedding_model in embedding_models.items():
        embedding_status[model_name] = 'loaded' if embedding_model is not None else 'failed_to_load'
    status['embedding_models'] = embedding_status
    
    return jsonify(status)

@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    global model
    try:
        if model is None:
            return jsonify({'error': 'Whisper model not available. Please check server logs for SSL/network issues.'}), 503
        
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
            video_clip.audio.write_audiofile(audio_temp_path, logger=None)
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
