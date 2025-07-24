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
import subprocess
import shutil
import wave
import math

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

def chunk_large_audio_file(audio_path, chunk_duration_seconds=30, max_file_size_mb=5):
    """
    Chunk large audio files into smaller segments for Whisper processing
    
    Args:
        audio_path: Path to the audio file
        chunk_duration_seconds: Duration of each chunk in seconds (default: 5 minutes)
        max_file_size_mb: Maximum file size in MB before chunking (default: 100MB)
    
    Returns:
        List of chunk file paths, or [audio_path] if no chunking needed
    """
    try:
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Audio file size: {file_size_mb:.1f}MB")
        
        if file_size_mb <= max_file_size_mb:
            logger.info(f"File size ({file_size_mb:.1f}MB) is within limit ({max_file_size_mb}MB), no chunking needed")
            return [audio_path]
        
        logger.info(f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_file_size_mb}MB), chunking into {chunk_duration_seconds}s segments")
        
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(f"Could not get audio duration: {result.stderr}")
            return [audio_path]  # Fallback to original file
        
        total_duration = float(result.stdout.strip())
        logger.info(f"Total audio duration: {total_duration:.1f} seconds")
        
        num_chunks = math.ceil(total_duration / chunk_duration_seconds)
        logger.info(f"Creating {num_chunks} chunks of {chunk_duration_seconds}s each")
        
        chunk_paths = []
        temp_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        for i in range(num_chunks):
            start_time = i * chunk_duration_seconds
            chunk_path = os.path.join(temp_dir, f"{base_name}_chunk_{i+1:03d}.wav")
            
            chunk_cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start_time),
                "-t", str(chunk_duration_seconds),
                "-c", "copy",
                chunk_path
            ]
            
            logger.info(f"Creating chunk {i+1}/{num_chunks}: {start_time}s-{start_time + chunk_duration_seconds}s")
            
            chunk_result = subprocess.run(chunk_cmd, capture_output=True, timeout=120)
            if chunk_result.returncode == 0:
                chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
                logger.info(f"Chunk {i+1} created: {chunk_size_mb:.1f}MB")
                chunk_paths.append(chunk_path)
            else:
                logger.warning(f"Failed to create chunk {i+1}: {chunk_result.stderr.decode()}")
        
        if chunk_paths:
            logger.info(f"Successfully created {len(chunk_paths)} audio chunks")
            return chunk_paths
        else:
            logger.warning("No chunks created, falling back to original file")
            return [audio_path]
            
    except Exception as e:
        logger.error(f"Error chunking audio file: {e}")
        return [audio_path]  # Fallback to original file

def transcribe_with_chunking(model, audio_path, max_file_size_mb=100):
    """
    Transcribe audio with automatic chunking for large files
    
    Args:
        model: Whisper model instance
        audio_path: Path to audio file
        max_file_size_mb: Maximum file size before chunking
    
    Returns:
        Combined transcription result
    """
    try:
        chunk_paths = chunk_large_audio_file(audio_path, max_file_size_mb=max_file_size_mb)
        
        if len(chunk_paths) == 1:
            logger.info("Transcribing single file (no chunking)")
            return model.transcribe(chunk_paths[0])
        
        logger.info(f"Transcribing {len(chunk_paths)} chunks")
        combined_text = ""
        total_duration = 0
        chunk_results = []
        
        for i, chunk_path in enumerate(chunk_paths):
            try:
                logger.info(f"Transcribing chunk {i+1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}")
                
                chunk_result = model.transcribe(chunk_path)
                
                chunk_text = chunk_result.get("text", "").strip()
                chunk_duration = chunk_result.get("duration", 0)
                
                if chunk_text:
                    combined_text += chunk_text + " "
                    
                total_duration += chunk_duration
                chunk_results.append(chunk_result)
                
                logger.info(f"Chunk {i+1} completed: {len(chunk_text)} characters, {chunk_duration:.1f}s")
                
            except Exception as chunk_error:
                logger.warning(f"Failed to transcribe chunk {i+1}: {chunk_error}")
                continue
        
        for chunk_path in chunk_paths:
            if chunk_path != audio_path:  # Don't delete the original file
                try:
                    os.unlink(chunk_path)
                    logger.info(f"Cleaned up chunk: {os.path.basename(chunk_path)}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up chunk {chunk_path}: {cleanup_error}")
        
        combined_result = {
            "text": combined_text.strip(),
            "duration": total_duration,
            "language": chunk_results[0].get("language", "en") if chunk_results else "en"
        }
        
        logger.info(f"Combined transcription completed: {len(combined_text)} characters, {total_duration:.1f}s total duration")
        return combined_result
        
    except Exception as e:
        logger.error(f"Error in chunked transcription: {e}")
        logger.info("Falling back to direct transcription")
        return model.transcribe(audio_path)

def safe_embedding_models_load():
    """Safely load embedding models with enhanced SSL configuration"""
    models = {}
    
    import os
    import ssl
    import urllib3
    
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl._create_default_https_context = lambda: ssl_context
    except Exception as ssl_error:
        logger.warning(f"Could not configure SSL context for embedding models: {ssl_error}")
    
    try:
        from sentence_transformers import SentenceTransformer
        import requests
        requests.packages.urllib3.disable_warnings()
        models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2', trust_remote_code=True)
        logger.info("Successfully loaded SentenceTransformer model")
    except Exception as e:
        logger.warning(f"Failed to load SentenceTransformer: {e}")
        models['sentence_transformer'] = None
    
    try:
        from keybert import KeyBERT
        import requests
        import urllib3
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        original_get = requests.get
        original_post = requests.post
        
        def patched_get(*args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = kwargs.get('timeout', 30)
            return original_get(*args, **kwargs)
        
        def patched_post(*args, **kwargs):
            kwargs['verify'] = False
            kwargs['timeout'] = kwargs.get('timeout', 30)
            return original_post(*args, **kwargs)
        
        requests.get = patched_get
        requests.post = patched_post
        
        # Initialize KeyBERT with SSL-disabled sentence transformer
        models['keybert'] = KeyBERT()
        
        requests.get = original_get
        requests.post = original_post
        
        logger.info("Successfully loaded KeyBERT model with SSL bypass")
    except Exception as e:
        logger.warning(f"Failed to load KeyBERT: {e}")
        models['keybert'] = None
    
    try:
        from bertopic import BERTopic
        import torch
        torch.hub.set_dir('/tmp/torch_cache')  # Use temp directory for model cache
        models['bertopic'] = BERTopic(verbose=True)
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

def extract_topics_with_keybert(text, model, encoded_data=None):
    """Extract topics using KeyBERT with optional pre-computed sentence transformer embeddings"""
    try:
        if encoded_data and encoded_data.get('embeddings') is not None:
            logger.info("Using pre-computed sentence transformer embeddings for KeyBERT")
            keywords = model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english',
                doc_embeddings=encoded_data['embeddings']
            )
        else:
            logger.info("Using text-based approach for KeyBERT (no pre-computed embeddings)")
            keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 2), 
                                            stop_words='english')
        
        topics = [keyword.title() for keyword, score in keywords[:5]]
        return topics if topics else []
    except Exception as e:
        logger.warning(f"KeyBERT topic extraction failed: {e}")
        return []

def extract_topics_with_bertopic(text, model, encoded_data=None):
    """Extract topics using BERTopic with optional pre-computed sentence transformer embeddings"""
    try:
        from nltk.tokenize import sent_tokenize
        
        if encoded_data and encoded_data.get('embeddings') is not None:
            logger.info("Using pre-computed sentence transformer embeddings for BERTopic")
            sentences = encoded_data['sentences']
            embeddings = encoded_data['embeddings']
            
            if len(sentences) < 2:
                return []
            
            topics, probs = model.fit_transform(sentences, embeddings=embeddings)
        else:
            logger.info("Using text-based approach for BERTopic (no pre-computed embeddings)")
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

def extract_topics_with_openai_embeddings(text, encoded_data=None):
    """Extract topics using OpenAI embeddings with optional sentence transformer preprocessing"""
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
        
        if encoded_data and encoded_data.get('sentences'):
            logger.info("Using sentence transformer preprocessing for OpenAI topic extraction")
            sentences = encoded_data['sentences'][:10]  # Top 10 sentences
            processed_text = ' '.join(sentences)[:2000]  # Limit text length
        else:
            logger.info("Using original text for OpenAI topic extraction")
            processed_text = text[:2000]  # Limit text length
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract 3-5 main topics from the following text. Return only topic names separated by commas, no explanations."},
                {"role": "user", "content": processed_text}
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

def encode_text_with_sentence_transformers(text):
    """Encode text using sentence transformers to create embeddings"""
    try:
        if not embedding_models.get('sentence_transformer'):
            logger.warning("SentenceTransformer model not available for encoding")
            return None
        
        model = embedding_models['sentence_transformer']
        
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        
        if len(sentences) > 50:
            sentences = sentences[:50]
            logger.info(f"Limited encoding to first 50 sentences out of {len(sent_tokenize(text))}")
        
        embeddings = model.encode(sentences, convert_to_tensor=False)
        
        logger.info(f"Successfully encoded {len(sentences)} sentences using SentenceTransformer")
        logger.info(f"Embedding shape: {embeddings.shape if hasattr(embeddings, 'shape') else f'{len(embeddings)} embeddings'}")
        
        return {
            'sentences': sentences,
            'embeddings': embeddings,
            'model_name': 'all-MiniLM-L6-v2'
        }
        
    except Exception as e:
        logger.warning(f"Failed to encode text with SentenceTransformer: {e}")
        return None

def extract_topics_with_sentence_transformer_clustering(encoded_data):
    """Extract topics using sentence transformer embeddings with clustering"""
    try:
        if not encoded_data or not encoded_data.get('embeddings') is not None:
            return []
        
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import Counter
        import re
        
        sentences = encoded_data['sentences']
        embeddings = np.array(encoded_data['embeddings'])
        
        n_sentences = len(sentences)
        n_clusters = min(max(3, n_sentences // 10), 7)
        
        logger.info(f"Clustering {n_sentences} sentences into {n_clusters} topic clusters")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        topics = []
        for cluster_id in range(n_clusters):
            cluster_sentences = [sentences[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if not cluster_sentences:
                continue
            
            cluster_text = ' '.join(cluster_sentences)
            
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform([cluster_text])
                feature_names = vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                top_indices = np.argsort(tfidf_scores)[-3:][::-1]  # Top 3 terms
                cluster_terms = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                
                if cluster_terms:
                    topic_name = ' '.join(cluster_terms).title()
                    topics.append(topic_name)
                    
            except Exception as cluster_error:
                logger.warning(f"Failed to extract terms for cluster {cluster_id}: {cluster_error}")
                continue
        
        logger.info(f"Extracted {len(topics)} topics using SentenceTransformer clustering: {topics}")
        return topics[:5]  # Return top 5 topics
        
    except Exception as e:
        logger.warning(f"SentenceTransformer clustering topic extraction failed: {e}")
        return []

def extract_topics_from_text(text):
    """Extract relevant topics from transcribed text using sentence transformers encoding as preprocessing for all methods"""
    if not text or len(text.strip()) < 10:
        return []
    
    logger.info("Starting topic extraction with SentenceTransformer encoding preprocessing")
    
    encoded_data = encode_text_with_sentence_transformers(text)
    
    if encoded_data:
        topics = extract_topics_with_sentence_transformer_clustering(encoded_data)
        if topics:
            logger.info(f"Successfully extracted topics using SentenceTransformer clustering: {topics}")
            return topics
    
    topics = extract_topics_with_openai_embeddings(text, encoded_data)
    if topics:
        logger.info(f"Successfully extracted topics using OpenAI with SentenceTransformer preprocessing: {topics}")
        return topics
    
    if embedding_models.get('keybert'):
        topics = extract_topics_with_keybert(text, embedding_models['keybert'], encoded_data)
        if topics:
            logger.info(f"Successfully extracted topics using KeyBERT with SentenceTransformer embeddings: {topics}")
            return topics
    
    if embedding_models.get('bertopic') and len(text.split()) > 50:
        topics = extract_topics_with_bertopic(text, embedding_models['bertopic'], encoded_data)
        if topics:
            logger.info(f"Successfully extracted topics using BERTopic with SentenceTransformer embeddings: {topics}")
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
        
        if os.name == 'nt':  # Windows
            storage_dir = r"C:\Ankit"
        else:  # Linux/Unix - create equivalent directory structure
            storage_dir = os.path.join(os.getcwd(), "Ankit")
        
        os.makedirs(storage_dir, exist_ok=True)
        logger.info(f"Using storage directory: {storage_dir} (OS: {os.name})")
        
        temp_file_path = os.path.join(storage_dir, filename)
        file.save(temp_file_path)
        
        logger.info(f"Processing video file: {filename}")
        
        try:
            logger.info("Extracting audio from video using moviepy...")
            logger.info(f"Video file path: {temp_file_path}")
            logger.info(f"Video file exists: {os.path.exists(temp_file_path)}")
            logger.info(f"Video file size: {os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else 'N/A'} bytes")
            
            try:
                video_clip = VideoFileClip(temp_file_path)
                logger.info("VideoFileClip loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load video file with VideoFileClip: {e}")
                return jsonify({'error': f'Failed to load video file: {str(e)}'}), 400
            
            if video_clip.audio is None:
                video_clip.close()
                return jsonify({'error': 'Video file does not contain an audio track'}), 400
            
            audio_temp_path = os.path.join(storage_dir, f"audio_{filename.rsplit('.', 1)[0]}.wav")
            logger.info(f"Audio file path: {audio_temp_path}")
            logger.info(f"Audio directory exists: {os.path.exists(os.path.dirname(audio_temp_path))}")
            
            try:
                video_clip.audio.write_audiofile(audio_temp_path, logger=None)
                video_clip.close()
                logger.info("Audio extraction completed")
            except Exception as e:
                video_clip.close()
                logger.error(f"Failed to extract audio with write_audiofile: {e}")
                return jsonify({'error': f'Failed to extract audio: {str(e)}'}), 400
            
            logger.info(f"Audio file exists: {os.path.exists(audio_temp_path)}")
            logger.info(f"Audio file size: {os.path.getsize(audio_temp_path) if os.path.exists(audio_temp_path) else 'N/A'} bytes")
            
            if not os.path.exists(audio_temp_path):
                logger.error(f"Audio file was not created: {audio_temp_path}")
                return jsonify({'error': 'Audio extraction failed - file not created'}), 500
            
            logger.info("Starting transcription with whisper...")
            logger.info(f"Whisper model type: {type(model)}")
            logger.info(f"Audio file path: {audio_temp_path}")
            logger.info(f"Audio file exists: {os.path.exists(audio_temp_path)}")
            logger.info(f"Audio file size: {os.path.getsize(audio_temp_path) if os.path.exists(audio_temp_path) else 'N/A'} bytes")
            
            try:
                import stat
                file_stat = os.stat(audio_temp_path)
                logger.info(f"File permissions: {stat.filemode(file_stat.st_mode)}")
                logger.info(f"File owner: {file_stat.st_uid}")
                logger.info(f"File group: {file_stat.st_gid}")
            except Exception as e:
                logger.warning(f"Could not get file stats: {e}")
            
            try:
                with open(audio_temp_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
                logger.info("File is readable by Python")
            except Exception as e:
                logger.error(f"File is not readable by Python: {e}")
                return jsonify({'error': f'Audio file is not readable: {str(e)}'}), 500
            
            logger.info("Starting Whisper transcription with chunking support for large files")
            logger.info(f"Audio file: {audio_temp_path}")
            logger.info(f"File exists: {os.path.exists(audio_temp_path)}")
            logger.info(f"File size: {os.path.getsize(audio_temp_path) if os.path.exists(audio_temp_path) else 'N/A'} bytes")
            
            try:
                result = transcribe_with_chunking(model, audio_temp_path, max_file_size_mb=5)
                transcribed_text = result["text"]
                logger.info(f"Transcription completed successfully: {len(transcribed_text)} characters")
                
            except Exception as e:
                logger.error(f"Chunked transcription failed: {e}")
                
                transcription_attempts = []
                transcription_success = False
                
                try:
                    logger.info("Fallback 1: Direct path transcription")
                    result = model.transcribe(audio_temp_path)
                    transcribed_text = result["text"]
                    logger.info("Direct path transcription completed successfully")
                    transcription_success = True
                except Exception as e1:
                    transcription_attempts.append(f"Direct path failed: {str(e1)}")
                    logger.warning(f"Direct path transcription failed: {e1}")
                    
                    try:
                        logger.info("Fallback 2: Copy to system temp directory")
                        import tempfile
                        import shutil
                        
                        temp_dir = tempfile.gettempdir()
                        simple_temp_path = os.path.join(temp_dir, f"whisper_audio_{os.getpid()}.wav")
                        
                        logger.info(f"Copying {os.path.getsize(audio_temp_path)} bytes to: {simple_temp_path}")
                        
                        with open(audio_temp_path, 'rb') as src, open(simple_temp_path, 'wb') as dst:
                            shutil.copyfileobj(src, dst, 1024*1024)
                        
                        logger.info(f"Copy completed. Temp file size: {os.path.getsize(simple_temp_path)} bytes")
                        
                        result = model.transcribe(simple_temp_path)
                        transcribed_text = result["text"]
                        logger.info("Temp file transcription completed successfully")
                        transcription_success = True
                        
                        try:
                            os.unlink(simple_temp_path)
                            logger.info("Temp file cleaned up successfully")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temp file: {cleanup_error}")
                            
                    except Exception as e2:
                        transcription_attempts.append(f"Temp file copy failed: {str(e2)}")
                        logger.error(f"Temp file copy transcription failed: {e2}")
                
                if not transcription_success:
                    logger.error(f"All transcription attempts failed for file: {audio_temp_path}")
                    logger.error(f"File exists: {os.path.exists(audio_temp_path)}")
                    logger.error(f"File size: {os.path.getsize(audio_temp_path) if os.path.exists(audio_temp_path) else 'N/A'} bytes")
                    logger.error(f"Attempts made: {'; '.join(transcription_attempts)}")
                    return jsonify({'error': f'Transcription failed after multiple attempts: {"; ".join(transcription_attempts)}'}), 500
            
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
                logger.info(f"Video and audio files stored in: {storage_dir}")
                logger.info(f"Video file: {temp_file_path}")
                if 'audio_temp_path' in locals():
                    logger.info(f"Audio file: {audio_temp_path}")
            except Exception as e:
                logger.warning(f"Failed to log file locations: {e}")
    
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
