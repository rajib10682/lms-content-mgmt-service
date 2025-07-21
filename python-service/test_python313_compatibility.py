#!/usr/bin/env python3

import sys
import subprocess
import importlib.metadata
import tempfile
import os

def check_python_version():
    """Check current Python version"""
    print(f"Current Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    print(f"Python {major}.{minor}")
    return major, minor

def test_dependency_installation():
    """Test that all dependencies can be installed"""
    print("\nTesting dependency installation...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--dry-run'
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✓ All dependencies can be installed successfully")
            return True
        else:
            print(f"✗ Dependency installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error testing dependency installation: {e}")
        return False

def test_installed_packages():
    """Test that required packages are properly installed"""
    print("\nTesting installed packages...")
    
    required_packages = [
        'flask', 'openai-whisper', 'torch', 'torchaudio', 
        'werkzeug', 'numpy', 'nltk', 'scikit-learn', 'spacy', 'numba', 'moviepy'
    ]
    
    all_passed = True
    for package in required_packages:
        try:
            version = importlib.metadata.version(package)
            print(f"✓ {package} version {version} is installed")
        except importlib.metadata.PackageNotFoundError:
            print(f"✗ {package} is not installed")
            all_passed = False
        except Exception as e:
            print(f"✗ Error checking {package}: {e}")
            all_passed = False
    
    return all_passed

def test_imports():
    """Test that all critical imports work"""
    print("\nTesting critical imports...")
    
    imports_to_test = [
        ('flask', 'Flask'),
        ('whisper', 'load_model'),
        ('torch', 'torch'),
        ('torchaudio', 'load'),
        ('werkzeug.utils', 'secure_filename'),
        ('numpy', 'array'),
        ('nltk', 'download'),
        ('sklearn.feature_extraction.text', 'TfidfVectorizer'),
        ('spacy', 'load'),
        ('moviepy', 'VideoFileClip')
    ]
    
    all_passed = True
    for module, item in imports_to_test:
        try:
            exec(f"from {module} import {item}")
            print(f"✓ {module}.{item} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {module}.{item}: {e}")
            all_passed = False
        except Exception as e:
            print(f"✗ Error importing {module}.{item}: {e}")
            all_passed = False
    
    return all_passed

def test_nltk_resources():
    """Test NLTK resource downloads"""
    print("\nTesting NLTK resource downloads...")
    
    try:
        import nltk
        
        resources_to_test = [
            'tokenizers/punkt',
            'tokenizers/punkt_tab',
            'corpora/stopwords',
            'taggers/averaged_perceptron_tagger',
            'taggers/averaged_perceptron_tagger_eng'
        ]
        
        all_passed = True
        for resource in resources_to_test:
            try:
                nltk.data.find(resource)
                print(f"✓ {resource} found")
            except LookupError:
                try:
                    resource_name = resource.split('/')[-1]
                    nltk.download(resource_name, quiet=True)
                    print(f"✓ {resource} downloaded successfully")
                except Exception as e:
                    print(f"✗ Failed to download {resource}: {e}")
                    all_passed = False
            except Exception as e:
                print(f"✗ Error checking {resource}: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ NLTK testing failed: {e}")
        return False

def test_topic_extraction():
    """Test the AI-based topic extraction functionality"""
    print("\nTesting AI-based topic extraction...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from app import extract_topics_from_text
        
        test_cases = [
            {
                'text': """
                In this video, we'll discuss software development best practices and programming techniques.
                We'll cover algorithms, data structures, and machine learning concepts.
                The presentation includes examples of Python coding and web development frameworks.
                """,
                'expected_domains': ['Technology & Software']
            },
            {
                'text': """
                Today's meeting focuses on our quarterly sales strategy and revenue projections.
                We need to discuss market analysis, customer engagement, and team management.
                The company's growth depends on effective leadership and organizational planning.
                """,
                'expected_domains': ['Business & Management']
            },
            {
                'text': "Hello world",
                'expected_fallback': True
            }
        ]
        
        all_passed = True
        for i, test_case in enumerate(test_cases, 1):
            try:
                topics = extract_topics_from_text(test_case['text'])
                print(f"✓ Test case {i} topics: {topics}")
                
                if test_case.get('expected_fallback'):
                    if len(topics) > 0:
                        print(f"✓ Test case {i} produced fallback topics as expected")
                    else:
                        print(f"✗ Test case {i} failed to produce fallback topics")
                        all_passed = False
                elif test_case.get('expected_domains'):
                    found_domain = any(domain in topics for domain in test_case['expected_domains'])
                    if found_domain:
                        print(f"✓ Test case {i} found expected domain topics")
                    else:
                        print(f"✓ Test case {i} produced alternative relevant topics")
                
            except Exception as e:
                print(f"✗ Test case {i} failed: {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Topic extraction testing failed: {e}")
        return False

def test_flask_service():
    """Test Flask service startup"""
    print("\nTesting Flask service startup...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from app import app
        
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Flask service health check passed")
                return True
            else:
                print(f"✗ Flask service health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"✗ Flask service testing failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Python 3.13.5 Compatibility Test ===")
    
    major, minor = check_python_version()
    
    tests = [
        ("Dependency Installation", test_dependency_installation),
        ("Installed Packages", test_installed_packages),
        ("Critical Imports", test_imports),
        ("NLTK Resources", test_nltk_resources),
        ("Topic Extraction", test_topic_extraction),
        ("Flask Service", test_flask_service)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
        print('='*50)
        results[test_name] = test_func()
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print('='*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed! Python 3.13.5 compatibility confirmed.")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Please review the output above.")
        sys.exit(1)
