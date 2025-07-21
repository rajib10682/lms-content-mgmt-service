# Installation Guide

## Quick Start

```bash
pip install -r requirements.txt
```

## Troubleshooting Build Issues

### BERTopic Installation Problems

If you encounter `ERROR: Failed building wheel for hdbscan`, you have several proven solutions:

#### Option 1: Use Minimal Requirements (Recommended for most users)
```bash
pip install -r requirements-minimal.txt
```
This excludes BERTopic but keeps OpenAI, KeyBERT, and NLTK topic extraction methods.

#### Option 2: Use Conda (Best for BERTopic)
```bash
# Install hdbscan via conda first
conda install -c conda-forge hdbscan

# Then install remaining dependencies
pip install -r requirements-minimal.txt

# Finally install BERTopic
pip install bertopic
```

#### Option 3: Platform-Specific Build Tools

**Windows:**
1. Install Microsoft Visual C++ Build Tools from https://visualstudio.microsoft.com/downloads/
2. Then run: `pip install -r requirements.txt`

**Linux/Ubuntu:**
```bash
# Install build essentials and Python headers
sudo apt-get update
sudo apt-get install gcc python3-dev

# Then install requirements
pip install -r requirements.txt
```

**Docker:**
```dockerfile
RUN apt-get update && \
    apt-get -y install gcc python3-dev
```

#### Option 4: Use Pre-compiled Wheels
```bash
pip install --only-binary=all bertopic
```

## Topic Extraction Hierarchy

The system uses a fallback hierarchy that works even without BERTopic:
1. **OpenAI embeddings** (if API key provided) - Most accurate
2. **KeyBERT** (always available) - Good keyword extraction
3. **BERTopic** (optional, may require build tools) - Advanced topic modeling
4. **NLTK/TF-IDF** (fallback) - Reliable baseline

Even without BERTopic, you'll get high-quality topic extraction from OpenAI and KeyBERT.

## Environment Variables

Copy `.env.example` to `.env` and configure:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Testing Installation

```bash
python test_python313_compatibility.py
```

## Common Error Solutions

**"Python.h: No such file or directory"** (Linux):
```bash
sudo apt-get install python3-dev
```

**"Microsoft Visual C++ 14.0 is required"** (Windows):
Install Visual Studio Build Tools or Visual Studio Community

**"LINK : fatal error LNK1327"** (Windows):
Use conda installation method (Option 2 above)
