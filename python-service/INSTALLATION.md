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

**Option 3a: Complete Visual Studio Build Tools (Recommended)**
1. Download Visual Studio Build Tools from https://visualstudio.microsoft.com/downloads/
2. During installation, select:
   - **C++ build tools** workload
   - **Windows 10/11 SDK** (latest version)
   - **MSVC v143 - VS 2022 C++ x64/x86 build tools**
3. Restart your command prompt/PowerShell
4. Then run: `pip install -r requirements.txt`

**Option 3b: Visual Studio Community (Alternative)**
1. Install Visual Studio Community from https://visualstudio.microsoft.com/vs/community/
2. During installation, select:
   - **Desktop development with C++** workload
   - Ensure **Windows SDK** is included
3. Restart your command prompt/PowerShell
4. Then run: `pip install -r requirements.txt`

**Option 3c: Fix Missing Windows SDK Headers**
If you get `Cannot open include file: 'io.h'` error:
1. Open Visual Studio Installer
2. Modify your installation
3. Add **Windows 10/11 SDK** component
4. Restart and try again

**Option 3d: Use Developer Command Prompt**
1. Search for "Developer Command Prompt for VS 2022" in Start Menu
2. Run as Administrator
3. Navigate to your project directory
4. Run: `pip install -r requirements.txt`

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

## SSL Certificate Issues

If you encounter SSL certificate verification errors:

**"certificate verify failed: unable to get local issuer certificate"**
- The application automatically disables SSL verification for compatibility
- This affects OpenAI API calls, model downloads, and NLTK resources
- No manual configuration needed - handled automatically

**Corporate Network Issues:**
- SSL verification is disabled by default for corporate environments
- If you need to enable SSL verification, modify the `configure_ssl_context()` function
- Consider using corporate certificate bundles if required by your organization

**Environment Variables:**
The application sets these automatically:
- `PYTHONHTTPSVERIFY=0`
- `CURL_CA_BUNDLE=''`
- `REQUESTS_CA_BUNDLE=''`

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
Install Visual Studio Build Tools or Visual Studio Community (see Option 3 above)

**"Cannot open include file: 'io.h'"** (Windows):
- Missing Windows SDK headers
- Install Windows 10/11 SDK via Visual Studio Installer
- Use Developer Command Prompt for VS 2022
- See Option 3c above for detailed steps

**"LINK : fatal error LNK1327"** (Windows):
Use conda installation method (Option 2 above)

**"cl.exe failed with exit code 2"** (Windows):
- Incomplete Visual Studio Build Tools installation
- Missing Windows SDK components
- Try Option 3a or 3b above for complete installation

**"fatal error C1083"** (Windows):
- Missing system headers (io.h, stdio.h, etc.)
- Install complete Windows SDK via Visual Studio Installer
- Ensure both C++ build tools AND Windows SDK are installed
