# OneDrive File Analyzer with Whisper Integration

A full-stack application that authenticates with Azure AD, accesses OneDrive files, and provides AI-powered video analysis using OpenAI's Whisper model.

## Features

- **Angular Frontend** with MSAL authentication for Azure Active Directory
- **Spring Boot Backend** with Microsoft Graph API integration
- **Python Whisper Service** for video transcription and topic extraction
- **File Management** - list, download, and analyze OneDrive files
- **Video Analysis** - extract topics from video content using AI transcription

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Angular        │    │  Spring Boot     │    │  Python Flask   │
│  Frontend       │◄──►│  Backend         │◄──►│  Whisper Service │
│  (Port 4200)    │    │  (Port 8080)     │    │  (Port 5000)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Azure AD       │    │  Microsoft       │
│  Authentication │    │  Graph API       │
└─────────────────┘    └──────────────────┘
```

## Setup Instructions

### Prerequisites

- Node.js 18+ and npm/yarn
- Java 11+
- Python 3.9+ (tested and compatible with Python 3.13.5)
- Azure AD App Registration

### 1. Azure AD Configuration

1. Create an Azure AD app registration
2. Configure redirect URI: `http://localhost:4200`
3. Add API permissions for Microsoft Graph:
   - `User.Read`
   - `Files.Read`
4. Update `frontend/src/app/app.config.ts` with your client ID and tenant ID

### 2. Backend Setup

```bash
cd backend
./gradlew bootRun
```

The Spring Boot backend will start on `http://localhost:8080`

### 3. Python Whisper Service Setup

```bash
cd python-service
pip install -r requirements.txt
python app.py
```

The Python service will start on `http://localhost:5000`

### 4. Frontend Setup

```bash
cd frontend
npm install
ng serve
```

The Angular frontend will start on `http://localhost:4200`

## Usage

1. **Authentication**: Sign in with your Microsoft account
2. **Browse Files**: View your OneDrive files and folders
3. **Download Files**: Download any file from your OneDrive
4. **Analyze Videos**: Click "Analyze" on video files to extract topics and get transcription

## Video Analysis Features

- **Automatic Transcription**: Uses OpenAI Whisper to transcribe audio from videos
- **Topic Extraction**: AI-powered analysis to identify relevant topics
- **Duration Detection**: Automatically detects video duration
- **Multiple Formats**: Supports mp4, avi, mov, mkv, webm, flv, m4v, wmv

## Configuration

### Backend Configuration (`backend/src/main/resources/application.properties`)

```properties
# Python Whisper Service
whisper.service.url=http://localhost:5000
whisper.service.timeout=300000

# File upload limits
spring.servlet.multipart.max-file-size=100MB
spring.servlet.multipart.max-request-size=100MB
```

### Frontend Configuration (`frontend/src/app/app.config.ts`)

```typescript
// Replace with your Azure AD app registration details
clientId: 'your-client-id-here'
authority: 'https://login.microsoftonline.com/your-tenant-id'
```

## API Endpoints

### Spring Boot Backend

- `GET /api/onedrive/files` - List OneDrive files
- `GET /api/onedrive/files/{fileId}/download` - Download file
- `POST /api/onedrive/files/{fileId}/analyze` - Analyze file

### Python Whisper Service

- `GET /health` - Health check
- `POST /analyze-video` - Analyze video file

## Development

### Running Tests

```bash
# Backend tests
cd backend
./gradlew test

# Frontend tests
cd frontend
ng test
```

### Building for Production

```bash
# Backend
cd backend
./gradlew build

# Frontend
cd frontend
ng build --prod
```

## Troubleshooting

1. **Authentication Issues**: Verify Azure AD app registration and redirect URIs
2. **File Access Issues**: Check Microsoft Graph API permissions
3. **Video Analysis Issues**: Ensure Python service is running and accessible
4. **Large File Issues**: Adjust file size limits in application.properties

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
