package com.onedrive.analyzer.service;

import com.onedrive.analyzer.dto.FileAnalysisDto;
import com.onedrive.analyzer.dto.OneDriveFileDto;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.apache.http.entity.mime.content.FileBody;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

@Service
public class FileAnalysisService {
    
    private static final Logger logger = LoggerFactory.getLogger(FileAnalysisService.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Autowired
    private GraphService graphService;
    
    @Value("${whisper.service.url}")
    private String whisperServiceUrl;
    
    @Value("${whisper.service.timeout:300000}")
    private int whisperServiceTimeout;

    public FileAnalysisDto analyzeFile(String fileId, String accessToken) {
        try {
            logger.info("Starting analysis for file ID: {}", fileId);
            
            OneDriveFileDto fileInfo = getFileInfo(fileId, accessToken);
            
            if (isVideoFile(fileInfo.getMimeType())) {
                return analyzeVideoFile(fileId, fileInfo, accessToken);
            } else {
                return analyzeNonVideoFile(fileId, fileInfo);
            }
            
        } catch (Exception e) {
            logger.error("Error analyzing file with ID: {}", fileId, e);
            throw new RuntimeException("Failed to analyze file: " + e.getMessage(), e);
        }
    }
    
    private OneDriveFileDto getFileInfo(String fileId, String accessToken) {
        List<OneDriveFileDto> files = graphService.getOneDriveFiles(accessToken);
        return files.stream()
                .filter(file -> file.getId().equals(fileId))
                .findFirst()
                .orElseThrow(() -> new RuntimeException("File not found: " + fileId));
    }
    
    private boolean isVideoFile(String mimeType) {
        if (mimeType == null) return false;
        return mimeType.startsWith("video/") || 
               mimeType.equals("application/mp4") ||
               mimeType.equals("application/x-msvideo");
    }
    
    private FileAnalysisDto analyzeVideoFile(String fileId, OneDriveFileDto fileInfo, String accessToken) {
        File tempFile = null;
        try {
            logger.info("Analyzing video file: {} ({})", fileInfo.getName(), fileInfo.getMimeType());
            
            byte[] fileContent = graphService.downloadFile(fileId, accessToken);
            
            tempFile = File.createTempFile("video_", getFileExtension(fileInfo.getName()));
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                fos.write(fileContent);
            }
            
            JsonNode whisperResponse = callWhisperService(tempFile);
            
            List<String> topics = extractTopicsFromResponse(whisperResponse);
            Long duration = whisperResponse.has("durationSeconds") ? 
                whisperResponse.get("durationSeconds").asLong() : null;
            String summary = whisperResponse.has("summary") ? 
                whisperResponse.get("summary").asText() : "Video analysis completed.";
            
            return new FileAnalysisDto(
                fileId,
                fileInfo.getName(),
                "video",
                topics,
                duration,
                summary,
                "completed"
            );
            
        } catch (Exception e) {
            logger.error("Error analyzing video file: {}", fileInfo.getName(), e);
            return createErrorAnalysis(fileId, fileInfo, "Video analysis failed: " + e.getMessage());
        } finally {
            if (tempFile != null && tempFile.exists()) {
                try {
                    Files.delete(tempFile.toPath());
                } catch (IOException e) {
                    logger.warn("Failed to delete temporary file: {}", tempFile.getPath());
                }
            }
        }
    }
    
    private FileAnalysisDto analyzeNonVideoFile(String fileId, OneDriveFileDto fileInfo) {
        logger.info("Analyzing non-video file: {} ({})", fileInfo.getName(), fileInfo.getMimeType());
        
        List<String> topics = Arrays.asList("Document Analysis", "Content Review");
        String summary = "Non-video file analyzed. For video content analysis, please upload a video file.";
        
        return new FileAnalysisDto(
            fileId,
            fileInfo.getName(),
            "document",
            topics,
            null,
            summary,
            "completed"
        );
    }
    
    private JsonNode callWhisperService(File videoFile) throws IOException {
        try (CloseableHttpClient httpClient = HttpClients.createDefault()) {
            HttpPost httpPost = new HttpPost(whisperServiceUrl + "/analyze-video");
            
            FileBody fileBody = new FileBody(videoFile);
            HttpEntity entity = MultipartEntityBuilder.create()
                    .addPart("file", fileBody)
                    .build();
            
            httpPost.setEntity(entity);
            
            try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
                String responseBody = EntityUtils.toString(response.getEntity());
                
                if (response.getStatusLine().getStatusCode() != 200) {
                    throw new RuntimeException("Whisper service error: " + responseBody);
                }
                
                return objectMapper.readTree(responseBody);
            }
        }
    }
    
    private List<String> extractTopicsFromResponse(JsonNode response) {
        if (response.has("topics") && response.get("topics").isArray()) {
            return objectMapper.convertValue(response.get("topics"), List.class);
        }
        return Arrays.asList("General Content");
    }
    
    private String getFileExtension(String filename) {
        int lastDotIndex = filename.lastIndexOf('.');
        return lastDotIndex > 0 ? filename.substring(lastDotIndex) : ".tmp";
    }
    
    private FileAnalysisDto createErrorAnalysis(String fileId, OneDriveFileDto fileInfo, String errorMessage) {
        return new FileAnalysisDto(
            fileId,
            fileInfo.getName(),
            "error",
            Arrays.asList("Analysis Failed"),
            null,
            errorMessage,
            "failed"
        );
    }
}
