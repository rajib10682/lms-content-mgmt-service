package com.onedrive.analyzer.service;

import com.onedrive.analyzer.dto.OneDriveFileDto;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

@Service
public class GraphService {
    
    private static final Logger logger = LoggerFactory.getLogger(GraphService.class);
    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();

    public List<OneDriveFileDto> getOneDriveFiles(String accessToken) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setBearerAuth(accessToken);
            HttpEntity<String> entity = new HttpEntity<>(headers);

            String url = "https://graph.microsoft.com/v1.0/me/drive/root/children";
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.GET, entity, String.class);
            
            JsonNode jsonResponse = objectMapper.readTree(response.getBody());
            JsonNode items = jsonResponse.get("value");
            
            List<OneDriveFileDto> files = new ArrayList<>();
            
            if (items != null && items.isArray()) {
                for (JsonNode item : items) {
                    OneDriveFileDto fileDto = convertJsonToDto(item);
                    files.add(fileDto);
                }
            }
            
            logger.info("Retrieved {} files from OneDrive", files.size());
            return files;
            
        } catch (Exception e) {
            logger.error("Error retrieving OneDrive files", e);
            throw new RuntimeException("Failed to retrieve OneDrive files: " + e.getMessage(), e);
        }
    }

    public byte[] downloadFile(String fileId, String accessToken) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setBearerAuth(accessToken);
            HttpEntity<String> entity = new HttpEntity<>(headers);

            String url = "https://graph.microsoft.com/v1.0/me/drive/items/" + fileId + "/content";
            ResponseEntity<byte[]> response = restTemplate.exchange(url, HttpMethod.GET, entity, byte[].class);
            
            logger.info("Downloaded file with ID: {}", fileId);
            return response.getBody();
            
        } catch (Exception e) {
            logger.error("Error downloading file with ID: {}", fileId, e);
            throw new RuntimeException("Failed to download file: " + e.getMessage(), e);
        }
    }

    private OneDriveFileDto convertJsonToDto(JsonNode item) {
        String id = item.get("id").asText();
        String name = item.get("name").asText();
        long size = item.has("size") ? item.get("size").asLong() : 0L;
        
        LocalDateTime createdDateTime = null;
        LocalDateTime lastModifiedDateTime = null;
        
        if (item.has("createdDateTime")) {
            createdDateTime = parseDateTime(item.get("createdDateTime").asText());
        }
        
        if (item.has("lastModifiedDateTime")) {
            lastModifiedDateTime = parseDateTime(item.get("lastModifiedDateTime").asText());
        }
        
        String mimeType = "application/octet-stream";
        if (item.has("file") && item.get("file").has("mimeType")) {
            mimeType = item.get("file").get("mimeType").asText();
        }
        
        String webUrl = item.has("webUrl") ? item.get("webUrl").asText() : null;
        String downloadUrl = item.has("@microsoft.graph.downloadUrl") ? 
            item.get("@microsoft.graph.downloadUrl").asText() : null;
        boolean isFolder = item.has("folder");

        return new OneDriveFileDto(
            id, name, size, createdDateTime, lastModifiedDateTime,
            mimeType, webUrl, downloadUrl, isFolder
        );
    }

    private LocalDateTime parseDateTime(String dateTimeString) {
        try {
            return LocalDateTime.parse(dateTimeString, DateTimeFormatter.ISO_DATE_TIME);
        } catch (Exception e) {
            logger.warn("Failed to parse datetime: {}", dateTimeString);
            return null;
        }
    }
}
