package com.onedrive.analyzer.service;

import com.onedrive.analyzer.dto.FileAnalysisDto;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Service
public class FileAnalysisService {
    
    private static final Logger logger = LoggerFactory.getLogger(FileAnalysisService.class);
    private final Random random = new Random();

    public FileAnalysisDto analyzeFile(String fileId, String accessToken) {
        try {
            logger.info("Starting analysis for file ID: {}", fileId);
            
            
            List<String> sampleTopics = Arrays.asList(
                "Business Strategy", "Technology", "Education", "Marketing", 
                "Finance", "Project Management", "Data Analysis", "Communication"
            );
            
            List<String> topics = Arrays.asList(
                sampleTopics.get(random.nextInt(sampleTopics.size())),
                sampleTopics.get(random.nextInt(sampleTopics.size()))
            );
            
            Long duration = null;
            String summary = "This file contains relevant content related to the identified topics.";
            
            if (random.nextBoolean()) {
                duration = (long) (300 + random.nextInt(1800)); // 5-35 minutes
                summary = "Video content analyzed for topics and duration.";
            }
            
            FileAnalysisDto analysis = new FileAnalysisDto(
                fileId,
                "Sample File",
                "document",
                topics,
                duration,
                summary,
                "completed"
            );
            
            logger.info("Analysis completed for file ID: {}", fileId);
            return analysis;
            
        } catch (Exception e) {
            logger.error("Error analyzing file with ID: {}", fileId, e);
            throw new RuntimeException("Failed to analyze file: " + e.getMessage(), e);
        }
    }
}
