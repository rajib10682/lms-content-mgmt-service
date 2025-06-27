package com.onedrive.analyzer.dto;

import java.util.List;

public class FileAnalysisDto {
    private String fileId;
    private String fileName;
    private String fileType;
    private List<String> topics;
    private Long durationSeconds;
    private String summary;
    private String analysisStatus;

    public FileAnalysisDto() {}

    public FileAnalysisDto(String fileId, String fileName, String fileType, List<String> topics, 
                          Long durationSeconds, String summary, String analysisStatus) {
        this.fileId = fileId;
        this.fileName = fileName;
        this.fileType = fileType;
        this.topics = topics;
        this.durationSeconds = durationSeconds;
        this.summary = summary;
        this.analysisStatus = analysisStatus;
    }

    public String getFileId() { return fileId; }
    public void setFileId(String fileId) { this.fileId = fileId; }

    public String getFileName() { return fileName; }
    public void setFileName(String fileName) { this.fileName = fileName; }

    public String getFileType() { return fileType; }
    public void setFileType(String fileType) { this.fileType = fileType; }

    public List<String> getTopics() { return topics; }
    public void setTopics(List<String> topics) { this.topics = topics; }

    public Long getDurationSeconds() { return durationSeconds; }
    public void setDurationSeconds(Long durationSeconds) { this.durationSeconds = durationSeconds; }

    public String getSummary() { return summary; }
    public void setSummary(String summary) { this.summary = summary; }

    public String getAnalysisStatus() { return analysisStatus; }
    public void setAnalysisStatus(String analysisStatus) { this.analysisStatus = analysisStatus; }
}
