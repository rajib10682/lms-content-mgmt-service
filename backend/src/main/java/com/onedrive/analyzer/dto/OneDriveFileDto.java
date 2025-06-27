package com.onedrive.analyzer.dto;

import java.time.LocalDateTime;

public class OneDriveFileDto {
    private String id;
    private String name;
    private long size;
    private LocalDateTime createdDateTime;
    private LocalDateTime lastModifiedDateTime;
    private String mimeType;
    private String webUrl;
    private String downloadUrl;
    private boolean isFolder;

    public OneDriveFileDto() {}

    public OneDriveFileDto(String id, String name, long size, LocalDateTime createdDateTime, 
                          LocalDateTime lastModifiedDateTime, String mimeType, String webUrl, 
                          String downloadUrl, boolean isFolder) {
        this.id = id;
        this.name = name;
        this.size = size;
        this.createdDateTime = createdDateTime;
        this.lastModifiedDateTime = lastModifiedDateTime;
        this.mimeType = mimeType;
        this.webUrl = webUrl;
        this.downloadUrl = downloadUrl;
        this.isFolder = isFolder;
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public long getSize() { return size; }
    public void setSize(long size) { this.size = size; }

    public LocalDateTime getCreatedDateTime() { return createdDateTime; }
    public void setCreatedDateTime(LocalDateTime createdDateTime) { this.createdDateTime = createdDateTime; }

    public LocalDateTime getLastModifiedDateTime() { return lastModifiedDateTime; }
    public void setLastModifiedDateTime(LocalDateTime lastModifiedDateTime) { this.lastModifiedDateTime = lastModifiedDateTime; }

    public String getMimeType() { return mimeType; }
    public void setMimeType(String mimeType) { this.mimeType = mimeType; }

    public String getWebUrl() { return webUrl; }
    public void setWebUrl(String webUrl) { this.webUrl = webUrl; }

    public String getDownloadUrl() { return downloadUrl; }
    public void setDownloadUrl(String downloadUrl) { this.downloadUrl = downloadUrl; }

    public boolean isFolder() { return isFolder; }
    public void setFolder(boolean folder) { isFolder = folder; }
}
