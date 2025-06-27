package com.onedrive.analyzer.controller;

import com.onedrive.analyzer.dto.FileAnalysisDto;
import com.onedrive.analyzer.dto.OneDriveFileDto;
import com.onedrive.analyzer.service.FileAnalysisService;
import com.onedrive.analyzer.service.GraphService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/onedrive")
@CrossOrigin(origins = "http://localhost:4200")
public class OneDriveController {

    @Autowired
    private GraphService graphService;

    @Autowired
    private FileAnalysisService fileAnalysisService;

    @GetMapping("/files")
    public ResponseEntity<List<OneDriveFileDto>> getFiles(@RequestHeader("Authorization") String authHeader) {
        try {
            String accessToken = authHeader.replace("Bearer ", "");
            List<OneDriveFileDto> files = graphService.getOneDriveFiles(accessToken);
            return ResponseEntity.ok(files);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    @GetMapping("/files/{fileId}/download")
    public ResponseEntity<byte[]> downloadFile(@PathVariable String fileId, 
                                             @RequestHeader("Authorization") String authHeader) {
        try {
            String accessToken = authHeader.replace("Bearer ", "");
            byte[] fileContent = graphService.downloadFile(fileId, accessToken);
            return ResponseEntity.ok(fileContent);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }

    @PostMapping("/files/{fileId}/analyze")
    public ResponseEntity<FileAnalysisDto> analyzeFile(@PathVariable String fileId,
                                                      @RequestHeader("Authorization") String authHeader) {
        try {
            String accessToken = authHeader.replace("Bearer ", "");
            FileAnalysisDto analysis = fileAnalysisService.analyzeFile(fileId, accessToken);
            return ResponseEntity.ok(analysis);
        } catch (Exception e) {
            return ResponseEntity.badRequest().build();
        }
    }
}
