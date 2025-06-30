import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AuthService } from './auth/auth.service';
import { OneDriveService, OneDriveFile, FileAnalysis } from './services/onedrive.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class AppComponent implements OnInit {
  title = 'OneDrive File Analyzer';
  isLoggedIn = false;
  user: any = null;
  files: OneDriveFile[] = [];
  loading = false;
  error: string | null = null;
  analysisResults: FileAnalysis[] = [];

  constructor(
    private authService: AuthService,
    private oneDriveService: OneDriveService
  ) {}

  ngOnInit() {
    this.checkLoginStatus();
  }

  checkLoginStatus() {
    this.isLoggedIn = this.authService.isLoggedIn();
    if (this.isLoggedIn) {
      this.user = this.authService.getUser();
      this.loadFiles();
    }
  }

  login() {
    this.loading = true;
    this.error = null;
    
    this.authService.login().subscribe({
      next: (result) => {
        console.log('Login successful', result);
        this.checkLoginStatus();
        this.loading = false;
      },
      error: (error) => {
        console.error('Login failed', error);
        this.error = 'Login failed. Please try again.';
        this.loading = false;
      }
    });
  }

  logout() {
    this.authService.logout();
    this.isLoggedIn = false;
    this.user = null;
    this.files = [];
    this.analysisResults = [];
  }

  loadFiles() {
    this.loading = true;
    this.error = null;

    this.oneDriveService.getFiles().subscribe({
      next: (files) => {
        this.files = files;
        this.loading = false;
        console.log('Files loaded:', files);
      },
      error: (error) => {
        console.error('Error loading files', error);
        this.error = 'Failed to load files. Please try again.';
        this.loading = false;
      }
    });
  }

  analyzeFile(file: OneDriveFile) {
    this.loading = true;
    this.error = null;

    this.oneDriveService.analyzeFile(file.id).subscribe({
      next: (analysis) => {
        this.analysisResults.push(analysis);
        this.loading = false;
        console.log('File analysis completed:', analysis);
      },
      error: (error) => {
        console.error('Error analyzing file', error);
        this.error = 'Failed to analyze file. Please try again.';
        this.loading = false;
      }
    });
  }

  downloadFile(file: OneDriveFile) {
    this.oneDriveService.downloadFile(file.id).subscribe({
      next: (blob) => {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = file.name;
        link.click();
        window.URL.revokeObjectURL(url);
      },
      error: (error) => {
        console.error('Error downloading file', error);
        this.error = 'Failed to download file. Please try again.';
      }
    });
  }

  getFileAnalysis(fileId: string): FileAnalysis | undefined {
    return this.analysisResults.find(analysis => analysis.fileId === fileId);
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  formatDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
}
