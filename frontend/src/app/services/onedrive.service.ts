import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AuthService } from '../auth/auth.service';
import { switchMap } from 'rxjs/operators';

export interface OneDriveFile {
  id: string;
  name: string;
  size: number;
  createdDateTime: string;
  lastModifiedDateTime: string;
  mimeType: string;
  webUrl: string;
  downloadUrl: string;
  isFolder: boolean;
}

export interface FileAnalysis {
  fileId: string;
  fileName: string;
  fileType: string;
  topics: string[];
  durationSeconds?: number;
  summary: string;
  analysisStatus: string;
}

@Injectable({
  providedIn: 'root'
})
export class OneDriveService {
  private apiUrl = 'http://localhost:8080/api/onedrive';

  constructor(
    private http: HttpClient,
    private authService: AuthService
  ) {}

  getFiles(): Observable<OneDriveFile[]> {
    return this.authService.getAccessToken().pipe(
      switchMap(token => {
        const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);
        return this.http.get<OneDriveFile[]>(`${this.apiUrl}/files`, { headers });
      })
    );
  }

  downloadFile(fileId: string): Observable<Blob> {
    return this.authService.getAccessToken().pipe(
      switchMap(token => {
        const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);
        return this.http.get(`${this.apiUrl}/files/${fileId}/download`, { 
          headers, 
          responseType: 'blob' 
        });
      })
    );
  }

  analyzeFile(fileId: string): Observable<FileAnalysis> {
    return this.authService.getAccessToken().pipe(
      switchMap(token => {
        const headers = new HttpHeaders().set('Authorization', `Bearer ${token}`);
        return this.http.post<FileAnalysis>(`${this.apiUrl}/files/${fileId}/analyze`, {}, { headers });
      })
    );
  }
}
