import { Injectable } from '@angular/core';
import { MsalService } from '@azure/msal-angular';
import { AuthenticationResult } from '@azure/msal-browser';
import { Observable, from } from 'rxjs';
import { map, switchMap } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  constructor(private msalService: MsalService) {
    this.initializeMsal();
  }

  private async initializeMsal(): Promise<void> {
    try {
      await this.msalService.instance.initialize();
      console.log('MSAL initialized successfully');
    } catch (error) {
      console.error('MSAL initialization failed:', error);
    }
  }

  login(): Observable<AuthenticationResult> {
    return from(this.msalService.instance.initialize()).pipe(
      switchMap(() => this.msalService.loginPopup({
        scopes: ['user.read', 'files.read']
      }))
    );
  }

  logout(): void {
    this.msalService.logout();
  }

  getAccessToken(): Observable<string> {
    return from(this.msalService.acquireTokenSilent({
      scopes: ['user.read', 'files.read'],
      account: this.msalService.instance.getAllAccounts()[0]
    })).pipe(
      map((result: any) => result.accessToken)
    );
  }

  isLoggedIn(): boolean {
    return this.msalService.instance.getAllAccounts().length > 0;
  }

  getUser(): any {
    const accounts = this.msalService.instance.getAllAccounts();
    return accounts.length > 0 ? accounts[0] : null;
  }
}
