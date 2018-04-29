import { BrowserModule } from '@angular/platform-browser';
import { NgModule, NO_ERRORS_SCHEMA } from '@angular/core';
import { MDBBootstrapModule } from 'angular-bootstrap-md';
import { AppComponent } from './app.component';
import { ROUTES } from './app.routes';
import { HomeComponent } from './home/home.component';
import { FooterComponent } from './footer/footer.component';
import {PreloadAllModules, RouterModule} from "@angular/router";
import { ClfService } from './home/Classification.service';
import { NgCircleProgressModule } from 'ng-circle-progress';
import {HttpModule} from "@angular/http";
import { HttpClient, HttpErrorResponse, HttpHeaders, HttpClientModule } from "@angular/common/http";
import {FormsModule, ReactiveFormsModule} from "@angular/forms";
import { InterceptorModule } from "./interceptor.module";
import { AlertsModule } from 'angular-alert-module';
import 'hammerjs';

@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    FooterComponent,
    
  ],
  imports: [
    BrowserModule,
    MDBBootstrapModule.forRoot(),
    RouterModule.forRoot(ROUTES, {useHash: false, preloadingStrategy: PreloadAllModules}),
    HttpModule,
    HttpClientModule,
    InterceptorModule,
    FormsModule,
    AlertsModule.forRoot(),
    ReactiveFormsModule,
    NgCircleProgressModule.forRoot({
      // set defaults here
      radius: 100,
      outerStrokeWidth: 12,
      innerStrokeWidth: 5,
      outerStrokeColor: "#4FC3F7",
      innerStrokeColor: "#039BE5",
      animationDuration: 2000,
      "unitsFontSize": "30",
      "titleFontSize" : "32",
      "outerStrokeLinecap": "square",
      "subtitleFontSize": "20",
      "subtitle" : "Akurasi Model" 

    })
  ],
  schemas: [ NO_ERRORS_SCHEMA ],
  providers: [ClfService],
  bootstrap: [AppComponent]
})
export class AppModule {
 
}
