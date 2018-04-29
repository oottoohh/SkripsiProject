import {Injectable} from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from "@angular/common/http";
import {Observable} from "rxjs/Observable";
import { of } from 'rxjs/observable/of';
import { catchError, map, tap } from 'rxjs/operators';
import 'rxjs/add/operator/map';
import { ClfOutput, ClfInput, ExplorerData, LinearsvcParameter, ModelAccuracy, InsertData, CheckBool } from "./classService";
import { environment } from './../../environments/environment';


const SERVER_URL: string = environment.ApiUrl;

@Injectable()
export class ClfService{
    constructor(private http: HttpClient){

    }

    public TrainModel(linearsvcParameters: LinearsvcParameter): Observable<ModelAccuracy>{        
        return this.http.post<ModelAccuracy>(SERVER_URL + 'train', linearsvcParameters);
        
    }
    public ClfSkripsi(clfInput: ClfInput): Observable<ClfOutput>{
        return this.http.post<ClfOutput>(SERVER_URL + 'BuildClassification', clfInput);
    }

    public LoadDatasets(){
        return this.http.get<ExplorerData>(SERVER_URL + 'CheckData').map(result => result);
      }
    
    public InsertDataset(insertData: InsertData): Observable<CheckBool>{
        return this.http.post<CheckBool>(SERVER_URL + 'InsertData', insertData);
    }
    // public ClfData(ClfData: ClfData): Observable<ClfData>{
    //     return this.http.get(`${SERVER_URL}ExploreData`, ClfData ).map((res) => res.json());    
    // }

}