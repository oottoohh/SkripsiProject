import { Component, OnInit } from '@angular/core';
import { ClfService } from "./Classification.service";
import { ClfInput, ClfOutput, ExplorerData, LinearsvcParameter, ModelAccuracy, InsertData, CheckBool} from "./classService";
import { AlertsService } from 'angular-alert-module';

@Component({
  selector: 'home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  public linearsvcParameters: LinearsvcParameter = new LinearsvcParameter();
  public modelAccuracy: ModelAccuracy;
  public clfInput: ClfInput = new ClfInput();
  public clfOutput: ClfOutput;
  public explorerData: ExplorerData = new ExplorerData();
  public chartType:string = 'bar';
  public insertData: InsertData = new InsertData();
  public checkBool: CheckBool;
  resultdata: Array<any> = [];
  chartLabels:Array<any> = ['Software Engineer', 'Artifical Inteligence', 'Network System', 'Data Science', 'Information Security', 'IoT'];
  public chartDatasets:Array<any> = [
    {data:[], label: 'Skripsi Datasets'},
  ];
  constructor(private clfService: ClfService, private alerts: AlertsService) { }
  ngOnInit() {

    this.clfService.LoadDatasets().subscribe((res => {this.explorerData = res;
      this.chartDatasets = [
        this.explorerData[0]['SE'], 
        this.explorerData[1]['Ai'],
        this.explorerData[2]['NS'],
        this.explorerData[3]['DS'],
        this.explorerData[4]['IS'],
        this.explorerData[5]['IoT'],
      ];
      console.log(this.chartDatasets)
      }))
  }
  
  public TrainModel(){
    this.clfService.TrainModel(this.linearsvcParameters).subscribe((modelAccuracy => {  
      this.modelAccuracy = modelAccuracy;
      console.log(modelAccuracy);
    }));
  }

  public Clfskripsi(){
    this.clfService.ClfSkripsi(this.clfInput).subscribe(clfOutput => {
      this.clfOutput = clfOutput[0].CategorySkripsi;
      console.log(clfOutput);
    })
  }
  public InsertDataset(){
    this.clfService.InsertDataset(this.insertData).subscribe((checkBool => {
      this.checkBool = checkBool;
      if(this.checkBool != null)
      {
        console.log("Second", this.checkBool[0].Result);
        if(this.checkBool[0].Result)        
          {this.alerts.setMessage('Insert Judul Skripsi Successfully','success');}
        else{this.alerts.setMessage('Insert Judul Skripsi Gagal','error');}
      }
    }))
  }

  public chartColors:Array<any> = [      
    {
        backgroundColor: [  
          "#ffbb33",
          "#00C851",
          "#33b5e5",
          "#9933CC",
          "#4285F4",
       ],
        borderColor: 'rgba(151,187,205,1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(151,187,205,1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(151,187,205,1)',
      }
  ];

  public chartOptions:any = { 
      responsive: true,
  };

  public chartClicked(e: any): void { 
       
  } 
  
  public chartHovered(e: any): void {
       
  }
  
}
  
