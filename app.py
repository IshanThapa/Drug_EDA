    
from src.Drug_EDA.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(

            CIC0 = float(request.form.get('CIC0')),
            SM1_Dz = float(request.form.get('SM1_Dz')),
            GATS1i = float(request.form.get('GATS1i')),
            NdssC = int(request.form.get('NdssC')),
            NdsCH = int(request.form.get('NdsCH')),
            MLOGP = float(request.form.get('MLOGP'))
        )
        
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=round(pred[0],2)
        
        return render_template("result.html",final_result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)