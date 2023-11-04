from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.utils.utils import load_object
from flask import Flask,request,render_template,jsonify
from src.DimondPricePrediction.pipelines.prediction_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def predict_price():
    print("11111111")
    if request.method == "GET":
        return render_template("details.html")
    elif request.method == "POST":
        data = CustomData(
            carat = float(request.form.get('carat')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get("table")),
            cut = request.form.get("cut"),
            color = request.form.get("color"),
            clarity = request.form.get("clarity")
        )
        final_data = data.get_as_dataframe()

        prediction_object = PredictPipeline()
        pred = prediction_object.predict(final_data)
        result = round(pred[0],2)
        output_string = "The Predicted price for the Diamond is: "+str(result)
        return render_template("details.html",final_result = output_string)

if __name__ == '__main__':
    app.run(debug=True)