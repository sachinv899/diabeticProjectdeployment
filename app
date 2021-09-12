import numpy as np
import  pandas as pd

from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    '''
    for rendering result in gui

    '''
    init_features=  [str(x) for x in request.form.values()]
    final_features=[np.array(init_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],2)

    if (int(output) == 1):
        p = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        p = "No need to fear. You have no dangerous symptoms of the disease"


    return render_template('index.html',prediction_text='  {}'.format(p))


if __name__ == '__main__':
    app.run(debug=True)
