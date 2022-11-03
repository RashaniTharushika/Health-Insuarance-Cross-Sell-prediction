from flask import Flask, render_template, request
import pickle
import bz2
import numpy as np

app = Flask(__name__)


def prediction(lst):

    ifile = bz2.BZ2File("model/rf",'rb')
    rf = pickle.load(ifile)
    ifile.close()

    pred_value = rf.predict(lst)
    return pred_value


@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')


@app.route('/classification', methods=['GET'])
def classification():
    return render_template('classification.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    pred_value = 0
    if request.method == 'POST':
        gender = request.form['gender']
        if gender == 'Male':
            gender =1
        else:
            gender = 0


        age = request.form['age']
        drivinglicense = request.form['drivinglicense']
        if drivinglicense == 'Yes':
            drivinglicense = 1
        else:
            drivinglicense = 0

        regioncode =request.form['regioncode']
        previouslyinsured =request.form['previouslyinsured']
        if previouslyinsured == 'Yes':
            previouslyinsured = 1
        else:
            previouslyinsured = 0


        vehicleage = request.form['vehicleage']
        if vehicleage == '1-2 Year':
            vehicleage = 0
        elif vehicleage == '< 1 Year':
            vehicleage = 1
        else:
            vehicleage = 2


        vehicledamage = request.form['vehicledamage']
        if vehicledamage == 'Yes':
            vehicledamage = 1
        else:
            vehicledamage = 0

        annualpremium = request.form['annualpremium']

        feature_list = []

        feature_list.append(int(age))
        feature_list.append(float(regioncode))
        feature_list.append(previouslyinsured)
        feature_list.append(vehicleage)
        feature_list.append(vehicledamage)
        feature_list.append(float(annualpremium))

        feature_list = np.array(feature_list).reshape((1,6))

        pred_value = prediction(feature_list)

        print(pred_value)

    return render_template('result.html', pred_value=pred_value[0])


if __name__ == '__main__':
    app.run(debug=True)
