from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)


def get_prediction(param):
    model = tf.keras.models.load_model(os.path.join("..", "model"))
    y_pred = model.predict(param)
    depth = np.round(y_pred[0][0], 2)
    width = np.round(y_pred[0][1], 2)

    return f'Глубина сварного шва {depth} \n Ширина сварного шва {width}'


@app.route('/', methods=['post', 'get'])
def processing():
    message = ''
    if request.method == 'POST':
        IW = request.form.get('IW')
        IF = request.form.get('IF')
        VW = request.form.get('VW')
        FP = request.form.get('FP')

        parameters = [float(IW), float(IF), float(VW), float(FP)]
        parameters = np.array([parameters])

        message = get_prediction(parameters)

    return render_template('predict.html', message=message)


if __name__ == '__main__':
    app.run()
