from flask import Flask,render_template, request,send_file
import numpy as np
import cv2
import re
import base64
from keras.models import load_model
import matplotlib
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())
    # load model yang udh di training
    model = load_model('mnist.h5')
    # read image yang udh diambil dari canvas
    img = cv2.imread('input_user.png', 0)
    # mengitung Inversi bit-wise dari sebuah elemen array.
    img = np.invert(img)
    # mengoperasikan operasi not bitwise
    img = cv2.bitwise_not(img)
    # resize image menjadi 28x28
    img = cv2.resize(img, (28, 28))
    # reshape to be [samples][width][height][channels]
    img = img.reshape(1, 28, 28, 1)
    # mengubah data type pada img menjadi nilai float
    img = img.astype('float32')
    # normalize inputs from 0-255 to 0-1
    # dinormalize agar nilai bias yang nanti dimiliki oleh data tidak terlalu besar range nya
    img = img / 255.0
    #predict image sebagai model apa
    out = model.predict(img)
    print(out)
    response = ""
    for idx, val in enumerate(out[0]):
        response += "<p>"+ str(idx) + " : " + "{:.3f}".format(round(val, 3)*100) + "% </p>"
    print(np.argmax(out, axis=1))
    response += "<h1>" + np.array_str(np.argmax(out, axis=1))+"</h1>"

    #plot pie diagram, save it to load in html
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'purple', 'orangered','lightyellow','aqua','pink','grey']
    explode = (0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
    plt.close('all')
    plt.pie(out[0], explode=explode,colors=colors, autopct='%1.1f%%', startangle=140, pctdistance=1.1)
    plt.legend(labels=[0,1,2,3,4,5,6,7,8,9])
    plt.axis('equal')
    plt.savefig('pie.png', transparent=True,dpi=80)
    sys.stdout.flush()
    return response

#Get Pie Diagram Image
@app.route('/pie', methods=['GET', 'POST'])
def pie():
    return send_file('pie.png',  mimetype='image/png')

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('input_user.png', 'wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.run(debug=True)
