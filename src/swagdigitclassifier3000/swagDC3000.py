from io import BytesIO
from flask import Flask
from flask import render_template
from flask import request
import base64
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("finalized_datad_model.sav", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    body = request.get_json()
    imageJson = body["image"]
    imageBase64 = imageJson.split(",")[1]
    decodeImage = base64.b64decode(imageBase64)
    img = Image.open(BytesIO(decodeImage))
    img.save("saved_images/original.png")
    img = Image.open(BytesIO(decodeImage)).convert("L")
    img.save("saved_images/greyscale.png")
    img = ImageOps.invert(img)
    img.save("saved_images/invert.png")
    # J'obtiens de meilleur pédictions sans bbox et crop l'image
    # imgbbox = img.getbbox()
    # img = img.crop(imgbbox)
    # img.save("saved_images/cropped.png")
    img = img.resize((8,8))
    img.save("saved_images/resize.png")
    img_array = numpy.array(img).reshape(1, -1)

    res = model.predict(img_array)
    return {
        "result": int(res[0])
    }
