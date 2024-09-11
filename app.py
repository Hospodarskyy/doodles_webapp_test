import flask
from flask import Flask, render_template, url_for, request
import base64
import numpy as np
import cv2
import tensorflow as tf

# Initialize the useless part of the base64 encoded image.
init_Base64 = 21

# Our dictionary
label_dict = {
    0: 'rhinoceros', 1: 'octopus', 2: 'firetruck', 3: 'laptop', 4: 'windmill',
    5: 'pineapple', 6: 'candle', 7: 'mosquito', 8: 'hot air balloon', 9: 'giraffe',
    10: 'crown', 11: 'rainbow', 12: 'toothbrush', 13: 'tornado', 14: 'paintbrush',
    15: 'helicopter', 16: 'snowman', 17: 'saxophone', 18: 'ambulance', 19: 'dragon'
}

# Load the pre-trained model using TensorFlow's Keras API
model = tf.keras.models.load_model('model_cnn.h5')

# Initializing new Flask instance. Find the HTML template in "templates".
app = flask.Flask(__name__, template_folder='templates')

# First route: Render the initial drawing template
@app.route('/')
def home():
    return render_template('draw.html')

# Second route: Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Preprocess the image: set the image to 128x128 shape
        # Access the image
        draw = request.form['url']
        # Removing the useless part of the url.
        draw = draw[init_Base64:]
        # Decoding
        draw_decoded = base64.b64decode(draw)
        image = np.asarray(bytearray(draw_decoded), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        # Resizing and reshaping to keep the ratio.
        resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        print(f'\n\tResized shape: {resized.shape}')
        vect = np.asarray(resized, dtype="uint8")
        # vect = vect.reshape(1, 128, 128, 1).astype('float32')
        vect = np.expand_dims(vect, axis=0).astype('float32')

        # Launch prediction
        my_prediction = model.predict(vect)
        print(f'\n\tPredictions: {my_prediction}')
        # Getting the index of the maximum prediction
        index = np.argmax(my_prediction[0])
        # Associating the index and its value within the dictionary
        final_pred = label_dict[index]

        return render_template('results.html', prediction=final_pred)

if __name__ == '__main__':
    app.run(debug=True)
