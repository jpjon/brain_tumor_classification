from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

input_shape = (200, 200)

model = load_model('CNN_model')

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def preprocess(image: Image.Image):
    image = image.resize(input_shape)
    image = img_to_array(image)
    image = image / 255
    # Return the preprocessed image
    return image

def predict(image: np.ndarray):
    # Add batch dimension to the input image
    image = np.expand_dims(image, axis=0)
    return np.argmax(model.predict(image), axis=-1)