from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

from prediction import read_image, predict, preprocess

encoding = {'meningioma':2, 'glioma':1, 'notumor':0, 'pituitary':3}
reverse_encoding = {2:'Meningioma', 1:'Glioma', 0:'No tumor', 3:'Pituitary'}

app = FastAPI()

# mount the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = Path("static/index.html")
    return index_file.read_text()

#need it be called file in arguement since the form in index.html has a key of "file"
@app.post('/api/predict')
def predict_image(file: bytes = File(...)):
    # Read the file uploaded by the user
    image = read_image(file)
    # Preprocess the file
    image = preprocess(image)
    # Output predictiona
    predictions = predict(image)
    # Get the prediction value (index) and the corresponding class name
    prediction_index = int(predictions[0])
    prediction_name = reverse_encoding[prediction_index]
    return {"prediction_index": prediction_index, "prediction_name": prediction_name}

if __name__ == "__main__":
    uvicorn.run(app, port=80, host='0.0.0.0')
