from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import ocr
import model

app = FastAPI()

@app.post("/predict")
async def predict(image: UploadFile):
    # Read the contents of the image file
    contents = await image.read()
    extracted_text = ocr.extract_text(contents)
    print(extracted_text)

    label = model.predict(extracted_text)
    print(label)
    # Return the extracted text
    return {"text": extracted_text, "label": label}

@app.post("/get_text")
async def get_text(image: UploadFile):
    # Read the contents of the image file
    contents = await image.read()
    extracted_text = ocr.extract_text(contents)
    return extracted_text

app.mount("/", StaticFiles(directory="ui", html=True), name="frontend")