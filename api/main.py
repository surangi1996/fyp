from fastapi import FastAPI, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from typing_extensions import Annotated
import ocr
import model
import caption

app = FastAPI()

@app.post("/predict")
async def predict(text: Annotated[str, Form()]):

    label = model.predict(text)
    captions = caption.get_caption(label)
    # Return the extracted text
    return {"text": text, "label": label, "caption": captions}

@app.post("/get_text")
async def get_text(image: UploadFile):
    # Read the contents of the image file
    contents = await image.read()
    extracted_text = ocr.extract_text(contents)
    return extracted_text

app.mount("/", StaticFiles(directory="ui", html=True), name="frontend")