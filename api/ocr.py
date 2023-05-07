import io
import os
import json
from google.cloud import vision
from spell_check import suggest

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ="E:\\projects\\apt-mark-382708-22af1afdddb6.json"

# Set up the Cloud Vision API client
client = vision.ImageAnnotatorClient()

def extract_text(image):
    # Read the image file and set up the image request
            
    image = vision.Image(content=image)
    features = [vision.Feature(type=vision.Feature.Type.TEXT_DETECTION)]
    request = vision.AnnotateImageRequest(image=image, features=features)

    # Submit the image request and save the JSON response to a file
    response = client.annotate_image(request)
    return [{
        "description": e.description,
        "vertices": e.boundingPoly.vertices,
        "suggestions": suggest(e.description)
    } for e in response.text_annotations if e.description.isalpha()]
    