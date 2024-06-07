from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from OCR import process_image
import os

app = FastAPI()

# Define the list of allowed origins
origins = [
    "https://localhost:44308",
    # Add other origins if needed
]

# Add the CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"./{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        # Process the image using OCR
        result = process_image(file_location)

        # Clean up the saved file
        os.remove(file_location)

        return result
    except Exception as e:
        return {"success": False, "error": str(e)}