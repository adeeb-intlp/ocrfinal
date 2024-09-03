from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from OCR import process_image

app = FastAPI()

# Define the list of allowed origins
origins = [
    "https://clientapp-dev.intelpeek.com",
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
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())

        # Process the image using OCR
        result = process_image(file.filename)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
