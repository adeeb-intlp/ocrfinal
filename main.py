from fastapi import FastAPI, UploadFile, File
from OCR import process_image

app = FastAPI()
# calling fastapi
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

