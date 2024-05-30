FROM python:3.9-slim
 
# Install Tesseract and dependencies
RUN apt-get update \
&& apt-get -y install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev
 
# Install additional dependencies
RUN apt update \
&& apt-get install ffmpeg libsm6 libxext6 -y
 
# Create a directory for Tesseract language data for version 5
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata
 
# Copy the Arabic traineddata file to Tesseract's tessdata directory for version 5
COPY tessfolder/Arabic.traineddata /usr/share/tesseract-ocr/5/tessdata/Arabic.traineddata
 
# Copy the project files to the working directory
COPY . /app
 
# Set the working directory
WORKDIR /app
 
# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip3 install uvicorn python-multipart
 
# Expose the port and set the entrypoint
ARG PORT
CMD uvicorn main:app --host 0.0.0.0 --port $PORT