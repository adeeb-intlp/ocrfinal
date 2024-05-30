FROM python:3.9-slim

# Install Tesseract and dependencies
RUN apt-get update \
  && apt-get -y install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev

# Install additional dependencies
RUN apt update \
  && apt-get install ffmpeg libsm6 libxext6 -y

# Create a directory for Tesseract language data for version 5
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata

# Copy the traineddata files to Tesseract's tessdata directory for version 5
COPY tessdata/ara.traineddata /usr/share/tesseract-ocr/5/tessdata/ara.traineddata
COPY tessdata/eng.traineddata /usr/share/tesseract-ocr/5/tessdata/eng.traineddata

# Set the TESSDATA_PREFIX environment variable
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/

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

