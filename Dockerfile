FROM python:3.9-slim
 
RUN apt-get update \
  && apt-get -y install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara libtesseract-dev libleptonica-dev
 
RUN apt update \
  && apt-get install ffmpeg libsm6 libxext6 -y
 
 
COPY . /app
WORKDIR /app
 
RUN pip install -r requirements.txt
RUN pip3 install uvicorn python-multipart
 
ARG PORT
 
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

