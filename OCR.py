import os
import cv2
import pytesseract
from PIL import Image
import re
from datetime import datetime, timedelta
import json

def process_image(image_path):
    try:
        # First, try to extract text without bounding boxes
        extracted_text = extract_text_from_image(image_path)

        if "Name" in extracted_text:
            # Extract information using the existing function
            data = {
                "extracted_data": {
                    "Name": extract_name_dob_sex(extracted_text)["Name"],
                    "DateOfBirth": extract_name_dob_sex(extracted_text)["DateOfBirth"],
                    "Sex": extract_name_dob_sex(extracted_text)["Sex"],
                    "IDNumber": extract_id_number(extracted_text),
                    "ExpiryDate": extract_expiry_date(extracted_text),
                    "IssuingDate": extract_issuing_date(extract_expiry_date(extracted_text)),
                    "Occupation": extract_occupation(extracted_text),
                    "Employer": extract_employer(extracted_text),
                    "IssuingPlace": extract_issuing_place(extracted_text)
                },
                "bounding_boxes": []
            }
            return {"success": True, "data": data}
        else:
            # Extract text with bounding boxes since the name was not detected
            extracted_text, bounding_boxes = extract_text_with_boxes_from_image(image_path, lang='ara')
            # Process the image for Arabic text extraction
            arabic_text = extract_arabic_text_from_image(image_path, lang='ara')
            # Return the extracted Arabic text along with bounding boxes
            return {"success": True, "data": {"extracted_data": None, "arabic_text": arabic_text}, "bounding_boxes": bounding_boxes}

    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_text_with_boxes_from_image(image_path, lang='eng+ara'):
    try:
        # Load the image
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance the image visibility
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.equalizeHist(gray)
        
        # Threshold the image to obtain binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours of the text regions
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        extracted_text = ""
        bounding_boxes = []
        
        # Iterate through each contour
        for contour in contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract the region of interest (ROI) from the binary image
            roi = binary[y:y+h, x:x+w]
            
            # Use Tesseract to recognize text in the ROI
            text = pytesseract.image_to_string(roi, lang=lang, config='--psm 6')
            
            # Append the extracted text and bounding box coordinates
            extracted_text += text.strip() + "\n"
            bounding_boxes.append({"text": text.strip(), "coordinates": (x, y, w, h)})
        
        return extracted_text.strip(), bounding_boxes
    
    except Exception as e:
        raise e

def extract_text_from_image(image_path):
    try:
        # Pre-processing for clearer images (resize, convert to grayscale, and enhance contrast)
        image = Image.open(image_path)
        image = image.resize((image.width * 2, image.height * 2))
        image = image.convert("L")
        
        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(image, lang='eng+ara', config='--psm 6')
        
        return extracted_text.strip()
    
    except Exception as e:
        raise e

def extract_arabic_text_from_image(image_path, lang='ara'):
    try:
        # Load the image
        image = cv2.imread(image_path)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance the image visibility
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.equalizeHist(gray)
        
        # Apply additional preprocessing steps for better OCR accuracy
        gray = cv2.bilateralFilter(gray, 9, 75, 75)  # Denoise image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use Tesseract to recognize text in the image
        extracted_text = pytesseract.image_to_string(binary, lang=lang, config='--psm 6')
        
        return extracted_text.strip()
    
    except Exception as e:
        raise e

def extract_name_dob_sex(text):
    name_pattern = r'Name:\s*(.*)'
    dob_pattern = r'Date of Birth:\s*(\d{2}/\d{2}/\d{4})'  # Regular expression pattern for date of birth
    sex_pattern = r'Sex:\s*([MF])'  # Regular expression pattern for sex (M or F)
    
    # Extract name
    name_match = re.search(name_pattern, text)
    name = name_match.group(1).strip() if name_match else None
    
    # Extract date of birth
    dob_match = re.search(dob_pattern, text)
    dob = dob_match.group(1).strip() if dob_match else None
    
    # Extract sex
    sex_match = re.search(sex_pattern, text)
    sex = sex_match.group(1).strip() if sex_match else None
    
    return {"Name": name, "DateOfBirth": dob, "Sex": sex}

def extract_id_number(text):
    id_number_pattern = r'784-\d{4}-\d{7}-\d'  # Regular expression pattern to match ID number starting with "784" and containing hyphens
    id_match = re.search(id_number_pattern, text)
    if id_match:
        return id_match.group()
    else:
        return None

def extract_expiry_date(text):
    expiry_date_pattern = r'[MF](\d{2})(\d{2})(\d{2})'  # Regular expression pattern for expiry date in format MMDDYY
    expiry_date_match = re.search(expiry_date_pattern, text)
    if expiry_date_match:
        day = expiry_date_match.group(1)
        month = expiry_date_match.group(2)
        year = expiry_date_match.group(3)
        return f"{year}-{month}-{day}"  # Adjust the year format
    else:
        return None

def extract_issuing_date(expiry_date):
    if expiry_date:
        expiry_datetime = datetime.strptime(expiry_date, '%y-%m-%d')
        issuing_datetime = expiry_datetime - timedelta(days=365) + timedelta(days=1)  # Subtract 1 year and add 1 day
        return issuing_datetime.strftime('%y-%m-%d')
    else:
        return None

def extract_occupation(text):
    occupation_pattern = r'Occupation:\s*(.*)'
    occupation_match = re.search(occupation_pattern, text, re.IGNORECASE)  # Case insensitive match
    return occupation_match.group(1).strip() if occupation_match else None

def extract_employer(text):
    employer_pattern = r'Employer:\s*(.*)'
    employer_match = re.search(employer_pattern, text, re.IGNORECASE)  # Case insensitive match
    return employer_match.group(1).strip() if employer_match else None

def extract_issuing_place(text):
    issuing_place_pattern = r'Issuing\s+Place:\s*(.*)'
    issuing_place_match = re.search(issuing_place_pattern, text)
    return issuing_place_match.group(1).strip() if issuing_place_match else None

# Path to the uploaded image
image_path = '/mnt/data/nmk_iqama2.jpeg'

# Process the image
result = process_image(image_path)

# Print the result
print(json.dumps(result, indent=4, ensure_ascii=False))




        







