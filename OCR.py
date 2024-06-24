import cv2
import pytesseract
import re
import json
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import os

# Set the path to the Tesseract executable if needed
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
# pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = '/app/src/tesseract-4.1.0'

def arabic_to_english(number):
    arabic_english_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3',
        '٤': '4', '٥': '5', '٦': '6', '٧': '7',
        '٨': '8', '٩': '9'
    }
    return ''.join(arabic_english_map.get(ch, ch) for ch in number)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness - 50
    alpha = contrast / 50.0 + 1.0
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def preprocess_image(image):
    # Adjust brightness and contrast
    image = adjust_brightness_contrast(image, brightness=100, contrast=50)
    
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Apply binary thresholding
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    return image

def extract_name(roi_name):
    # Adjust brightness and contrast
    roi_name = adjust_brightness_contrast(roi_name, brightness=70, contrast=30)

    # Apply preprocessing to improve OCR accuracy
    roi_name = cv2.GaussianBlur(roi_name, (3, 3), 0)
    roi_name = cv2.threshold(roi_name, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Use OCR to extract text from the ROI
    custom_config = r'--oem 3 --psm 6 -l eng+ara'
    name_text = pytesseract.image_to_string(roi_name, config=custom_config)
    print("Extracted Text for Name:", name_text)  # Debugging line

    # Extract Arabic name
    name_arabic = re.findall(r'[\u0600-\u06FF]+ [\u0600-\u06FF]+ [\u0600-\u06FF]+ [\u0600-\u06FF]+', name_text)

    return name_text, name_arabic[0] if name_arabic else None

def extract_dob(roi_dob):
    # Preprocess the ROI
    roi_dob = preprocess_image(roi_dob)

    # Use OCR to extract text from the ROI
    dob_text = pytesseract.image_to_string(roi_dob, config='--psm 6')
    print("Extracted Text for DOB:", dob_text)  # Debugging line

    # Extract and convert Arabic date of birth
    dob_arabic = re.findall(r'[\u0660-\u0669]{4}/[\u0660-\u0669]{2}/[\u0660-\u0669]{2}', dob_text)
    if dob_arabic:
        dob_english = arabic_to_english(dob_arabic[0])
    else:
        dob_english = None

    return dob_english

def extract_text_from_image(image_path):
    try:
        # Pre-processing for clearer images (convert to grayscale and enhance contrast)
        image = Image.open(image_path)
        image = image.convert("L")
        
        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(image, lang='eng+ara', config='--psm 6')
        
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

def extract_details(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define regions of interest (ROI) for ID, DOB, and name based on provided coordinates
    roi_name = gray[130:200, 650:gray.shape[1]]  # Expand vertically and to the right for name in Arabic
    roi_dob = gray[300:350, 300:600]  # Further adjusted for DOB in Arabic
    roi_id = gray[600:650, 20:270]    # Adjusted for ID number in English

    # Extract name
    name_text, name_arabic = extract_name(roi_name)
    
    # Extract DOB
    dob = extract_dob(roi_dob)

    # Apply preprocessing to improve OCR accuracy for ID
    roi_id = cv2.GaussianBlur(roi_id, (5, 5), 0)  # Reduce noise with Gaussian Blur
    roi_id = cv2.threshold(roi_id, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Use OCR to extract text from each ROI
    id_text = pytesseract.image_to_string(roi_id, config='--psm 6')

    # Extract required details
    id_number = re.findall(r'\b\d{10}\b', id_text)

    return id_number[0] if id_number else None, dob, name_text, name_arabic

def process_image(image_path):
    try:
        # Extract text from image
        extracted_text = extract_text_from_image(image_path)
        print("Extracted Text from Image:", extracted_text)  # Display the extracted text

        if "Name" in extracted_text:
            # Extract information using the existing function
            data = {
                "Name": extract_name_dob_sex(extracted_text)["Name"],
                "DateOfBirth": extract_name_dob_sex(extracted_text)["DateOfBirth"],
                "Sex": extract_name_dob_sex(extracted_text)["Sex"],
                "IDNumber": extract_id_number(extracted_text),
                "ExpiryDate": extract_expiry_date(extracted_text),
                "IssuingDate": extract_issuing_date(extract_expiry_date(extracted_text)),
                "Occupation": extract_occupation(extracted_text),
                "Employer": extract_employer(extracted_text),
                "IssuingPlace": extract_issuing_place(extracted_text)
            }
            return {"success": True, "data": data}
        else:
            # Extract details using the Arabic text extraction function
            id_number, dob, name_text, name_arabic = extract_details(image_path)
            data = {
                "IDNumber": id_number,
                "DateOfBirth": dob,
                "Name": name_text,
                "ArabicName": name_arabic
            }
            return {"success": True, "data": data}

    except Exception as e:
        return {"success": False, "error": str(e)}


 













        







