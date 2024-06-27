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

    # Use OCR to extract text from each ROI
    id_text = pytesseract.image_to_string(roi_id, config='--psm 6')

    # Extract required details
    id_number = re.findall(r'\b\d{10}\b', id_text)

    return id_number[0] if id_number else None, dob, name_text, name_arabic

def extract_passport_details(text):
    def extract_name(text):
        name_pattern = r'P<[^<]+<<([^<]+)'  # Regex to match the name pattern after 'P<'
        name_match = re.search(name_pattern, text)
        if name_match:
            name_parts = name_match.group(1).replace('<', ' ').strip()
            names = ' '.join(name_parts.split())  # Normalize spaces
            # Ensure the name contains at least two words
            name_list = names.split()
            if len(name_list) >= 2:
                return ' '.join(name_list[:2])  # Return the first two words as the name
            else:
                return names  # Return the name as is if it's a single word
        else:
            return None

    def extract_gender(text):
        gender_pattern = r'([MF])\d{7}'  # Regex to match the gender pattern (M or F followed by exactly 7 digits)
        gender_match = re.search(gender_pattern, text)
        if gender_match:
            return gender_match.group(1).strip()
        else:
            return None

    def extract_passport_id(text):
        id_pattern = r'\b([A-Z]\d{7})\b'  # Regex to match the passport ID pattern (1 letter followed by 7 digits)
        id_match = re.search(id_pattern, text)
        if id_match:
            return id_match.group(1).strip()
        else:
            return None

    def extract_expiry_date(text):
        expiry_pattern = r'([MF])(\d{6})\d'  # Regex to match the gender (M/F) followed by 6 digits and 1 insignificant digit
        expiry_match = re.search(expiry_pattern, text)
        if expiry_match:
            expiry_digits = expiry_match.group(2).strip()
            expiry_digits = expiry_digits.replace('O', '0')  # Replace 'O' with '0'
            return f"{expiry_digits[4:6]}{expiry_digits[2:4]}{expiry_digits[0:2]}"  # Rearrange digits to DDMMYY format
        else:
            return None

    def extract_dob(text):
        dob_pattern = r'(\d{7}[MF])'  # Regex to match the 7 digits before the gender (M/F)
        dob_match = re.search(dob_pattern, text)
        if dob_match:
            dob_digits = dob_match.group(1)[:-1]  # Exclude the last character (M/F)
            # Clean non-digit characters, replace 'O' with '0'
            dob_digits = re.sub(r'[^0-9]', '', dob_digits.replace('O', '0'))
            if len(dob_digits) == 6:  # Ensure we have exactly 6 digits
                return f"{dob_digits[0:2]}{dob_digits[2:4]}{dob_digits[4:6]}"  # Rearrange digits to DDMMYY format
        return None

    return {
        "ExtractedText": text,
        "Name": extract_name(text),
        "Gender": extract_gender(text),
        "PassportID": extract_passport_id(text),
        "ExpiryDate": extract_expiry_date(text),
        "DateOfBirth": extract_dob(text)
    }

def process_image(image_path):
    try:
        # Process the image
        extracted_text = extract_text_from_image(image_path)

        if "UNITED" in extracted_text:
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
        elif "REPUBLIC" in extracted_text:
            # Extract information using the extract_passport_details function
            passport_details = extract_passport_details(extracted_text)
            data = {
                "IDNumber": passport_details["PassportID"],
                "DateOfBirth": passport_details["DateOfBirth"],
                "Name": passport_details["Name"],
                "Sex": passport_details["Gender"],
                "ExpiryDate": passport_details["ExpiryDate"]
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




 













        







