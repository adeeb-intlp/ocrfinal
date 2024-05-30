import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import json
from datetime import datetime, timedelta
import os

# Function to process the image
def process_image(image_path):
    try:
        # Process the image
        extracted_text = extract_text_from_image(image_path)
        print("Initial Extracted Text:", extracted_text)

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
                }
            }
            return {"success": True, "data": data}
        else:
            # Pass the image to Arabic text extraction function
            crop_box = (0, 0, 1000, 500)  # Define the crop box based on the document layout
            arabic_text = crop_and_extract(image_path, crop_box)
            print("Arabic Text After Cropping:", arabic_text)

            return {"success": True, "data": {"extracted_data": None, "arabic_text": arabic_text}}

    except Exception as e:
        return {"success": False, "error": str(e)}

def crop_and_extract(image_path, crop_box):
    try:
        # Load the image
        image = Image.open(image_path)

        # Crop the image and retry extraction
        cropped_image = image.crop(crop_box)
        cropped_text = extract_arabic_text(cropped_image)
        return cropped_text

    except Exception as e:
        return str(e)

# Function to extract text from image
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

# Function to extract Arabic text from image
def extract_arabic_text(image):
    # Convert to grayscale
    image = image.convert('L')

    # Apply sharpening filter
    image = image.filter(ImageFilter.SHARPEN)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)

    # Resize the image
    basewidth = 1200
    wpercent = (basewidth / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((basewidth, hsize), Image.LANCZOS)

    # Binarize the image
    image = image.point(lambda x: 0 if x < 128 else 255, '1')

    # Perform OCR with custom configurations
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(image, lang='ara', config=custom_config)

    return extracted_text.strip()

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

        







