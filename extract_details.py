import cv2
import pytesseract
import re
import numpy as np
import matplotlib.pyplot as plt
from ArabicOcr import arabicocr

def arabic_to_english(number):
    arabic_english_map = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3',
        '٤': '4', '٥': '5', '٦': '6', '٧': '7',
        '٨': '8', '٩': '9'
    }
    return ''.join(arabic_english_map.get(ch, ch) for ch in number)

def preprocess_image(image, alpha=1.0, beta=0):
    # Adjust brightness and contrast
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def extract_name(roi_name):
    # Preprocess the ROI with minimal adjustments
    roi_name = preprocess_image(roi_name, alpha=1.2, beta=20)

    # Use OCR to extract text from the ROI
    custom_config = r'--oem 3 --psm 6 -l eng+ara'
    name_text = pytesseract.image_to_string(roi_name, config=custom_config)
    print("Extracted Text for Name:", name_text)  # Debugging line

    # Extract Arabic name
    name_arabic = re.findall(r'[\u0600-\u06FF]+ [\u0600-\u06FF]+ [\u0600-\u06FF]+ [\u0600-\u06FF]+', name_text)

    return name_text, name_arabic[0] if name_arabic else None

def extract_dob(roi_dob):
    # Experimenting with different preprocessing techniques
    roi_dob = preprocess_image(roi_dob, alpha=1.2, beta=30)

    # Apply additional preprocessing steps
    roi_dob = cv2.GaussianBlur(roi_dob, (3, 3), 0)
    roi_dob = cv2.threshold(roi_dob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Save the ROI for debugging
    cv2.imwrite("roi_dob.jpg", roi_dob)

    # Display the ROI
    plt.imshow(roi_dob, cmap='gray')
    plt.title('DOB Region')
    plt.show()

    # Use ArabicOcr to extract text from the ROI
    with open("image_part.jpg", 'wb') as f:
        f.write(cv2.imencode('.jpg', roi_dob)[1])
    results = arabicocr.arabic_ocr("image_part.jpg", "out_image.jpg")
    
    # Debugging output for ArabicOCR results
    print("ArabicOCR Results:", results)

    dob_text = " ".join([item[1] for item in results])
    print("Extracted Text for DOB:", dob_text)  # Debugging line

    # Extract and convert Arabic date of birth
    dob_arabic = re.findall(r'[\u0660-\u0669]{4}/[\u0660-\u0669]{2}/[\u0660-\u0669]{2}', dob_text)
    if dob_arabic:
        dob_english = arabic_to_english(dob_arabic[0])
    else:
        dob_english = None

    return dob_english

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

def visualize_rois(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check the image size and adjust ROIs accordingly
    height, width = gray.shape

    # Define regions of interest (ROI) for ID, DOB, and name based on provided coordinates
    roi_name = gray[130:200, 650:width]  # Expand vertically and to the right for name in Arabic
    roi_dob = gray[300:350, 300:600]  # Further adjusted for DOB in Arabic
    roi_id = gray[600:650, 20:270]    # Adjusted for ID number in English

    # Display the ROIs
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(roi_name, cmap='gray')
    plt.title('Name Region')

    plt.subplot(1, 3, 2)
    plt.imshow(roi_dob, cmap='gray')
    plt.title('DOB Region')

    plt.subplot(1, 3, 3)
    plt.imshow(roi_id, cmap='gray')
    plt.title('ID Region')

    plt.show()

if __name__ == "__main__":
    # Visualize ROIs to verify correctness
    image_path = "Image.jpeg"  # Replace with your image path
    visualize_rois(image_path)
    
    # Extract details
    id_number, dob, name_text, name_arabic = extract_details(image_path)
    print(f"ID Number: {id_number}")
    print(f"Date of Birth: {dob}")
    print(f"Extracted Text for Name: {name_text}")
    print(f"Arabic Name: {name_arabic}")












