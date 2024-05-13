import requests

# URL of the online image
image_url = 'https://www.edarabia.com/wp-content/uploads/2018/02/all-you-need-know-about-emirates-id.jpg'

# Download the image
response = requests.get(image_url)

# Check if the request was successful
if response.status_code == 200:
    # Save the image to a file
    with open('image.jpg', 'wb') as f:
        f.write(response.content)
    
    # Upload the downloaded image
    url = 'https://ocr-intelpeek-dev-b568.up.railway.app/upload'
    files = {'file': open('image.jpg', 'rb')}
    resp = requests.post(url=url, files=files)
    
    # Print the response
    print(resp.json())
else:
    print("Failed to download the image")  