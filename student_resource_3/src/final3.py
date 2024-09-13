import os
import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image
import pytesseract
import torch
from torchvision import models, transforms
from io import BytesIO
import re
import easyocr
from collections import defaultdict
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None
def preprocess_image(image):
    # Ensure the image is in RGB format if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])
    
    image_tensor = transform(image)  # Apply transformations
    return image_tensor.unsqueeze(0).to(device)  # Add batch dimension (1, C, H, W)

# def preprocess_image(image):
#     # Ensure the image is in RGB format if it's not already
#     if image.mode != 'RGB':
#         image = image.convert('RGB')

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),  # Convert the image to a tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
#     ])
    
#     image_tensor = transform(image)  # Apply transformations
#     return image_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device


def extract_text_from_image(image):
    reader = easyocr.Reader(['en'], gpu= True)  # Set gpu=True if GPU is available
    result = reader.readtext(np.array(image), detail=0)
    return result

# def extract_image_features(image):
#     model = models.resnet18(pretrained=True).to(device)  # Move the model to MPS
#     model.eval()

#     with torch.no_grad():
#         features = model(image)

#     return features.cpu().numpy()  # Return feature vector, move back to CPU if needed

def extract_entity_value(text_list, entity_name):
    text = ' '.join(text_list)  # Combine all detected text
    
    patterns = {
        'item_weight': r'(\d+(\.\d+)?)\s?(gram|kg|kilogram|ounce|oz|pound|lb)',
        'item_volume': r'(\d+(\.\d+)?)\s?(ml|litre|liter|gallon)',
        'item_dimensions': r'(\d+(\.\d+)?)\s?(cm|inch|millimetre|mm)'
    }
    
    pattern = patterns.get(entity_name)
    if pattern:
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            unit = match.group(3)
            return f'{value} {unit}'
    
    return ''  # Return empty string if no match found

def extract_image_features(image):
    model = models.resnet50(pretrained=True)
    model.eval()
    
    with torch.no_grad():
        features = model(image)
    
    return features.numpy()  # Extracted feature vector

def process_image_and_extract_value(index, image_link, entity_name):
    # Download the image
    image = download_image(image_link)
    if image is None:
        return ''
    
    # Preprocess the image for CNN (if needed for future extensions)
    preprocessed_image = preprocess_image(image)
    
    # Extract text using OCR
    extracted_text = extract_text_from_image(image)
    # extracted_text=extract_image_features(image)
    
    # Extract the entity value using regex
    prediction = extract_entity_value(extracted_text, entity_name)
    
    # Return the prediction (formatted as 'x unit')
    return prediction

# Load test dataset
test_df = pd.read_csv('dataset/test2.csv')

# Prepare list to store predictions
predictions = []

for i, row in test_df.iterrows():
    index = row['index']
    image_link = row['image_link']
    entity_name = row['entity_name']
    
    # Process each image and extract the predicted value
    prediction = process_image_and_extract_value(index, image_link, entity_name)
    
    # Append the result to the predictions list
    predictions.append({
        'index': index,
        'prediction': prediction
    })

# Convert predictions to DataFrame and save to CSV
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('predictions.csv', index=False)

