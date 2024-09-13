import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import os
import multiprocessing

# Assuming the utility functions are in a file named utils.py
from utils import download_images, parse_string

def setup_mps():
    try:
        # Check if MPS is available
        if tf.test.is_built_with_mps() and tf.config.experimental.list_physical_devices('MPS'):
            print("MPS is available. Using MPS backend.")
            tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('MPS')[0], 'MPS')
        else:
            print("MPS is not available. Using CPU.")
    except:
        print("Error setting up MPS. Falling back to CPU.")

def load_and_preprocess_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    all_images = pd.concat([train_data['image_link'], test_data['image_link']]).unique()
    download_folder = 'downloaded_images'
    download_images(all_images, download_folder)
    
    train_data['parsed_value'], train_data['parsed_unit'] = zip(*train_data['entity_value'].apply(parse_string))
    
    return train_data, test_data

def extract_image_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = model.predict(x)
    return features.flatten()

def prepare_data(data, le_entity_name, is_train=True):
    X = []
    for _, row in data.iterrows():
        image_path = os.path.join('downloaded_images', os.path.basename(row['image_link']))
        image_features = extract_image_features(image_path)
        
        features = np.concatenate([
            image_features,
            [row['group_id']],
            le_entity_name.transform([row['entity_name']])
        ])
        X.append(features)
    
    X = np.array(X)
    
    if is_train:
        y = data['parsed_value'].values
        return X, y
    else:
        return X

def train_and_predict():
    setup_mps()

    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data('dataset/train2.csv', 'dataset/test2.csv')

    # Encode categorical variables
    le_entity_name = LabelEncoder()
    le_entity_name.fit(pd.concat([train_data['entity_name'], test_data['entity_name']]))

    # Prepare training data
    X_train, y_train = prepare_data(train_data, le_entity_name)

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation MSE: {mse}")

    # Prepare test data and make predictions
    X_test = prepare_data(test_data, le_entity_name, is_train=False)
    test_predictions = model.predict(X_test)

    # Create submission file
    test_data['predicted_value'] = test_predictions
    test_data[['index', 'predicted_value']].to_csv('submission.csv', index=False)

    print("Predictions saved to submission.csv")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_and_predict()