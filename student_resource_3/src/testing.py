import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from contextlib import contextmanager
import constants
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the start method for multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Step 1: Set up the environment and import necessary libraries
DATASET_FOLDER = 'dataset'
IMAGE_FOLDER = 'images'

# Function to get the appropriate device (CPU, CUDA, or MPS)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# DataLoader context manager
@contextmanager
def create_dataloader(*args, **kwargs):
    loader = DataLoader(*args, **kwargs)
    try:
        yield loader
    finally:
        if hasattr(loader, '_iterator'):
            del loader._iterator

# Improved download_images function
def download_single_image(image_link, folder):
    try:
        response = requests.get(image_link, timeout=10)
        response.raise_for_status()
        filename = os.path.join(folder, os.path.basename(image_link))
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Error downloading {image_link}: {str(e)}")
        return False

def download_images(image_links, folder):
    os.makedirs(folder, exist_ok=True)
    successful_downloads = 0
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_single_image, link, folder) for link in image_links]
        for future in tqdm(as_completed(futures), total=len(image_links), desc="Downloading images"):
            if future.result():
                successful_downloads += 1
    
    logger.info(f"Downloaded {successful_downloads} out of {len(image_links)} images.")

# Step 2: Implement image processing
class ProductDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.is_train = is_train
        self.entity_unit_map = constants.entity_unit_map
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(self.entity_unit_map.keys())}
        self.unit_to_idx = {unit: idx for idx, unit in enumerate(set.union(*self.entity_unit_map.values()))}

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch) if batch else None
    def parse_string(self, s):
        try:
            s = s.strip()
            # Remove square brackets if present
            s = s.strip('[]')
            
            # Check if the string contains a range with 'to'
            if ' to ' in s:
                parts = s.split(' to ')
                if len(parts) != 2:
                    raise ValueError(f"Invalid 'to' range format: {s}")
                
                start_value, end_value = parts
                start_num = float(start_value.split()[0])
                end_num = float(end_value.split()[0])
                value = (start_num + end_num) / 2  # Use average of range
                unit = ' '.join(end_value.split()[1:])  # Use the unit from the end part
            else:
                # Use regex to separate numeric part and unit part
                match = re.match(r'([\d.,]+)\s*(.+)', s)
                if not match:
                    raise ValueError(f"Invalid format: {s}")
                
                numeric_part, unit = match.groups()
                
                # Handle comma-separated ranges (e.g., "100.0, 240.0")
                if ',' in numeric_part:
                    values = [float(v.strip()) for v in numeric_part.split(',')]
                    value = sum(values) / len(values)  # Use average of range
                else:
                    value = float(numeric_part)
            
            # Remove any trailing brackets from the unit
            unit = unit.rstrip('])')
            
            return value, unit.strip()
        except Exception as e:
            logger.error(f"Error parsing string '{s}': {str(e)}")
            return None, None

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.image_folder, os.path.basename(self.data.iloc[idx]['image_link']))
            if not os.path.exists(img_name):
                logger.error(f"Image file not found: {img_name}")
                return None

            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)

            group_id = self.data.iloc[idx]['group_id']
            entity_name = self.data.iloc[idx]['entity_name']
            entity_idx = self.entity_to_idx.get(entity_name)
            if entity_idx is None:
                logger.error(f"Unknown entity name: {entity_name}")
                return None

            if self.is_train:
                value, unit = self.parse_string(self.data.iloc[idx]['entity_value'])
                if value is None or unit is None:
                    return None
                value = np.float32(value)
                unit_idx = self.unit_to_idx.get(unit)
                if unit_idx is None:
                    logger.error(f"Unknown unit: {unit}")
                    return None
                return image, group_id, entity_idx, value, unit_idx
            else:
                return image, group_id, entity_idx, self.data.iloc[idx]['index']
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            return None

# Step 3: Create an improved prediction model
class ImprovedModel(nn.Module):
    def __init__(self, num_groups, num_entities, num_units):
        super(ImprovedModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        resnet_out_features = 512  # ResNet output size after pooling
        input_size = resnet_out_features + num_groups + num_entities  # Total input size for fc1
        
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_value = nn.Linear(128, 1)
        self.fc_unit = nn.Linear(128, num_units)

    def forward(self, x, group_id, entity_idx):
        x = self.resnet(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fix one-hot encoding: Ensure `num_groups` is used correctly
        group_one_hot = nn.functional.one_hot(group_id, num_classes=num_groups).float()
        entity_one_hot = nn.functional.one_hot(entity_idx, num_classes=num_entities).float()
        
        logger.info(f"Group one-hot shape: {group_one_hot.shape}")  # Check this value
        logger.info(f"Entity one-hot shape: {entity_one_hot.shape}")  # Check this value
        
        # Concatenate ResNet features and one-hot encodings
        x = torch.cat((x, group_one_hot, entity_one_hot), dim=1)
        logger.info(f"Final concatenated input shape: {x.shape}")
        
        # Pass through the fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        value = self.fc_value(x)
        unit = self.fc_unit(x)
        return value, unit

# Step 4: Implement training procedure
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            images, group_ids, entity_idxs, values, unit_idxs = batch
            logger.info(f"Batch shapes: images {images.shape}, group_ids {group_ids.shape}, entity_idxs {entity_idxs.shape}")
            
            images = images.to(device)
            group_ids = group_ids.to(device).long()
            entity_idxs = entity_idxs.to(device).long()
            values = values.to(device).float().unsqueeze(1)
            unit_idxs = unit_idxs.to(device)

            optimizer.zero_grad()
            try:
                value_pred, unit_pred = model(images, group_ids, entity_idxs)
            except Exception as e:
                logger.error(f"Error in forward pass: {str(e)}")
                raise

            loss_value = criterion(value_pred, values)
            loss_unit = nn.functional.cross_entropy(unit_pred, unit_idxs)
            loss = loss_value + loss_unit
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, group_ids, entity_idxs, values, unit_idxs = batch
                images = images.to(device)
                group_ids = group_ids.to(device)
                entity_idxs = entity_idxs.to(device)
                values = values.to(device).float().unsqueeze(1)
                unit_idxs = unit_idxs.to(device)

                value_pred, unit_pred = model(images, group_ids, entity_idxs)
                loss_value = criterion(value_pred, values)
                loss_unit = nn.functional.cross_entropy(unit_pred, unit_idxs)
                loss = loss_value + loss_unit
                val_loss += loss.item()

        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    logger.info("Training completed.")

# Step 5: Process the test data and generate predictions
def process_test_data(model, test_loader, device, unit_idx_to_unit):
    model.eval()
    predictions = []
    indices = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing test data"):
            if batch is None:
                continue
            images, group_ids, entity_idxs, idx = batch
            images = images.to(device)
            group_ids = group_ids.to(device)
            entity_idxs = entity_idxs.to(device)

            try:
                value_pred, unit_pred = model(images, group_ids, entity_idxs)
                value = value_pred.squeeze().cpu().numpy()
                unit_idx = torch.argmax(unit_pred, dim=1).cpu().numpy()
                unit = [unit_idx_to_unit[idx] for idx in unit_idx]
            except Exception as e:
                logger.error(f"Error in test processing: {str(e)}")
                value, unit = None, None  # Handle failed prediction gracefully

            for v, u in zip(value, unit):
                predictions.append((v, u) if v is not None and u else ("", ""))  # Handle missing data

            indices.extend(idx.numpy())

    return indices, predictions

# Step 6: Format and save the output
def format_prediction(prediction):
    value, unit = prediction
    if not unit:  # Handle missing units
        return ""
    return f"{value:.2f} {unit}"

def save_predictions(indices, predictions, output_file):
    df = pd.DataFrame({
        'index': indices,
        'prediction': [format_prediction(pred) for pred in predictions]
    })
    df.to_csv(output_file, index=False)

# Main execution
if __name__ == "__main__":
    # Download images (for both train and test sets)
    logger.info("Downloading images...")
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train2.csv'))
    test_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'test2.csv'))
    all_image_links = pd.concat([train_df['image_link'], test_df['image_link']])
    download_images(all_image_links, IMAGE_FOLDER)

    # Set up data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ProductDataset(os.path.join(DATASET_FOLDER, 'train2.csv'), IMAGE_FOLDER, transform, is_train=True)
    
    # Create a small validation set (10% of train data)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    test_dataset = ProductDataset(os.path.join(DATASET_FOLDER, 'test2.csv'), IMAGE_FOLDER, transform, is_train=False)


    # Set up model
    device = get_device()
    logger.info(f"Using device: {device}")
    # num_groups = train_df['group_id'].nunique()

    # Access the original dataset through the `dataset` attribute
    original_train_dataset = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset

    # num_entities = len(original_train_dataset.entity_to_idx)
    # num_units = len(original_train_dataset.unit_to_idx)
    num_groups = 1000  # Define a reasonable number of groups
    num_entities = 1000  # Example number of entities
    num_units = 100  # Example output units

    model = ImprovedModel(num_groups=num_groups, num_entities=num_entities, num_units=num_units)

    logger.info(f"Number of groups: {num_groups}")
    logger.info(f"Number of entities: {num_entities}")
    logger.info(f"Number of units: {num_units}")
    model = ImprovedModel(num_groups, num_entities, num_units).to(device)
    
    # Define output_filename here
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')

    # Train the model
    try:
        with create_dataloader(train_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=ProductDataset.collate_fn) as train_loader, \
             create_dataloader(val_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=ProductDataset.collate_fn) as val_loader:
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            num_epochs = 10  # Adjust as needed
            train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

        # Load the best model
        model.load_state_dict(torch.load('best_model.pth', weights_only=True))

        # Process test data
        with create_dataloader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=ProductDataset.collate_fn) as test_loader:
            original_train_dataset = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
            unit_idx_to_unit = {idx: unit for unit, idx in original_train_dataset.unit_to_idx.items()}
            indices, predictions = process_test_data(model, test_loader, device, unit_idx_to_unit)

        # Save predictions
        save_predictions(indices, predictions, output_filename)

        logger.info(f"Prediction process completed. Output saved to: {output_filename}")

        # Run sanity check
        from sanity import sanity_check
        sanity_check(os.path.join(DATASET_FOLDER, 'test2.csv'), output_filename)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise