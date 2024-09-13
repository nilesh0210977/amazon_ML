import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
from transformers import BertTokenizer, BertModel

# class CustomDataset(Dataset):
#     def __init__(self, csv_file, is_test=False):
#         self.data = pd.read_csv(csv_file)
#         self.is_test = is_test
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
#         # Dynamically create entity_types from the data
#         if is_test:
#             self.entity_types = sorted(self.data['entity_name'].unique())
#         else:
#             self.entity_types = sorted(set(self.data['entity_name'].unique()) | 
#                                        {'item_weight', 'item_volume', 'height', 'width', 'depth', 'wattage'})
        
#         self.entity_to_index = {entity: idx for idx, entity in enumerate(self.entity_types)}

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         if self.is_test:
#             image_url = self.data.iloc[idx, 1]  # Image URL is in the second column for test data
#             group_id = self.data.iloc[idx, 2]  # Group ID is in the third column for test data
#             entity_name = self.data.iloc[idx, 3]  # Entity name is in the fourth column for test data
#         else:
#             image_url = self.data.iloc[idx, 0]  # Image URL is in the first column for train data
#             group_id = self.data.iloc[idx, 1]  # Group ID is in the second column for train data
#             entity_name = self.data.iloc[idx, 2]  # Entity name is in the third column for train data

#         response = requests.get(image_url)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#         image = self.transform(image)
        
#         entity_index = self.entity_to_index[entity_name]
        
#         # Combine group_id and entity_name for text input
#         text_input = f"{group_id} {entity_name}"
#         text_encoding = self.tokenizer(text_input, return_tensors='pt', padding='max_length', truncation=True, max_length=32)
        
#         if self.is_test:
#             return image, text_encoding['input_ids'].squeeze(), text_encoding['attention_mask'].squeeze(), torch.tensor(entity_index, dtype=torch.long)
#         else:
#             # Parse entity value and unit for training data
#             value_unit = self.data.iloc[idx, 3].split()
#             entity_value = float(value_unit[0])
#             unit = value_unit[1] if len(value_unit) > 1 else ''
            
#             # Normalize values to a common unit (e.g., grams for weight, liters for volume)
#             if entity_name == 'item_weight':
#                 if unit == 'milligram':
#                     entity_value /= 1000
#                 elif unit == 'kilogram':
#                     entity_value *= 1000
#             elif entity_name == 'item_volume':
#                 if unit == 'cup':
#                     entity_value *= 0.236588  # Convert cups to liters
            
#             return image, text_encoding['input_ids'].squeeze(), text_encoding['attention_mask'].squeeze(), torch.tensor(entity_index, dtype=torch.long), torch.tensor(entity_value, dtype=torch.float)

# class ImageTextModel(nn.Module):
#     def __init__(self, num_entity_types):
#         super().__init__()
#         self.cnn = torchvision.models.resnet18(pretrained=True)
#         self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.entity_embedding = nn.Embedding(num_entity_types, 64)
#         self.fc = nn.Linear(128 + 768 + 64, 1)  # 768 is BERT's hidden size

#     def forward(self, image, input_ids, attention_mask, entity_index):
#         image_features = self.cnn(image)
#         text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         text_features = text_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
#         entity_features = self.entity_embedding(entity_index)
#         combined = torch.cat((image_features, text_features, entity_features), dim=1)
#         return self.fc(combined)

# # Training loop
# def train_model(model, train_loader, criterion, optimizer, num_epochs):
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for images, input_ids, attention_mask, entity_indices, entity_values in train_loader:
#             images = images.float()
#             outputs = model(images, input_ids, attention_mask, entity_indices)
#             loss = criterion(outputs.squeeze(), entity_values)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_loss = total_loss / len(train_loader)
#         print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# # Prediction function
# def predict(model, test_loader):
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for images, input_ids, attention_mask, entity_indices in test_loader:
#             images = images.float()
#             outputs = model(images, input_ids, attention_mask, entity_indices)
#             predictions.extend(outputs.squeeze().tolist())
#     return predictions

# # Main execution
# train_dataset = CustomDataset('dataset/train2.csv', is_test=False)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# num_entity_types = len(train_dataset.entity_to_index)
# model = ImageTextModel(num_entity_types)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# # Generate predictions and save to CSV
# test_dataset = CustomDataset('dataset/test2.csv', is_test=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# predictions = predict(model, test_loader)
# results = pd.DataFrame({'index': range(len(predictions)), 'prediction': predictions})
# results['prediction'] = results['prediction'].apply(lambda x: f"{x:.2f}")  # Format as desired
# results.to_csv('predictions.csv', index=False)




class CustomDataset(Dataset):
    def __init__(self, csv_file, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.is_test = is_test
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Use a fixed set of entity types for both train and test
        self.entity_types = ['item_weight', 'item_volume', 'height', 'width', 'depth', 'wattage']
        self.entity_to_index = {entity: idx for idx, entity in enumerate(self.entity_types)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_test:
            image_url = self.data.iloc[idx, 1]
            group_id = self.data.iloc[idx, 2]
            entity_name = self.data.iloc[idx, 3]
        else:
            image_url = self.data.iloc[idx, 0]
            group_id = self.data.iloc[idx, 1]
            entity_name = self.data.iloc[idx, 2]

        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = self.transform(image)
        
        entity_index = self.entity_to_index.get(entity_name, 0)  # Use 0 as default if entity not found
        
        text_input = f"{group_id} {entity_name}"
        text_encoding = self.tokenizer(text_input, return_tensors='pt', padding='max_length', truncation=True, max_length=32)
        
        if self.is_test:
            return image, text_encoding['input_ids'].squeeze(), text_encoding['attention_mask'].squeeze(), torch.tensor(entity_index, dtype=torch.long)
        else:
            value_unit = self.data.iloc[idx, 3].split()
            entity_value = float(value_unit[0])
            unit = value_unit[1] if len(value_unit) > 1 else ''
            
            if entity_name == 'item_weight':
                if unit == 'milligram':
                    entity_value /= 1000
                elif unit == 'kilogram':
                    entity_value *= 1000
            elif entity_name == 'item_volume':
                if unit == 'cup':
                    entity_value *= 0.236588
            
            return image, text_encoding['input_ids'].squeeze(), text_encoding['attention_mask'].squeeze(), torch.tensor(entity_index, dtype=torch.long), torch.tensor(entity_value, dtype=torch.float)

class ImageTextModel(nn.Module):
    def __init__(self, num_entity_types):
        super().__init__()
        self.cnn = torchvision.models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.entity_embedding = nn.Embedding(num_entity_types, 64)
        self.fc = nn.Linear(128 + 768 + 64, 1)

    def forward(self, image, input_ids, attention_mask, entity_index):
        image_features = self.cnn(image)
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        entity_features = self.entity_embedding(entity_index)
        combined = torch.cat((image_features, text_features, entity_features), dim=1)
        return self.fc(combined)

def predict(model, test_loader, dataset):
    model.eval()
    predictions = []
    entity_types = []
    with torch.no_grad():
        for images, input_ids, attention_mask, entity_indices in test_loader:
            images = images.float()
            outputs = model(images, input_ids, attention_mask, entity_indices)
            predictions.extend(outputs.squeeze().tolist())
            entity_types.extend([dataset.entity_types[i] for i in entity_indices])
    return predictions, entity_types


def convert_to_unit(value, entity_type):
    if entity_type == 'item_weight':
        return f"{value:.2f} gram"
    elif entity_type == 'item_volume':
        return f"{value:.2f} liter"
    elif entity_type in ['height', 'width', 'depth']:
        return f"{value:.2f} foot"
    elif entity_type == 'wattage':
        if value >= 1000:
            return f"{value/1000:.2f} kilowatt"
        else:
            return f"{value:.2f} watt"
    else:
        return f"{value:.2f}"


# Main execution
train_dataset = CustomDataset('dataset/train2.csv', is_test=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
num_entity_types = len(train_dataset.entity_types)
model = ImageTextModel(num_entity_types)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Uncomment the following line to train the model
# train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Generate predictions and save to CSV
test_dataset = CustomDataset('dataset/test2.csv', is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
predictions, entity_types = predict(model, test_loader, test_dataset)

results = pd.DataFrame({'index': range(len(predictions)), 'prediction': predictions, 'entity_type': entity_types})
results['prediction'] = results.apply(lambda row: convert_to_unit(row['prediction'], row['entity_type']), axis=1)
results = results.drop('entity_type', axis=1)
results.to_csv('predictions.csv', index=False)