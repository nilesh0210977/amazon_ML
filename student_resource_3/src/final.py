import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import requests
from io import BytesIO

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.entity_to_index = {entity: idx for idx, entity in enumerate(self.data['entity_name'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_url = self.data.iloc[idx, 0]
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = self.transform(image)
        entity_name = self.data.iloc[idx, 2]
        entity_index = self.entity_to_index[entity_name]
        entity_value = float(self.data.iloc[idx, 3].split()[0])
        return image, torch.tensor(entity_index, dtype=torch.long), torch.tensor(entity_value, dtype=torch.float)

class ImageTextModel(nn.Module):
    def __init__(self, num_entities):
        super().__init__()
        self.cnn = torchvision.models.resnet18(weights=None)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 128)
        self.text_embedding = nn.Embedding(num_entities, 64)
        self.fc = nn.Linear(128 + 64, 1)

    def forward(self, image, text):
        image_features = self.cnn(image)
        text_features = self.text_embedding(text)
        combined = torch.cat((image_features, text_features), dim=1)
        return self.fc(combined)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, entity_indices, entity_values in train_loader:
            images = images.float()  # Ensure images are float
            outputs = model(images, entity_indices)
            loss = criterion(outputs.squeeze(), entity_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Main execution
dataset = CustomDataset('dataset/train2.csv')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
num_entities = len(dataset.entity_to_index)
model = ImageTextModel(num_entities)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert model parameters to float
model.float()

# train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Prediction function (commented out in original code)
def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, entity_indices, _ in test_loader:
            images = images.float()  # Ensure images are float
            outputs = model(images, entity_indices)
            predictions.extend(outputs.squeeze().tolist())
    return predictions

# Generate predictions and save to CSV

test_dataset = CustomDataset('dataset/train2.csv')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
predictions = predict(model, test_loader)
results = pd.DataFrame({'index': range(len(predictions)), 'prediction': predictions})
results['prediction'] = results['prediction'].apply(lambda x: f"{x:.2f}")  # Format as desired
results.to_csv('predictions.csv', index=False)