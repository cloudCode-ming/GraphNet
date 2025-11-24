import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

X_test = np.load('data/X_test.npy')
X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')
print('Testing Data loading successful')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleTransformer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

input_dim = X_train.shape[1]
num_classes = len(set(y_train))

model = SimpleTransformer(input_dim, num_classes)
model.load_state_dict(torch.load('model/simple_transformer_model.pth'))

model.to(device)
model.eval()

batch_size = 100
correct = 0
total = 0

# DataLoader
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the network on the test data: {accuracy:.2f}%')

print(classification_report(all_labels, all_preds))