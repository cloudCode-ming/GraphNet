import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

X_train = np.load('data/X_train.npy')
y_train = np.load('data/y_train.npy')
print('Training data loading successful')

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

model = SimpleTransformer(input_dim, num_classes).to(device)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 3

print('Start training:')
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'model/simple_transformer_model.pth')

