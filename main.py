import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def preprocess_data(file_path):
    data_df = pd.read_csv(file_path, header=None, low_memory=False, nrows=100000)

    data_df.fillna('0', inplace=True)
    features = data_df.iloc[:, :-1]
    labels = data_df.iloc[:, -1].values

    COL_TYPES = {
        'object': [0, 1, 2, 3, 4, 5, 13, 47],
        'numeric': list(set(range(44)) - {0, 1, 2, 3, 4, 5, 13, 47})
    }

    # 语义列映射
    COL_MAP = {
        'src_ip': 0,
        'src_port': 1,
        'dst_ip': 2,
        'dst_port': 3,
        'protocol': 4,
        'conn_state': 5,
        'duration': 6,
        'service': 13,
        'label': -1
    }

    for port_col in [COL_MAP['src_port'], COL_MAP['dst_port']]:
        features[port_col] = pd.to_numeric(features[port_col], errors='coerce')
        features[port_col] = features[port_col].fillna(-1)
        features[port_col] = features[port_col].clip(0, 65535)

    categorical_cols = {
        COL_MAP['protocol']: 'protocol',
        COL_MAP['conn_state']: 'conn_state',
        COL_MAP['service']: 'service'
    }
    label_encoders = {}
    for col, name in categorical_cols.items():
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col].astype(str))
        features[col] = features[col].replace(-1, len(le.classes_))  # 新增未知类别
        label_encoders[name] = le

    ip_encoder = LabelEncoder()
    all_ips = pd.concat([features[COL_MAP['src_ip']],
                         features[COL_MAP['dst_ip']]]).astype(str)
    ip_encoder.fit(all_ips)

    numeric_cols = [i for i in COL_TYPES['numeric'] if i not in [COL_MAP['src_port'], COL_MAP['dst_port']]]
    scaler = MinMaxScaler()
    features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

    port_scaler = MinMaxScaler(feature_range=(0, 1))
    ports = features[[COL_MAP['src_port'], COL_MAP['dst_port']]].values.astype(float)
    features[[COL_MAP['src_port'], COL_MAP['dst_port']]] = port_scaler.fit_transform(ports)

    edge_index = torch.tensor([
        ip_encoder.transform(features[COL_MAP['src_ip']].astype(str)),
        ip_encoder.transform(features[COL_MAP['dst_ip']].astype(str))
    ], dtype=torch.long)

    edge_attr_components = [
        features[COL_MAP['protocol']].values.reshape(-1, 1),  # 协议
        features[COL_MAP['conn_state']].values.reshape(-1, 1),  # 连接状态
        features[COL_MAP['src_port']].values.reshape(-1, 1),  # 源端口
        features[COL_MAP['dst_port']].values.reshape(-1, 1),  # 目的端口
        features[numeric_cols].values
    ]
    edge_attr = np.hstack(edge_attr_components).astype(np.float32)

    data = Data(
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_attr),
        y=torch.tensor(labels, dtype=torch.long)
    )

    indices = np.arange(edge_index.shape[1])
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42)

    data.train_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    data.val_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    data.test_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data, len(ip_encoder.classes_), edge_attr.shape[1]


class GNNFeatureExtractor(nn.Module):
    def __init__(self, num_nodes, emb_dim=64, edge_feat_dim=None, out_dim=128):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.feature_net = nn.Sequential(
            nn.Linear(emb_dim * 2 + edge_feat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, out_dim)
        )

    def forward(self, edge_index, edge_attr):
        src, dst = edge_index
        h_src = self.node_emb(src)
        h_dst = self.node_emb(dst)
        combined = torch.cat([h_src, h_dst, edge_attr], dim=1)
        return self.feature_net(combined)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, num_classes=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class GNNTranformerModel(nn.Module):
    def __init__(self, gnn, transformer):
        super().__init__()
        self.gnn = gnn
        self.transformer = transformer

    def forward(self, edge_index, edge_attr):
        gnn_features = self.gnn(edge_index, edge_attr)
        combined = torch.cat([edge_attr, gnn_features], dim=1)
        return self.transformer(combined)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "data/UNSW-NB15_1.csv"

    data, num_nodes, edge_attr_dim = preprocess_data(data_path)
    data = data.to(device)

    gnn = GNNFeatureExtractor(num_nodes, edge_feat_dim=edge_attr_dim).to(device)
    transformer = TransformerClassifier(
        input_dim=edge_attr_dim + 64*2  # GNN输出64维
    ).to(device)

    model = GNNTranformerModel(gnn, transformer)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.edge_index, data.edge_attr)
            val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred[data.val_mask] == data.y[data.val_mask]).float().mean()

        scheduler.step(val_acc)

        print(f'Epoch {epoch + 1:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            patience = 0
        else:
            patience += 1
            if patience > 10:
                print("Early stopping!")
                break

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        test_out = model(data.edge_index, data.edge_attr)
        test_pred = test_out.argmax(dim=1)
        test_acc = (test_pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    print(f'Test Accuracy: {test_acc:.4f}')


if __name__ == "__main__":
    main()