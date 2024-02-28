import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
import torch.optim as optim
import numpy as np

from sklearn.metrics import roc_auc_score
from collections import defaultdict

# Base GNN Model
class BaseGNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BaseGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Base GNN Model with Learnable Node Parameters
class BaseGNNLearnableNodeParams(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels):
        super(BaseGNNLearnableNodeParams, self).__init__()
        self.num_nodes = num_nodes
        self.node_features = nn.Parameter(torch.randn(num_nodes, in_channels))
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)
    
    def forward(self, data):
        x = self.node_features
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Node Prediction Model
class NodePredictionModel(nn.Module):
    def __init__(self, base_model, n_classes=10):
        super(NodePredictionModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.conv2.out_channels, n_classes)  #classes for node prediction
    
    def forward(self, data):
        x = self.base_model(data)
        x = F.relu(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# Link Prediction Model
class LinkPredictionModel(nn.Module):
    def __init__(self, base_model):
        super(LinkPredictionModel, self).__init__()
        self.base_model = base_model
        self.link_classifier = nn.Linear(256, 1)
        
    def forward(self, data, edge_index_pos, edge_index_neg):
        x = self.base_model(data)
        x_pos = torch.cat([x[edge_index_pos[0]], x[edge_index_pos[1]]], dim=1)
        x_neg = torch.cat([x[edge_index_neg[0]], x[edge_index_neg[1]]], dim=1)
        
        x_pos = self.link_classifier(x_pos).squeeze(-1)
        x_neg = self.link_classifier(x_neg).squeeze(-1)
        
        return x_pos, x_neg

# Joint Model for Node and Link Prediction
class JointModel(nn.Module):
    def __init__(self, in_channels):
        super(JointModel, self).__init__()
        self.base_model = BaseGNN(in_channels, 128)
        self.node_model = NodePredictionModel(self.base_model)
        self.link_model = LinkPredictionModel(self.base_model)

    def forward(self, data, edge_index_pos, edge_index_neg):
        node_pred = self.node_model(data)
        link_pred_pos, link_pred_neg = self.link_model(data, edge_index_pos, edge_index_neg)
        return node_pred, link_pred_pos, link_pred_neg


def joint_loss(node_pred, node_labels, link_pred_pos, link_pred_neg):
    node_loss = F.nll_loss(node_pred, node_labels)
    
    link_labels_pos = torch.ones(link_pred_pos.shape[0]).to(link_pred_pos.device)
    link_labels_neg = torch.zeros(link_pred_neg.shape[0]).to(link_pred_neg.device)
    
    link_loss = F.binary_cross_entropy_with_logits(link_pred_pos, link_labels_pos) + \
                F.binary_cross_entropy_with_logits(link_pred_neg, link_labels_neg)
    
    return node_loss + link_loss


def sample_edges(edge_index, num_samples, balance_degree=False):
    """
    Sample positive and negative edges from a graph represented by its edge index.

    Parameters:
        edge_index (Tensor): The edge index tensor.
        num_samples (int): Number of negative samples to generate.
        balance_degree (bool): Whether to use degree-based sampling (need to add, as first attempt was sloooow)

    Returns:
        pos_samples (Tensor): Tensor of positive edge samples.
        neg_samples (Tensor): Tensor of negative edge samples.
    """
    print("Step 1: Determine number of nodes in the graph")
    num_nodes = edge_index.max().item() + 1

    print("Step 2: Select positive samples")
    # Positive samples
    pos_samples = edge_index.t()
    
    if pos_samples.shape[0] < num_samples:
        raise ValueError(f"Not enough edges in graph to sample {num_samples} positive samples.")

    pos_samples = pos_samples[:num_samples]

    print(f"Positive samples:\n{len(pos_samples)}")

    print("Step 3: Create an adjacency set for fast lookup")
    # Create an adjacency set for fast lookup
    adjacency_set = set([(u.item(), v.item()) for u, v in pos_samples])

    print("Step 4: Perform negative sampling")
    # Negative sampling
    neg_samples = set()

    print("Sampling candidate negative samples...")
    while len(neg_samples) < num_samples:
        u, v = np.random.randint(0, num_nodes, 2)
        if u != v and (u, v) not in adjacency_set and (v, u) not in adjacency_set:
            print(f"Adding negative sample: ({u}, {v})") if u+v %10 ==0 else None
            neg_samples.add((u, v))

    neg_samples = torch.tensor(list(neg_samples), dtype=torch.long)

    print(f"Negative samples:\n{len(neg_samples)}")

    return pos_samples.t(), neg_samples.t()

if __name__ == '__main__':
    # use Planetoid's CORA dataset as an example.

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and create train/validation Data objects
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)

    train_data = data  # For this example, same data for training and validation (usually these should be different)
    val_data = data

    train_loader = DataLoader([train_data], batch_size=32, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=32, shuffle=False)

    joint_model = JointModel(dataset.num_features).to(device)
    optimizer = optim.Adam(joint_model.parameters(), lr=0.01)

    edge_index_pos_train, edge_index_neg_train = sample_edges(train_data.edge_index, train_data.num_edges)
    edge_index_pos_val, edge_index_neg_val = sample_edges(val_data.edge_index, 100)

    # Make sure to move sampled edges to the same device as your model
    edge_index_pos_train, edge_index_neg_train = edge_index_pos_train.to(device), edge_index_neg_train.to(device)
    edge_index_pos_val, edge_index_neg_val = edge_index_pos_val.to(device), edge_index_neg_val.to(device)

    # Training Loop
    joint_model.train()
    for epoch in range(100):
        for batch in train_loader:
            optimizer.zero_grad()

            batch = batch.to(device)

            node_pred, link_pred_pos, link_pred_neg = joint_model(batch, edge_index_pos_train, edge_index_neg_train)

            loss = joint_loss(node_pred, batch.y, link_pred_pos, link_pred_neg)

            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch+1}, Train Loss: {loss.item()}')

        # Validation Loop
        joint_model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                node_pred_val, link_pred_pos_val, link_pred_neg_val = joint_model(batch, edge_index_pos_val, edge_index_neg_val)
                val_loss = joint_loss(node_pred_val, batch.y, link_pred_pos_val, link_pred_neg_val)

                # Node prediction metrics
                pred = node_pred_val.argmax(dim=1)
                node_correct = pred.eq(batch.y).sum().item()
                node_accuracy = node_correct / pred.shape[0]

            # Link prediction metrics
            link_labels = torch.cat([torch.ones(link_pred_pos_val.shape[0]), torch.zeros(link_pred_neg_val.shape[0])]).to(device)
            link_preds = torch.cat([link_pred_pos_val, link_pred_neg_val])
            roc_score = roc_auc_score(link_labels.detach().cpu(), link_preds.detach().cpu())

            print(f'-- Validation Loss: {val_loss.item()}, Node Classification Accuracy: {node_accuracy}, Link Prediction ROC: {roc_score}')
            print()
