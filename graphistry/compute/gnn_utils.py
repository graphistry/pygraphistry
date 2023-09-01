import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
import torch.optim as optim

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
    
    def forward(self, data, edge_index_pos, edge_index_neg):
        x = self.base_model(data)
        x_pos = torch.cat([x[edge_index_pos[0]], x[edge_index_pos[1]]], dim=1)
        x_neg = torch.cat([x[edge_index_neg[0]], x[edge_index_neg[1]]], dim=1)
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


def sample_edges(edge_index, num_samples):
    """
    Sample positive and negative edges considering degree dominance.
    
    Parameters:
        edge_index (Tensor): The edge index tensor.
        num_samples (int): The number of positive/negative samples needed.
        
    Returns:
        edge_index_pos (Tensor): Tensor for positive samples.
        edge_index_neg (Tensor): Tensor for negative samples.
    """
    # Create an edge list and a degree dictionary
    edge_list = edge_index.t().tolist()
    degree_dict = defaultdict(int)

    for u, v in edge_list:
        degree_dict[u] += 1
        degree_dict[v] += 1

    # Sort the edge list based on node degree (sum of degrees of both nodes)
    sorted_edge_list = sorted(edge_list, key=lambda x: degree_dict[x[0]] + degree_dict[x[1]])

    # Split into high-degree and low-degree edges
    mid_point = len(sorted_edge_list) // 2
    high_degree_edges = sorted_edge_list[mid_point:]
    low_degree_edges = sorted_edge_list[:mid_point]

    # Sample equally from high-degree and low-degree edges for positive samples
    positive_samples = random.sample(high_degree_edges, num_samples // 2) + random.sample(low_degree_edges, num_samples // 2)
    random.shuffle(positive_samples)

    # Generate negative samples ensuring they are not in the graph
    negative_samples = set()
    while len(negative_samples) < num_samples:
        u, v = random.choice(list(degree_dict.keys())), random.choice(list(degree_dict.keys()))
        if u != v and (u, v) not in edge_list and (v, u) not in edge_list:
            # Take into account the degree to balance high and low degree nodes
            if random.random() < (degree_dict[u] + degree_dict[v]) / (2 * sum(degree_dict.values())):
                negative_samples.add((u, v))

    # Convert to PyTorch tensors
    edge_index_pos = torch.tensor(positive_samples, dtype=torch.long).t().contiguous()
    edge_index_neg = torch.tensor(list(negative_samples), dtype=torch.long).t().contiguous()
    
    return edge_index_pos, edge_index_neg

if __name__ == '__main__':
    # use Planetoid's CORA dataset as an example.
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    # Create data loaders (use your own data loaders if you have custom datasets)
    train_loader = DataLoader([data], batch_size=32, shuffle=True)
    val_loader = DataLoader([data], batch_size=32, shuffle=False)


    # Initialize model and optimizer
    joint_model = JointModel(dataset.num_features)
    optimizer = optim.Adam(joint_model.parameters(), lr=0.003)

    # Split edges for positive and negative samples using degree dominance
    edge_index_pos, edge_index_neg = sample_edges(data.edge_index, num_samples=2 * data.num_edges)
    edge_index_pos_val, edge_index_neg_val = sample_edges(data.edge_index, 100)

    # Training Loop
    joint_model.train()
    for epoch in range(100):
        for batch in train_loader:
            optimizer.zero_grad()
            
            node_pred, link_pred_pos, link_pred_neg = joint_model(batch, edge_index_pos, edge_index_neg)
            
            loss = joint_loss(node_pred, batch.y, link_pred_pos, link_pred_neg)
            
            loss.backward()
            optimizer.step()
            
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Evaluation Loop (simplified)
    joint_model.eval()
    with torch.no_grad():
        correct = 0
        for batch in val_loader:
            node_pred_val, link_pred_pos_val, link_pred_neg_val = joint_model(batch, edge_index_pos_val, edge_index_neg_val)
            val_loss = joint_loss(node_pred_val, batch.y, link_pred_pos_val, link_pred_neg_val)

            # Node prediction metrics
            pred = node_pred_val.argmax(dim=1)
            node_correct = pred.eq(batch.y).sum().item()
            node_accuracy = node_correct / len(val_loader.dataset)
            
            # Link prediction metrics
            link_labels = torch.cat([torch.ones(link_pred_pos_val.shape[0]), torch.zeros(link_pred_neg_val.shape[0])])
            link_preds = torch.cat([link_pred_pos_val, link_pred_neg_val])
            roc_score = roc_auc_score(link_labels.detach().cpu(), link_preds.detach().cpu())

            print(f'Epoch {epoch+1}, Validation Loss: {val_loss.item()}, Node Classification Accuracy: {node_accuracy}, Link Prediction ROC: {roc_score}')            
