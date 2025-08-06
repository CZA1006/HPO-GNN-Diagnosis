import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
import torch.nn.functional as F
from torch.optim import AdamW
import obonet
import networkx as nx
from transformers import AutoTokenizer, AutoModel

# --- 1. Load HPO graph ---

def load_hpo_graph(obo_path, namespace_filter=None):
    graph = obonet.read_obo(obo_path)
    G = nx.DiGraph(graph)
    if namespace_filter:
        nodes = [n for n,d in G.nodes(data=True) if d.get('namespace') == namespace_filter]
        G = G.subgraph(nodes).copy()
    edge_index = torch.tensor(list(G.to_undirected().edges())).t().contiguous()
    return G, edge_index

# --- 2. Extract TSDAE embeddings as node features ---

def load_tsdae_embeddings(checkpoint_dir, hpo_terms, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    embeddings = []
    for term_id, term_text in hpo_terms:
        inputs = tokenizer(
            term_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, return_dict=True)
            # mean-pool last hidden state
            hid = out.last_hidden_state
            emb = hid.mean(dim=1).squeeze(0).cpu()
        embeddings.append(emb)
    x = torch.stack(embeddings, dim=0)
    return x

# --- 3. Prepare PyG Data ---

def build_data_obj(obo_path, checkpoint_dir, namespace='ALL'):
    graph, edge_index = load_hpo_graph(obo_path, None if namespace == 'ALL' else namespace)
    terms = [
        (n, f"{d['name']} : {d.get('def', '')}")
        for n, d in graph.nodes(data=True)
    ]
    x = load_tsdae_embeddings(checkpoint_dir, terms)
    data = Data(x=x, edge_index=edge_index)
    return data, terms

# --- 4. GIN + Contrastive Learning ---

class GINEncoder(torch.nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hid_dim),
            torch.nn.BatchNorm1d(hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, hid_dim)
        )
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim),
            torch.nn.BatchNorm1d(hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, hid_dim)
        )
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# Augmentations: edge dropout and feature masking

def augment(x, edge_index, drop_edge_prob=0.2, mask_feat_prob=0.1):
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) >= drop_edge_prob
    edge_index_aug = edge_index[:, mask]
    x_aug = x.clone()
    mask_feat = torch.rand(x_aug.shape) < mask_feat_prob
    x_aug[mask_feat] = 0
    return x_aug, edge_index_aug

# Contrastive loss (InfoNCE)

def contrastive_loss(z1, z2, tau=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    sim_matrix = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(N).to(z1.device)
    loss1 = F.cross_entropy(sim_matrix, labels)
    loss2 = F.cross_entropy(sim_matrix.t(), labels)
    return (loss1 + loss2) / 2

# Training loop

def train_gcl(data, in_dim, hid_dim=768, epochs=100, lr=1e-3, device='cpu'):
    data = data.to(device)
    model = GINEncoder(in_dim, hid_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        x1, e1 = augment(data.x, data.edge_index)
        x2, e2 = augment(data.x, data.edge_index)
        z1 = model(x1.to(device), e1.to(device))
        z2 = model(x2.to(device), e2.to(device))
        loss = contrastive_loss(z1, z2)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model(data.x.to(device), data.edge_index.to(device)).cpu()
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(z, 'checkpoints/hpo_gcl_embeddings.pt')
    return z

# --- Main ---

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obo', type=str, default='hp.obo')
    parser.add_argument('--tsdae_ckpt', type=str, default='checkpoints/hpo_tsdae_encoder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    data, terms = build_data_obj(args.obo, args.tsdae_ckpt)
    print(data)
    z = train_gcl(
        data,
        in_dim=data.num_node_features,
        epochs=args.epochs,
        device=(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    )
    print("GCL training complete, embeddings saved.")