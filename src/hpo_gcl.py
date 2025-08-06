import os
import torch
import obonet
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

# --- 1. Parse HPO .obo into term list and graph ---
def parse_hpo_terms_and_graph(obo_path, namespace_filter=None):
    graph = obonet.read_obo(obo_path)
    G = nx.DiGraph(graph)
    if namespace_filter:
        keep = {n for n,d in G.nodes(data=True) if d.get("namespace")==namespace_filter}
        G = G.subgraph(keep).copy()

    # extract term texts
    terms = []
    for n, d in G.nodes(data=True):
        name = d.get("name", "")
        definition = d.get("def", "").strip('"')
        text = f"{name} : {definition}"
        terms.append((n, text))

    # build node→index mapping
    node_list = [n for n,_ in terms]
    node2idx   = {n:i for i,n in enumerate(node_list)}

    # build undirected edge_index
    edges = []
    for u, v in G.to_undirected().edges():
        i, j = node2idx[u], node2idx[v]
        edges.append((i, j))
        edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return terms, edge_index

# --- 2. Load TSDAE embeddings for each term ---
def load_tsdae_embeddings(checkpoint_dir, terms, model_name, device="cpu"):
    # tokenizer from original model to get vocab
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model weights from your TSDAE checkpoint
    model = AutoModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    embs = []
    with torch.no_grad():
        for term_id, text in terms:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            ).to(device)
            out = model(**inputs, return_dict=True)
            hid = out.last_hidden_state      # [1, seq_len, dim]
            emb = hid.mean(dim=1).squeeze(0).cpu()
            embs.append(emb)
    return torch.stack(embs, dim=0)

# --- 3. Prepare PyG Data ---
def build_data_obj(obo_path, tsdae_ckpt, model_name, namespace=None):
    terms, edge_index = parse_hpo_terms_and_graph(obo_path, namespace)
    x = load_tsdae_embeddings(tsdae_ckpt, terms, model_name)
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
            torch.nn.Linear(hid_dim, hid_dim),
        )
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim),
            torch.nn.BatchNorm1d(hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, hid_dim),
        )
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

def augment(x, edge_index, drop_edge_prob=0.2, mask_feat_prob=0.1):
    mask = torch.rand(edge_index.size(1)) >= drop_edge_prob
    edge_index_aug = edge_index[:, mask]
    x_aug = x.clone()
    feat_mask = torch.rand(x_aug.shape) < mask_feat_prob
    x_aug[feat_mask] = 0
    return x_aug, edge_index_aug

def contrastive_loss(z1, z2, tau=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(z1.size(0), device=sim.device)
    return 0.5*(F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))

def train_gcl(data, hid_dim=768, epochs=100, lr=1e-3, device="cpu"):
    data = data.to(device)
    model = GINEncoder(data.num_node_features, hid_dim).to(device)
    optim = AdamW(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        optim.zero_grad()
        x1, e1 = augment(data.x, data.edge_index)
        x2, e2 = augment(data.x, data.edge_index)
        z1 = model(x1.to(device), e1.to(device))
        z2 = model(x2.to(device), e2.to(device))
        loss = contrastive_loss(z1, z2)
        loss.backward()
        optim.step()
        if ep % 10 == 0:
            print(f"Epoch {ep}/{epochs} – Loss: {loss:.4f}")

    model.eval()
    with torch.no_grad():
        z = model(data.x.to(device), data.edge_index.to(device)).cpu()

    torch.save(z, "checkpoints/hpo_gcl_embeddings.pt")
    print("Saved GCL embeddings to checkpoints/hpo_gcl_embeddings.pt")
    return z

# --- Main CLI ---
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--obo",        default="hp.obo",                            type=str)
    p.add_argument("--tsdae_ckpt", default="checkpoints/hpo_tsdae_encoder",     type=str)
    p.add_argument("--model_name", default="dmis-lab/biobert-v1.1",             type=str)
    p.add_argument("--epochs",     default=100,                                 type=int)
    p.add_argument("--device",     default=None,                                type=str)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data, terms = build_data_obj(args.obo, args.tsdae_ckpt, args.model_name)
    print(data)

    # save ordered list of term IDs for downstream
    os.makedirs("checkpoints", exist_ok=True)
    torch.save([t[0] for t in terms], "checkpoints/node_list.pt")
    print(f"Saved {len(terms)} term IDs to checkpoints/node_list.pt")

    train_gcl(data, epochs=args.epochs, device=device)
