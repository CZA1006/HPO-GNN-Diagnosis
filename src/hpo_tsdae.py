import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.optim import AdamW

class HPOTSDAEDataset(Dataset):
    def __init__(self, obo_file, tokenizer, max_length=128):
        self.terms = []  # list of (term_id, text)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Parse HPO .obo file
        with open(obo_file) as f:
            lines = f.read().split("\n")
        cur = {}
        for line in lines:
            if line == "[Term]":
                cur = {}
            elif line == "":
                if 'id' in cur and 'name' in cur and 'def' in cur:
                    text = cur['name'] + ' : ' + cur['def']
                    self.terms.append((cur['id'], text))
                cur = {}
            else:
                if line.startswith('id: '): cur['id'] = line.split('id: ')[1]
                elif line.startswith('name: '): cur['name'] = line.split('name: ')[1]
                elif line.startswith('def: '): cur['def'] = line.split('def: ')[1].strip('"')

    def __len__(self):
        return len(self.terms)

    def __getitem__(self, idx):
        term_id, text = self.terms[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return term_id, inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0)

class TSDAEModel(torch.nn.Module):
    def __init__(self, model_name, mask_token_id, use_safetensors=True):
        super().__init__()
        self.mask_token_id = mask_token_id
        load_kwargs = {"low_cpu_mem_usage": True}
        if use_safetensors:
            load_kwargs["use_safetensors"] = True
        self.encoder = AutoModelForMaskedLM.from_pretrained(
            model_name,
            **load_kwargs
        )

    def forward(self, input_ids, attention_mask, noise_prob=0.1):
        noisy = input_ids.clone()
        mask = torch.rand(noisy.shape, device=noisy.device) < noise_prob
        noisy[mask] = self.mask_token_id
        outputs = self.encoder(noisy, attention_mask=attention_mask, labels=input_ids)
        return outputs.loss


def train_tsdae(
    obo_path,
    model_name='dmis-lab/biobert-v1.1',
    batch_size=16,
    lr=5e-5,
    epochs=3,
    device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    dataset = HPOTSDAEDataset(obo_path, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TSDAEModel(
        model_name,
        mask_token_id=tokenizer.mask_token_id,
        use_safetensors=True
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for _, input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            loss = model(input_ids, attention_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    model.encoder.save_pretrained('checkpoints/hpo_tsdae_encoder')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpo_obo', default='hp.obo', type=str)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    train_tsdae(args.hpo_obo, epochs=args.epochs)
