import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, f1_score
from utils import read_data, read_test, reverse_seq, get_embedding, get_interaction_map, \
    get_interaction_map_for_test, get_interaction_map_for_test_short, specificity_score, NPV

# ============================
# Dataset
# ============================
class myDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mirna, mrna, label = self.data[index]
        reverse_mrna = reverse_seq(mrna)
        mirna = mirna + 'X' * (30 - len(mirna))
        emb_m = get_embedding(reverse_mrna)
        emb_mi = get_embedding(mirna)
        pairing_m, pairing_mi = get_interaction_map(mirna, reverse_mrna)

        return {
            'fea1': torch.tensor(emb_m),
            'fea2': torch.tensor(emb_mi),
            'fea3': torch.tensor(pairing_m),
            'fea4': torch.tensor(pairing_mi),
            'label': torch.tensor(label, dtype=torch.float)
        }

# ============================
# BiTransformer_DST با 16 لایه
# ============================
class BiTransformer_DST(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=16, num_heads=8, dropout=0.1, num_classes=2):
        super(BiTransformer_DST, self).__init__()
        self.num_classes = num_classes

        # mRNA embeddings
        self.embedding_m = nn.Embedding(input_size, hidden_size)
        self.position_encoding_m = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_m = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_m, mean=0, std=0.1)

        # miRNA embeddings
        self.embedding_mi = nn.Embedding(input_size, hidden_size)
        self.position_encoding_mi = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_mi = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_mi, mean=0, std=0.1)

        # Transformer encoders
        encoder_layer_m = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_m = nn.TransformerEncoder(encoder_layer_m, num_layers)
        encoder_layer_mi = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_mi = nn.TransformerEncoder(encoder_layer_mi, num_layers)

        # Cross-attention
        self.cross_attention_m_to_mi = nn.MultiheadAttention(hidden_size, num_heads)
        self.cross_attention_mi_to_m = nn.MultiheadAttention(hidden_size, num_heads)

        # Fusion + DST layers
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 12)
        self.fc2 = nn.Linear(12, num_classes)

    def forward(self, emb_m, emb_mi, pairing_m, pairing_mi):
        # Embedding + position + interaction
        m_emb = self.embedding_m(emb_m) + self.position_encoding_m[:, :emb_m.size(1), :] + self.interaction_embedding_m(pairing_m)
        mi_emb = self.embedding_mi(emb_mi) + self.position_encoding_mi[:, :emb_mi.size(1), :] + self.interaction_embedding_mi(pairing_mi)

        # Transformer expects (seq_len, batch, hidden)
        m_emb = m_emb.permute(1, 0, 2)
        mi_emb = mi_emb.permute(1, 0, 2)

        # Encoder
        encoder_output_m = self.encoder_m(m_emb)
        encoder_output_mi = self.encoder_mi(mi_emb)

        # Cross attention
        attn_m_to_mi, _ = self.cross_attention_m_to_mi(encoder_output_m, encoder_output_mi, encoder_output_mi)
        attn_mi_to_m, _ = self.cross_attention_mi_to_m(encoder_output_mi, encoder_output_m, encoder_output_m)

        # Mean pooling
        attn_m_to_mi = attn_m_to_mi.permute(1, 0, 2).mean(dim=1)
        attn_mi_to_m = attn_mi_to_m.permute(1, 0, 2).mean(dim=1)

        # Fusion
        fused = torch.cat([attn_m_to_mi, attn_mi_to_m], dim=1)
        fused = self.fusion(fused)

        # Evidence layer
        x = torch.relu(self.fc1(fused))
        evidence = torch.relu(self.fc2(x))

        # DST belief and uncertainty
        S = torch.sum(evidence, dim=1, keepdim=True) + self.num_classes
        belief = evidence / S
        uncertainty = self.num_classes / S

        return belief, uncertainty

# ============================
# Training / Validation
# ============================
def Deep_train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for data in dataloader:
        features1 = data['fea1']
        features2 = data['fea2']
        features3 = data['fea3']
        features4 = data['fea4']
        target = data['label']

        belief, uncertainty = model(features1, features2, features3, features4)
        # DST log
        logits = torch.log(belief + 1e-6)
        # CrossEntropy expects probabilities for DST
        loss = -torch.mean(torch.sum(torch.nn.functional.one_hot(target.long(), num_classes=2) * logits, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def Deep_validate(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data in dataloader:
            features1 = data['fea1']
            features2 = data['fea2']
            features3 = data['fea3']
            features4 = data['fea4']
            target = data['label']

            belief, uncertainty = model(features1, features2, features3, features4)
            preds = torch.argmax(belief, dim=1).cpu().numpy().tolist()
            all_predictions.extend(preds)
            all_targets.extend(target.cpu().numpy().astype(int).tolist())

    acc = accuracy_score(all_targets, all_predictions)
    pre = average_precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    spec = specificity_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    npv = NPV(all_targets, all_predictions)
    print(f'acc: {acc:.4f}, precision: {pre:.4f}, recall: {recall:.4f}, specificity: {spec:.4f}, f1: {f1:.4f}, npv: {npv:.4f}')
    return acc

# ============================
# Train model
# ============================
def perform_train():
    batchsize = 256
    learningrate = 1e-4
    epochs = 60

    train, val = read_data('miRAW_Train_Validation.txt')
    train_loader = DataLoader(myDataset(train), batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(myDataset(val), batch_size=batchsize, shuffle=True)

    model = BiTransformer_DST(input_size=5, hidden_size=64, num_layers=16, num_heads=8, dropout=0.1, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=learningrate, weight_decay=1e-5)

    best_val_acc = 0
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        train_loss = Deep_train(model, train_loader, optimizer)
        val_acc = Deep_validate(model, val_loader)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

# ============================
# Segment mRNA
# ============================
def get_cts(rmrna, stepsize):
    if len(rmrna) >= 40:
        return [rmrna[i:i+40] for i in range(0, len(rmrna), stepsize) if i+40 <= len(rmrna)]
    else:
        return [rmrna + 'X' * (40 - len(rmrna))]

# ============================
# Predict on kmers with DST combination
# ============================
def kmers_predict_dst_combined(kmers, mirna, model):
    mirna = mirna + 'X' * (30 - len(mirna))
    fea1, fea2, fea3, fea4 = [], [], [], []

    for kmer in kmers:
        fea1.append(get_embedding(kmer))
        fea2.append(get_embedding(mirna))
        if 'X' in kmer:
            p_m, p_mi = get_interaction_map_for_test_short(mirna, kmer)
        else:
            p_m, p_mi = get_interaction_map_for_test(mirna, kmer)
        fea3.append(p_m)
        fea4.append(p_mi)

    t1 = torch.tensor(fea1, dtype=torch.long)
    t2 = torch.tensor(fea2)
    t3 = torch.tensor(fea3)
    t4 = torch.tensor(fea4)

    belief, uncertainty = model(t1, t2, t3, t4)
    combined_belief = torch.mean(belief, dim=0)
    combined_uncertainty = torch.mean(uncertainty, dim=0)
    pred_class = int(torch.argmax(combined_belief))
    return pred_class, float(combined_uncertainty.max())

# ============================
# Final test
# ============================
def perform_test(pathfile, stepsize):
    test = read_test(pathfile)
    y_true, y_pred, uncertainty_list = [], [], []

    model = BiTransformer_DST(input_size=5, hidden_size=64, num_layers=16, num_heads=8, dropout=0.1, num_classes=2)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    for fasta in test:
        mirna = fasta[0].upper().replace('T', 'U')
        mrna = fasta[1].upper().replace('T', 'U')
        reverse_mrna = reverse_seq(mrna)
        y_true.append(fasta[2])
        kmers = get_cts(reverse_mrna, stepsize)

        if kmers is None:
            y_pred.append(0)
            uncertainty_list.append(1.0)
        else:
            pred_class, pred_uncertainty = kmers_predict_dst_combined(kmers, mirna, model)
            y_pred.append(pred_class)
            uncertainty_list.append(pred_uncertainty)

    acc = accuracy_score(y_true, y_pred)
    pre = average_precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    npv = NPV(y_true, y_pred)
    print(f'acc: {acc:.4f}, precision: {pre:.4f}, recall: {recall:.4f}, specificity: {spec:.4f}, f1: {f1:.4f}, npv: {npv:.4f}')
    print('Average uncertainty:', np.mean(uncertainty_list))

# ============================
# Main
# ============================
if __name__ == "__main__":
    perform_train()
    for i in range(10):
        filename = f"miRAW_Test{i}.txt"
        perform_test(filename, stepsize=5)
