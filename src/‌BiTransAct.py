import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, f1_score
from utils import read_data, read_test, reverse_seq, get_embedding, get_interaction_map, get_interaction_map_for_test, get_interaction_map_for_test_short, decision_for_whole, specificity_score, NPV

# Dataset class
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

# BiTransAct
class BiTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, output_size):
        super(BiTransformer, self).__init__()
        self.embedding_m = nn.Embedding(input_size, hidden_size)
        self.position_encoding_m = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_m = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_m, mean=0, std=0.1)

        self.embedding_mi = nn.Embedding(input_size, hidden_size)
        self.position_encoding_mi = nn.Parameter(torch.zeros(1, 100, hidden_size))
        self.interaction_embedding_mi = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.position_encoding_mi, mean=0, std=0.1)

        encoder_layer_m = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_m = nn.TransformerEncoder(encoder_layer_m, num_layers)

        encoder_layer_mi = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder_mi = nn.TransformerEncoder(encoder_layer_mi, num_layers)

        self.cross_attention_m_to_mi = nn.MultiheadAttention(hidden_size, num_heads)
        self.cross_attention_mi_to_m = nn.MultiheadAttention(hidden_size, num_heads)

        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 12)
        self.fc2 = nn.Linear(12, output_size)

    def forward(self, emb_m, emb_mi, pairing_m, pairing_mi):
        m_emb = self.embedding_m(emb_m) + self.position_encoding_m[:, :emb_m.size(1), :] + self.interaction_embedding_m(pairing_m)
        mi_emb = self.embedding_mi(emb_mi) + self.position_encoding_mi[:, :emb_mi.size(1), :] + self.interaction_embedding_mi(pairing_mi)

        m_emb = m_emb.permute(1, 0, 2)
        mi_emb = mi_emb.permute(1, 0, 2)

        encoder_output_m = self.encoder_m(m_emb)
        encoder_output_mi = self.encoder_mi(mi_emb)

        attn_m_to_mi, _ = self.cross_attention_m_to_mi(encoder_output_m, encoder_output_mi, encoder_output_mi)
        attn_mi_to_m, _ = self.cross_attention_mi_to_m(encoder_output_mi, encoder_output_m, encoder_output_m)

        attn_m_to_mi = attn_m_to_mi.permute(1, 0, 2).mean(dim=1)
        attn_mi_to_m = attn_mi_to_m.permute(1, 0, 2).mean(dim=1)

        fused = torch.cat([attn_m_to_mi, attn_mi_to_m], dim=1)
        fused = self.fusion(fused)

        x = self.fc1(fused)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Training loop
def Deep_train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in dataloader:
        features1 = data['fea1']
        features2 = data['fea2']
        features3 = data['fea3']
        features4 = data['fea4']
        target = data['label']
        outputs = model(features1, features2, features3, features4)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation loop
def Deep_validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data in dataloader:
            features1 = data['fea1']
            features2 = data['fea2']
            features3 = data['fea3']
            features4 = data['fea4']
            target = data['label']
            outputs = model(features1, features2, features3, features4)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            preds = [1 if o[1] > 0.5 else 0 for o in outputs.cpu().numpy()]
            all_predictions.extend(preds)
            all_targets.extend([t[1] for t in target.cpu().numpy()])
    acc = accuracy_score(all_targets, all_predictions)
    pre = average_precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    spec = specificity_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    npv = NPV(all_targets, all_predictions)
    print(f'acc: {acc:.4f}, precision: {pre:.4f}, recall: {recall:.4f}, specificity: {spec:.4f}, f1: {f1:.4f}, npv: {npv:.4f}')
    return total_loss / len(dataloader)

# Train model
def perform_train():
    batchsize = 256
    learningrate = 1e-4
    epochs = 100
    train, val = read_data('miRAW_Train_Validation.txt')
    train_loader = DataLoader(myDataset(train), batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(myDataset(val), batch_size=batchsize, shuffle=True)
    model = BiTransformer(input_size=5, hidden_size=64, num_layers=16, num_heads=8, dropout=0.1, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningrate, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
       print(f'Epoch {epoch+1}/{epochs}')
       train_loss = Deep_train(model, train_loader, optimizer, criterion)
       val_loss = Deep_validate(model, val_loader, criterion)

       
       train_losses.append(train_loss)
       val_losses.append(val_loss)

       
       if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_epoch = epoch + 1
          torch.save(model.state_dict(), 'best_model.pth') 

       print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Epoch: {best_epoch}')

# Segment mRNA
def get_cts(rmrna, stepsize):
    if len(rmrna) >= 40:
        return [rmrna[i:i+40] for i in range(0, len(rmrna), stepsize) if i+40 <= len(rmrna)]
    else:
        return [rmrna + 'X' * (40 - len(rmrna))]

# Predict on kmers
def kmers_predict(kmers, mirna, model):
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
    t1 = torch.tensor(fea1,
dtype=torch.long)
    t2 = torch.tensor(fea2)
    t3 = torch.tensor(fea3)
    t4 = torch.tensor(fea4)
    outputs = model(t1, t2, t3, t4).detach().numpy().tolist()
    return decision_for_whole(outputs)

# Final test function
def perform_test(pathfile, stepsize):
    test = read_test(pathfile)
    y_true = []
    y_pred = []
    model = BiTransformer(input_size=5, hidden_size=64, num_layers=16, num_heads=8, dropout=0.1, output_size=2)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    print('تعداد نمونه‌ها:', len(test))
    for fasta in test:
        mirna = fasta[0].upper().replace('T', 'U')
        mrna = fasta[1].upper().replace('T', 'U')
        reverse_mrna = reverse_seq(mrna)
        y_true.append(fasta[2])
        kmers = get_cts(reverse_mrna, stepsize)
        if kmers is None:
            y_pred.append(0)
        else:
            pred = kmers_predict(kmers, mirna, model)
            y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred)
    pre = average_precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    spec = specificity_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    npv = NPV(y_true, y_pred)
    print(f'acc: {acc:.4f}, precision: {pre:.4f}, recall: {recall:.4f}, specificity: {spec:.4f}, f1: {f1:.4f}, npv: {npv:.4f}')


if __name__ == "__main__":
    #perform_train()
    for i in range(10):
        filename = f"miRAW_Test{i}.txt"
        perform_test(filename, stepsize=5)