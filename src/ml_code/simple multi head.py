import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample parameters
num_continuous = 5        # Number of continuous features
cat_cardinalities = [10, 6, 4]  # Number of categories per categorical variable
embedding_dim = 4          # Embedding dimension for categorical features
num_classes = 2            # Binary classification

class TabularAttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Embedding layers for categorical variables
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_size, embedding_dim)
            for cat_size in cat_cardinalities
        ])
        
        # BatchNorm for continuous variables
        self.bn_cont = nn.BatchNorm1d(num_continuous)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=num_continuous + embedding_dim*len(cat_cardinalities),
            num_heads=1, batch_first=True
        )
        
        # Classification MLP
        self.fc = nn.Sequential(
            nn.Linear(num_continuous + embedding_dim*len(cat_cardinalities), 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x_cont, x_cat):
        # x_cont: [batch_size, num_continuous]
        # x_cat: list of [batch_size] tensors for each categorical variable
        
        # Embed categorical variables and concatenate
        x_cat_emb = [emb(x_cat[i]) for i, emb in enumerate(self.embeddings)]
        x_cat_emb = torch.cat(x_cat_emb, dim=1)
        
        # Normalize continuous variables
        x_cont = self.bn_cont(x_cont)
        
        # Concatenate all features
        x = torch.cat([x_cont, x_cat_emb], dim=1)
        
        # Attention expects input: (batch_size, seq_len, embed_dim)
        # We'll treat each sample as a "sequence" of length 1
        x_attn, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x_attn = x_attn.squeeze(1)  # Back to [batch_size, embed_dim]
        
        # Classification
        out = self.fc(x_attn)
        return out

# Example usage
batch_size = 8
x_cont = torch.randn(batch_size, num_continuous)
x_cat = [torch.randint(0, size, (batch_size,)) for size in cat_cardinalities]

model = TabularAttentionClassifier()
logits = model(x_cont, x_cat)
print(logits.shape)  # [batch_size, num_classes]
