import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd

class RSAAnalyzer:
    def __init__(self, device="cuda"):
        self.device = device

    def get_layerwise_hidden(self, sentences, model, tokenizer):
        all_sent_hiddens = []
        for sent in tqdm(sentences, desc="Getting hidden states"):
            encoding = tokenizer(sent, is_split_into_words=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**{k: v.to(self.device) for k,v in encoding.items()}, output_hidden_states=True)
            hiddens = outputs.hidden_states  # Tuple: (num_layers+1, seq_len, hidden_dim)
            word_ids = encoding.word_ids(batch_index=0)
            reps = []
            for layer_h in hiddens:
                layer_reps = []
                for i in range(len(sent)):
                    # Average subword vectors for each word
                    token_idxs = [j for j, wid in enumerate(word_ids) if wid == i]
                    subtok = torch.stack([layer_h[0, tid] for tid in token_idxs], dim=0)
                    layer_reps.append(subtok.mean(dim=0).cpu().numpy())
                layer_reps = np.stack(layer_reps)  # (num_tokens, hidden_dim)
                reps.append(layer_reps)
            all_sent_hiddens.append(reps)
        return all_sent_hiddens

    def compute_rsa(self, sent_reps1, sent_reps2):
        # Pearson correlation for each layer between two models
        n_layers = len(sent_reps1[0])
        rsa_sim = []
        for layer in range(n_layers):
            # For each layer, get mean-pooled sentence representations
            mat1 = [x[layer] for x in sent_reps1]
            mat2 = [x[layer] for x in sent_reps2]
            mats1 = [x.reshape(-1, x.shape[-1]).mean(0) for x in mat1]
            mats2 = [x.reshape(-1, x.shape[-1]).mean(0) for x in mat2]
            mats1 = np.stack(mats1)
            mats2 = np.stack(mats2)
            # Compute pairwise cosine similarity matrices for all sentences
            def pdist(X):
                normed = X / np.linalg.norm(X, axis=1, keepdims=True)
                return np.dot(normed, normed.T)
            D1 = pdist(mats1)
            D2 = pdist(mats2)
            # Extract upper triangle 
            idx = np.triu_indices(len(mats1), k=1)
            x = D1[idx]
            y = D2[idx]
            # Pearson correlation between similarity matrices
            sim = np.corrcoef(x, y)[0,1]
            rsa_sim.append(sim)
        return rsa_sim

    def plot_rsa(self, rsa_dict, base_name="Base", language="English", y_limlow=0, y_limup=1, max_sentences=None):
        sns.set(style="whitegrid", font_scale=1.2)
        layers = np.arange(len(list(rsa_dict.values())[0]))
        df_records = []
        for name, rsa_vals in rsa_dict.items():
            for i, score in enumerate(rsa_vals):
                df_records.append({"Layer": i, "RSA Similarity": score, "Comparison": name})
        df = pd.DataFrame(df_records)
        plt.figure(figsize=(8,6))
        sns.lineplot(data=df, x="Layer", y="RSA Similarity", hue="Comparison", marker="o")
        plt.title(f"Cross Model RSA Similarity ({language})")
        plt.ylim(y_limlow, y_limup)
        if max_sentences is not None:
            plt.gcf().text(0.99, 0.01, f"max_sentences: {max_sentences}", 
                           ha='right', va='bottom', fontsize=12, color='gray', alpha=0.8)
        plt.show()
