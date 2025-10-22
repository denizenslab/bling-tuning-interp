import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge  # Switch to ridge from LinearRegression (the env problem is solved using setting solver='svd')
from scipy.stats import spearmanr
from scipy.sparse.csgraph import minimum_spanning_tree
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from conllu import parse_incr

class StructuralProbeAnalyzer:
    # Initialize with list of models, tokenizers, and device for computation
    def __init__(self, models, tokenizers, device="cuda"):
        self.models = models if isinstance(models, list) else [models]
        self.tokenizers = tokenizers if isinstance(tokenizers, list) else [tokenizers]
        self.device = device

    # Reads sentences from the file up to max_sentences; returns list of dicts with words and heads
    @staticmethod
    def read_ud_sentences(conllu_path, max_sentences=None):
        sentences = []
        with open(conllu_path, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                words = []
                heads = []
                for token in tokenlist:
                    if isinstance(token['id'], int):
                        words.append(token['form'])
                        # Use -1 as depth/root marker for tokens with head=0 or None
                        if token['head'] is None or token['head'] == 0:
                            heads.append(-1)
                        else:
                            heads.append(token['head'] - 1)
                sentences.append({
                    "words": words,
                    "heads": heads,
                    "sent_id": tokenlist.metadata.get('sent_id', None),
                })
                # Stop reading early if max_sentences is set
                if max_sentences and len(sentences) >= max_sentences:
                    break
        return sentences

    # Extracts BERT layerwise representations for each word in each sentence
    def extract_bert_layerwise_reps(self, sentences, model, tokenizer):
        all_reps = []
        for sent in tqdm(sentences, desc="Extracting BERT representations"):
            words = sent['words']
            # Tokenize sentence and prepare input for BERT
            encoding = tokenizer(words, is_split_into_words=True, return_tensors='pt')
            with torch.no_grad():  
                outputs = model(**{k: v.to(self.device) for k,v in encoding.items()}, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # List of tensors per BERT layer
            word_ids = encoding.word_ids(batch_index=0)  # Map tokens to word indices
            reps_by_word = []
            # For each word, average subword representations to get single vector per layer
            for i in range(len(words)):
                token_idxs = [j for j, wid in enumerate(word_ids) if wid == i]
                layer_reps = []
                for l in range(len(hidden_states)):
                    subtoks = torch.stack([hidden_states[l][0, tid] for tid in token_idxs], dim=0)
                    mean_rep = subtoks.mean(dim=0)
                    layer_reps.append(mean_rep.cpu())
                reps_by_word.append(layer_reps)
            all_reps.append({
                "words": words,
                "hidden_states": reps_by_word,
                "heads": sent['heads'],
                "sent_id": sent['sent_id'],
            })
        return all_reps

    # Computes token depths from gold-headed tree by traversing parents recursively
    @staticmethod
    def compute_tree_depths(heads):
        n = len(heads)
        depths = [-1] * n
        for idx in range(n):
            depth = 0
            cur = idx
            visited = set()
            while heads[cur] != -1:
                if cur in visited or heads[cur] < 0 or heads[cur] >= n:
                    depth = -1  # Invalid tree/cycle detection
                    break
                visited.add(cur)
                cur = heads[cur]
                depth += 1
            if depth != -1:
                depths[idx] = depth
        return depths

    # Computes shortest path distances between all tokens in gold tree
    @staticmethod
    def compute_tree_distances(heads):
        n = len(heads)
        adj = np.zeros((n, n), dtype=int)
        # Build undirected adjacency for tree edges
        for i, h in enumerate(heads):
            if h is not None and h != -1 and 0 <= h < n:
                adj[i, h] = 1
                adj[h, i] = 1
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        dist[adj == 1] = 1
        # Floyd-Warshall to compute all pair shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    # Computes Unlabeled Undirected Attachment Score (UUAS)
    @staticmethod
    def compute_uuas(gold_heads, pred_dist):
        n = len(gold_heads)
        gold_edges = set()
        for i, h in enumerate(gold_heads):
            if h != -1:
                gold_edges.add(tuple(sorted((i, h))))
        mst = minimum_spanning_tree(pred_dist + pred_dist.T).toarray()  # MST based on predicted distances
        pred_edges = set()
        for i in range(n):
            for j in range(n):
                if i != j and (mst[i, j] != 0 or mst[j, i] != 0):
                    pred_edges.add(tuple(sorted((i, j))))
        if len(gold_edges) == 0:
            return None
        correct = len(gold_edges & pred_edges)
        return correct / len(gold_edges)

    # Main probe routine for depth and distance tasks at each layer
    def run_probe(self, layerwise_reps, max_distance_train=100, max_distance_eval=10):
        num_layers = len(layerwise_reps[0]['hidden_states'][0])
        # Precompute depths and distances for each sentence
        for sent in layerwise_reps:
            sent['depths'] = self.compute_tree_depths(sent['heads'])
            sent['distances'] = self.compute_tree_distances(sent['heads'])

        root_accs = []
        depth_spearmans = []
        dist_spearmans = []
        uuas_list = []

        for layer in range(num_layers):
            # Depth probe: Fit ridge regression to predict depth from word reps
            X_all = []
            y_all = []
            for sent in layerwise_reps:
                for i, word_reps in enumerate(sent['hidden_states']):
                    X_all.append(word_reps[layer].numpy())
                    y_all.append(sent['depths'][i])
            X_all = np.stack(X_all)
            y_all = np.array(y_all)
            reg_depth = Ridge(alpha=1.0, solver='svd').fit(X_all, y_all)  # Solve with svd for stabilit

            y_pred = reg_depth.predict(X_all)
            depth_corr = spearmanr(y_all, y_pred).correlation  # Spearman corr for monotonicity
            depth_spearmans.append(depth_corr)

            n_sent = 0
            n_correct = 0
            for sent in layerwise_reps:
                X = np.stack([word_reps[layer].numpy() for word_reps in sent['hidden_states']])
                y_true = sent['depths']
                gold_root = np.argmin(y_true)
                pred_depths = reg_depth.predict(X)
                pred_root = np.argmin(pred_depths)
                if gold_root == pred_root:
                    n_correct += 1
                n_sent += 1
            root_accs.append(n_correct / n_sent)  # Accuracy of root prediction

            # Distance probe: Fit ridge regression on pairs of word reps differences to predict sqrt distance
            X_pairs_all = []
            y_pairs_all = []
            for sent in layerwise_reps[:max_distance_train]:
                X = np.stack([word_reps[layer].numpy() for word_reps in sent['hidden_states']])
                D = np.array(sent['distances'])
                for i in range(len(D)):
                    for j in range(i+1, len(D)):
                        X_pairs_all.append(X[i] - X[j])
                        y_pairs_all.append(np.sqrt(D[i,j] + 1e-8))
            X_pairs_all = np.stack(X_pairs_all)
            y_pairs_all = np.array(y_pairs_all)
            reg_dist = Ridge(alpha=1.0, solver='svd').fit(X_pairs_all, y_pairs_all)  # Fit ridge regression

            spearman_vals = []
            uuas_vals = []
            for sent in layerwise_reps[:max_distance_eval]:
                n = len(sent['words'])
                X = np.stack([word_reps[layer].numpy() for word_reps in sent['hidden_states']])
                pred_dist = np.zeros((n, n))
                for i in range(n):
                    for j in range(i+1, n):
                        dpred = reg_dist.predict((X[i] - X[j]).reshape(1, -1))[0]
                        pred_dist[i, j] = dpred ** 2  # square back since y was sqrt(dist)
                        pred_dist[j, i] = dpred ** 2
                gold_dist = np.array(sent['distances'])
                idx_triu = np.triu_indices(n, k=1)
                s = spearmanr(gold_dist[idx_triu], pred_dist[idx_triu])
                if s.correlation is not None:
                    spearman_vals.append(s.correlation)
                uuas_val = self.compute_uuas(sent['heads'], pred_dist)
                if uuas_val is not None:
                    uuas_vals.append(uuas_val)
            dist_spearmans.append(np.mean(spearman_vals))
            uuas_list.append(np.mean(uuas_vals))

        return {
            "root_accs": root_accs,
            "depth_spearmans": depth_spearmans,
            "dist_spearmans": dist_spearmans,
            "uuas_list": uuas_list,
            "num_layers": num_layers
        }

    # Plot all probe results in subplots for clear comparison
    def plot_probe_results(self, results_list, model_names=None, language="English", y_lowlim=0, max_sentences=None):

        if not isinstance(results_list, list):
            results_list = [results_list]
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(results_list))]
        num_layers = results_list[0]["num_layers"]
        layers = np.arange(num_layers)

        # Prepare DataFrame for seaborn
        records = []
        for res, name in zip(results_list, model_names):
            for i in range(num_layers):
                records.append({
                    "Layer": i,
                    "Root Accuracy": res["root_accs"][i],
                    "Depth Spearman": res["depth_spearmans"][i],
                    "Distance Spearman": res["dist_spearmans"][i],
                    "UUAS": res["uuas_list"][i],
                    "Model": name
                })
        df = pd.DataFrame(records)

        # Ensure all columns are numeric and drop rows with NaN/None
        for col in ["Root Accuracy", "Depth Spearman", "Distance Spearman", "UUAS"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Root Accuracy", "Depth Spearman", "Distance Spearman", "UUAS"])

        # Use a large color palette for distinct lines
        palette = sns.color_palette("tab10", n_colors=len(model_names))
        sns.set(style="whitegrid", font_scale=1.2)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        # Add style and dashes to distinguish lines even if colors overlap
        line_styles = ['-', '--', '-.', ':']
        ylabels = ["Root Accuracy", "Depth Spearman Correlation", "UUAS", "Distance Spearman Correlation"]
        for idx, (metric, ax, ylabel) in enumerate(zip(
            ["Root Accuracy", "Depth Spearman", "UUAS", "Distance Spearman"],
            [axes[0,0], axes[0,1], axes[1,0], axes[1,1]],
            ylabels
        )):
            for i, name in enumerate(model_names):
                subdf = df[df['Model'] == name]
                ax.plot(subdf["Layer"], subdf[metric], label=name, 
                        color=palette[i % len(palette)], 
                        linestyle=line_styles[i % len(line_styles)], 
                        marker="o")
            ax.set(
                title=f"{language} {metric}",
                xlabel="Layer",
                ylabel=ylabel,
                ylim=(y_lowlim,1)
            )
            ax.legend(loc='best', fontsize='medium')

        plt.tight_layout()
        # Add annotation for number of sentences
        fig.text(0.99, 0.01, f"Number of sentences: {max_sentences}", 
                 ha='right', va='bottom', fontsize=14, color='black', alpha=0.8)
        plt.show()

    # Compares embedding and combined query-key-value weights for each brain-tuned model to base, summarizing differences
    def compare_to_base_attention_embedding_combined(self, model_names=None):
        results = []
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(self.models))]
        base_model = self.models[0]
        base_name = model_names[0]

        # Detect number of encoder layers automatically
        try:
            n_layers = len(base_model.encoder.layer)
        except AttributeError:
            n_layers = len(base_model.bert.encoder.layer)

        for idx in range(1, len(self.models)):  # Compare models 1..N to base (0)
            model = self.models[idx]
            name = model_names[idx]

            # Embedding weights
            try:
                emb_base = base_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
                emb_model = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
            except AttributeError:
                emb_base = base_model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
                emb_model = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
            results.append({
                "Comparison": f"{name} vs {base_name}",
                "Layer": "Embedding",
                "Type": "Embedding",
                "Max_Diff": np.abs(emb_base - emb_model).max(),
                "Mean_Diff": np.abs(emb_base - emb_model).mean()
            })

            # Attention QKV combined difference per layer
            for layer in range(n_layers):
                try:
                    attn_base = base_model.encoder.layer[layer].attention.self
                    attn_model = model.encoder.layer[layer].attention.self
                except AttributeError:
                    attn_base = base_model.bert.encoder.layer[layer].attention.self
                    attn_model = model.bert.encoder.layer[layer].attention.self

                # Concatenate query, key, value weights for joint comparison
                q_base = attn_base.query.weight.detach().cpu().numpy()
                k_base = attn_base.key.weight.detach().cpu().numpy()
                v_base = attn_base.value.weight.detach().cpu().numpy()
                q_model = attn_model.query.weight.detach().cpu().numpy()
                k_model = attn_model.key.weight.detach().cpu().numpy()
                v_model = attn_model.value.weight.detach().cpu().numpy()

                qkv_base = np.concatenate([q_base, k_base, v_base], axis=0)
                qkv_model = np.concatenate([q_model, k_model, v_model], axis=0)

                max_diff = np.abs(qkv_base - qkv_model).max()
                mean_diff = np.abs(qkv_base - qkv_model).mean()

                results.append({
                    "Comparison": f"{name} vs {base_name}",
                    "Layer": layer,
                    "Type": "QKV_combined",
                    "Max_Diff": max_diff,
                    "Mean_Diff": mean_diff
                })

        df = pd.DataFrame(results)
        return df
