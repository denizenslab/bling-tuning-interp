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
import networkx as nx

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
    def run_probe(self, train_layerwise_reps, test_layerwise_reps, max_distance_train=100, max_distance_eval=10):
        num_layers = len(train_layerwise_reps[0]['hidden_states'][0])

        # Precompute depths and distances for train and test separately
        for sent in train_layerwise_reps:
            sent['depths'] = self.compute_tree_depths(sent['heads'])
            sent['distances'] = self.compute_tree_distances(sent['heads'])
        for sent in test_layerwise_reps:
            sent['depths'] = self.compute_tree_depths(sent['heads'])
            sent['distances'] = self.compute_tree_distances(sent['heads'])

        root_accs = []
        depth_spearmans = []
        fitted_probes = {}
        dist_spearmans = []
        uuas_list = []

        for layer in range(num_layers):
            # Train depth probe on train data
            X_train = []
            y_train = []
            for sent in train_layerwise_reps:
                for i, word_reps in enumerate(sent['hidden_states']):
                    X_train.append(word_reps[layer].numpy())
                    y_train.append(sent['depths'][i])
            X_train = np.stack(X_train)
            y_train = np.array(y_train)
            reg_depth = Ridge(alpha=1.0, solver='svd').fit(X_train, y_train)

            # Evaluate depth probe on test data
            X_test = []
            y_test = []
            for sent in test_layerwise_reps:
                for i, word_reps in enumerate(sent['hidden_states']):
                    X_test.append(word_reps[layer].numpy())
                    y_test.append(sent['depths'][i])
            X_test = np.stack(X_test)
            y_test = np.array(y_test)

            y_pred = reg_depth.predict(X_test)
            depth_corr = spearmanr(y_test, y_pred).correlation
            depth_spearmans.append(depth_corr)

            # Root accuracy on test set
            n_sent = 0
            n_correct = 0
            for sent in test_layerwise_reps:
                X = np.stack([word_reps[layer].numpy() for word_reps in sent['hidden_states']])
                y_true = sent['depths']
                gold_root = np.argmin(y_true)
                pred_depths = reg_depth.predict(X)
                pred_root = np.argmin(pred_depths)
                if gold_root == pred_root:
                    n_correct += 1
                n_sent += 1
            root_accs.append(n_correct / n_sent)

            # Train distance probe on train data pairs (up to max_distance_train sentences)
            X_pairs_train = []
            y_pairs_train = []
            for sent in train_layerwise_reps[:max_distance_train]:
                X = np.stack([word_reps[layer].numpy() for word_reps in sent['hidden_states']])
                D = np.array(sent['distances'])
                for i in range(len(D)):
                    for j in range(i+1, len(D)):
                        X_pairs_train.append(X[i] - X[j])
                        y_pairs_train.append(np.sqrt(D[i, j] + 1e-8))
            X_pairs_train = np.stack(X_pairs_train)
            y_pairs_train = np.array(y_pairs_train)
            reg_dist = Ridge(alpha=1.0, solver='svd').fit(X_pairs_train, y_pairs_train)

            # Evaluate distance probe on test data pairs (up to max_distance_eval sentences)
            spearman_vals = []
            uuas_vals = []
            for sent in test_layerwise_reps[:max_distance_eval]:
                n = len(sent['words'])
                X = np.stack([word_reps[layer].numpy() for word_reps in sent['hidden_states']])
                pred_dist = np.zeros((n, n))
                for i in range(n):
                    for j in range(i+1, n):
                        dpred = reg_dist.predict((X[i] - X[j]).reshape(1, -1))[0]
                        pred_dist[i, j] = dpred ** 2
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
            
            # --- NEW: Store fitted probe objects for this layer ---
            fitted_probes[layer] = {'depth': reg_depth, 'dist': reg_dist}

        return {
            "root_accs": root_accs,
            "depth_spearmans": depth_spearmans,
            "dist_spearmans": dist_spearmans,
            "uuas_list": uuas_list,
            "num_layers": num_layers,
            "fitted_probes": fitted_probes    # <--- nananaanannananaaselenaselana
        }


    
    def plot_tree_comparison_ax(self, ax_gold, ax_pred, sentence_data, layerwise_reps_for_sentence, fitted_probes, layer_to_plot, model_name=""):
        words = sentence_data['words']
        heads = sentence_data['heads']
        n = len(words)

        if layer_to_plot not in fitted_probes:
            print(f"Error: Probes for layer {layer_to_plot} are not fitted. Run 'run_probe' first.")
            return

        reg_dist = fitted_probes[layer_to_plot]['dist']
        X = np.stack([word_reps[layer_to_plot].numpy() for word_reps in layerwise_reps_for_sentence])

        pred_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dpred = reg_dist.predict((X[i] - X[j]).reshape(1, -1))[0]
                pred_dist[i, j] = dpred ** 2
                pred_dist[j, i] = pred_dist[i, j]

        pred_dist_sym = (pred_dist + pred_dist.T) / 2
        pred_dist_sym[pred_dist_sym < 0] = 0 
        np.fill_diagonal(pred_dist_sym, 0)

        mst = minimum_spanning_tree(pred_dist_sym).toarray()
        pred_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if mst[i, j] != 0 or mst[j, i] != 0:
                    pred_edges.append((i, j))

        gold_edges = []
        for i, h in enumerate(heads):
            if h != -1:
                gold_edges.append(tuple(sorted((i, h))))

        labels = {i: w for i, w in enumerate(words)}

        # Gold Tree Plot
        G_gold = nx.Graph()
        G_gold.add_nodes_from(range(n))
        G_gold.add_edges_from(gold_edges)
        pos_gold = nx.spring_layout(G_gold, seed=42)
        nx.draw(G_gold, pos=pos_gold, labels=labels, with_labels=True, node_color='lightblue', 
                ax=ax_gold, node_size=2000, font_size=10)
        ax_gold.set_title(f"{model_name} Gold Tree", fontsize=12)

        # Predicted MST Plot
        G_pred = nx.Graph()
        G_pred.add_nodes_from(range(n))
        G_pred.add_edges_from(pred_edges)
        try:
            pos_pred = nx.spring_layout(G_pred, pos=pos_gold, seed=42)
        except:
             pos_pred = nx.spring_layout(G_pred, seed=42)

        correct_edges = set(gold_edges) & set(pred_edges)
        incorrect_edges = set(pred_edges) - set(gold_edges)

        nx.draw_networkx_nodes(G_pred, pos=pos_pred, ax=ax_pred, node_color='lightgray', node_size=2000)
        nx.draw_networkx_labels(G_pred, pos=pos_pred, labels=labels, ax=ax_pred, font_size=10)
        nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=correct_edges, ax=ax_pred, edge_color='green', width=2)
        nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=incorrect_edges, ax=ax_pred, edge_color='red', width=2, style='dashed')
        ax_pred.set_title(f"{model_name} Predicted MST - Layer {layer_to_plot}", fontsize=12)

        # Hide axis for clean look
        ax_gold.axis('off')
        ax_pred.axis('off')
    
    def plot_gold_and_predicted_trees(self, test_sentences, results_list, sentence_indices, layer_to_plot):
        model_names = ["BERT-base", "BERT-whole", "BERT-semantic", "BERT-language"]

        for sentence_idx in sentence_indices:
            sentence_data = test_sentences[sentence_idx]
            words = sentence_data['words']
            heads = sentence_data['heads']
            n = len(words)

            fig, axes = plt.subplots(1, 5, figsize=(25, 6))
            fig.suptitle(f"Sentence {sentence_idx} Parse Trees: Gold and Predicted (Layer {layer_to_plot})", fontsize=18)

            # Gold Tree
            gold_edges = []
            for i, h in enumerate(heads):
                if h != -1:
                    gold_edges.append(tuple(sorted((i, h))))
            labels = {i: w for i, w in enumerate(words)}

            G_gold = nx.Graph()
            G_gold.add_nodes_from(range(n))
            G_gold.add_edges_from(gold_edges)
            pos_gold = nx.spring_layout(G_gold, seed=42)
            nx.draw(G_gold, pos=pos_gold, labels=labels, with_labels=True, node_color='lightblue',
                    ax=axes[0], node_size=2000, font_size=12)
            axes[0].set_title("Gold Tree", fontsize=16)
            axes[0].axis('off')

            # Predicted MST for each model
            for i, (model, tokenizer, name) in enumerate(zip(self.models, self.tokenizers, model_names)):
                layerwise_reps_dict = self.extract_bert_layerwise_reps([sentence_data], model, tokenizer)[0]
                hidden_states = layerwise_reps_dict['hidden_states']
                fitted_probes = results_list[i]['fitted_probes']

                reg_dist = fitted_probes[layer_to_plot]['dist']
                X = np.stack([word_reps[layer_to_plot].numpy() for word_reps in hidden_states])
                pred_dist = np.zeros((n, n))
                for a in range(n):
                    for b in range(a+1, n):
                        dpred = reg_dist.predict((X[a] - X[b]).reshape(1, -1))[0]
                        pred_dist[a, b] = dpred ** 2
                        pred_dist[b, a] = pred_dist[a, b]

                pred_dist_sym = (pred_dist + pred_dist.T) / 2
                pred_dist_sym[pred_dist_sym < 0] = 0
                np.fill_diagonal(pred_dist_sym, 0)

                mst = minimum_spanning_tree(pred_dist_sym).toarray()
                pred_edges = []
                for a in range(n):
                    for b in range(a+1, n):
                        if mst[a, b] != 0 or mst[b, a] != 0:
                            pred_edges.append((a, b))

                correct_edges = set(gold_edges) & set(pred_edges)
                incorrect_edges = set(pred_edges) - set(gold_edges)

                G_pred = nx.Graph()
                G_pred.add_nodes_from(range(n))
                G_pred.add_edges_from(pred_edges)

                try:
                    pos_pred = nx.spring_layout(G_pred, pos=pos_gold, seed=42)
                except:
                    pos_pred = nx.spring_layout(G_pred, seed=42)

                ax = axes[i+1]
                nx.draw_networkx_nodes(G_pred, pos=pos_pred, ax=ax, node_color='lightgray', node_size=2000)
                nx.draw_networkx_labels(G_pred, pos=pos_pred, labels=labels, ax=ax, font_size=12)
                nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=correct_edges, ax=ax, edge_color='green', width=2)
                nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=incorrect_edges, ax=ax, edge_color='red', width=2, style='dashed')
                ax.set_title(f"{name}\nPredicted MST", fontsize=16)
                ax.axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    
    # --- NEW METHOD TO PLOT GRAPHS ---
    def plot_tree_comparison(self, sentence_data, layerwise_reps_for_sentence, fitted_probes, layer_to_plot, model_name=""):
        
        words = sentence_data['words']
        heads = sentence_data['heads']
        n = len(words)
        
        if layer_to_plot not in fitted_probes:
            print(f"Error: Probes for layer {layer_to_plot} are not fitted. Run 'run_probe' first.")
            return

        # --- 1. Get Predicted Distances & MST ---
        reg_dist = fitted_probes[layer_to_plot]['dist']
        X = np.stack([word_reps[layer_to_plot].numpy() for word_reps in layerwise_reps_for_sentence])
        
        pred_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dpred = reg_dist.predict((X[i] - X[j]).reshape(1, -1))[0]
                pred_dist[i, j] = dpred ** 2
                pred_dist[j, i] = pred_dist[i, j]

        # Get predicted edges from MST
        pred_dist_sym = (pred_dist + pred_dist.T) / 2
        pred_dist_sym[pred_dist_sym < 0] = 0 
        np.fill_diagonal(pred_dist_sym, 0)
        
        mst = minimum_spanning_tree(pred_dist_sym).toarray()
        pred_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if mst[i, j] != 0 or mst[j, i] != 0:
                    pred_edges.append((i, j))

        # --- 2. Get Gold Edges ---
        gold_edges = []
        for i, h in enumerate(heads):
            if h != -1:
                gold_edges.append(tuple(sorted((i, h))))
        
        # --- 3. Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        labels = {i: w for i, w in enumerate(words)}
        
        # Plot Gold Tree
        G_gold = nx.Graph()
        G_gold.add_nodes_from(range(n))
        G_gold.add_edges_from(gold_edges)
        pos_gold = nx.spring_layout(G_gold, seed=42)
        nx.draw(G_gold, pos=pos_gold, labels=labels, with_labels=True, node_color='lightblue', 
                ax=axes[0], node_size=2000, font_size=12)
        axes[0].set_title("Gold Dependency Tree (Undirected)", fontsize=16)

        # Plot Predicted MST
        G_pred = nx.Graph()
        G_pred.add_nodes_from(range(n))
        G_pred.add_edges_from(pred_edges)
        
        # Try to use the same layout for easier comparison
        try:
            pos_pred = nx.spring_layout(G_pred, pos=pos_gold, seed=42)
        except:
             pos_pred = nx.spring_layout(G_pred, seed=42) # Fallback
             
        # Highlight correct/incorrect edges
        correct_edges = set(gold_edges) & set(pred_edges)
        incorrect_edges = set(pred_edges) - set(gold_edges)
        missing_edges = set(gold_edges) - set(pred_edges) # Will show up in gold plot

        nx.draw_networkx_nodes(G_pred, pos=pos_pred, ax=axes[1], node_color='lightgray', node_size=2000)
        nx.draw_networkx_labels(G_pred, pos=pos_pred, labels=labels, ax=axes[1], font_size=12)
        nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=correct_edges, ax=axes[1], edge_color='green', width=2)
        nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=incorrect_edges, ax=axes[1], edge_color='red', width=2, style='dashed')
        
        axes[1].set_title(f"Predicted Geometric Tree (MST) - {model_name} Layer {layer_to_plot}", fontsize=16)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=2, label='Correct Edge (in Gold & Pred)'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Incorrect Edge (in Pred only)')
        ]
        axes[1].legend(handles=legend_elements, loc='lower center', fontsize='medium')

        plt.tight_layout()
        plt.show()
    
    
    # Plot all probe results in subplots for clear comparison
    def plot_probe_results(self, results_list, model_names=None, language="English", y_lowlim=0, max_sentences=None):

        if not isinstance(results_list, list):
            results_list = [results_list]
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(results_list))]
        num_layers = results_list[0]["num_layers"]

        # Exclude the embedding layer (layer 0)
        layers = np.arange(1, num_layers)
        layer_labels = [f"{i}" for i in layers]

        # Prepare DataFrame for seaborn
        records = []
        for res, name in zip(results_list, model_names):
            for i in layers:
                records.append({
                    "Layer": i,
                    "Layer Label": f"Layer {i}" if i == 1 else f"Transformer {i}",
                    "Root Accuracy": res["root_accs"][i],
                    "Depth Spearman": res["depth_spearmans"][i],
                    "Distance Spearman": res["dist_spearmans"][i],
                    "UUAS": res["uuas_list"][i],
                    "Model": name
                })
        df = pd.DataFrame(records)

        for col in ["Root Accuracy", "Depth Spearman", "Distance Spearman", "UUAS"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Root Accuracy", "Depth Spearman", "Distance Spearman", "UUAS"])

        palette = sns.color_palette("tab10", n_colors=len(model_names))
        sns.set(style="whitegrid", font_scale=1.2)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        line_styles = ['-', '--', '-.', ':']
        ylabels = [
            "Root Accuracy", 
            "Depth Spearman Correlation", 
            "UUAS", 
            "Distance Spearman Correlation"
        ]
        metrics = ["Root Accuracy", "Depth Spearman", "UUAS", "Distance Spearman"]
        axes_grid = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

        for idx, (metric, ax, ylabel) in enumerate(zip(metrics, axes_grid, ylabels)):
            for i, name in enumerate(model_names):
                subdf = df[df['Model'] == name]
                ax.plot(subdf["Layer"], subdf[metric], label=name, 
                        color=palette[i % len(palette)], 
                        linestyle=line_styles[i % len(line_styles)], 
                        marker="o")
            ax.set(
                title=f"{language} {metric}",
                xlabel="Transformer Block Output (1-12)",
                ylabel=ylabel,
                ylim=(y_lowlim,1),
                xticks=layers,
                xticklabels=layer_labels
            )
            ax.legend(loc='best', fontsize='medium')

        plt.tight_layout()
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
    
  # --- NEW/UPDATED METHOD ---
    @staticmethod
    def _get_params(param_obj):
        """Helper to safely get .weight and .bias, detaching and converting to numpy."""
        w = param_obj.weight.detach().cpu().numpy()
        b = param_obj.bias.detach().cpu().numpy()
        return w, b

    @staticmethod
    def _get_diff_dict(comp, layer, type, base_w, model_w, base_b=None, model_b=None):
        """Helper to create the results dictionary for a component."""
        # Handle weights
        w_diff = np.abs(base_w - model_w)
        
        # Handle biases if provided
        if base_b is not None and model_b is not None:
            b_diff = np.abs(base_b - model_b)
            # Concatenate weights and biases for a combined metric
            combined_base = np.concatenate([base_w.ravel(), base_b.ravel()])
            combined_model = np.concatenate([model_w.ravel(), model_b.ravel()])
            combined_diff = np.abs(combined_base - combined_model)
            mean_diff = combined_diff.mean()
            max_diff = combined_diff.max()
        else:
            # Only weights
            mean_diff = w_diff.mean()
            max_diff = w_diff.max()
            
        return {
            "Comparison": comp,
            "Layer": layer,
            "Type": type,
            "Max_Diff": max_diff,
            "Mean_Diff": mean_diff
        }


    def analyze_model_weights(self, model_names=None):
        """
        Compares all major weights and biases for each model against the base model.
        (Embedding, LayerNorms, QKV, Attention Output, MLP, Pooler)
        """
        results = []
        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(self.models))]
        
        base_model = self.models[0].to(self.device).eval()
        base_name = model_names[0]

        try:
            n_layers = len(base_model.encoder.layer)
            base_prefix = "" 
        except AttributeError:
            n_layers = len(base_model.bert.encoder.layer)
            base_prefix = "bert."
        
        print(f"Base model '{base_name}' has {n_layers} layers. Base prefix: '{base_prefix}'")

        for idx in range(1, len(self.models)):  # Compare models 1..N to base (0)
            model = self.models[idx].to(self.device).eval()
            name = model_names[idx]
            comp_name = f"{name} vs {base_name}"
            print(f"Analyzing: {comp_name}")

            try:
                len(model.encoder.layer)
                model_prefix = ""
            except AttributeError:
                len(model.bert.encoder.layer)
                model_prefix = "bert."
                
            # --- 1. Embedding Layer (Weight only) ---
            try:
                emb_base_w = eval(f"base_model.{base_prefix}embeddings.word_embeddings.weight").detach().cpu().numpy()
                emb_model_w = eval(f"model.{model_prefix}embeddings.word_embeddings.weight").detach().cpu().numpy()
                results.append(self._get_diff_dict(comp_name, "Embedding", "Embedding", emb_base_w, emb_model_w))
            except Exception as e:
                print(f"Error comparing Embedding weights for {name}: {e}")

            # --- 2. Embedding LayerNorm (Weight + Bias) ---
            try:
                base_w, base_b = self._get_params(eval(f"base_model.{base_prefix}embeddings.LayerNorm"))
                model_w, model_b = self._get_params(eval(f"model.{model_prefix}embeddings.LayerNorm"))
                results.append(self._get_diff_dict(comp_name, "Embedding", "Emb_LayerNorm", base_w, model_w, base_b, model_b))
            except Exception as e:
                print(f"Error comparing Embedding LayerNorm for {name}: {e}")

            # --- 4. Per-layer weights ---
            for layer in tqdm(range(n_layers), desc=f"Comparing layers for {name}"):
                try:
                    # --- 4a. Attention QKV (Weights + Biases) ---
                    attn_base = eval(f"base_model.{base_prefix}encoder.layer[{layer}].attention.self")
                    attn_model = eval(f"model.{model_prefix}encoder.layer[{layer}].attention.self")
                    
                    q_base_w, q_base_b = self._get_params(attn_base.query)
                    k_base_w, k_base_b = self._get_params(attn_base.key)
                    v_base_w, v_base_b = self._get_params(attn_base.value)

                    q_model_w, q_model_b = self._get_params(attn_model.query)
                    k_model_w, k_model_b = self._get_params(attn_model.key)
                    v_model_w, v_model_b = self._get_params(attn_model.value)

                    qkv_base_w = np.concatenate([q_base_w, k_base_w, v_base_w], axis=0)
                    qkv_model_w = np.concatenate([q_model_w, k_model_w, v_model_w], axis=0)
                    qkv_base_b = np.concatenate([q_base_b, k_base_b, v_base_b], axis=0)
                    qkv_model_b = np.concatenate([q_model_b, k_model_b, v_model_b], axis=0)
                    
                    results.append(self._get_diff_dict(comp_name, layer, "QKV_combined", qkv_base_w, qkv_model_w, qkv_base_b, qkv_model_b))

                    # --- 4b. Attention Output (Weight + Bias) ---
                    base_w, base_b = self._get_params(eval(f"base_model.{base_prefix}encoder.layer[{layer}].attention.output.dense"))
                    model_w, model_b = self._get_params(eval(f"model.{model_prefix}encoder.layer[{layer}].attention.output.dense"))
                    results.append(self._get_diff_dict(comp_name, layer, "Attn_Output", base_w, model_w, base_b, model_b))

                    # --- 4c. Attention LayerNorm (Weight + Bias) ---
                    base_w, base_b = self._get_params(eval(f"base_model.{base_prefix}encoder.layer[{layer}].attention.output.LayerNorm"))
                    model_w, model_b = self._get_params(eval(f"model.{model_prefix}encoder.layer[{layer}].attention.output.LayerNorm"))
                    results.append(self._get_diff_dict(comp_name, layer, "Attn_LayerNorm", base_w, model_w, base_b, model_b))

                    # --- 4d. MLP (Intermediate + Output, Weights + Biases) ---
                    base_inter_w, base_inter_b = self._get_params(eval(f"base_model.{base_prefix}encoder.layer[{layer}].intermediate.dense"))
                    base_out_w, base_out_b = self._get_params(eval(f"base_model.{base_prefix}encoder.layer[{layer}].output.dense"))

                    model_inter_w, model_inter_b = self._get_params(eval(f"model.{model_prefix}encoder.layer[{layer}].intermediate.dense"))
                    model_out_w, model_out_b = self._get_params(eval(f"model.{model_prefix}encoder.layer[{layer}].output.dense"))

                    # Transpose output weights to concatenate
                    mlp_base_w = np.concatenate([base_inter_w, base_out_w.T], axis=0)
                    mlp_model_w = np.concatenate([model_inter_w, model_out_w.T], axis=0)
                    # Biases can be concatenated directly
                    mlp_base_b = np.concatenate([base_inter_b, base_out_b], axis=0)
                    mlp_model_b = np.concatenate([model_inter_b, model_out_b], axis=0)
                    
                    results.append(self._get_diff_dict(comp_name, layer, "MLP_combined", mlp_base_w, mlp_model_w, mlp_base_b, mlp_model_b))
                    
                    # --- 4e. Final FFN LayerNorm (Weight + Bias) ---
                    base_w, base_b = self._get_params(eval(f"base_model.{base_prefix}encoder.layer[{layer}].output.LayerNorm"))
                    model_w, model_b = self._get_params(eval(f"model.{model_prefix}encoder.layer[{layer}].output.LayerNorm"))
                    results.append(self._get_diff_dict(comp_name, layer, "FF_LayerNorm", base_w, model_w, base_b, model_b))
                    
                except Exception as e:
                    print(f"Error comparing weights for {name} at layer {layer}: {e}")

        df = pd.DataFrame(results)
        return df

    # --- NEW PLOTTING METHOD ---
    def plot_weight_analysis(self, weight_df):
        """
        Plots the results from analyze_model_weights in a 3x3 grid
        to show all analyzed components clearly.
        """
        # Call the correct 3x3 plotting function
        self.plot_weight_analysis_3x3(weight_df)

    
    def plot_weight_analysis_3x3(self, weight_df):
        """
        Plots the results from analyze_model_weights in a 3x3 grid
        to show all analyzed components clearly.
        """
        if weight_df.empty:
            print("Weight DataFrame is empty. Cannot plot.")
            return

        comparisons = weight_df['Comparison'].unique()
        palette = sns.color_palette("tab10", n_colors=len(comparisons))
        
        fig, axes = plt.subplots(3, 3, figsize=(24, 20))
        fig.suptitle('Model Weight Differences vs. Base Model (Mean Absolute Difference)', fontsize=20, y=1.02)
        
        # --- Row 1: Bar Plots (Non-layer-specific) ---
        bar_metrics = ["Embedding", "Emb_LayerNorm"]
        bar_axes = [axes[0,0], axes[0,1]]
        
        for metric, ax in zip(bar_metrics, bar_axes):
            metric_df = weight_df[weight_df['Type'] == metric]
            if not metric_df.empty:
                sns.barplot(x='Comparison', y='Mean_Diff', data=metric_df, ax=ax, palette=palette)
                ax.set_title(f'{metric} Difference')
                ax.set_ylabel('Mean Abs. Diff (Weight+Bias)')
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=15)
            else:
                ax.set_title(f'{metric} (No Data)')
                ax.set_xlabel('')
        
        # Hide the unused axes in the first row
        axes[0, 2].set_visible(False)

        # --- Row 2: Core Transformer Layers (Line Plots) ---
        line_metrics_row2 = ["QKV_combined", "Attn_Output", "MLP_combined"]
        line_axes_row2 = [axes[1,0], axes[1,1], axes[1,2]]
        line_styles_row2 = ['-', ':', '--']
        
        for metric, ax, style in zip(line_metrics_row2, line_axes_row2, line_styles_row2):
            ax.set_title(f'{metric} Difference')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Abs. Diff (Weight+Bias)')
            for i, comp in enumerate(comparisons):
                comp_df = weight_df[(weight_df['Comparison'] == comp) & (weight_df['Type'] == metric)]
                if not comp_df.empty:
                    ax.plot(comp_df['Layer'], comp_df['Mean_Diff'], label=comp, 
                            color=palette[i], marker='.', linestyle=style)
            ax.legend(loc='best', fontsize='small')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # --- Row 3: LayerNorms (Line Plots) + Empty Plot ---
        line_metrics_row3 = ["Attn_LayerNorm", "FF_LayerNorm"]
        line_axes_row3 = [axes[2,0], axes[2,1]]
        line_styles_row3 = ['-', '--']

        for metric, ax, style in zip(line_metrics_row3, line_axes_row3, line_styles_row3):
            ax.set_title(f'{metric} Difference')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Abs. Diff (Weight+Bias)')
            for i, comp in enumerate(comparisons):
                comp_df = weight_df[(weight_df['Comparison'] == comp) & (weight_df['Type'] == metric)]
                if not comp_df.empty:
                    ax.plot(comp_df['Layer'], comp_df['Mean_Diff'], label=comp, 
                            color=palette[i], marker='.', linestyle=style)
            ax.legend(loc='best', fontsize='small')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
        # Turn off the last unused axis
        axes[2, 2].set_visible(False) 

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust for suptitle
        plt.show()
