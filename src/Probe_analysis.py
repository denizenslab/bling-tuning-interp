import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from scipy.sparse.csgraph import minimum_spanning_tree
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from conllu import parse_incr
import networkx as nx
import pickle

class StructuralProbeAnalyzer:

    def __init__(self, models, tokenizers, device='cuda'):
        self.models = models if isinstance(models, list) else [models]
        self.tokenizers = tokenizers if isinstance(tokenizers, list) else [tokenizers]
        self.device = device

    @staticmethod
    def read_ud_sentences(conllu_path, max_sentences=None):
        sentences = []
        with open(conllu_path, 'r', encoding='utf-8') as f:
            for tokenlist in parse_incr(f):
                words = []
                heads = []
                for token in tokenlist:
                    if isinstance(token['id'], int):
                        words.append(token['form'])
                        if token['head'] is None or token['head'] == 0:
                            heads.append(-1)
                        else:
                            heads.append(token['head'] - 1)
                sentences.append({'words': words, 'heads': heads, 'sent_id': tokenlist.metadata.get('sent_id', None)})
                if max_sentences and len(sentences) >= max_sentences:
                    break
        return sentences

    def extract_bert_layerwise_reps(self, sentences, model, tokenizer, layer_output_mode='pre_norm'):
        all_reps = []
        hook_handles = []
        layer_activations = {}

        def get_hook(layer_idx):

            def hook(module, input, output):
                layer_activations[layer_idx] = input[0].detach()
            return hook
        if layer_output_mode == 'pre_norm':
            if hasattr(model, 'bert'):
                encoder_layers = model.bert.encoder.layer
            elif hasattr(model, 'encoder'):
                encoder_layers = model.encoder.layer
            else:
                encoder_layers = model.transformer.h
            for i, layer in enumerate(encoder_layers):
                if hasattr(layer, 'output') and hasattr(layer.output, 'LayerNorm'):
                    target_module = layer.output.LayerNorm
                else:
                    target_module = layer.output.LayerNorm
                handle = target_module.register_forward_hook(get_hook(i))
                hook_handles.append(handle)
        try:
            for sent in tqdm(sentences, desc=f'Extracting BERT reps ({layer_output_mode})'):
                words = sent['words']
                encoding = tokenizer(words, is_split_into_words=True, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(**{k: v.to(self.device) for k, v in encoding.items()}, output_hidden_states=True)
                if layer_output_mode == 'post_norm':
                    hidden_states = outputs.hidden_states
                else:
                    num_layers = len(hook_handles)
                    layers_list = [outputs.hidden_states[0]]
                    for i in range(num_layers):
                        layers_list.append(layer_activations[i])
                    hidden_states = tuple(layers_list)
                word_ids = encoding.word_ids(batch_index=0)
                reps_by_word = []
                for i in range(len(words)):
                    token_idxs = [j for j, wid in enumerate(word_ids) if wid == i]
                    layer_reps = []
                    for l in range(len(hidden_states)):
                        subtoks = torch.stack([hidden_states[l][0, tid] for tid in token_idxs], dim=0)
                        mean_rep = subtoks.mean(dim=0)
                        layer_reps.append(mean_rep.cpu())
                    reps_by_word.append(layer_reps)
                all_reps.append({'words': words, 'hidden_states': reps_by_word, 'heads': sent['heads'], 'sent_id': sent['sent_id']})
        finally:
            for handle in hook_handles:
                handle.remove()
        return all_reps

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
                    depth = -1
                    break
                visited.add(cur)
                cur = heads[cur]
                depth += 1
            if depth != -1:
                depths[idx] = depth
        return depths

    @staticmethod
    def compute_tree_distances(heads):
        n = len(heads)
        adj = np.zeros((n, n), dtype=int)
        for i, h in enumerate(heads):
            if h is not None and h != -1 and (0 <= h < n):
                adj[i, h] = 1
                adj[h, i] = 1
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        dist[adj == 1] = 1
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, j] > dist[i, k] + dist[k, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        return dist

    @staticmethod
    def compute_uuas(gold_heads, pred_dist):
        n = len(gold_heads)
        gold_edges = set()
        for i, h in enumerate(gold_heads):
            if h != -1:
                gold_edges.add(tuple(sorted((i, h))))
        mst = minimum_spanning_tree(pred_dist + pred_dist.T).toarray()
        pred_edges = set()
        for i in range(n):
            for j in range(n):
                if i != j and (mst[i, j] != 0 or mst[j, i] != 0):
                    pred_edges.add(tuple(sorted((i, j))))
        if len(gold_edges) == 0:
            return None
        correct = len(gold_edges & pred_edges)
        return correct / len(gold_edges)

    def run_probe(self, train_layerwise_reps, test_layerwise_reps, max_distance_train=100, max_distance_eval=10):
        num_layers = len(train_layerwise_reps[0]['hidden_states'][0])
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
            X_train = []
            y_train = []
            for sent in train_layerwise_reps:
                for i, word_reps in enumerate(sent['hidden_states']):
                    X_train.append(word_reps[layer].numpy())
                    y_train.append(sent['depths'][i])
            X_train = np.stack(X_train)
            y_train = np.array(y_train)
            reg_depth = Ridge(alpha=1.0, solver='svd').fit(X_train, y_train)
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
            X_pairs_train = []
            y_pairs_train = []
            for sent in train_layerwise_reps[:max_distance_train]:
                X = np.stack([word_reps[layer].numpy() for word_reps in sent['hidden_states']])
                D = np.array(sent['distances'])
                for i in range(len(D)):
                    for j in range(i + 1, len(D)):
                        X_pairs_train.append(X[i] - X[j])
                        y_pairs_train.append(np.sqrt(D[i, j] + 1e-08))
            X_pairs_train = np.stack(X_pairs_train)
            y_pairs_train = np.array(y_pairs_train)
            reg_dist = Ridge(alpha=1.0, solver='svd').fit(X_pairs_train, y_pairs_train)
            spearman_vals = []
            uuas_vals = []
            for sent in test_layerwise_reps[:max_distance_eval]:
                n = len(sent['words'])
                X = np.stack([word_reps[layer].numpy() for word_reps in sent['hidden_states']])
                pred_dist = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
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
            fitted_probes[layer] = {'depth': reg_depth, 'dist': reg_dist}
        return {'root_accs': root_accs, 'depth_spearmans': depth_spearmans, 'dist_spearmans': dist_spearmans, 'uuas_list': uuas_list, 'num_layers': num_layers, 'fitted_probes': fitted_probes}

    def plot_tree_comparison_ax(self, ax_gold, ax_pred, sentence_data, layerwise_reps_for_sentence, fitted_probes, layer_to_plot, model_name=''):
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
            for j in range(i + 1, n):
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
        G_gold = nx.Graph()
        G_gold.add_nodes_from(range(n))
        G_gold.add_edges_from(gold_edges)
        pos_gold = nx.spring_layout(G_gold, seed=42)
        nx.draw(G_gold, pos=pos_gold, labels=labels, with_labels=True, node_color='lightblue', ax=ax_gold, node_size=2000, font_size=10)
        ax_gold.set_title(f'{model_name} Gold Tree', fontsize=12)
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
        ax_pred.set_title(f'{model_name} Predicted MST - Layer {layer_to_plot}', fontsize=12)
        ax_gold.axis('off')
        ax_pred.axis('off')

    def plot_tree_comparison(self, sentence_data, layerwise_reps_for_sentence, fitted_probes, layer_to_plot, model_name=''):
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
            for j in range(i + 1, n):
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
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        labels = {i: w for i, w in enumerate(words)}
        G_gold = nx.Graph()
        G_gold.add_nodes_from(range(n))
        G_gold.add_edges_from(gold_edges)
        pos_gold = nx.spring_layout(G_gold, seed=42)
        nx.draw(G_gold, pos=pos_gold, labels=labels, with_labels=True, node_color='lightblue', ax=axes[0], node_size=2000, font_size=12)
        axes[0].set_title('Gold Dependency Tree (Undirected)', fontsize=16)
        G_pred = nx.Graph()
        G_pred.add_nodes_from(range(n))
        G_pred.add_edges_from(pred_edges)
        try:
            pos_pred = nx.spring_layout(G_pred, pos=pos_gold, seed=42)
        except:
            pos_pred = nx.spring_layout(G_pred, seed=42)
        correct_edges = set(gold_edges) & set(pred_edges)
        incorrect_edges = set(pred_edges) - set(gold_edges)
        missing_edges = set(gold_edges) - set(pred_edges)
        nx.draw_networkx_nodes(G_pred, pos=pos_pred, ax=axes[1], node_color='lightgray', node_size=2000)
        nx.draw_networkx_labels(G_pred, pos=pos_pred, labels=labels, ax=axes[1], font_size=12)
        nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=correct_edges, ax=axes[1], edge_color='green', width=2)
        nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=incorrect_edges, ax=axes[1], edge_color='red', width=2, style='dashed')
        axes[1].set_title(f'Predicted Geometric Tree (MST) - {model_name} Layer {layer_to_plot}', fontsize=16)
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='green', lw=2, label='Correct Edge (in Gold & Pred)'), Line2D([0], [0], color='red', lw=2, linestyle='--', label='Incorrect Edge (in Pred only)')]
        axes[1].legend(handles=legend_elements, loc='lower center', fontsize='medium')
        plt.tight_layout()
        plt.show()

    def compare_to_base_attention_embedding_combined(self, model_names=None):
        results = []
        if model_names is None:
            model_names = [f'Model_{i + 1}' for i in range(len(self.models))]
        base_model = self.models[0]
        base_name = model_names[0]
        try:
            n_layers = len(base_model.encoder.layer)
        except AttributeError:
            n_layers = len(base_model.bert.encoder.layer)
        for idx in range(1, len(self.models)):
            model = self.models[idx]
            name = model_names[idx]
            try:
                emb_base = base_model.embeddings.word_embeddings.weight.detach().cpu().numpy()
                emb_model = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
            except AttributeError:
                emb_base = base_model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
                emb_model = model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
            results.append({'Comparison': f'{name} vs {base_name}', 'Layer': 'Embedding', 'Type': 'Embedding', 'Max_Diff': np.abs(emb_base - emb_model).max(), 'Mean_Diff': np.abs(emb_base - emb_model).mean()})
            for layer in range(n_layers):
                try:
                    attn_base = base_model.encoder.layer[layer].attention.self
                    attn_model = model.encoder.layer[layer].attention.self
                except AttributeError:
                    attn_base = base_model.bert.encoder.layer[layer].attention.self
                    attn_model = model.bert.encoder.layer[layer].attention.self
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
                results.append({'Comparison': f'{name} vs {base_name}', 'Layer': layer, 'Type': 'QKV_combined', 'Max_Diff': max_diff, 'Mean_Diff': mean_diff})
        df = pd.DataFrame(results)
        return df

    @staticmethod
    def _get_params(param_obj):
        w = param_obj.weight.detach().cpu().numpy()
        b = param_obj.bias.detach().cpu().numpy()
        return (w, b)

    @staticmethod
    def _get_diff_dict(comp, layer, type, base_w, model_w, base_b=None, model_b=None):
        w_diff = np.abs(base_w - model_w)
        if base_b is not None and model_b is not None:
            b_diff = np.abs(base_b - model_b)
            combined_base = np.concatenate([base_w.ravel(), base_b.ravel()])
            combined_model = np.concatenate([model_w.ravel(), model_b.ravel()])
            combined_diff = np.abs(combined_base - combined_model)
            mean_diff = combined_diff.mean()
            max_diff = combined_diff.max()
        else:
            mean_diff = w_diff.mean()
            max_diff = w_diff.max()
        return {'Comparison': comp, 'Layer': layer, 'Type': type, 'Max_Diff': max_diff, 'Mean_Diff': mean_diff}

    def analyze_model_weights(self, model_names=None):
        results = []
        if model_names is None:
            model_names = [f'Model_{i + 1}' for i in range(len(self.models))]
        base_model = self.models[0].to(self.device).eval()
        base_name = model_names[0]
        try:
            n_layers = len(base_model.encoder.layer)
            base_prefix = ''
        except AttributeError:
            n_layers = len(base_model.bert.encoder.layer)
            base_prefix = 'bert.'
        print(f"Base model '{base_name}' has {n_layers} layers. Base prefix: '{base_prefix}'")
        for idx in range(1, len(self.models)):
            model = self.models[idx].to(self.device).eval()
            name = model_names[idx]
            comp_name = f'{name} vs {base_name}'
            print(f'Analyzing: {comp_name}')
            try:
                len(model.encoder.layer)
                model_prefix = ''
            except AttributeError:
                len(model.bert.encoder.layer)
                model_prefix = 'bert.'
            try:
                emb_base_w = eval(f'base_model.{base_prefix}embeddings.word_embeddings.weight').detach().cpu().numpy()
                emb_model_w = eval(f'model.{model_prefix}embeddings.word_embeddings.weight').detach().cpu().numpy()
                results.append(self._get_diff_dict(comp_name, 'Embedding', 'Embedding', emb_base_w, emb_model_w))
            except Exception as e:
                print(f'Error comparing Embedding weights for {name}: {e}')
            try:
                base_w, base_b = self._get_params(eval(f'base_model.{base_prefix}embeddings.LayerNorm'))
                model_w, model_b = self._get_params(eval(f'model.{model_prefix}embeddings.LayerNorm'))
                results.append(self._get_diff_dict(comp_name, 'Embedding', 'Emb_LayerNorm', base_w, model_w, base_b, model_b))
            except Exception as e:
                print(f'Error comparing Embedding LayerNorm for {name}: {e}')
            for layer in tqdm(range(n_layers), desc=f'Comparing layers for {name}'):
                try:
                    attn_base = eval(f'base_model.{base_prefix}encoder.layer[{layer}].attention.self')
                    attn_model = eval(f'model.{model_prefix}encoder.layer[{layer}].attention.self')
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
                    results.append(self._get_diff_dict(comp_name, layer, 'QKV_combined', qkv_base_w, qkv_model_w, qkv_base_b, qkv_model_b))
                    base_w, base_b = self._get_params(eval(f'base_model.{base_prefix}encoder.layer[{layer}].attention.output.dense'))
                    model_w, model_b = self._get_params(eval(f'model.{model_prefix}encoder.layer[{layer}].attention.output.dense'))
                    results.append(self._get_diff_dict(comp_name, layer, 'Attn_Output', base_w, model_w, base_b, model_b))
                    base_w, base_b = self._get_params(eval(f'base_model.{base_prefix}encoder.layer[{layer}].attention.output.LayerNorm'))
                    model_w, model_b = self._get_params(eval(f'model.{model_prefix}encoder.layer[{layer}].attention.output.LayerNorm'))
                    results.append(self._get_diff_dict(comp_name, layer, 'Attn_LayerNorm', base_w, model_w, base_b, model_b))
                    base_inter_w, base_inter_b = self._get_params(eval(f'base_model.{base_prefix}encoder.layer[{layer}].intermediate.dense'))
                    base_out_w, base_out_b = self._get_params(eval(f'base_model.{base_prefix}encoder.layer[{layer}].output.dense'))
                    model_inter_w, model_inter_b = self._get_params(eval(f'model.{model_prefix}encoder.layer[{layer}].intermediate.dense'))
                    model_out_w, model_out_b = self._get_params(eval(f'model.{model_prefix}encoder.layer[{layer}].output.dense'))
                    mlp_base_w = np.concatenate([base_inter_w, base_out_w.T], axis=0)
                    mlp_model_w = np.concatenate([model_inter_w, model_out_w.T], axis=0)
                    mlp_base_b = np.concatenate([base_inter_b, base_out_b], axis=0)
                    mlp_model_b = np.concatenate([model_inter_b, model_out_b], axis=0)
                    results.append(self._get_diff_dict(comp_name, layer, 'MLP_combined', mlp_base_w, mlp_model_w, mlp_base_b, mlp_model_b))
                    base_w, base_b = self._get_params(eval(f'base_model.{base_prefix}encoder.layer[{layer}].output.LayerNorm'))
                    model_w, model_b = self._get_params(eval(f'model.{model_prefix}encoder.layer[{layer}].output.LayerNorm'))
                    results.append(self._get_diff_dict(comp_name, layer, 'FF_LayerNorm', base_w, model_w, base_b, model_b))
                except Exception as e:
                    print(f'Error comparing weights for {name} at layer {layer}: {e}')
        df = pd.DataFrame(results)
        return df

    def plot_weight_analysis(self, weight_df):
        self.plot_weight_analysis_3x3(weight_df)

    def plot_weight_analysis_3x3(self, weight_df, models, subject, title='Model Weight Differences vs. Base Model (Mean Absolute Difference) - BERT'):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        if weight_df.empty:
            print('Weight DataFrame is empty. Cannot plot.')
            return
        comparisons = weight_df['Comparison'].unique()
        palette = sns.color_palette('tab10', n_colors=len(comparisons))
        fig, axes = plt.subplots(3, 3, figsize=(24, 20))
        fig.suptitle(f'{title} - Subject {subject}', fontsize=20, y=1.02)
        bar_metrics = ['Embedding', 'Emb_LayerNorm']
        bar_axes = [axes[0, 0], axes[0, 1]]
        vals_row1 = weight_df[weight_df['Type'].isin(bar_metrics)]['Mean_Diff']
        min_row1, max_row1 = (vals_row1.min(), vals_row1.max())
        for metric, ax in zip(bar_metrics, bar_axes):
            metric_df = weight_df[weight_df['Type'] == metric]
            if not metric_df.empty:
                ax = sns.barplot(x='Comparison', y='Mean_Diff', data=metric_df, ax=ax, palette=palette)
                ax.set_title(f'{metric} Difference')
                ax.set_ylabel('Mean Abs. Diff (Weight+Bias)')
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=15)
                ax.set_ylim(min_row1, max_row1)
                for idx_patch, patch in enumerate(ax.patches):
                    value = patch.get_height()
                    x = patch.get_x() + patch.get_width() / 2.0
                    label = f'{value:.6f}'
                    y_offset = (max_row1 - min_row1) * 0.01
                    if value > max_row1 - y_offset * 2:
                        ax.text(x, value - y_offset, label, ha='center', va='top', fontsize=12, color='white', rotation=0)
                    else:
                        ax.text(x, value + y_offset, label, ha='center', va='bottom', fontsize=12, color='black', rotation=0)
            else:
                ax.set_title(f'{metric} (No Data)')
                ax.set_xlabel('')
        axes[0, 2].set_visible(False)
        line_metrics_row2 = ['QKV_combined', 'Attn_Output', 'MLP_combined']
        line_axes_row2 = [axes[1, 0], axes[1, 1], axes[1, 2]]
        line_styles_row2 = ['-', ':', '--']
        vals_row2 = weight_df[weight_df['Type'].isin(line_metrics_row2)]['Mean_Diff']
        min_row2, max_row2 = (vals_row2.min(), vals_row2.max())
        for metric, ax, style in zip(line_metrics_row2, line_axes_row2, line_styles_row2):
            ax.set_title(f'{metric} Difference')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Abs. Diff (Weight+Bias)')
            for i, comp in enumerate(comparisons):
                comp_df = weight_df[(weight_df['Comparison'] == comp) & (weight_df['Type'] == metric)]
                if not comp_df.empty:
                    ax.plot(comp_df['Layer'], comp_df['Mean_Diff'], label=comp, color=palette[i], marker='.', linestyle=style)
            ax.legend(loc='best', fontsize='small')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylim(min_row2, max_row2)
        line_metrics_row3 = ['Attn_LayerNorm', 'FF_LayerNorm']
        line_axes_row3 = [axes[2, 0], axes[2, 1]]
        line_styles_row3 = ['-', '--']
        vals_row3 = weight_df[weight_df['Type'].isin(line_metrics_row3)]['Mean_Diff']
        min_row3, max_row3 = (vals_row3.min(), vals_row3.max())
        for metric, ax, style in zip(line_metrics_row3, line_axes_row3, line_styles_row3):
            ax.set_title(f'{metric} Difference')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean Abs. Diff (Weight+Bias)')
            for i, comp in enumerate(comparisons):
                comp_df = weight_df[(weight_df['Comparison'] == comp) & (weight_df['Type'] == metric)]
                if not comp_df.empty:
                    ax.plot(comp_df['Layer'], comp_df['Mean_Diff'], label=comp, color=palette[i], marker='.', linestyle=style)
            ax.legend(loc='best', fontsize='small')
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_ylim(min_row3, max_row3)
        axes[2, 2].set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        save_dir = f'probe_results/subject_{subject}/weight_analysis'
        os.makedirs(save_dir, exist_ok=True)
        plot_data = {'weight_df': weight_df, 'title': title, 'subject': subject, 'comparisons': comparisons.tolist()}
        save_path = os.path.join(save_dir, f'weight_analysis_3x3_subject_{subject}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f'Saved plot data: {save_path}')
        plt.show()

    def plot_probe_results(self, results_list, models, subject, model_names=None, language='English', y_lowlim=0, max_sentences=None, title=''):
        import pandas as pd
        if not isinstance(results_list, list):
            results_list = [results_list]
        if not isinstance(models, list):
            models = [models]
        if model_names is None:
            model_names = [f'Model {i + 1}' for i in range(len(results_list))]
        num_layers = results_list[0]['num_layers']
        layers = np.arange(1, num_layers)
        layer_labels = [f'{i}' for i in layers]
        records = []
        for res, name in zip(results_list, model_names):
            for i in layers:
                records.append({'Layer': i, 'Layer Label': f'Layer {i}' if i == 1 else f'Transformer {i}', 'Root Accuracy': res['root_accs'][i], 'Depth Spearman': res['depth_spearmans'][i], 'Distance Spearman': res['dist_spearmans'][i], 'UUAS': res['uuas_list'][i], 'Model': name})
        df = pd.DataFrame(records)
        for col in ['Root Accuracy', 'Depth Spearman', 'Distance Spearman', 'UUAS']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Root Accuracy', 'Depth Spearman', 'Distance Spearman', 'UUAS'])
        palette = sns.color_palette('tab10', n_colors=len(model_names))
        sns.set(style='whitegrid', font_scale=1.2)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        line_styles = ['-', '--', '-.', ':']
        ylabels = ['Root Accuracy', 'Depth Spearman Correlation', 'UUAS', 'Distance Spearman Correlation']
        metrics = ['Root Accuracy', 'Depth Spearman', 'UUAS', 'Distance Spearman']
        axes_grid = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        for idx, (metric, ax, ylabel) in enumerate(zip(metrics, axes_grid, ylabels)):
            for i, name in enumerate(model_names):
                subdf = df[df['Model'] == name]
                ax.plot(subdf['Layer'], subdf[metric], label=name, color=palette[i % len(palette)], linestyle=line_styles[i % len(line_styles)], marker='o')
            ax.set(title=f'{language} {metric}', xlabel='Transformer Block Output (1-12)', ylabel=ylabel, ylim=(y_lowlim, 1), xticks=layers, xticklabels=layer_labels)
            ax.legend(loc='best', fontsize='medium')
        fig.suptitle(f'{title} - Subject {subject}', fontsize=20, y=0.98)
        plt.tight_layout()
        fig.text(0.99, 0.01, f'Number of sentences: {max_sentences}', ha='right', va='bottom', fontsize=14, color='black', alpha=0.8)
        save_dir = f'probe_results/subject_{subject}/probe_results'
        os.makedirs(save_dir, exist_ok=True)
        plot_data = {'df': df, 'results_list': results_list, 'model_names': model_names, 'language': language, 'y_lowlim': y_lowlim, 'max_sentences': max_sentences, 'title': title, 'subject': subject, 'metrics': metrics, 'ylabels': ylabels}
        save_path = os.path.join(save_dir, f'probe_results_{language.lower()}_subject_{subject}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f'Saved plot data: {save_path}')
        plt.show()

    def plot_gold_and_predicted_trees(self, test_sentences, results_list, sentence_indices, layer_to_plot, subject, title='', model_names=['BERT-base', 'BERT-whole', 'BERT-semantic', 'BERT-language'], layer_output_mode='pre_norm'):
        import matplotlib.pyplot as plt
        tree_results = {}
        for sentence_idx in sentence_indices:
            sentence_data = test_sentences[sentence_idx]
            words = sentence_data['words']
            heads = sentence_data['heads']
            n = len(words)
            fig, axes = plt.subplots(1, 5, figsize=(25, 6))
            try:
                if all((len(w) == 1 for w in words)):
                    sentence_str = ''.join(words)
                else:
                    sentence_str = ' '.join(words)
            except Exception:
                sentence_str = ' '.join(words)
            fig.text(0.5, 1.02, f'Sentence {sentence_idx}: {sentence_str}', ha='center', va='bottom', fontsize=18)
            fig.suptitle(f'{title} - Subject {subject} ({layer_output_mode})', fontsize=24, y=1.15)
            fig.text(0.5, 0.96, f'Sentence {sentence_idx} Parse Trees: Gold and Predicted (Layer {layer_to_plot})', ha='center', va='bottom', fontsize=16)
            gold_edges = []
            for i, h in enumerate(heads):
                if h != -1:
                    gold_edges.append(tuple(sorted((i, h))))
            labels = {i: w for i, w in enumerate(words)}
            G_gold = nx.Graph()
            G_gold.add_nodes_from(range(n))
            G_gold.add_edges_from(gold_edges)
            pos_gold = nx.spring_layout(G_gold, seed=42)
            nx.draw(G_gold, pos=pos_gold, labels=labels, with_labels=True, node_color='lightblue', ax=axes[0], node_size=2000, font_size=12)
            axes[0].set_title('Gold Tree', fontsize=16)
            axes[0].axis('off')
            total_gold_edges = len(gold_edges)
            sentence_tree_data = {'sentence_idx': sentence_idx, 'sentence_str': sentence_str, 'words': words, 'gold_edges': gold_edges, 'predicted_models': {}}
            for i, (model, tokenizer, name) in enumerate(zip(self.models, self.tokenizers, model_names)):
                layerwise_reps_dict = self.extract_bert_layerwise_reps([sentence_data], model, tokenizer, layer_output_mode=layer_output_mode)[0]
                hidden_states = layerwise_reps_dict['hidden_states']
                fitted_probes = results_list[i]['fitted_probes']
                reg_dist = fitted_probes[layer_to_plot]['dist']
                X = np.stack([word_reps[layer_to_plot].numpy() for word_reps in hidden_states])
                pred_dist = np.zeros((n, n))
                for a in range(n):
                    for b in range(a + 1, n):
                        dpred = reg_dist.predict((X[a] - X[b]).reshape(1, -1))[0]
                        pred_dist[a, b] = dpred ** 2
                        pred_dist[b, a] = pred_dist[a, b]
                pred_dist_sym = (pred_dist + pred_dist.T) / 2
                pred_dist_sym[pred_dist_sym < 0] = 0
                np.fill_diagonal(pred_dist_sym, 0)
                mst = minimum_spanning_tree(pred_dist_sym).toarray()
                pred_edges = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if mst[a, b] != 0 or mst[b, a] != 0:
                            pred_edges.append((a, b))
                correct_edges = set(gold_edges) & set(pred_edges)
                incorrect_edges = set(pred_edges) - set(gold_edges)
                num_correct = len(correct_edges)
                total_pred = len(pred_edges)
                sentence_tree_data['predicted_models'][name] = {'predicted_edges': pred_edges, 'correct_edges': list(correct_edges), 'incorrect_edges': list(incorrect_edges), 'num_correct': num_correct, 'total_gold_edges': total_gold_edges, 'total_pred_edges': total_pred}
                G_pred = nx.Graph()
                G_pred.add_nodes_from(range(n))
                G_pred.add_edges_from(pred_edges)
                try:
                    pos_pred = nx.spring_layout(G_pred, pos=pos_gold, seed=42)
                except Exception:
                    pos_pred = nx.spring_layout(G_pred, seed=42)
                ax = axes[i + 1]
                nx.draw_networkx_nodes(G_pred, pos=pos_pred, ax=ax, node_color='lightgray', node_size=2000)
                nx.draw_networkx_labels(G_pred, pos=pos_pred, labels=labels, ax=ax, font_size=12)
                nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=correct_edges, ax=ax, edge_color='green', width=2)
                nx.draw_networkx_edges(G_pred, pos=pos_pred, edgelist=incorrect_edges, ax=ax, edge_color='red', width=2, style='dashed')
                ax.set_title(f'{name}\nPredicted MST', fontsize=16)
                ax.text(0.5, -0.08, f'Correct edges: {num_correct}/{total_gold_edges}', ha='center', va='top', fontsize=11, transform=ax.transAxes)
                ax.axis('off')
            tree_results[sentence_idx] = sentence_tree_data
            plt.tight_layout(rect=[0, 0.06, 1, 0.95])
            plt.show()
        save_dir = f'probe_results/subject_{subject}/tree_comparison'
        os.makedirs(save_dir, exist_ok=True)
        save_data = {'title': title, 'subject': subject, 'layer_to_plot': layer_to_plot, 'model_names': model_names, 'tree_results': tree_results, 'layer_output_mode': layer_output_mode}
        save_path = os.path.join(save_dir, f'tree_comparison_layer{layer_to_plot}_subject_{subject}_{layer_output_mode}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f'Saved tree results: {save_path}')
