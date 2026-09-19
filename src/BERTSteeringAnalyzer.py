import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cosine

class SteeringVectorAnalyzer:

    class Results:

        def __init__(self):
            self.layer_vectors = {}

        def add(self, layer, model, concept, vec):
            if layer not in self.layer_vectors:
                self.layer_vectors[layer] = {}
            if model not in self.layer_vectors[layer]:
                self.layer_vectors[layer][model] = {}
            self.layer_vectors[layer][model][concept] = vec.cpu().detach()

        def get(self, layer, model, concept):
            return self.layer_vectors.get(layer, {}).get(model, {}).get(concept)

        def similarity(self, layer, model_a, concept_a, model_b, concept_b):
            va, vb = (self.get(layer, model_a, concept_a), self.get(layer, model_b, concept_b))
            if va is None or vb is None:
                return 0.0
            v1, v2 = (va.float().numpy().flatten(), vb.float().numpy().flatten())
            if np.all(v1 == 0) or np.all(v2 == 0):
                return 0.0
            return 1 - cosine(v1, v2)

    def __init__(self, device='cuda', layers=None, csv_path='llm_insight_dataset.csv'):
        self.device = device
        self.layers = layers if layers else list(range(12))
        self.csv_path = csv_path
        self.results = self.Results()
        self.load_dataset()
        self.categories = {'Abstract': ['good', 'bad', 'love', 'hate'], 'Adjective': ['big', 'small'], 'Concrete': ['computer', 'animal', 'music'], 'Scientific': ['thermodynamics']}
        self.zh_map = {'good': '好', 'bad': '坏', 'big': '大', 'small': '小', 'love': '爱', 'hate': '恨', 'thermodynamics': '热力学', 'computer': '电脑', 'animal': '动物', 'music': '音乐'}
        self.language_vectors: dict = {}
        self.cloze_steering_results = {}

    def load_dataset(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f'Dataset not found at {self.csv_path}.')
        self.dataset = pd.read_csv(self.csv_path)
        print(f'Analyzer initialized. Dataset size: {len(self.dataset)}')

    def _get_steering_data(self, concept_word, lang):
        df = self.dataset
        lang = lang.lower()
        pos_rows = df[(df['word'] == concept_word) & (df['lang'] == lang) & (df['type'] == 'sentence')]
        pos_texts = pos_rows['text'].tolist()
        if not pos_texts:
            return ([], [])
        pair = pos_rows.iloc[0]['pair']
        if pd.notna(pair):
            neg_texts = df[(df['word'] == pair) & (df['lang'] == lang) & (df['type'] == 'sentence')]['text'].tolist()
        else:
            neg_texts = df[(df['word'] == 'GENERAL_SET') & (df['lang'] == lang)]['text'].tolist()
        return (pos_texts, neg_texts)

    def _compute_hidden_means_all_layers(self, model, tokenizer, texts):
        if not texts:
            return None
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            layer_means = {}
            for layer_idx in self.layers:
                if layer_idx + 1 >= len(out.hidden_states):
                    continue
                hs = out.hidden_states[layer_idx + 1]
                layer_means[layer_idx] = hs.mean(dim=1)
            return layer_means

    def run_batch_analysis(self, model_families, family_name: str, subject: str, use_pre_norm: bool=True, save_dir: str='steering_vectors'):
        print(f'--- Running Concept Vector Analysis (Cosine) [Pre-Norm={use_pre_norm}] ---')
        flat_models = {}
        if isinstance(model_families, dict) and 'model' in list(model_families.values())[0]:
            flat_models = model_families
        else:
            for fam, models in model_families.items():
                for name, cfg in models.items():
                    flat_models[f'{fam}_{name}'] = cfg
        target_concepts = self.dataset[(self.dataset['lang'] == 'en') & (self.dataset['word'] != 'GENERAL_SET')]['word'].unique().tolist()
        for m_key, config in tqdm(flat_models.items(), desc='Models'):
            model, tokenizer = (config['model'], config['tokenizer'])
            model.to(self.device)
            model.eval()
            for word in target_concepts:
                pos_en, neg_en = self._get_steering_data(word, 'en')
                if pos_en:
                    pos_means = self._extract_layer_means(model, tokenizer, pos_en, use_pre_norm)
                    neg_means = self._extract_layer_means(model, tokenizer, neg_en, use_pre_norm)
                    for layer in self.layers:
                        if layer in pos_means and layer in neg_means:
                            v = pos_means[layer] - neg_means[layer]
                            self.results.add(layer, m_key, f'{word}_en', v)
                zh_word = self.zh_map.get(word)
                if zh_word:
                    pos_zh, neg_zh = self._get_steering_data(zh_word, 'zh')
                    if pos_zh:
                        pos_means = self._extract_layer_means(model, tokenizer, pos_zh, use_pre_norm)
                        neg_means = self._extract_layer_means(model, tokenizer, neg_zh, use_pre_norm)
                        for layer in self.layers:
                            if layer in pos_means and layer in neg_means:
                                v = pos_means[layer] - neg_means[layer]
                                self.results.add(layer, m_key, f'{word}_zh', v)
        self._save_steering_similarity_results(family_name=family_name, subject=subject, use_pre_norm=use_pre_norm, save_dir=save_dir)

    def _save_steering_similarity_results(self, family_name: str, subject: str, use_pre_norm: bool, save_dir: str='steering_vectors') -> None:
        os.makedirs(save_dir, exist_ok=True)
        out_dir = os.path.join(save_dir, f'subject_{subject}', family_name)
        os.makedirs(out_dir, exist_ok=True)
        norm_tag = 'prenorm' if use_pre_norm else 'postnorm'
        fname = f'steering_{family_name}_{subject}_{norm_tag}.pkl'
        fpath = os.path.join(out_dir, fname)
        payload = {'results': self.results, 'meta': {'family_name': family_name, 'subject': subject, 'use_pre_norm': use_pre_norm}}
        with open(fpath, 'wb') as f:
            pickle.dump(payload, f)
        print(f'[Steering] Saved similarity results to {fpath}')

    def _get_mask_index(self, input_ids, tokenizer):
        mask_id = tokenizer.mask_token_id
        mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)
        if len(mask_positions) != 1:
            return None
        return mask_positions[0, 1].item()

    def _get_language_direction(self, layer_idx, model_name, steer_lang):
        layer_dict = self.language_vectors.get(layer_idx, {}).get(model_name, {})
        v_en = layer_dict.get('en', None)
        v_zh = layer_dict.get('zh', None)
        if v_en is None or v_zh is None:
            return None
        if steer_lang == 'en':
            return v_en - v_zh
        elif steer_lang == 'zh':
            return v_zh - v_en
        return None

    def _extract_layer_means(self, model, tokenizer, texts, use_pre_norm=True):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        captured_activations = {l: [] for l in self.layers}
        handles = []
        if use_pre_norm:

            def get_hook(layer_idx):

                def hook(module, input):
                    captured_activations[layer_idx].append(input[0].detach().cpu())
                return hook
            for layer_idx in self.layers:
                if hasattr(model, 'bert'):
                    target_module = model.bert.encoder.layer[layer_idx].output.LayerNorm
                elif hasattr(model, 'encoder'):
                    target_module = model.encoder.layer[layer_idx].output.LayerNorm
                else:
                    continue
                handles.append(target_module.register_forward_pre_hook(get_hook(layer_idx)))
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=not use_pre_norm)
        for h in handles:
            h.remove()
        layer_means = {}
        for layer_idx in self.layers:
            if use_pre_norm:
                if not captured_activations[layer_idx]:
                    continue
                hs = torch.cat(captured_activations[layer_idx], dim=0)
            elif layer_idx + 1 < len(out.hidden_states):
                hs = out.hidden_states[layer_idx + 1]
            else:
                continue
            layer_means[layer_idx] = hs.mean(dim=(0, 1)).detach().cpu()
        return layer_means

    def compute_language_identity_vectors(self, model_families, languages=('en', 'zh'), max_sentences_per_lang=2000, use_pre_norm=True):
        print(f'--- Computing GENERIC Vectors [Pre-Norm={use_pre_norm}] ---')
        self.language_vectors = {}
        from torch.utils.data import DataLoader
        df = self.dataset
        lang_texts = {}
        for lang in languages:
            texts = df[(df['lang'] == lang) & (df['type'] == 'sentence')]['text'].tolist()
            if max_sentences_per_lang:
                texts = texts[:max_sentences_per_lang]
            lang_texts[lang] = texts
        flat_models = {}
        if isinstance(model_families, dict) and 'model' in list(model_families.values())[0]:
            flat_models = model_families
        else:
            for family, models in model_families.items():
                for name, cfg in models.items():
                    flat_models[f'{family}_{name}'] = cfg
        for m_name, cfg in tqdm(flat_models.items(), desc='Models'):
            model, tokenizer = (cfg['model'], cfg['tokenizer'])
            model.to(self.device)
            model.eval()
            for lang in languages:
                texts = lang_texts.get(lang, [])
                if not texts:
                    continue
                loader = DataLoader(texts, batch_size=16, shuffle=False)
                layer_sums = {l: None for l in self.layers}
                layer_counts = 0
                for batch in loader:
                    batch_means = self._extract_layer_means(model, tokenizer, list(batch), use_pre_norm)
                    for layer_idx, vec in batch_means.items():
                        if layer_sums[layer_idx] is None:
                            layer_sums[layer_idx] = vec
                        else:
                            layer_sums[layer_idx] += vec
                    layer_counts += 1
                for layer_idx in self.layers:
                    if layer_sums[layer_idx] is None:
                        continue
                    vec = (layer_sums[layer_idx] / layer_counts).detach().cpu()
                    self.language_vectors.setdefault(layer_idx, {})
                    self.language_vectors[layer_idx].setdefault(m_name, {})
                    self.language_vectors[layer_idx][m_name][lang] = vec

    def compute_targeted_vectors_from_list(self, model_families, facts_en, facts_zh, use_pre_norm=True):
        print(f'--- Computing TARGETED Vectors [Pre-Norm={use_pre_norm}] ---')
        self.language_vectors = {}
        flat_models = {}
        if isinstance(model_families, dict) and 'model' in list(model_families.values())[0]:
            flat_models = model_families
        else:
            for family, models in model_families.items():
                for name, cfg in models.items():
                    flat_models[f'{family}_{name}'] = cfg
        for m_name, cfg in tqdm(flat_models.items(), desc='Models'):
            model, tokenizer = (cfg['model'], cfg['tokenizer'])
            model.to(self.device)
            model.eval()
            for lang, fact_list in [('en', facts_en), ('zh', facts_zh)]:
                texts = [f['prompt'] for f in fact_list]
                layer_means = self._extract_layer_means(model, tokenizer, texts, use_pre_norm)
                for layer_idx, vec in layer_means.items():
                    self.language_vectors.setdefault(layer_idx, {})
                    self.language_vectors[layer_idx].setdefault(m_name, {})
                    self.language_vectors[layer_idx][m_name][lang] = vec
            print('Targeted vectors computed.')

    def run_language_steering_cloze(self, model_configs, facts_en, facts_zh, layers=None, scale=1.0, use_pre_norm=True):
        import torch.nn.functional as F
        if layers is None:
            layers = self.layers
        self.cloze_steering_results = {}
        for model_name, cfg in model_configs.items():
            model, tokenizer = (cfg['model'], cfg['tokenizer'])
            model.to(self.device)
            model_results = []
            for prompt_lang, facts in [('en', facts_en), ('zh', facts_zh)]:
                for steer_lang in ['en', 'zh']:
                    for layer_idx in layers:
                        lang_vec = self._get_language_direction(layer_idx, model_name, steer_lang)
                        for fact in facts:
                            prompt, target = (fact['prompt'], fact['target'])
                            logits_base, logits_steered, mask_idx = self._run_mlm_single(model, tokenizer, prompt, lang_vec, layer_idx, scale, use_pre_norm)
                            if logits_base is None:
                                continue
                            target_ids = tokenizer.encode(target, add_special_tokens=False)
                            if len(target_ids) == 0:
                                continue
                            tid = target_ids[0]
                            probs_base = F.softmax(logits_base, dim=-1)
                            probs_steered = F.softmax(logits_steered, dim=-1) if logits_steered is not None else probs_base
                            delta_p = probs_steered[tid].item() - probs_base[tid].item()
                            delta_logp = probs_steered[tid].log().item() - probs_base[tid].log().item()
                            _, sorted_idx = torch.sort(probs_base, descending=True)
                            rank_base = (sorted_idx == tid).nonzero(as_tuple=False)[0, 0].item() + 1
                            _, sorted_idx_s = torch.sort(probs_steered, descending=True)
                            rank_steered = (sorted_idx_s == tid).nonzero(as_tuple=False)[0, 0].item() + 1
                            model_results.append({'model': model_name, 'prompt_lang': prompt_lang, 'steer_lang': steer_lang, 'layer': layer_idx, 'delta_p': delta_p, 'delta_logp': delta_logp, 'rank_base': rank_base, 'rank_steered': rank_steered})
            self.cloze_steering_results[model_name] = model_results
            print(f'Completed cloze for {model_name}: {len(model_results)} samples')

    def _run_mlm_single(self, model, tokenizer, prompt, lang_vec, layer_idx, scale, use_pre_norm):
        model.eval()
        inputs = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True).to(self.device)
        mask_idx = self._get_mask_index(inputs['input_ids'], tokenizer)
        if mask_idx is None:
            return (None, None, None)
        with torch.no_grad():
            out_base = model(**inputs, output_hidden_states=True)
        logits_base = out_base.logits[0, mask_idx, :]
        logits_steered = None
        if lang_vec is not None:
            lang_vec = lang_vec.to(self.device)
            handle = None

            def hook_fn(module, input, output=None):
                if use_pre_norm:
                    return (input[0] + scale * lang_vec.unsqueeze(0).unsqueeze(0),)
                else:
                    if isinstance(output, tuple):
                        return (output[0] + scale * lang_vec.unsqueeze(0).unsqueeze(0),) + output[1:]
                    return output + scale * lang_vec.unsqueeze(0).unsqueeze(0)
            if use_pre_norm:
                if hasattr(model, 'bert'):
                    target = model.bert.encoder.layer[layer_idx].output.LayerNorm
                else:
                    target = model.encoder.layer[layer_idx].output.LayerNorm
                handle = target.register_forward_pre_hook(hook_fn)
            else:
                if hasattr(model, 'bert'):
                    target = model.bert.encoder.layer[layer_idx]
                else:
                    target = model.encoder.layer[layer_idx]
                handle = target.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
            with torch.no_grad():
                out_steered = model(**inputs, output_hidden_states=False)
            handle.remove()
            logits_steered = out_steered.logits[0, mask_idx, :]
        return (logits_base, logits_steered, mask_idx)

    def generate_capital_split_from_data(self, facts_en, facts_zh, split_ratio=0.5):
        if len(facts_en) != len(facts_zh):
            min_len = min(len(facts_en), len(facts_zh))
            facts_en = facts_en[:min_len]
            facts_zh = facts_zh[:min_len]
        split_idx = int(len(facts_en) * split_ratio)
        train_en, train_zh = (facts_en[:split_idx], facts_zh[:split_idx])
        test_en, test_zh = (facts_en[split_idx:], facts_zh[split_idx:])
        print(f'Data Split: {len(train_en)} Train, {len(test_en)} Test.')
        return ((train_en, train_zh), (test_en, test_zh))

    def summarize_language_steering_cloze(self, family_name: str, subject: str, target: str, save_dir: str='steering_vectors'):
        rows = []
        for m, res in self.cloze_steering_results.items():
            rows.extend(res)
        if not rows:
            return (pd.DataFrame(), pd.DataFrame())
        df = pd.DataFrame(rows)
        df['delta_rank'] = df['rank_steered'] - df['rank_base']
        summary = df.groupby(['model', 'prompt_lang', 'steer_lang', 'layer'], as_index=False).agg(mean_delta_p=('delta_p', 'mean'), mean_delta_rank=('delta_rank', 'mean'))
        os.makedirs(save_dir, exist_ok=True)
        out_dir = os.path.join(save_dir, f'subject_{subject}', family_name)
        os.makedirs(out_dir, exist_ok=True)
        safe_target = str(target).replace(' ', '_')
        df_path = os.path.join(out_dir, f'steering_cloze_{family_name}_{subject}_{safe_target}_full.pkl')
        summary_path = os.path.join(out_dir, f'steering_cloze_{family_name}_{subject}_{safe_target}_summary.pkl')
        with open(df_path, 'wb') as f:
            pickle.dump(df, f)
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        print(f'[Steering] Saved cloze steering full results to {df_path}')
        print(f'[Steering] Saved cloze steering summary to {summary_path}')
        return (df, summary)

    def plot_best_layer_bar(self, summary_df, model_names, metric='mean_delta_p', title='Steering Effectiveness (Best Layer per Model)'):
        df = summary_df[summary_df['model'].isin(model_names)].copy()
        if df.empty:
            print('No data to plot.')
            return
        if 'rank' in metric:
            idx = df.groupby(['model', 'prompt_lang', 'steer_lang'])[metric].idxmin()
        else:
            idx = df.groupby(['model', 'prompt_lang', 'steer_lang'])[metric].idxmax()
        best_df = df.loc[idx].copy()
        print(f'\n--- Best Layers Selected for {title} ---')
        print(best_df[['model', 'prompt_lang', 'steer_lang', 'layer', metric]].to_string(index=False))
        print('-' * 60)
        best_df['Prompt Language'] = best_df['prompt_lang'].map({'en': 'English Prompt', 'zh': 'Chinese Prompt'})
        best_df['Steering Vector'] = best_df['steer_lang'].map({'en': 'English Vector', 'zh': 'Chinese Vector'})
        g = sns.catplot(data=best_df, kind='bar', y='model', x=metric, hue='Steering Vector', col='Prompt Language', palette={'English Vector': '#1f77b4', 'Chinese Vector': '#ff7f0e'}, height=6, aspect=1.2, edgecolor='black', linewidth=1, sharey=True, legend=False)
        metric_label = {'mean_delta_p': 'Max Δ Probability (Best Layer)', 'mean_delta_logp': 'Max Δ LogProb (Best Layer)', 'mean_delta_rank': 'Best Δ Rank improvement'}.get(metric, metric)
        g.set_axis_labels(metric_label, '')
        g.set_titles('{col_name}')
        for ax in g.axes.flat:
            ax.axvline(0, color='black', linewidth=1)
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            x_min, x_max = ax.get_xlim()
            padding = (x_max - x_min) * 0.02
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Steering Vector')
        plt.subplots_adjust(top=0.85, right=0.85)
        g.fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.show()
        return best_df

    def plot_layerwise_mechanism_grid(self, summary_df, model_family_name, model_names, metric='mean_delta_p', title=None):
        df = summary_df[summary_df['model'].isin(model_names)].copy()
        if df.empty:
            return
        df['condition'] = 'Prompt ' + df['prompt_lang'].str.upper() + ' / Steer ' + df['steer_lang'].str.upper()
        palette = {'Prompt EN / Steer EN': '#1f77b4', 'Prompt EN / Steer ZH': '#aec7e8', 'Prompt ZH / Steer EN': '#d62728', 'Prompt ZH / Steer ZH': '#ff9896'}
        dashes = {'Prompt EN / Steer EN': '', 'Prompt EN / Steer ZH': (2, 2), 'Prompt ZH / Steer EN': '', 'Prompt ZH / Steer ZH': (2, 2)}
        num_models = len(model_names)
        rows = (num_models + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, model in enumerate(model_names):
            ax = axes[i]
            sns.lineplot(data=df[df['model'] == model], x='layer', y=metric, hue='condition', style='condition', palette=palette, dashes=dashes, markers=True, ax=ax, linewidth=2.5)
            ax.set_title(model)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            if i > 0:
                ax.legend().remove()
        plt.suptitle(title if title else f'{model_family_name}: Layer-wise Steering')
        plt.tight_layout()
        plt.show()

    def print_layerwise_statistics(self, summary_df, model_names, metric='mean_delta_p'):
        df = summary_df[summary_df['model'].isin(model_names)].copy()
        if df.empty:
            return
        df['Condition'] = 'P_' + df['prompt_lang'].str.upper() + '_S_' + df['steer_lang'].str.upper()
        pivot = df.pivot_table(index=['model', 'layer'], columns='Condition', values=metric)
        print(f'\nLAYER-WISE STATISTICS: {metric}\n')
        for model in model_names:
            if model in pivot.index.get_level_values(0):
                print(f'--- Model: {model} ---')
                print(pivot.loc[model].to_string(float_format='%.4f'))
                print('\n')

    def plot_bert_layerwise_stability(self, family_name, base_model_name, comparison_models, title=None):
        layers_to_plot = self.layers
        if len(layers_to_plot) > 12:
            layers_to_plot = layers_to_plot[:12]
        fig, axes = plt.subplots(4, 3, figsize=(22, 18), sharey=True)
        axes = axes.flatten()
        base_key = f'{family_name}_{base_model_name}'
        comp_keys = [f'{family_name}_{m}' for m in comparison_models]
        for i, layer in enumerate(layers_to_plot):
            if i >= len(axes):
                break
            ax = axes[i]
            data = []
            for cat, words in self.categories.items():
                for m_name in comparison_models:
                    m_key = f'{family_name}_{m_name}'
                    sims = []
                    for w in words:
                        s = self.results.similarity(layer, base_key, f'{w}_en', m_key, f'{w}_en')
                        if s != 0:
                            sims.append(s)
                    if sims:
                        data.append({'Topic': cat, 'Model': m_name, 'Similarity': np.mean(sims)})
            if data:
                df = pd.DataFrame(data)
                bar_plot = sns.barplot(data=df, x='Similarity', y='Topic', hue='Model', palette='viridis', ax=ax)
                ax.set_xlim(0.4, 1.02)
                ax.set_title(f'Layer {layer}', fontweight='bold', fontsize=12)
                ax.set_ylabel('')
                ax.set_xlabel('')
                ax.get_legend().remove()
                for container in bar_plot.containers:
                    ax.bar_label(container, fmt='%.2f', label_type='edge', padding=-25, color='white', fontweight='bold', fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=len(comparison_models), title='Model', fontsize=12)
        final_title = title if title else f'{family_name}: Layer-wise Concept Stability by Topic'
        plt.suptitle(final_title, fontsize=18, y=1.05)
        plt.tight_layout()
        plt.show()

    def plot_mbert_grid_analysis(self, family_name, model_names, title=None):
        layers_to_plot = self.layers
        fig, axes = plt.subplots(3, 4, figsize=(24, 15), sharey=False)
        axes = axes.flatten()
        all_concepts = [w for cat in self.categories.values() for w in cat]
        for i, layer in enumerate(layers_to_plot):
            if i >= len(axes):
                break
            ax = axes[i]
            means = []
            labels = []
            for m_name in model_names:
                m_key = f'{family_name}_{m_name}'
                sims = []
                for word in all_concepts:
                    s = self.results.similarity(layer, m_key, f'{word}_en', m_key, f'{word}_zh')
                    if s != 0:
                        sims.append(s)
                if sims:
                    means.append(np.mean(sims))
                    labels.append(m_name)
            if means:
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(labels)]
                bars = ax.bar(labels, means, color=colors, alpha=0.8)
                ax.set_ylim(0.0, 1.0)
                ax.set_title(f'Layer {layer}', fontweight='bold', fontsize=12)
                ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
                ax.set_ylabel('Cosine Similarity', fontsize=10)
                ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
        final_title = title if title else f'{family_name}: Layer-wise Cross-Lingual Alignment (EN vs ZH)'
        plt.suptitle(final_title, fontsize=18, y=1.02)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.25)
        plt.show()

    def plot_mbert_line_analysis(self, family_name, model_names, concept_wise=False, title=None):
        if not concept_wise:
            all_concepts = [w for cat in self.categories.values() for w in cat]
            plt.figure(figsize=(12, 8))
            for m_name in model_names:
                m_key = f'{family_name}_{m_name}'
                layer_scores = []
                for layer in self.layers:
                    sims = []
                    for word in all_concepts:
                        s = self.results.similarity(layer, m_key, f'{word}_en', m_key, f'{word}_zh')
                        if s != 0:
                            sims.append(s)
                    layer_scores.append(np.mean(sims) if sims else 0)
                plt.plot(self.layers, layer_scores, marker='o', linewidth=2.5, label=m_name)
            plt.ylim(0.0, 1.0)
            plt.xlabel('Layer Index', fontsize=14)
            plt.ylabel('Avg Cross-Lingual Similarity (EN vs ZH)', fontsize=14)
            plt.xticks(self.layers)
            final_title = title if title else f'{family_name}: Overall Cross-Lingual Alignment'
            plt.title(final_title, fontsize=16, fontweight='bold')
            plt.legend(title='Model', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            all_concepts = [w for cat in self.categories.values() for w in cat]
            n_concepts = len(all_concepts)
            cols = 4
            rows = (n_concepts + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows), sharey=False, sharex=True)
            axes = axes.flatten()
            plt.subplots_adjust(hspace=0.3, wspace=0.25)
            for i, word in enumerate(all_concepts):
                ax = axes[i]
                for m_name in model_names:
                    m_key = f'{family_name}_{m_name}'
                    layer_scores = []
                    for layer in self.layers:
                        s = self.results.similarity(layer, m_key, f'{word}_en', m_key, f'{word}_zh')
                        layer_scores.append(s)
                    ax.plot(self.layers, layer_scores, marker='.', linewidth=2, label=m_name)
                ax.set_title(word.upper(), fontweight='bold', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0.0, 1.0)
                ax.set_ylabel('Cosine Sim', fontsize=10)
                ax.set_xlabel('Layer Index', fontsize=10)
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(model_names), title='Model', fontsize=12)
            final_title = title if title else f'{family_name}: Concept-wise Cross-Lingual Alignment'
            plt.suptitle(final_title, fontsize=18, y=1.03)
            plt.show()

    def inspect_layer_architecture(self, model_key, model_configs, use_pre_norm=True):
        if model_key not in model_configs:
            print(f'Model {model_key} not found.')
            return
        cfg = model_configs[model_key]
        model = cfg['model']
        print(f'\n{'=' * 80}')
        print(f'ARCHITECTURAL INSPECTION: {model_key}')
        print(f'Mode: {('PRE-NORM (Steer MLP Output)' if use_pre_norm else 'POST-NORM (Steer Block Output)')}')
        print(f'{'=' * 80}')
        try:
            encoder = model.bert.encoder if hasattr(model, 'bert') else model.encoder
        except AttributeError:
            print('Could not find standard BERT encoder location.')
            return
        print(f'Model Type: {type(model).__name__}')
        print(f'Encoder: {type(encoder).__name__}')
        print(f'\n{'Layer':<5} | {'Steering Injection Point (Hook)':<55} | {'Signal Source'}')
        print('-' * 90)
        for layer_idx in self.layers:
            block = encoder.layer[layer_idx]
            if use_pre_norm:
                if hasattr(block, 'output') and hasattr(block.output, 'LayerNorm'):
                    target_module = 'block.output.LayerNorm'
                    hook_type = 'Pre-Hook (Input to Norm)'
                    signal_desc = 'MLP Output + Residual'
                else:
                    target_module = 'UNKNOWN STRUCTURE'
                    hook_type = '???'
                    signal_desc = '???'
            else:
                target_module = f'encoder.layer[{layer_idx}]'
                hook_type = 'Post-Hook (Output of Block)'
                signal_desc = 'Final Layer Output (Post-Norm)'
            print(f'{layer_idx:<5} | {target_module:<30} ({hook_type:<22}) | {signal_desc}')
        print('-' * 90)
        print("NOTE: 'Feature Extraction' for vector calculation uses the SAME point as steering")
        print('      if use_pre_norm=True (via hooks), otherwise it uses standard hidden_states.\n')

    def print_transformer_block_structure(self, model_key, model_configs):
        if model_key not in model_configs:
            print(f'Model {model_key} not found.')
            return
        cfg = model_configs[model_key]
        model = cfg['model']
        print(f'\n{'=' * 60}')
        print(f'BLOCK STRUCTURE INSPECTION: {model_key}')
        print(f'{'=' * 60}')
        try:
            if hasattr(model, 'bert'):
                block = model.bert.encoder.layer[0]
            elif hasattr(model, 'encoder'):
                block = model.encoder.layer[0]
            else:
                print('Could not find standard BERT encoder layer to inspect.')
                return
            print(f'Inspecting: {type(block).__name__} (Layer 0)\n')

            def print_modules(module, indent=0):
                spaces = '  ' * indent
                for name, child in module.named_children():
                    print(f'{spaces}- {name} ({type(child).__name__})')
                    print_modules(child, indent + 1)
            print_modules(block)
        except Exception as e:
            print(f'Error inspecting block: {e}')
        print('-' * 60 + '\n')

    def plot_pos_steering_from_pickle(self, pickle_path: str, pos_dataset_path: str='pos_cloze_dataset.csv', metric: str='delta_p', aggregation: str='best', title_suffix: str=''):
        print(f'Loading results from: {pickle_path}')
        try:
            df_raw_results = pd.read_pickle(pickle_path)
        except Exception as e:
            print(f'Error loading pickle: {e}')
            return
        df_pos_ref = pd.read_csv(pos_dataset_path)
        df_zh_ref = df_pos_ref[df_pos_ref['language'] == 'zh'].reset_index(drop=True)
        n_facts = len(df_zh_ref)
        pos_list = df_zh_ref['pos'].values
        annotated_chunks = []
        df_raw_results['condition_key'] = list(zip(df_raw_results['model'], df_raw_results['prompt_lang'], df_raw_results['steer_lang']))
        unique_groups = df_raw_results['condition_key'].unique()
        for key in unique_groups:
            model, p_lang, s_lang = key
            sub_df = df_raw_results[(df_raw_results['model'] == model) & (df_raw_results['prompt_lang'] == p_lang) & (df_raw_results['steer_lang'] == s_lang)].copy()
            sub_df = sub_df.sort_values('layer')
            num_layers_found = len(sub_df) // n_facts
            if num_layers_found > 0:
                tiled_pos = np.tile(pos_list, num_layers_found)
                if len(tiled_pos) > len(sub_df):
                    tiled_pos = tiled_pos[:len(sub_df)]
                elif len(tiled_pos) < len(sub_df):
                    tiled_pos = np.pad(tiled_pos, (0, len(sub_df) - len(tiled_pos)), constant_values='UNKNOWN')
                sub_df['pos'] = tiled_pos
                annotated_chunks.append(sub_df)
        df_mapped = pd.concat(annotated_chunks)
        df_mapped['condition'] = 'Prompt ' + df_mapped['prompt_lang'].str.upper() + ' / Steer ' + df_mapped['steer_lang'].str.upper()
        df_mapped['delta_rank'] = -df_mapped['delta_rank']
        zh_models = [m for m in df_mapped['model'].unique() if 'zh' in m or 'Chinese' in m]
        if not zh_models:
            zh_models = df_mapped['model'].unique()
        plot_df = df_mapped[df_mapped['model'].isin(zh_models)].copy()
        df_overall = plot_df.copy()
        df_overall['pos'] = 'OVERALL'
        df_combined = pd.concat([plot_df, df_overall])
        layer_means = df_combined.groupby(['model', 'condition', 'layer', 'pos'], as_index=False)[['delta_p', 'delta_rank']].mean()
        if aggregation == 'best':
            final_df = layer_means.groupby(['model', 'condition', 'pos'], as_index=False)[['delta_p', 'delta_rank']].max()
            agg_label = 'Peak (Best Layer)'
        else:
            final_df = layer_means.groupby(['model', 'condition', 'pos'], as_index=False)[['delta_p', 'delta_rank']].mean()
            agg_label = 'Average (All Layers)'
        conditions = ['Prompt EN / Steer EN', 'Prompt EN / Steer ZH', 'Prompt ZH / Steer EN', 'Prompt ZH / Steer ZH']
        custom_order = ['OVERALL', 'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CCONJ']
        actual_order = [o for o in custom_order if o in final_df['pos'].unique()]
        metric_label = 'Probability Increase' if metric == 'delta_p' else 'Rank Improvement'
        palette = 'viridis' if metric == 'delta_p' else 'magma'
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)
        axes = axes.flatten()
        for i, cond in enumerate(conditions):
            ax = axes[i]
            subset = final_df[final_df['condition'] == cond]
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(cond)
                continue
            sns.barplot(data=subset, x='pos', y=metric, hue='model', palette=palette, order=actual_order, ax=ax, edgecolor='black', linewidth=0.5)
            ax.set_title(cond, fontsize=14, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(f'{metric_label}' if i % 2 == 0 else '')
            ax.axhline(0, color='black', linewidth=1)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=15)
            if i == 1:
                ax.legend(loc='upper right', title='Model', fontsize=9, framealpha=0.95)
            elif ax.get_legend():
                ax.get_legend().remove()
        plt.suptitle(f'STEERING IMPACT: {metric_label} ({agg_label}) {title_suffix}', fontsize=18, y=0.98)
        plt.tight_layout()
        plt.show()
