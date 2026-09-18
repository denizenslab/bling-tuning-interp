import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Any, Optional, List
import os
import pickle

class PosLogitLensAnalyzer:

    def __init__(self, model_families: Dict[str, Dict[str, Dict[str, Any]]], device: Optional[torch.device]=None, max_layers: Optional[int]=None):
        self.model_families = model_families
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_layers = max_layers
        self._vocab_lang_masks = {}
        self._decoder_cache = {}

    @staticmethod
    def _token_has_latin(token_str: str) -> bool:
        return any(('A' <= ch <= 'Z' or 'a' <= ch <= 'z' for ch in token_str))

    @staticmethod
    def _token_has_cjk(token_str: str) -> bool:
        for ch in token_str:
            code = ord(ch)
            if 19968 <= code <= 40959 or 13312 <= code <= 19903 or 131072 <= code <= 173791 or (173824 <= code <= 177983) or (177984 <= code <= 178207) or (178208 <= code <= 183983):
                return True
        return False

    def _build_vocab_lang_masks(self, tokenizer, model) -> None:
        key = id(tokenizer)
        if key in self._vocab_lang_masks:
            return
        vocab_size = model.get_output_embeddings().weight.shape[0]
        tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]
        en_mask = np.zeros(vocab_size, dtype=bool)
        zh_mask = np.zeros(vocab_size, dtype=bool)
        for i, tok in enumerate(tokens):
            if tok.startswith('[') and tok.endswith(']'):
                continue
            core = tok.lstrip('#')
            if self._token_has_cjk(core):
                zh_mask[i] = True
            elif self._token_has_latin(core):
                en_mask[i] = True
        en_mask_t = torch.from_numpy(en_mask).to(self.device)
        zh_mask_t = torch.from_numpy(zh_mask).to(self.device)
        self._vocab_lang_masks[key] = (en_mask_t, zh_mask_t)

    def run_pos_logit_lens(self, df: pd.DataFrame, subject: str, prompt_language: Optional[str]=None, max_examples_per_pos: Optional[int]=None, progress: bool=True, use_pre_norm: bool=True) -> pd.DataFrame:
        if prompt_language is not None:
            df = df[df['language'] == prompt_language].copy()
        if max_examples_per_pos is not None:
            df = df.groupby(['language', 'pos'], group_keys=False).apply(lambda g: g.sample(n=min(len(g), max_examples_per_pos), random_state=0)).reset_index(drop=True)
        all_rows: List[Dict[str, Any]] = []
        for family_name, family in self.model_families.items():
            if progress:
                print(f'\n=== Analyzing family: {family_name} [Pre-Norm={use_pre_norm}] ===')
            for model_name, bundle in family.items():
                model = bundle['model'].to(self.device)
                tokenizer = bundle['tokenizer']
                model.eval()
                if progress:
                    print(f'  -> Model: {model_name}')
                self._build_vocab_lang_masks(tokenizer, model)
                en_mask, zh_mask = self._vocab_lang_masks[id(tokenizer)]
                decoder = self._get_decoder(model)
                captured_layers = {}
                handles = []
                if use_pre_norm:

                    def get_hook(layer_idx):

                        def hook(module, input):
                            captured_layers[layer_idx] = input[0].detach()
                        return hook
                    if hasattr(model, 'bert'):
                        layers = model.bert.encoder.layer
                    elif hasattr(model, 'encoder'):
                        layers = model.encoder.layer
                    else:
                        layers = []
                    for i, layer_module in enumerate(layers):
                        target = layer_module.output.LayerNorm
                        handles.append(target.register_forward_pre_hook(get_hook(i)))
                with torch.no_grad():
                    for idx, row in df.iterrows():
                        prompt = row['prompt']
                        pos = row['pos']
                        lang = row['language']
                        enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
                        input_ids = enc['input_ids'].to(self.device)
                        attention_mask = enc['attention_mask'].to(self.device)
                        mask_id = tokenizer.mask_token_id
                        mask_positions = (input_ids == mask_id).nonzero(as_tuple=False)
                        if mask_positions.numel() == 0:
                            continue
                        mask_pos = mask_positions[0, 1].item()
                        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=not use_pre_norm)
                        if use_pre_norm:
                            num_layers = len(captured_layers)
                        else:
                            hidden_states = out.hidden_states
                            num_layers = len(hidden_states) - 1
                        if self.max_layers is not None:
                            num_layers = min(num_layers, self.max_layers)
                        for layer_idx in range(1, num_layers + 1):
                            if use_pre_norm:
                                h_all = captured_layers[layer_idx - 1]
                            else:
                                h_all = hidden_states[layer_idx]
                            h = h_all[0, mask_pos, :].unsqueeze(0)
                            logits = decoder(h)
                            probs = F.softmax(logits[0], dim=-1)
                            p_en = probs[en_mask].sum().item()
                            p_zh = probs[zh_mask].sum().item()
                            denom = p_en + p_zh + 1e-09
                            p_zh_share = p_zh / denom
                            all_rows.append({'model_family': family_name, 'model_name': model_name, 'language': lang, 'pos': pos, 'layer_idx': layer_idx, 'p_en': p_en, 'p_zh': p_zh, 'p_zh_share': p_zh_share})
                for h in handles:
                    h.remove()
        results_df = pd.DataFrame(all_rows)
        self._save_logit_lens_results(results_df, subject, family_name, prompt_language)
        return results_df

    def _save_logit_lens_results(self, results_df: pd.DataFrame, subject: str, family_name: str, prompt_language: str):
        save_dir = f'logitlense_results/subject_{subject}'
        os.makedirs(save_dir, exist_ok=True)
        filename = f'logitlens_{family_name}_{prompt_language}_{subject}.pkl'
        save_path = os.path.join(save_dir, filename)
        save_data = {'results_df': results_df, 'subject': subject, 'family_name': family_name, 'n_rows': len(results_df)}
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f' Saved: {save_path}')

    def load_logit_lens_results(self, subject: str, family_name: str):
        save_dir = f'logitlense_results/subject_{subject}'
        filename = f'logitlens_{family_name}_{subject}.pkl'
        load_path = os.path.join(save_dir, filename)
        if not os.path.exists(load_path):
            print(f'File not found: {load_path}')
            return None
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        print(f' Loaded: {load_path}')
        return data['results_df']

    @staticmethod
    def plot_chinese_routing_by_pos(results_df: pd.DataFrame, family_name: str, model_order: Optional[List[str]]=None, pos_order: Optional[List[str]]=None):
        df = results_df[(results_df['model_family'] == family_name) & (results_df['language'] == 'zh')].copy()
        if df.empty:
            raise ValueError("No rows for given family_name and language == 'zh'.")
        grouped = df.groupby(['model_name', 'pos'], as_index=False)['p_zh_share'].mean().rename(columns={'p_zh_share': 'mean_p_zh_share'})
        if model_order is not None:
            grouped['model_name'] = pd.Categorical(grouped['model_name'], categories=model_order, ordered=True)
        if pos_order is not None:
            grouped['pos'] = pd.Categorical(grouped['pos'], categories=pos_order, ordered=True)
        plt.figure(figsize=(10, 4))
        sns.barplot(data=grouped, x='pos', y='mean_p_zh_share', hue='model_name')
        plt.ylim(0.0, 1.0)
        plt.ylabel('Mean Chinese prob. share (p_zh / (p_en + p_zh))')
        plt.xlabel('Part of speech')
        plt.title(f'Chinese routing by POS (family: {family_name}, prompts in ZH)')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_en_vs_zh_prob_by_pos(results_df: pd.DataFrame, family_name: str, model_order: Optional[List[str]]=None, pos_order: Optional[List[str]]=None):
        df = results_df[(results_df['model_family'] == family_name) & (results_df['language'] == 'zh')].copy()
        if df.empty:
            raise ValueError("No rows for given family_name and language == 'zh'.")
        grouped = df.groupby(['model_name', 'pos'], as_index=False)[['p_en', 'p_zh']].mean()
        melted = grouped.melt(id_vars=['model_name', 'pos'], value_vars=['p_en', 'p_zh'], var_name='prob_type', value_name='mean_prob')
        melted['prob_type'] = melted['prob_type'].map({'p_en': 'English prob.', 'p_zh': 'Chinese prob.'})
        if model_order is not None:
            melted['model_name'] = pd.Categorical(melted['model_name'], categories=model_order, ordered=True)
        if pos_order is not None:
            melted['pos'] = pd.Categorical(melted['pos'], categories=pos_order, ordered=True)
        plt.figure(figsize=(10, 4))
        sns.barplot(data=melted, x='pos', y='mean_prob', hue='prob_type')
        plt.ylabel('Mean probability mass')
        plt.xlabel('Part of speech')
        plt.title(f'English vs Chinese probability by POS\n(family: {family_name}, prompts in ZH)')
        plt.legend(title='')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_layerwise_pos_grid(results_df: pd.DataFrame, family_name: str, language: str='zh', stat: str='share', which_share: str='zh', base_model_name: str='Base', pos_order: Optional[List[str]]=None, max_cols: int=3):
        sns.set_style('whitegrid')
        df = results_df[(results_df['model_family'] == family_name) & (results_df['language'] == language)].copy()
        if df.empty:
            raise ValueError(f'No rows for family={family_name}, language={language}.')
        denom = df['p_en'] + df['p_zh'] + 1e-09
        df['p_en_share'] = df['p_en'] / denom
        df['p_zh_share'] = df['p_zh'] / denom
        if which_share == 'zh':
            share_col = 'p_zh_share'
            share_label = 'Chinese prob. share'
        elif which_share == 'en':
            share_col = 'p_en_share'
            share_label = 'English prob. share'
        else:
            raise ValueError("which_share must be 'en' or 'zh'.")
        agg = df.groupby(['model_name', 'pos', 'layer_idx'], as_index=False)[[share_col]].mean()
        if stat == 'delta_vs_base':
            base = agg[agg['model_name'] == base_model_name].rename(columns={share_col: 'base_share'})[['pos', 'layer_idx', 'base_share']]
            merged = agg.merge(base, on=['pos', 'layer_idx'], how='left')
            merged['value'] = merged[share_col] - merged['base_share']
            y_label = f'Δ {share_label} vs {base_model_name}'
        elif stat == 'share':
            merged = agg.copy()
            merged['value'] = merged[share_col]
            y_label = f'Mean {share_label}'
        else:
            raise ValueError("stat must be 'share' or 'delta_vs_base'.")
        if pos_order is None:
            pos_list = sorted(merged['pos'].unique())
        else:
            pos_list = pos_order
        n_pos = len(pos_list)
        n_cols = min(max_cols, n_pos)
        n_rows = int(np.ceil(n_pos / n_cols))
        model_names = sorted(merged['model_name'].unique())
        palette = sns.color_palette('tab10', n_colors=len(model_names))
        linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
        style_map = {m: {'color': palette[i % len(palette)], 'linestyle': linestyles[i % len(linestyles)]} for i, m in enumerate(model_names)}
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), sharey=True, sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.reshape(axes, (n_rows, n_cols))
        min_layer = int(merged['layer_idx'].min())
        max_layer = int(merged['layer_idx'].max())
        x_ticks = list(range(min_layer, max_layer + 1))
        y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i, pos in enumerate(pos_list):
            r = i // n_cols
            c = i % n_cols
            ax = axes[r, c]
            sub = merged[merged['pos'] == pos]
            if sub.empty:
                ax.axis('off')
                continue
            for m_name, sub_m in sub.groupby('model_name'):
                style = style_map[m_name]
                ax.plot(sub_m['layer_idx'], sub_m['value'], marker='o', markersize=3, linewidth=1.8, label=m_name, **style)
            ax.set_title(pos, fontsize=11)
            ax.set_xlim(min_layer, max_layer)
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_xlabel('Layer', fontsize=9)
            ax.set_ylabel(y_label, fontsize=9)
        for j in range(n_pos, n_rows * n_cols):
            r = j // n_cols
            c = j % n_cols
            axes[r, c].axis('off')
        first_ax = axes.flatten()[0]
        handles, labels = first_ax.get_legend_handles_labels()
        fig.suptitle(f'Layer-wise Chinese routing by POS\nfamily={family_name}, prompts in {language}, stat={stat}', fontsize=13, y=0.93)
        fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.9))
        plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.88])
        plt.show()

    @staticmethod
    def plot_english_routing_by_pos(results_df: pd.DataFrame, family_name: str, language: str='en', which_share: str='en', model_order: Optional[List[str]]=None, pos_order: Optional[List[str]]=None):
        df = results_df[(results_df['model_family'] == family_name) & (results_df['language'] == language)].copy()
        if df.empty:
            raise ValueError(f'No rows for family={family_name}, language={language}.')
        denom = df['p_en'] + df['p_zh'] + 1e-09
        df['p_en_share'] = df['p_en'] / denom
        df['p_zh_share'] = df['p_zh'] / denom
        if which_share == 'en':
            value_col = 'p_en_share'
            y_label = 'Mean English prob. share (p_en / (p_en + p_zh))'
            title_lang = 'English routing'
        elif which_share == 'zh':
            value_col = 'p_zh_share'
            y_label = 'Mean Chinese prob. share (p_zh / (p_en + p_zh))'
            title_lang = 'Chinese routing'
        else:
            raise ValueError("which_share must be 'en' or 'zh'.")
        grouped = df.groupby(['model_name', 'pos'], as_index=False)[value_col].mean().rename(columns={value_col: 'mean_share'})
        if model_order is not None:
            grouped['model_name'] = pd.Categorical(grouped['model_name'], categories=model_order, ordered=True)
        if pos_order is not None:
            grouped['pos'] = pd.Categorical(grouped['pos'], categories=pos_order, ordered=True)
        plt.figure(figsize=(10, 4))
        sns.barplot(data=grouped, x='pos', y='mean_share', hue='model_name')
        plt.ylim(0.0, 1.0)
        plt.ylabel(y_label)
        plt.xlabel('Part of speech')
        plt.title(f'{title_lang} by POS (family: {family_name}, prompts in {language})')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_en_vs_zh_prob_by_pos_for_en_prompts(results_df: pd.DataFrame, family_name: str, language: str='en', model_order: Optional[List[str]]=None, pos_order: Optional[List[str]]=None):
        df = results_df[(results_df['model_family'] == family_name) & (results_df['language'] == language)].copy()
        if df.empty:
            raise ValueError(f'No rows for family={family_name}, language={language}.')
        grouped = df.groupby(['model_name', 'pos'], as_index=False)[['p_en', 'p_zh']].mean()
        melted = grouped.melt(id_vars=['model_name', 'pos'], value_vars=['p_en', 'p_zh'], var_name='prob_type', value_name='mean_prob')
        melted['prob_type'] = melted['prob_type'].map({'p_en': 'English prob.', 'p_zh': 'Chinese prob.'})
        if model_order is not None:
            melted['model_name'] = pd.Categorical(melted['model_name'], categories=model_order, ordered=True)
        if pos_order is not None:
            melted['pos'] = pd.Categorical(melted['pos'], categories=pos_order, ordered=True)
        plt.figure(figsize=(10, 4))
        sns.barplot(data=melted, x='pos', y='mean_prob', hue='prob_type')
        plt.ylabel('Mean probability mass')
        plt.xlabel('Part of speech')
        plt.title(f'English vs Chinese probability by POS\n(family: {family_name}, prompts in {language})')
        plt.legend(title='')
        plt.tight_layout()
        plt.show()

    def _get_decoder(self, model):
        if hasattr(model, 'cls') and hasattr(model.cls, 'predictions'):

            def decoder(hidden):
                return model.cls.predictions(hidden)
            return decoder
        out_emb = model.get_output_embeddings()
        if out_emb is None:
            raise ValueError('Model has no output embeddings / MLM head.')

        def decoder(hidden):
            return out_emb(hidden)
        return decoder

    def run_pos_target_prob_lens(self, dataset_df: pd.DataFrame, family_name: str, subject: str, language: str='zh', batch_size: int=16, max_examples_per_pos=None, include_embedding_layer: bool=False, random_state: int=0, use_pre_norm: bool=True, save: bool=True, dynamic_target_masking: bool=True, mask_expand_languages: tuple=('zh',), require_adjacent_two_masks: bool=True) -> pd.DataFrame:
        family_dict = self.model_families[family_name]
        results = []
        subset = dataset_df[dataset_df['language'] == language].copy()
        if subset.empty:
            print(f'Warning: No data found for language={language}')
            return pd.DataFrame()
        target_col = self._targets_for_language(subset, language)
        if max_examples_per_pos is not None:
            subset = subset.groupby(['language', 'pos'], group_keys=False).apply(lambda g: g.sample(n=min(len(g), max_examples_per_pos), random_state=random_state)).reset_index(drop=True)
        for pos_tag, pos_group in subset.groupby('pos'):
            prompts = pos_group['prompt'].tolist()
            targets = pos_group[target_col].tolist()
            for model_name, model_info in family_dict.items():
                model = model_info['model'].to(self.device)
                tokenizer = model_info['tokenizer']
                model.eval()
                decoder = self.get_decoder(model)
                mask_tok = tokenizer.mask_token
                mask_id = tokenizer.mask_token_id
                if mask_tok is None or mask_id is None:
                    raise ValueError('Tokenizer missing mask_token/mask_token_id.')
                valid_items = []
                for p, t in zip(prompts, targets):
                    target_ids = tokenizer.encode(str(t), add_special_tokens=False)
                    if len(target_ids) == 0:
                        continue
                    if not dynamic_target_masking:
                        valid_items.append(dict(prompt=p, kind='one', t1=target_ids[0], t2=None, was_expanded=False))
                        continue
                    if len(target_ids) == 1:
                        valid_items.append(dict(prompt=p, kind='one', t1=target_ids[0], t2=None, was_expanded=False))
                    elif len(target_ids) == 2:
                        p2 = p
                        was_expanded = False
                        if language in set(mask_expand_languages) and p2.count(mask_tok) == 1:
                            p2 = p2.replace(mask_tok, f'{mask_tok} {mask_tok}', 1)
                            was_expanded = True
                        valid_items.append(dict(prompt=p2, kind='two', t1=target_ids[0], t2=target_ids[1], was_expanded=was_expanded))
                    else:
                        continue
                if len(valid_items) == 0:
                    continue
                captured_layers = {}
                handles = []
                layers = None
                if use_pre_norm:

                    def get_hook(layer_idx):

                        def hook(module, inputs):
                            captured_layers[layer_idx] = inputs[0].detach()
                        return hook
                    if hasattr(model, 'bert'):
                        layers = model.bert.encoder.layer
                    elif hasattr(model, 'encoder'):
                        layers = model.encoder.layer
                    if layers is None:
                        raise ValueError('Model does not expose encoder layers for pre-norm hooks.')
                    for li, layer_module in enumerate(layers):
                        handles.append(layer_module.output.LayerNorm.register_forward_pre_hook(get_hook(li)))
                try:
                    for kind in ('one', 'two'):
                        kind_items = [x for x in valid_items if x['kind'] == kind]
                        if not kind_items:
                            continue
                        for i in range(0, len(kind_items), batch_size):
                            batch = kind_items[i:i + batch_size]
                            batch_prompts = [x['prompt'] for x in batch]
                            B_raw = len(batch)
                            if kind == 'one':
                                batch_t1 = torch.tensor([x['t1'] for x in batch], device=self.device, dtype=torch.long)
                            else:
                                batch_t1 = torch.tensor([x['t1'] for x in batch], device=self.device, dtype=torch.long)
                                batch_t2 = torch.tensor([x['t2'] for x in batch], device=self.device, dtype=torch.long)
                            inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True).to(self.device)
                            input_ids = inputs.input_ids
                            attn = inputs.attention_mask
                            B, S = input_ids.shape
                            mask_counts = (input_ids == mask_id).sum(dim=1)
                            expected_masks = 1 if kind == 'one' else 2
                            keep = mask_counts == expected_masks
                            if keep.sum().item() == 0:
                                continue
                            if not bool(keep.all()):
                                input_ids = input_ids[keep]
                                attn = attn[keep]
                                batch_t1 = batch_t1[keep]
                                if kind == 'two':
                                    batch_t2 = batch_t2[keep]
                                B = input_ids.shape[0]
                            if kind == 'one':
                                mask_pos = (input_ids == mask_id).int().argmax(dim=1)
                            else:
                                nz = (input_ids == mask_id).nonzero(as_tuple=False)
                                mask_cols = nz[:, 1].view(B, 2)
                                mask_pos_a = mask_cols[:, 0]
                                mask_pos_b = mask_cols[:, 1]
                                if require_adjacent_two_masks:
                                    adj = mask_pos_b == mask_pos_a + 1
                                    if adj.sum().item() == 0:
                                        continue
                                    if not bool(adj.all()):
                                        input_ids = input_ids[adj]
                                        attn = attn[adj]
                                        batch_t1 = batch_t1[adj]
                                        batch_t2 = batch_t2[adj]
                                        mask_pos_a = mask_pos_a[adj]
                                        mask_pos_b = mask_pos_b[adj]
                                        B = input_ids.shape[0]
                            with torch.no_grad():
                                out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=not use_pre_norm)

                                def score_layer(h_all, layer_idx_for_df: int):
                                    if kind == 'one':
                                        vec = h_all[torch.arange(B, device=self.device), mask_pos]
                                        logits = decoder(vec)
                                        probs = torch.softmax(logits, dim=-1)
                                        tp = probs.gather(1, batch_t1.unsqueeze(1)).squeeze(1)
                                        results.append({'family_name': family_name, 'model_name': model_name, 'language': language, 'pos': pos_tag, 'layer': int(layer_idx_for_df), 'target_prob': float(tp.mean().item()), 'n_examples': int(B), 'n_multitoken_targets': 0, 'target_token_strategy': 'dynamic:one_mask', 'use_pre_norm': bool(use_pre_norm), **({'include_embedding_layer': bool(include_embedding_layer)} if not use_pre_norm else {})})
                                    else:
                                        vec_a = h_all[torch.arange(B, device=self.device), mask_pos_a]
                                        vec_b = h_all[torch.arange(B, device=self.device), mask_pos_b]
                                        logits_a = decoder(vec_a)
                                        logits_b = decoder(vec_b)
                                        logp_a = torch.log_softmax(logits_a, dim=-1)
                                        logp_b = torch.log_softmax(logits_b, dim=-1)
                                        lp1 = logp_a.gather(1, batch_t1.unsqueeze(1)).squeeze(1)
                                        lp2 = logp_b.gather(1, batch_t2.unsqueeze(1)).squeeze(1)
                                        mean_lp = 0.5 * (lp1 + lp2)
                                        geom_mean_prob = mean_lp.exp()
                                        results.append({'family_name': family_name, 'model_name': model_name, 'language': language, 'pos': pos_tag, 'layer': int(layer_idx_for_df), 'target_prob': float(geom_mean_prob.mean().item()), 'target_logprob': float(mean_lp.mean().item()), 'n_examples': int(B), 'n_multitoken_targets': int(B), 'target_token_strategy': 'dynamic:two_mask_joint', 'use_pre_norm': bool(use_pre_norm), 'mask_expand_languages': ','.join(mask_expand_languages), 'require_adjacent_two_masks': bool(require_adjacent_two_masks), **({'include_embedding_layer': bool(include_embedding_layer)} if not use_pre_norm else {})})
                                if use_pre_norm:
                                    num_layers = len(captured_layers)
                                    if num_layers == 0:
                                        continue
                                    for li in range(1, num_layers + 1):
                                        score_layer(captured_layers[li - 1], layer_idx_for_df=li)
                                else:
                                    hidden_states = out.hidden_states
                                    start_layer = 0 if include_embedding_layer else 1
                                    for li in range(start_layer, len(hidden_states)):
                                        score_layer(hidden_states[li], layer_idx_for_df=li)
                finally:
                    for h in handles:
                        h.remove()
        results_df = pd.DataFrame(results)
        if save:
            self._save_pos_target_prob_lens_results(results_df=results_df, subject=subject, family_name=family_name, prompt_language=language, use_pre_norm=use_pre_norm, include_embedding_layer=include_embedding_layer, target_token_strategy='dynamic' if dynamic_target_masking else 'first', batch_size=batch_size, max_examples_per_pos=max_examples_per_pos, random_state=random_state)
        return results_df

    def _save_pos_target_prob_lens_results(self, results_df: pd.DataFrame, subject: str, family_name: str, prompt_language: str, use_pre_norm: bool, include_embedding_layer: bool, target_token_strategy: str, batch_size: Optional[int], max_examples_per_pos: Optional[int], random_state: int):
        save_dir = f'logitlense_results/subject_{subject}'
        os.makedirs(save_dir, exist_ok=True)
        filename = f'logitlens_{family_name}_{prompt_language}_{subject}.pkl'
        save_path = os.path.join(save_dir, filename)
        save_data = {'results_df': results_df, 'subject': subject, 'family_name': family_name, 'prompt_language': prompt_language, 'use_pre_norm': use_pre_norm, 'include_embedding_layer': include_embedding_layer, 'target_token_strategy': target_token_strategy, 'batch_size': batch_size, 'max_examples_per_pos': max_examples_per_pos, 'random_state': random_state, 'n_rows': len(results_df)}
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f' Saved: {save_path}')
        return save_path

    def get_decoder(self, model):
        if hasattr(model, 'cls') and hasattr(model.cls, 'predictions'):

            def decoder(hidden):
                return model.cls.predictions(hidden)
            return decoder
        out_emb = model.get_output_embeddings()
        if out_emb is None:
            raise ValueError('Model has no output embeddings / MLM head.')

        def decoder(hidden):
            return out_emb(hidden)
        return decoder

    @staticmethod
    def _targets_for_language(df: pd.DataFrame, language: str):
        if language == 'en':
            return 'target_en' if 'target_en' in df.columns else 'target'
        if language == 'zh':
            return 'target_zh' if 'target_zh' in df.columns else 'target'
        raise ValueError("language must be 'en' or 'zh'")

    def _model_palette(self):
        return {'Base': '#1f77b4', 'Whole': '#ff7f0e', 'Semantic': '#2ca02c', 'Language': '#d62728'}

    def _model_order(self):
        return ['Base', 'Whole', 'Semantic', 'Language']

    def plot_target_prob_change(self, results_df: pd.DataFrame, title: str='Average Target Probability by POS'):
        if results_df.empty:
            print('No results to plot.')
            return
        aggregated = results_df.groupby(['model_name', 'pos'], as_index=False)['target_prob'].mean()
        model_order = [m for m in self._model_order() if m in set(aggregated['model_name'])]
        palette = self._model_palette()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=aggregated, x='pos', y='target_prob', hue='model_name', hue_order=model_order, palette=palette)
        plt.title(title)
        plt.ylabel('Avg Probability of Correct Token (All Layers)')
        plt.xlabel('Part of Speech')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()

    def plot_target_prob_bar_at_layer(self, results_df: pd.DataFrame, layer: int, title=None):
        if results_df.empty:
            print('No results to plot.')
            return
        df = results_df[results_df['layer'] == layer].copy()
        if df.empty:
            print(f'No rows for layer={layer}.')
            return
        aggregated = df.groupby(['model_name', 'pos'], as_index=False)['target_prob'].mean()
        model_order = [m for m in self._model_order() if m in set(aggregated['model_name'])]
        palette = self._model_palette()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=aggregated, x='pos', y='target_prob', hue='model_name', hue_order=model_order, palette=palette)
        plt.title(title or f'Target Probability by POS at Layer {layer}')
        plt.ylabel('P(target)')
        plt.xlabel('Part of Speech')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()

    def plot_target_prob_layers_grid(self, results_df: pd.DataFrame, title: str='Target Probability Across Layers (2x2)', pos_order=None, layer_order=None, model_order=None):
        if results_df.empty:
            print('No results to plot.')
            return
        df = results_df.copy()
        df = df.groupby(['model_name', 'pos', 'layer'], as_index=False)['target_prob'].mean()
        if model_order is None:
            model_order = ['Base', 'Whole', 'Semantic', 'Language']
        present_models = [m for m in model_order if m in set(df['model_name'])]
        if not present_models:
            present_models = sorted(df['model_name'].unique())
        if pos_order is None:
            pos_order = sorted(df['pos'].unique())
        df['pos'] = pd.Categorical(df['pos'], categories=pos_order, ordered=True)
        if layer_order is None:
            layer_order = sorted(df['layer'].unique())
        layer_order = list(layer_order)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i >= len(present_models):
                ax.axis('off')
                continue
            mname = present_models[i]
            sub = df[df['model_name'] == mname].copy()
            if sub.empty:
                ax.axis('off')
                continue
            for pos_tag in pos_order:
                subpos = sub[sub['pos'] == pos_tag].sort_values('layer')
                if subpos.empty:
                    continue
                ax.plot(subpos['layer'], subpos['target_prob'], marker='o', linewidth=1.6, markersize=3, label=str(pos_tag))
            ax.set_title(mname)
            ax.set_xlabel('Layer')
            ax.set_ylabel('P(target-first-subtoken)')
            ax.set_xticks(layer_order)
            ax.grid(True, alpha=0.3)
        handles, labels = (None, None)
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            if len(h) > 0:
                handles, labels = (h, l)
                break
        if handles is not None:
            fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 6), frameon=False)
        fig.suptitle(title, y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_target_prob_delta_grid(results_df: pd.DataFrame, family_name: str, language: str='en', base_model_name: str='Base', pos_order: Optional[List[str]]=None, max_cols: int=3):
        sns.set_style('whitegrid')
        df = results_df[(results_df['family_name'] == family_name) & (results_df['language'] == language)].copy()
        if df.empty:
            raise ValueError(f'No rows for family={family_name}, language={language}.')
        agg = df.groupby(['model_name', 'pos', 'layer'], as_index=False)['target_prob'].mean()
        base_df = agg[agg['model_name'] == base_model_name].rename(columns={'target_prob': 'base_prob'})[['pos', 'layer', 'base_prob']]
        if base_df.empty:
            print(f"Warning: Base model '{base_model_name}' not found for delta calculation.")
        merged = agg.merge(base_df, on=['pos', 'layer'], how='left')
        merged['base_prob'] = merged['base_prob'].fillna(0)
        merged['value'] = merged['target_prob'] - merged['base_prob']
        y_label = f'Δ Target Prob. vs {base_model_name}'
        CUSTOM_STYLE_MAP = {'Base': {'color': 'tab:blue', 'linestyle': '-', 'marker': 'o', 'markersize': 4}, 'Language': {'color': 'tab:red', 'linestyle': '-', 'marker': '*', 'markersize': 6}, 'Semantic': {'color': 'tab:green', 'linestyle': ':', 'marker': 'd', 'markersize': 4}, 'Whole': {'color': 'tab:orange', 'linestyle': '--', 'marker': 's', 'markersize': 4}}
        DEFAULT_STYLE = {'color': 'gray', 'linestyle': '-', 'marker': '.', 'markersize': 3}
        if pos_order is None:
            pos_list = sorted(merged['pos'].unique())
        else:
            pos_list = pos_order
        n_pos = len(pos_list)
        n_cols = min(max_cols, n_pos)
        n_rows = int(np.ceil(n_pos / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), sharey=True, sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.reshape(axes, (n_rows, n_cols))
        min_layer = int(merged['layer'].min())
        max_layer = int(merged['layer'].max())
        x_ticks = list(range(min_layer, max_layer + 1))
        model_order = ['Base', 'Language', 'Semantic', 'Whole']
        existing_models = [m for m in model_order if m in merged['model_name'].unique()]
        others = [m for m in merged['model_name'].unique() if m not in model_order]
        plot_order = existing_models + sorted(others)
        for i, pos in enumerate(pos_list):
            r = i // n_cols
            c = i % n_cols
            ax = axes[r, c]
            sub = merged[merged['pos'] == pos]
            if sub.empty:
                ax.axis('off')
                continue
            for m_name in plot_order:
                sub_m = sub[sub['model_name'] == m_name]
                if sub_m.empty:
                    continue
                style = CUSTOM_STYLE_MAP.get(m_name, DEFAULT_STYLE)
                ax.plot(sub_m['layer'], sub_m['value'], linewidth=1.5, label=m_name, **style)
            ax.set_title(pos, fontsize=11)
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.4, linestyle='-')
            ax.set_xlim(min_layer, max_layer)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x_ticks)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_xlabel('Layer', fontsize=9)
            ax.set_ylabel(y_label, fontsize=9)
        for j in range(n_pos, n_rows * n_cols):
            axes[j // n_cols, j % n_cols].axis('off')
        handles, labels = axes.flatten()[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ordered_handles = [by_label[m] for m in plot_order if m in by_label]
        ordered_labels = [m for m in plot_order if m in by_label]
        fig.suptitle(f'Layer-wise Target Prob Change by POS\nfamily={family_name}, language={language}, stat=delta_vs_base', fontsize=13, y=0.95)
        fig.legend(ordered_handles, ordered_labels, loc='upper center', ncol=len(ordered_labels), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.92))
        plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.88])
        plt.show()
