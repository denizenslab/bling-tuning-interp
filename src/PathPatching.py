import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle
import random

class HeadAttributionAnalyzer:

    def __init__(self, device='cuda', seed=42):
        self.device = device
        self.seed = seed

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _get_subject_indices(self, input_ids, subject_tokens, tokenizer):
        input_list = input_ids[0].tolist()
        subject_list = subject_tokens[0].tolist()
        if subject_list and subject_list[0] == tokenizer.bos_token_id:
            subject_list = subject_list[1:]
        if subject_list and subject_list[-1] == tokenizer.eos_token_id:
            subject_list = subject_list[:-1]
        len_subj = len(subject_list)
        if len_subj == 0:
            return None
        for i in range(len(input_list) - len_subj + 1):
            if input_list[i:i + len_subj] == subject_list:
                return range(i, i + len_subj)
        return None

    def prepare_facts(self, facts_list):
        prepared = []
        for f in facts_list:
            f_new = f.copy()
            if 'subject' in f and f['subject']:
                prepared.append(f_new)
                continue
            prompt = f['prompt']
            if '[MASK]' in prompt:
                parts = prompt.split('[MASK]')
                pre_context = parts[0].strip()
                if not pre_context and len(parts) > 1:
                    pre_context = parts[1].strip()
            else:
                pre_context = prompt
            f_new['subject'] = pre_context
            prepared.append(f_new)
        return prepared

    def run_head_attribution(self, model, tokenizer, facts, noise_scale=0.1, top_k=50):
        self._set_seed()
        'Runs patching analysis for a single model.'
        model.to(self.device)
        model.eval()
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size
        head_dim = hidden_size // n_heads
        if hasattr(model, 'bert'):
            layers = model.bert.encoder.layer
        elif hasattr(model, 'roberta'):
            layers = model.roberta.encoder.layer
        else:
            layers = model.encoder.layer
        total_attribution = torch.zeros((n_layers, n_heads)).to(self.device)
        valid_count = 0
        facts_subset = self.prepare_facts(facts[:top_k])
        for fact in tqdm(facts_subset, desc='Path Patching', leave=False):
            prompt = fact['prompt']
            target = fact['target']
            subject = fact['subject']
            inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            mask_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask_indices[0]) == 0:
                continue
            mask_pos = mask_indices[1][0].item()
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids:
                continue
            target_id = target_ids[0]
            subj_tokens = tokenizer(subject, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
            subj_range = self._get_subject_indices(input_ids, subj_tokens, tokenizer)
            if subj_range is None:
                continue
            clean_activations = {}

            def get_clean_hook(layer_idx):

                def hook(module, args, output):
                    ctx = output[0].detach()
                    b, s, _ = ctx.shape
                    clean_activations[layer_idx] = ctx.view(b, s, n_heads, head_dim)
                return hook
            handles = []
            for i, layer in enumerate(layers):
                handles.append(layer.attention.self.register_forward_hook(get_clean_hook(i)))
            try:
                with torch.no_grad():
                    model(**inputs, output_attentions=True)
            finally:
                for h in handles:
                    h.remove()
            emb_module = model.get_input_embeddings()
            clean_emb = emb_module(input_ids)
            noise = torch.randn_like(clean_emb[:, list(subj_range), :]) * noise_scale
            corr_emb = clean_emb.clone()
            corr_emb[:, list(subj_range), :] += noise
            with torch.no_grad():
                out_corr = model(inputs_embeds=corr_emb, attention_mask=attention_mask, output_attentions=True)
                if not hasattr(out_corr, 'logits'):
                    raise AttributeError("Model output missing 'logits'.")
                prob_corr = torch.softmax(out_corr.logits[0, mask_pos], dim=-1)[target_id].item()
            for l in range(n_layers):
                for h in range(n_heads):

                    def patch_hook(module, args, output):
                        ctx = output[0]
                        b, s, _ = ctx.shape
                        ctx_view = ctx.view(b, s, n_heads, head_dim)
                        clean_head = clean_activations[l][:, :, h, :]
                        patched = ctx_view.clone()
                        patched[:, :, h, :] = clean_head
                        return (patched.view(b, s, hidden_size),) + output[1:]
                    handle = layers[l].attention.self.register_forward_hook(patch_hook)
                    try:
                        with torch.no_grad():
                            out_patch = model(inputs_embeds=corr_emb, attention_mask=attention_mask, output_attentions=True)
                            prob_patch = torch.softmax(out_patch.logits[0, mask_pos], dim=-1)[target_id].item()
                            total_attribution[l, h] += prob_patch - prob_corr
                    finally:
                        handle.remove()
            valid_count += 1
        if valid_count == 0:
            return None
        return (total_attribution / valid_count).cpu().numpy()

    def run_family_attribution(self, family_dict, facts, family_name, subject, language, folder='path_patching', top_k=50):
        results = {}
        models_order = ['Base', 'Whole', 'Semantic', 'Language']
        print(f'Running Family Attribution for {family_name} ({language}) on Top {top_k} facts...')
        for name in models_order:
            if name not in family_dict:
                continue
            print(f'  -> Tracing {name}...')
            bundle = family_dict[name]
            matrix = self.run_head_attribution(bundle['model'], bundle['tokenizer'], facts, noise_scale=0.1, top_k=top_k)
            results[name] = matrix
        os.makedirs(folder, exist_ok=True)
        filename = f'{family_name}_Subject-{subject}_{language}.pkl'
        save_path = os.path.join(folder, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f'Saved results to: {save_path}')
        return results

    def plot_family_grid(self, results_dict, title_suffix=''):
        models_order = ['Base', 'Whole', 'Semantic', 'Language']
        valid_mats = [m for m in results_dict.values() if m is not None]
        if not valid_mats:
            print('No valid results to plot.')
            return
        vmin = min((m.min() for m in valid_mats))
        vmax = max((m.max() for m in valid_mats))
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for i, name in enumerate(models_order):
            ax = axes[i]
            if name in results_dict and results_dict[name] is not None:
                sns.heatmap(results_dict[name], ax=ax, cmap='viridis', vmin=vmin, vmax=vmax, cbar_kws={'label': 'Restoration Score'} if i % 2 != 0 else None)
                ax.set_title(f'{name} Model', fontsize=14, fontweight='bold')
                ax.set_xlabel('Head ID')
                ax.set_ylabel('Layer ID')
            else:
                ax.text(0.5, 0.5, 'Model Not Found', ha='center')
                ax.axis('off')
        plt.suptitle(f'Attention Head Attribution: {title_suffix}', fontsize=18, y=0.98)
        plt.tight_layout()
        plt.show()

    def _build_script_indices(self, tokenizer, model):
        vocab_size = model.get_output_embeddings().weight.shape[0]
        en_idx, zh_idx = ([], [])
        for idx in range(vocab_size):
            tok = tokenizer.convert_ids_to_tokens(idx)
            if tok is None:
                continue
            if tok.startswith('[') and tok.endswith(']'):
                continue
            core = tok.lstrip('#')
            has_cjk = any(('一' <= ch <= '鿿' or '㐀' <= ch <= '䶿' for ch in core))
            has_lat = any(('a' <= ch.lower() <= 'z' for ch in core))
            if has_cjk:
                zh_idx.append(idx)
            elif has_lat:
                en_idx.append(idx)
        en_t = torch.tensor(en_idx, device=self.device, dtype=torch.long)
        zh_t = torch.tensor(zh_idx, device=self.device, dtype=torch.long)
        return (en_t, zh_t)

    def run_family_mass_attribution(self, family_dict, facts, family_name, subject, language, folder='path_patching', top_k=50, metric='pzh_share', noise_scale=0.1):
        results = {}
        models_order = ['Base', 'Whole', 'Semantic', 'Language']
        print(f'Running Family Attribution for {family_name} ({language}) metric={metric} top_k={top_k}...')
        for name in models_order:
            if name not in family_dict:
                continue
            print(f'  -> Tracing {name}...')
            bundle = family_dict[name]
            matrix = self.run_head_mass_attribution(bundle['model'], bundle['tokenizer'], facts, noise_scale=noise_scale, top_k=top_k, metric=metric)
            results[name] = matrix
        os.makedirs(folder, exist_ok=True)
        filename = f'{family_name}_Subject-{subject}_{language}_{metric}.pkl'
        save_path = os.path.join(folder, filename)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f'Saved results to: {save_path}')
        return results

    def run_head_mass_attribution(self, model, tokenizer, facts, noise_scale=0.1, top_k=50, metric='pzh_share', eps=1e-09):
        if metric not in ('pzh_share', 'pen_share'):
            raise ValueError("metric must be 'pzh_share' or 'pen_share'.")
        model.to(self.device)
        model.eval()
        n_layers = model.config.num_hidden_layers
        n_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size
        head_dim = hidden_size // n_heads
        if hasattr(model, 'bert'):
            layers = model.bert.encoder.layer
        elif hasattr(model, 'roberta'):
            layers = model.roberta.encoder.layer
        else:
            layers = model.encoder.layer
        en_idx, zh_idx = self._build_script_indices(tokenizer, model)
        if en_idx.numel() == 0 or zh_idx.numel() == 0:
            raise ValueError('Script index sets empty; check tokenizer/model vocab or heuristics.')

        def share_from_logits(logits_1v, mask_pos: int):
            probs = torch.softmax(logits_1v[0, mask_pos], dim=-1)
            p_en = probs.index_select(0, en_idx).sum()
            p_zh = probs.index_select(0, zh_idx).sum()
            denom = p_en + p_zh + eps
            if metric == 'pzh_share':
                return p_zh / denom
            else:
                return p_en / denom
        total_attribution = torch.zeros((n_layers, n_heads), device=self.device)
        valid_count = 0
        facts_subset = self.prepare_facts(facts[:top_k])
        for fact in tqdm(facts_subset, desc='Path Patching', leave=False):
            prompt = fact['prompt']
            subject = fact['subject']
            inputs = tokenizer(prompt, return_tensors='pt').to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            mask_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
            if len(mask_indices[0]) == 0:
                continue
            mask_pos = mask_indices[1][0].item()
            subj_tokens = tokenizer(subject, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
            subj_range = self._get_subject_indices(input_ids, subj_tokens, tokenizer)
            if subj_range is None:
                continue
            clean_activations = {}

            def get_clean_hook(layer_idx):

                def hook(module, args, output):
                    ctx = output[0] if isinstance(output, tuple) else output
                    ctx = ctx.detach()
                    b, s, _ = ctx.shape
                    clean_activations[layer_idx] = ctx.view(b, s, n_heads, head_dim)
                return hook
            handles = []
            for i, layer in enumerate(layers):
                handles.append(layer.attention.self.register_forward_hook(get_clean_hook(i)))
            try:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            finally:
                for h in handles:
                    h.remove()
            if len(clean_activations) != n_layers:
                continue
            emb_module = model.get_input_embeddings()
            clean_emb = emb_module(input_ids)
            corr_emb = clean_emb.clone()
            noise = torch.randn_like(corr_emb[:, list(subj_range), :]) * noise_scale
            corr_emb[:, list(subj_range), :] += noise
            with torch.no_grad():
                out_corr = model(inputs_embeds=corr_emb, attention_mask=attention_mask, output_attentions=True)
                if not hasattr(out_corr, 'logits'):
                    raise AttributeError("Model output missing 'logits'.")
                share_corr = share_from_logits(out_corr.logits, mask_pos).item()
            for l in range(n_layers):
                for h in range(n_heads):

                    def patch_hook(module, args, output, l=l, h=h):
                        ctx = output[0] if isinstance(output, tuple) else output
                        b, s, _ = ctx.shape
                        ctx_view = ctx.view(b, s, n_heads, head_dim)
                        clean_head = clean_activations[l][:, :, h, :]
                        patched = ctx_view.clone()
                        patched[:, :, h, :] = clean_head
                        patched_flat = patched.view(b, s, hidden_size)
                        if isinstance(output, tuple):
                            return (patched_flat,) + output[1:]
                        return patched_flat
                    handle = layers[l].attention.self.register_forward_hook(patch_hook)
                    try:
                        with torch.no_grad():
                            out_patch = model(inputs_embeds=corr_emb, attention_mask=attention_mask, output_attentions=True)
                            share_patch = share_from_logits(out_patch.logits, mask_pos).item()
                            total_attribution[l, h] += share_patch - share_corr
                    finally:
                        handle.remove()
            valid_count += 1
        if valid_count == 0:
            return None
        return (total_attribution / valid_count).cpu().numpy()

    def load_family_attribution_avg(self, family_name: str, language: str, subjects, folder_path: str='patching_results', models_order=('Base', 'Whole', 'Semantic', 'Language'), strict_shapes: bool=True, return_per_subject: bool=False):
        per_model_mats = {m: [] for m in models_order}
        counts = {m: 0 for m in models_order}
        per_subject = {}
        for subj in subjects:
            filename = f'{family_name}_Subject-{subj}_{language}.pkl'
            path = os.path.join(folder_path, filename)
            if not os.path.exists(path):
                continue
            with open(path, 'rb') as f:
                res = pickle.load(f)
            if return_per_subject:
                per_subject[subj] = res
            for m in models_order:
                mat = res.get(m, None)
                if mat is None:
                    continue
                mat = np.asarray(mat)
                if strict_shapes and len(per_model_mats[m]) > 0:
                    if mat.shape != per_model_mats[m][0].shape:
                        raise ValueError(f'Shape mismatch for model {m}: got {mat.shape} (subj={subj})')
                per_model_mats[m].append(mat)
                counts[m] += 1
        avg_results = {}
        std_results = {}
        for m in models_order:
            if counts[m] == 0:
                avg_results[m] = None
                std_results[m] = None
            else:
                stacked = np.stack(per_model_mats[m], axis=0)
                avg_results[m] = stacked.mean(axis=0)
                std_results[m] = stacked.std(axis=0)
        if return_per_subject:
            return (avg_results, std_results, counts, per_subject)
        return (avg_results, std_results, counts)

    def plot_family_grid_avg_diff(self, avg_results: dict, std_results: dict, counts=None, title_suffix: str='', models_order=('Base', 'Whole', 'Semantic', 'Language'), cmap_main='viridis', cmap_diff='coolwarm', cmap_std='magma'):
        valid_avg = [avg_results.get(m) for m in models_order if avg_results.get(m) is not None]
        valid_std = [std_results.get(m) for m in models_order if std_results.get(m) is not None]
        if not valid_avg:
            print('No valid results to plot.')
            return
        base = avg_results.get('Base', None)
        vmin_mean, vmax_mean = (min((m.min() for m in valid_avg)), max((m.max() for m in valid_avg)))
        vmax_std = max((m.max() for m in valid_std)) if valid_std else 1.0
        diffs = [avg_results.get(m) - base for m in models_order if avg_results.get(m) is not None]
        max_abs_diff = max((np.abs(d).max() for d in diffs)) if diffs else 1.0
        plt.rcParams.update({'font.size': 14})
        fig, axes = plt.subplots(3, 4, figsize=(32, 22), constrained_layout=True)
        for col, m in enumerate(models_order):
            mat_avg = avg_results.get(m, None)
            mat_std = std_results.get(m, None)
            n_txt = f' (n={counts.get(m, 0)})' if counts is not None else ''
            ax = axes[0, col]
            if mat_avg is not None:
                sns.heatmap(mat_avg, ax=ax, cmap=cmap_main, vmin=vmin_mean, vmax=vmax_mean, cbar=col == 3, cbar_kws={'label': 'Mean Score'} if col == 3 else None)
                ax.set_title(f'{m} Mean{n_txt}', fontsize=22, fontweight='bold')
            ax = axes[1, col]
            if mat_avg is not None and base is not None:
                diff = mat_avg - base
                sns.heatmap(diff, ax=ax, cmap=cmap_diff, vmin=-max_abs_diff, vmax=max_abs_diff, center=0.0, cbar=col == 3, cbar_kws={'label': 'Δ vs Base'} if col == 3 else None)
                ax.set_title(f'{m} - Base', fontsize=22, fontweight='bold')
            ax = axes[2, col]
            if mat_std is not None:
                sns.heatmap(mat_std, ax=ax, cmap=cmap_std, vmin=0, vmax=vmax_std, cbar=col == 3, cbar_kws={'label': 'Std Dev'} if col == 3 else None)
                ax.set_title(f'{m} Std Dev', fontsize=22, fontweight='bold')
            for row_idx in range(3):
                axes[row_idx, col].set_xlabel('Head ID', fontsize=18)
                axes[row_idx, col].set_ylabel('Layer ID', fontsize=18)
                axes[row_idx, col].tick_params(labelsize=14)
        fig.suptitle(f'Attention Head Attribution: {title_suffix}', fontsize=32, fontweight='bold', y=1.02)
        plt.show()
