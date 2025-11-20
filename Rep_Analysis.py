import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy.stats import spearmanr
from datasets import load_dataset

class RSAAnalyzer:
    def __init__(self, device="cuda"):
        self.device = device

    def get_layerwise_hidden(self, sentences, model, tokenizer):
        all_sent_hiddens = []
        for sent in tqdm(sentences, desc="Getting hidden states"):
            encoding = tokenizer(sent, is_split_into_words=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**{k: v.to(self.device) for k,v in encoding.items()}, output_hidden_states=True)
                hiddens = outputs.hidden_states
                word_ids = encoding.word_ids(batch_index=0)
                reps = []
                for layer_h in hiddens:
                    layer_reps = []
                    for i in range(len(sent)):
                        token_idxs = [j for j, wid in enumerate(word_ids) if wid == i]
                        subtok = torch.stack([layer_h[0, tid] for tid in token_idxs], dim=0)
                        layer_reps.append(subtok.mean(dim=0).cpu().numpy())
                    layer_reps = np.stack(layer_reps)
                    reps.append(layer_reps)
                all_sent_hiddens.append(reps)
        return all_sent_hiddens
    
    def get_layerwise_hidden_strings(self, sentences, model, tokenizer):
        all_sent_hiddens = []
        for sent in tqdm(sentences, desc="Getting hidden states"):
            encoding = tokenizer(sent, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**{k: v.to(self.device) for k,v in encoding.items()}, output_hidden_states=True)
                hiddens = outputs.hidden_states
                # Mean pool over all tokens for sentence representation at each layer
                reps = [layer_h[0].mean(dim=0).cpu().numpy() for layer_h in hiddens]
                all_sent_hiddens.append(reps)
        return all_sent_hiddens

    def compute_rsa(self, sent_reps1, sent_reps2):
        n_layers = len(sent_reps1[0])
        rsa_sim = []
        for layer in range(n_layers):
            mat1 = [x[layer] for x in sent_reps1]
            mat2 = [x[layer] for x in sent_reps2]
            mats1 = [x.reshape(-1, x.shape[-1]).mean(0) for x in mat1]
            mats2 = [x.reshape(-1, x.shape[-1]).mean(0) for x in mat2]
            mats1 = np.stack(mats1)
            mats2 = np.stack(mats2)

            def pdist(X):
                normed = X / np.linalg.norm(X, axis=1, keepdims=True)
                return np.dot(normed, normed.T)

            D1 = pdist(mats1)
            D2 = pdist(mats2)
            idx = np.triu_indices(len(mats1), k=1)
            x = D1[idx]
            y = D2[idx]
            sim, _ = spearmanr(x, y)
            rsa_sim.append(sim)
        return rsa_sim

    def plot_rsa(self, rsa_dict, base_name="Base", language="English", y_limlow=0, y_limup=1, max_sentences=None):
        sns.set(style="whitegrid", font_scale=1.2)
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

    # ===== NEW METHODS FOR XNLI =====
    
    def load_xnli_parallel(self, lang1="en", lang2="zh", max_sentences=1000):
        xnli1 = load_dataset("xnli", name=lang1)['train']
        xnli2 = load_dataset("xnli", name=lang2)['train']
        
        premises1 = [xnli1[i]["premise"] for i in range(min(max_sentences, len(xnli1)))]
        premises2 = [xnli2[i]["premise"] for i in range(min(max_sentences, len(xnli2)))]
        hypotheses1 = [xnli1[i]["hypothesis"] for i in range(min(max_sentences, len(xnli1)))]
        hypotheses2 = [xnli2[i]["hypothesis"] for i in range(min(max_sentences, len(xnli2)))]
        labels = [xnli1[i]["label"] for i in range(min(max_sentences, len(xnli1)))]
        
        return {
            f'premises_{lang1}': premises1,
            f'premises_{lang2}': premises2,
            f'hypotheses_{lang1}': hypotheses1,
            f'hypotheses_{lang2}': hypotheses2,
            'labels': labels
        }
    
    def compute_rsa_xnli(self, xnli_data, model_base, model_whole, model_semantic, model_language, 
                         tokenizer, lang1="en", lang2="zh"):
        results = {}
        label_names = {0: "contradiction", 1: "neutral", 2: "entailment"}
        
        for sent_type in ['premises', 'hypotheses']:
            sentences_lang1 = xnli_data[f'{sent_type}_{lang1}']
            sentences_lang2 = xnli_data[f'{sent_type}_{lang2}']
            labels = xnli_data['labels']
            
            # Group by label
            for label_id in [0, 1, 2]:
                indices = [i for i, lbl in enumerate(labels) if lbl == label_id]
                if len(indices) == 0:
                    continue
                
                sents_l1 = [sentences_lang1[i] for i in indices]
                sents_l2 = [sentences_lang2[i] for i in indices]
                
                # Get representations for all four models for lang1
                print(f"\nProcessing {sent_type} - {label_names[label_id]} ({len(sents_l1)} sentences)")
                reps_base = self.get_layerwise_hidden_strings(sents_l1, model_base, tokenizer)
                reps_whole = self.get_layerwise_hidden_strings(sents_l1, model_whole, tokenizer)
                reps_semantic = self.get_layerwise_hidden_strings(sents_l1, model_semantic, tokenizer)
                reps_language = self.get_layerwise_hidden_strings(sents_l1, model_language, tokenizer)
                
                # Compute RSA vs base
                rsa_scores = {
                    "Whole vs Base": self.compute_rsa(reps_whole, reps_base),
                    "Semantic vs Base": self.compute_rsa(reps_semantic, reps_base),
                    "Language vs Base": self.compute_rsa(reps_language, reps_base)
                }
                
                key = f"{sent_type}_{label_names[label_id]}"
                results[key] = {
                    'rsa_scores': rsa_scores,
                    'n_sentences': len(sents_l1),
                    'label_id': label_id,
                    'sent_type': sent_type
                }
        
        return results
    
    def plot_rsa_xnli(self, xnli_results, language="English-Chinese", y_limlow=0.3, y_limup=1.1):
        """
        Plot RSA results from XNLI analysis with clear labeling
        Skips layer 0 (embedding layer) and shows only transformer blocks 1-12
        """
        sns.set(style="whitegrid", font_scale=1.0)

        # Create 2x3 subplot grid: 2 rows (premises, hypotheses) x 3 cols (contradiction, neutral, entailment)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Cross-Model RSA Similarity - XNLI ({language})", fontsize=16)

        label_names = ["contradiction", "neutral", "entailment"]
        sent_types = ["premises", "hypotheses"]

        for row_idx, sent_type in enumerate(sent_types):
            for col_idx, label_name in enumerate(label_names):
                ax = axes[row_idx, col_idx]
                key = f"{sent_type}_{label_name}"

                if key not in xnli_results:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center')
                    ax.set_title(f"{sent_type.capitalize()} - {label_name.capitalize()}")
                    continue

                result = xnli_results[key]
                rsa_scores = result['rsa_scores']
                n_sents = result['n_sentences']

                df_records = []
                for name, rsa_vals in rsa_scores.items():
                    # Skip layer 0 (embedding layer), only plot layers 1-12
                    for i in range(1, len(rsa_vals)):
                        df_records.append({"Layer": i, "RSA Similarity": rsa_vals[i], "Comparison": name})
                df = pd.DataFrame(df_records)

                sns.lineplot(data=df, x="Layer", y="RSA Similarity", hue="Comparison", marker="o", ax=ax)
                ax.set_title(f"{sent_type.capitalize()} - {label_name.capitalize()}\n(n={n_sents})", fontsize=12)
                ax.set_ylim(y_limlow, y_limup)
                ax.set_xlabel("Transformer Block Output (1-12)" if row_idx == 1 else "")
                ax.set_ylabel("RSA Similarity" if col_idx == 0 else "")

                # Set x-axis to show layers 1-12
                ax.set_xticks(range(1, 13))

                if row_idx == 0 or col_idx > 0:
                    ax.legend().set_visible(False)
                else:
                    ax.legend(loc='best', fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    def compute_rsa_xnli_per_language(self, xnli_data, language="en", 
                                       model_base=None, model_whole=None, 
                                       model_semantic=None, model_language=None,
                                       tokenizer=None):
        results = {}
        label_names = {0: "contradiction", 1: "neutral", 2: "entailment"}

        for sent_type in ['premises', 'hypotheses']:
            sentences = xnli_data[f'{sent_type}_{language}']
            labels = xnli_data['labels']

            # Group by label
            for label_id in [0, 1, 2]:
                indices = [i for i, lbl in enumerate(labels) if lbl == label_id]
                if len(indices) == 0:
                    continue

                sents = [sentences[i] for i in indices]

                # Get representations for all four models
                print(f"\nProcessing {sent_type} - {label_names[label_id]} ({len(sents)} sentences)")
                reps_base = self.get_layerwise_hidden_strings(sents, model_base, tokenizer)
                reps_whole = self.get_layerwise_hidden_strings(sents, model_whole, tokenizer)
                reps_semantic = self.get_layerwise_hidden_strings(sents, model_semantic, tokenizer)
                reps_language = self.get_layerwise_hidden_strings(sents, model_language, tokenizer)

                # Compute RSA vs base
                rsa_scores = {
                    "Whole vs Base": self.compute_rsa(reps_whole, reps_base),
                    "Semantic vs Base": self.compute_rsa(reps_semantic, reps_base),
                    "Language vs Base": self.compute_rsa(reps_language, reps_base)
                }

                key = f"{sent_type}_{label_names[label_id]}"
                results[key] = {
                    'rsa_scores': rsa_scores,
                    'n_sentences': len(sents),
                    'label_id': label_id,
                    'sent_type': sent_type
                }

        return results
