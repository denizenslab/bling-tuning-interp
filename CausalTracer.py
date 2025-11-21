import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings
from tqdm import tqdm
import copy

warnings.filterwarnings('ignore')

class CausalTracingAnalyzer:
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        self.n_layers = model.config.num_hidden_layers
        self.embed_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        self.has_logits = hasattr(model, 'predictions') or hasattr(model, 'lm_head')
        
        if not self.has_logits:
            self.lm_head = nn.Linear(self.embed_dim, self.vocab_size).to(device)
            self.lm_head.eval()
        
        self.results = {}
        self.all_scores = [] 
        
    def _get_subject_indices(self, input_ids, subject_tokens):

        input_list = input_ids[0].tolist()
        subject_list = subject_tokens[0].tolist()
        
        if subject_list and subject_list[0] == self.tokenizer.bos_token_id:
            subject_list = subject_list[1:]
        if subject_list and subject_list[-1] == self.tokenizer.eos_token_id:
            subject_list = subject_list[:-1]
            
        len_subj = len(subject_list)
        if len_subj == 0:
            return range(1, min(5, len(input_list)))
        
        for i in range(len(input_list) - len_subj + 1):
            if input_list[i:i+len_subj] == subject_list:
                return range(i, i+len_subj)
        
        return range(1, min(5, len(input_list)))

    def get_embeddings_with_noise(self, input_ids, subject_range, noise_scale):
        embeddings = self.model.get_input_embeddings()(input_ids)
        noise = torch.randn_like(embeddings[0, list(subject_range), :]) * noise_scale
        corrupted_embeddings = embeddings.clone()
        corrupted_embeddings[0, list(subject_range), :] += noise
        return corrupted_embeddings

    def get_logits_from_output(self, output):
        if hasattr(output, 'logits') and output.logits is not None:
            return output.logits
        elif hasattr(output, 'last_hidden_state'):
            return self.lm_head(output.last_hidden_state)
        else:
            raise ValueError("Cannot extract logits from model output")

    def trace_with_patch(self, 
                            input_ids: torch.Tensor, 
                            subject_range: range, 
                            target_id: int,
                            noise_scale: float = 0.1,
                            samples: int = 5) -> np.ndarray:

            mask_token_id = self.tokenizer.mask_token_id
            try:
                mask_idx = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0].item()
            except IndexError:
                mask_idx = -1 

            with torch.no_grad():
                clean_outputs = self.model(input_ids, output_hidden_states=True)
                clean_hidden_states = clean_outputs.hidden_states
                clean_logits = self.get_logits_from_output(clean_outputs)

                clean_prob = torch.softmax(clean_logits[0, mask_idx, :], dim=-1)[target_id].item()

            # 2. Corrupted Run
            corrupted_probs = []
            with torch.no_grad():
                for _ in range(samples):
                    corrupted_embeds = self.get_embeddings_with_noise(input_ids, subject_range, noise_scale)
                    out = self.model(inputs_embeds=corrupted_embeds, output_hidden_states=True)
                    logits = self.get_logits_from_output(out)
                    # FIX: Use mask_idx
                    probs = torch.softmax(logits[0, mask_idx, :], dim=-1)
                    corrupted_probs.append(probs[target_id].item())

            avg_corrupted_prob = np.mean(corrupted_probs)

            seq_len = input_ids.shape[1]
            scores = np.zeros((self.n_layers, seq_len))

            def get_patching_hook(layer_idx, token_idx, clean_activation):
                def hook(module, args, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output

                    # Patch the state
                    hs[:, token_idx, :] = clean_activation[:, token_idx, :]

                    if isinstance(output, tuple):
                        return (hs,) + output[1:]
                    return hs
                return hook

            corrupted_embeds_fixed = self.get_embeddings_with_noise(input_ids, subject_range, noise_scale)

            # Pre-calculate layer modules to avoid repeated lookups
            if hasattr(self.model, 'bert'):
                layers = self.model.bert.encoder.layer
            elif hasattr(self.model, 'model'): # DistilBERT/Roberta
                layers = self.model.model.layers 
            else:
                layers = self.model.model.layers # Llama/Mistral

            for layer_idx in range(self.n_layers):
                for token_idx in range(seq_len):
                    layer_module = layers[layer_idx]
                    clean_act = clean_hidden_states[layer_idx + 1]

                    handle = layer_module.register_forward_hook(
                        get_patching_hook(layer_idx, token_idx, clean_act)
                    )

                    try:
                        with torch.no_grad():
                            out = self.model(inputs_embeds=corrupted_embeds_fixed, output_hidden_states=True)
                            logits = self.get_logits_from_output(out)
                            # FIX: Use mask_idx
                            restored_prob = torch.softmax(logits[0, mask_idx, :], dim=-1)[target_id].item()
                            scores[layer_idx, token_idx] = restored_prob - avg_corrupted_prob
                    finally:
                        handle.remove()

            return scores

    def analyze_batch(self, 
                     facts: List[Dict], 
                     noise_scale: float = 0.1):
        print(f"Processing {len(facts)} facts...")
        lang_scores = []
        
        for fact in tqdm(facts, desc="Processing facts"):
            prompt = fact['prompt']
            subject = fact['subject']
            target = fact['target']
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids
            
            # Get Subject Range
            if subject:
                subj_tokens = self.tokenizer(subject, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                subj_range = self._get_subject_indices(input_ids, subj_tokens)
            else:
                subj_range = range(1, min(5, input_ids.shape[1]))
            
            # Get Target ID
            ids = self.tokenizer(target, add_special_tokens=False).input_ids
            target_id = ids[0][0] if isinstance(ids[0], list) else ids[0]
            
            # Run Trace
            scores = self.trace_with_patch(input_ids, subj_range, target_id, noise_scale)
            layer_impacts = np.max(scores, axis=1)  # Max over token dimension
            lang_scores.append(layer_impacts)
            self.all_scores.append(layer_impacts)
        
        # Average across all facts in this batch
        if lang_scores:
            avg_scores = np.mean(np.array(lang_scores), axis=0)
            self.results['batch'] = avg_scores
            print(f"Completed {len(facts)} facts. Average scores stored.")

    def plot_fact_localization(self, save_path: Optional[str] = None):
        if 'batch' not in self.results:
            print("No results to plot. Run analyze_batch first.")
            return
        
        plt.figure(figsize=(10, 6))
        scores = self.results['batch']
        layers = np.arange(len(scores))
        
        plt.plot(layers, scores, marker='o', markersize=6, linewidth=2.5, color='#1f77b4', label='Batch Average')
        
        # Add confidence band if we have individual scores
        if self.all_scores:
            all_scores_array = np.array(self.all_scores)
            std_scores = np.std(all_scores_array, axis=0)
            plt.fill_between(layers, scores - std_scores, scores + std_scores, alpha=0.2, color='#1f77b4')
        
        plt.title("Causal Tracing: Fact Localization (Batch Average)")
        plt.xlabel("Layer Index")
        plt.ylabel("Average Indirect Effect (AIE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_individual_facts_heatmap(self, save_path: Optional[str] = None):
        if not self.all_scores:
            print("No individual fact scores to plot. Run analyze_batch first.")
            return
        
        all_scores_array = np.array(self.all_scores)
        
        plt.figure(figsize=(12, max(4, len(self.all_scores) / 3)))
        sns.heatmap(all_scores_array, cmap='YlOrRd', cbar_kws={'label': 'AIE Score'}, 
                    xticklabels=[f"L{i}" for i in range(all_scores_array.shape[1])],
                    yticklabels=[f"Fact {i+1}" for i in range(all_scores_array.shape[0])])
        plt.title("Causal Tracing: Individual Facts Heatmap")
        plt.xlabel("Layer")
        plt.ylabel("Fact Index")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_average_heatmap(self, save_path: Optional[str] = None):
        if 'batch' not in self.results:
            print("No batch results to plot. Run analyze_batch first.")
            return
        
        avg_scores = self.results['batch']
        avg_scores_reshaped = avg_scores.reshape(1, -1)
        
        plt.figure(figsize=(12, 2))
        sns.heatmap(avg_scores_reshaped, cmap='YlOrRd', cbar_kws={'label': 'AIE Score'}, 
                    xticklabels=[f"L{i}" for i in range(len(avg_scores))],
                    yticklabels=['Avg'])
        plt.title("Causal Tracing: Average Across All Facts (Heatmap)")
        plt.xlabel("Layer")
        plt.ylabel("")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def localization_statistics(self) -> Dict:
        if 'batch' not in self.results:
            return {}
        
        scores = self.results['batch']
        early_layers = scores[:self.n_layers//3]
        mid_layers = scores[self.n_layers//3 : 2*self.n_layers//3]
        late_layers = scores[2*self.n_layers//3:]
        
        return {
            'peak_layer': int(np.argmax(scores)),
            'peak_score': float(np.max(scores)),
            'early_layer_avg': float(np.mean(early_layers)),
            'mid_layer_avg': float(np.mean(mid_layers)),
            'late_layer_avg': float(np.mean(late_layers)),
            'total_impact': float(np.sum(scores))
        }
