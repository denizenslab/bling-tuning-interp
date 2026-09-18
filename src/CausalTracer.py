import os
import pickle
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, "/mlama")
from dataset.reader import MLama


class CausalTracingAnalyzer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.n_layers = model.config.num_hidden_layers
        self.embed_dim = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        self.has_logits = hasattr(model, "predictions") or hasattr(model, "lm_head")
        if not self.has_logits:
            self.lm_head = nn.Linear(self.embed_dim, self.vocab_size).to(device).eval()
        self.results = {}
        self.all_scores = []

    def _layers(self):
        if hasattr(self.model, "bert"):
            return self.model.bert.encoder.layer
        if hasattr(self.model, "model"):
            return self.model.model.layers
        return self.model.encoder.layer

    @staticmethod
    def _pre_norm_target(layer):
        if hasattr(layer, "output") and hasattr(layer.output, "LayerNorm"):
            return layer.output.LayerNorm
        if hasattr(layer, "post_attention_layernorm"):
            return layer.post_attention_layernorm
        return layer

    def _get_subject_indices(self, input_ids, subject_tokens):
        input_ids = input_ids[0].tolist()
        subject_ids = subject_tokens[0].tolist()
        if subject_ids and subject_ids[0] == self.tokenizer.bos_token_id:
            subject_ids = subject_ids[1:]
        if subject_ids and subject_ids[-1] == self.tokenizer.eos_token_id:
            subject_ids = subject_ids[:-1]

        fallback = range(1, min(5, len(input_ids)))
        if not subject_ids:
            return fallback
        width = len(subject_ids)
        for start in range(len(input_ids) - width + 1):
            if input_ids[start : start + width] == subject_ids:
                return range(start, start + width)
        return fallback

    @staticmethod
    def filter_all_datasets_by_base_knowledge(
        base_model, tokenizer, all_facts_en, all_facts_zh, top_k=10, device="cuda"
    ):
        def is_known(fact):
            inputs = tokenizer(fact["prompt"], return_tensors="pt").to(device)
            mask_positions = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero(
                as_tuple=True
            )[0]
            target_ids = tokenizer.encode(fact["target"], add_special_tokens=False)
            if not len(mask_positions) or not target_ids:
                return False
            output = base_model(**inputs)
            logits = (
                output.logits
                if hasattr(output, "logits")
                else output.last_hidden_state @ base_model.embeddings.word_embeddings.weight.T
            )
            predictions = torch.topk(logits[0, mask_positions.item()], k=top_k).indices
            return target_ids[0] in predictions

        print(f"{'=' * 60}\nSTARTING BASE-ANCHORED FILTERING (Top-{top_k} Accuracy)\n{'=' * 60}")
        base_model.to(device).eval()
        filtered_en, filtered_zh = {}, {}

        with torch.no_grad():
            for relation, facts_en in all_facts_en.items():
                if relation not in all_facts_zh:
                    print(f"Skipping '{relation}': Not found in Chinese dataset.")
                    continue
                facts_zh = all_facts_zh[relation]
                n_pairs = min(len(facts_en), len(facts_zh))
                print(f"Processing '{relation}'... (Input: {n_pairs} pairs)")
                valid = [
                    index
                    for index in tqdm(range(n_pairs), desc="  -> Checking Base Model")
                    if is_known(facts_en[index]) and is_known(facts_zh[index])
                ]
                filtered_en[relation] = [facts_en[index] for index in valid]
                filtered_zh[relation] = [facts_zh[index] for index in valid]
                print(f"  -> Kept {len(valid)} facts.")

        print(f"{'=' * 60}\nFILTERING COMPLETE.")
        return filtered_en, filtered_zh

    def get_embeddings_with_noise(self, input_ids, subject_range, noise_scale):
        embeddings = self.model.get_input_embeddings()(input_ids)
        indices = list(subject_range)
        corrupted = embeddings.clone()
        corrupted[0, indices] += torch.randn_like(embeddings[0, indices]) * noise_scale
        return corrupted

    def get_logits_from_output(self, output):
        if getattr(output, "logits", None) is not None:
            return output.logits
        if hasattr(output, "last_hidden_state"):
            return self.lm_head(output.last_hidden_state)
        raise ValueError("Cannot extract logits from model output")

    def _capture_clean_run(self, input_ids, use_pre_norm=False):
        if not use_pre_norm:
            with torch.no_grad():
                output = self.model(input_ids, output_hidden_states=True)
            return output.hidden_states[1:], output

        activations = {}

        def capture(layer_index):
            def hook(_, inputs):
                activations[layer_index] = inputs[0].detach()

            return hook

        handles = [
            self._pre_norm_target(layer).register_forward_pre_hook(capture(index))
            for index, layer in enumerate(self._layers())
        ]
        try:
            with torch.no_grad():
                output = self.model(input_ids, output_hidden_states=False)
        finally:
            for handle in handles:
                handle.remove()
        return activations, output

    def trace_with_patch(
        self,
        input_ids: torch.Tensor,
        subject_range: range,
        target_id: int,
        noise_scale: float = 0.1,
        samples: int = 5,
        use_pre_norm: bool = False,
    ) -> np.ndarray:
        mask_idx = (input_ids[0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
        clean_activations, _ = self._capture_clean_run(input_ids, use_pre_norm)

        with torch.no_grad():
            corrupted_probs = [
                torch.softmax(
                    self.get_logits_from_output(
                        self.model(
                            inputs_embeds=self.get_embeddings_with_noise(
                                input_ids, subject_range, noise_scale
                            ),
                            output_hidden_states=False,
                        )
                    )[0, mask_idx],
                    dim=-1,
                )[target_id].item()
                for _ in range(samples)
            ]
        baseline = np.mean(corrupted_probs)
        scores = np.zeros((self.n_layers, input_ids.shape[1]))
        corrupted_embeddings = self.get_embeddings_with_noise(input_ids, subject_range, noise_scale)

        for layer_index, layer in enumerate(self._layers()):
            clean_activation = clean_activations[layer_index]
            for token_index in range(input_ids.shape[1]):
                def patch(_, inputs, output=None, token_index=token_index, clean_activation=clean_activation):
                    if use_pre_norm:
                        hidden_states = inputs[0]
                        hidden_states[:, token_index] = clean_activation[:, token_index]
                        return (hidden_states,)
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        hidden_states[:, token_index] = clean_activation[:, token_index]
                        return (hidden_states,) + output[1:]
                    output[:, token_index] = clean_activation[:, token_index]
                    return output

                target = self._pre_norm_target(layer) if use_pre_norm else layer
                register = target.register_forward_pre_hook if use_pre_norm else target.register_forward_hook
                handle = register(patch)
                try:
                    with torch.no_grad():
                        logits = self.get_logits_from_output(
                            self.model(inputs_embeds=corrupted_embeddings, output_hidden_states=False)
                        )
                        restored = torch.softmax(logits[0, mask_idx], dim=-1)[target_id].item()
                        scores[layer_index, token_index] = restored - baseline
                finally:
                    handle.remove()
        return scores

    def analyze_batch(
        self, facts: List[Dict], noise_scale: float = 0.1, use_pre_norm: bool = True
    ):
        print(f"Processing {len(facts)} facts (Pre-Norm={use_pre_norm})...")
        self.all_scores = []
        for fact in tqdm(facts, desc="Processing facts"):
            inputs = self.tokenizer(fact["prompt"], return_tensors="pt").to(self.device)
            if fact["subject"]:
                subject_ids = self.tokenizer(
                    fact["subject"], return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.device)
                subject_range = self._get_subject_indices(inputs.input_ids, subject_ids)
            else:
                subject_range = range(1, min(5, inputs.input_ids.shape[1]))

            target_ids = self.tokenizer(fact["target"], add_special_tokens=False).input_ids
            target_id = target_ids[0][0] if isinstance(target_ids[0], list) else target_ids[0]
            scores = self.trace_with_patch(
                inputs.input_ids, subject_range, target_id, noise_scale, use_pre_norm=use_pre_norm
            )
            self.all_scores.append(np.max(scores, axis=1))

        if self.all_scores:
            self.results["batch"] = np.mean(self.all_scores, axis=0)
            print("Completed. Average scores stored.")

    def plot_fact_localization(self, save_path: Optional[str] = None):
        if "batch" not in self.results:
            print("No results to plot. Run analyze_batch first.")
            return
        scores = self.results["batch"]
        layers = np.arange(len(scores))
        plt.figure(figsize=(10, 6))
        plt.plot(layers, scores, marker="o", markersize=6, linewidth=2.5, color="#1f77b4", label="Batch Average")
        if self.all_scores:
            deviation = np.std(self.all_scores, axis=0)
            plt.fill_between(layers, scores - deviation, scores + deviation, alpha=0.2, color="#1f77b4")
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
        scores = np.array(self.all_scores)
        plt.figure(figsize=(12, max(4, len(scores) / 3)))
        sns.heatmap(
            scores,
            cmap="YlOrRd",
            cbar_kws={"label": "AIE Score"},
            xticklabels=[f"L{i}" for i in range(scores.shape[1])],
            yticklabels=[f"Fact {i + 1}" for i in range(scores.shape[0])],
        )
        plt.title("Causal Tracing: Individual Facts Heatmap")
        plt.xlabel("Layer")
        plt.ylabel("Fact Index")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_average_heatmap(self, save_path: Optional[str] = None):
        if "batch" not in self.results:
            print("No batch results to plot. Run analyze_batch first.")
            return
        scores = self.results["batch"]
        plt.figure(figsize=(12, 2))
        sns.heatmap(
            scores.reshape(1, -1),
            cmap="YlOrRd",
            cbar_kws={"label": "AIE Score"},
            xticklabels=[f"L{i}" for i in range(len(scores))],
            yticklabels=["Avg"],
        )
        plt.title("Causal Tracing: Average Across All Facts (Heatmap)")
        plt.xlabel("Layer")
        plt.ylabel("")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def localization_statistics(self) -> Dict:
        if "batch" not in self.results:
            return {}
        scores = self.results["batch"]
        first, second = self.n_layers // 3, 2 * self.n_layers // 3
        return {
            "peak_layer": int(np.argmax(scores)),
            "peak_score": float(np.max(scores)),
            "early_layer_avg": float(np.mean(scores[:first])),
            "mid_layer_avg": float(np.mean(scores[first:second])),
            "late_layer_avg": float(np.mean(scores[second:])),
            "total_impact": float(np.sum(scores)),
        }

    @staticmethod
    def build_fact_list(fact_dict, relation_type="capital"):
        facts = []
        for prompt, answer in fact_dict.items():
            try:
                subject = prompt.split("of ")[1].split(" is")[0].strip()
            except IndexError:
                subject = ""
            facts.append({"prompt": prompt, "subject": subject, "target": answer})
        return facts

    def run_all_models(
        self,
        subject: str,
        model_families,
        facts_dict,
        noise_scale: float = 0.1,
        fact_size: Optional[int] = None,
        save_dir: str = "causal_tracing_results",
    ) -> Dict[str, Dict[str, Dict[str, "CausalTracingAnalyzer"]]]:
        results = {}
        for fact_type, facts in facts_dict.items():
            current_facts = facts[:fact_size] if fact_size is not None else facts
            print(f"\n{'=' * 20}\nProcessing Dataset: {fact_type}\n{'=' * 20}")
            if not current_facts:
                print(f"No facts found for {fact_type}. Skipping.")
                continue
            results[fact_type] = {}
            for family_name, models in model_families.items():
                results[fact_type][family_name] = {}
                for model_name, model_info in models.items():
                    print(f"Running {family_name} - {model_name} on {len(current_facts)} facts...")
                    if isinstance(model_info, dict) and {"model", "tokenizer"} <= model_info.keys():
                        model, tokenizer = model_info["model"], model_info["tokenizer"]
                    else:
                        model, tokenizer = model_info, self.tokenizer
                    analyzer = CausalTracingAnalyzer(model, tokenizer, self.device)
                    analyzer.analyze_batch(current_facts, noise_scale)
                    results[fact_type][family_name][model_name] = analyzer
                    self._save_causal_tracing_result(
                        analyzer, subject, fact_type, family_name, model_name,
                        noise_scale, len(current_facts), save_dir,
                    )
        return results

    def _save_causal_tracing_result(
        self, analyzer, subject, fact_type, family_name, model_name,
        noise_scale, fact_size, save_dir,
    ) -> None:
        directory = os.path.join(save_dir, f"subject_{subject}", fact_type, family_name)
        os.makedirs(directory, exist_ok=True)
        filename = f"causal_{family_name}_{model_name}_{fact_type}_ns{noise_scale:g}_n{fact_size}.pkl"
        filepath = os.path.join(directory, filename)
        payload = {
            "analyzer": analyzer,
            "meta": {
                "subject": subject,
                "fact_type": fact_type,
                "family_name": family_name,
                "model_name": model_name,
                "noise_scale": noise_scale,
                "fact_size": fact_size,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
            },
        }
        with open(filepath, "wb") as file:
            pickle.dump(payload, file)
        print(f"  -> Saved causal tracing result to {filepath}")

    def collect_prompts_answers_all_relations(
        self, data_folder, languages=["en", "zh"], mask_token="[MASK]"
    ):
        mlama = MLama(data_folder)
        mlama.load()
        facts = {}
        for language in languages:
            for relation, entry in mlama.data[language].items():
                template = entry["template"]
                facts[f"{relation}-{language}"] = {
                    template.replace("[X]", triple["sub_label"]).replace("[Y]", mask_token): triple["obj_label"]
                    for triple in entry["triples"].values()
                }
        return facts

    def plot_family_grid(
        self, analyzers_dict, family_name, specific_indices=None,
        facts_list=None, random_fact_count=10, random_seed=42,
    ):
        if random_seed is not None:
            random.seed(random_seed)
        model_names = list(analyzers_dict)
        if specific_indices is None:
            total_facts = len(analyzers_dict[model_names[0]].all_scores)
            specific_indices = sorted(random.sample(range(total_facts), min(total_facts, random_fact_count)))

        _, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 20))
        for row, model_name in enumerate(model_names[:4]):
            analyzer = analyzers_dict[model_name]
            line_axis, heatmap_axis = axes[row]
            batch_scores = analyzer.results.get("batch")
            if batch_scores is None:
                line_axis.text(0.5, 0.5, "No batch scores", ha="center", va="center")
            else:
                layers = range(len(batch_scores))
                line_axis.plot(layers, batch_scores, marker="o", linewidth=2)
                line_axis.set_title(f"{family_name} {model_name} - Overall Average AIE")
                line_axis.set_xlabel("Layer Index")
                line_axis.set_ylabel("AIE Score")
                line_axis.grid(True, alpha=0.3)
                line_axis.set_xticks(layers)

            valid_indices = [i for i in specific_indices if i < len(analyzer.all_scores)]
            if not valid_indices:
                heatmap_axis.text(0.5, 0.5, "No fact scores", ha="center", va="center")
                continue
            scores = np.array(analyzer.all_scores)[valid_indices]
            labels = [f"Fact {i}" for i in valid_indices]
            sns.heatmap(
                scores, ax=heatmap_axis, cmap="YlOrRd", cbar=True,
                xticklabels=[f"L{i}" for i in range(scores.shape[1])], yticklabels=labels,
            )
            heatmap_axis.set_title(f"{family_name} {model_name} - Specific Facts")
            heatmap_axis.set_xlabel("Layer Index")
            heatmap_axis.set_yticklabels(labels, rotation=0, fontsize=10, va="center")

        plt.suptitle(f"Causal Tracing Analysis: {family_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def plot_family_overall_heatmaps(self, analyzers_dict, family_name):
        base_name = next((name for name in analyzers_dict if "base" in name.lower()), None)
        if base_name is None:
            print(f"Warning: No 'Base' model found in {family_name}. Cannot compute differences.")
            return
        base_scores = analyzers_dict[base_name].results.get("batch")
        if base_scores is None:
            print("Base model has no batch results yet.")
            return

        model_names = list(analyzers_dict)
        scores = [
            analyzers_dict[name].results.get("batch", np.zeros_like(base_scores))
            for name in model_names
        ]
        differences = [score - base_scores for score in scores]
        _, axes = plt.subplots(2, len(model_names), figsize=(4 * len(model_names), 8), sharey=True)
        axes = np.asarray(axes).reshape(2, -1)
        abs_min, abs_max = 0, np.max(np.vstack(scores))
        diff_max = np.max(np.abs(np.vstack(differences)))

        for index, name in enumerate(model_names):
            sns.heatmap(
                scores[index].reshape(1, -1), ax=axes[0, index], cmap="YlOrRd", cbar=True,
                vmin=abs_min, vmax=abs_max,
                xticklabels=[f"L{i}" for i in range(len(base_scores))], yticklabels=[name],
            )
            axes[0, index].set_title(f"{name} (Absolute)")
            axes[0, index].set_xlabel("")
            axes[0, index].set_yticks([])
            sns.heatmap(
                differences[index].reshape(1, -1), ax=axes[1, index], cmap="bwr", center=0,
                cbar=True, vmin=-diff_max, vmax=diff_max,
                xticklabels=[f"L{i}" for i in range(len(base_scores))], yticklabels=[f"Δ {name}"],
            )
            axes[1, index].set_title(f"Diff from Base ({name} - Base)")
            axes[1, index].set_xlabel("Transformer Block Outputs")
            axes[1, index].set_yticks([])

        plt.suptitle(f"{family_name} Family: Absolute vs. Relative Causal Traces", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
