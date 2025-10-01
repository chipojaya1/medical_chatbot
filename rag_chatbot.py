"""Medical chatbot implementation for Milestone 3: Embeddings + RAG.

This module prepares the disease/symptom dataset, builds dense embeddings
using SentenceTransformers, and uses a quantized Llama-2 model to generate
RAG-based responses that include a disclaimer.  The code mirrors the steps in
Milestone 3 of the NSDC medical chatbot project while exposing a reusable
``MedicalRAGChatbot`` class and a simple CLI interface.

The default model IDs match the tutorial instructions.  Loading the language
model requires significant resources; run this script in an environment with a
GPU (for example Kaggle with the GPU accelerator enabled).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer


DisclaimerText = (
    "Note: This is not a medical diagnosis. Always consult a licensed physician."
)


@dataclass
class GenerationConfig:
    """Configuration parameters for language model generation."""

    max_new_tokens: int = 300
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True

    @property
    def sampling_kwargs(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and clean the disease/symptom dataset.

    Steps mirror the milestone instructions:
    1. Replace underscores with spaces in all ``Symptom_*`` columns.
    2. Combine symptoms for each row into a comma-separated string.
    3. Drop duplicate symptom lists.
    4. Group by disease and join all symptom sentences.
    5. Build a ``Text`` column that concatenates disease and symptoms.
    """

    df = pd.read_csv(csv_path)
    symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]

    # Step 2: Remove underscores from the symptom text.
    df[symptom_cols] = df[symptom_cols].replace("_", " ", regex=True)

    # Step 3: Combine symptoms into a single comma-separated column.
    df["Symptoms"] = df[symptom_cols].apply(
        lambda row: ", ".join(symptom for symptom in row if pd.notna(symptom)),
        axis=1,
    )

    # Remove the original symptom columns and drop duplicate symptom lists.
    df = df[["Disease", "Symptoms"]].drop_duplicates(subset=["Symptoms"])

    # Step 4: Group all symptom entries for each disease into one string.
    grouped = (
        df.groupby("Disease", as_index=False)["Symptoms"]
        .agg(lambda symptoms: ", ".join(symptoms))
        .copy()
    )

    # Step 5: Create the text that will serve as the retrieval corpus.
    grouped["Text"] = grouped.apply(
        lambda row: f"Disease: {row['Disease']}. Symptoms: {row['Symptoms']}",
        axis=1,
    )
    return grouped


class MedicalRAGChatbot:
    """Retrieval Augmented Generation chatbot for medical symptom queries."""

    def __init__(
        self,
        csv_path: Path | str = "dataset.csv",
        *,
        embed_model_name: str = "all-MiniLM-L6-v2",
        llm_model_id: str = "TheBloke/Llama-2-7B-Chat-GPTQ",
        top_k: int = 2,
        generation_config: GenerationConfig | None = None,
        tokenizer_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        if generation_config is None:
            generation_config = GenerationConfig()
        self.generation_config = generation_config
        self.top_k = top_k

        # Prepare dataset and retrieval corpus.
        self.dataset = _load_dataset(self.csv_path)
        self.corpus: List[str] = self.dataset["Text"].tolist()

        # Step 4: Encode the corpus into embeddings.
        self.embed_model = SentenceTransformer(embed_model_name)
        self.corpus_embeddings = self.embed_model.encode(
            self.corpus, convert_to_tensor=True
        )

        # Step 5: Load tokenizer and LLM.
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_id, use_fast=True, **tokenizer_kwargs
        )

        model_kwargs = model_kwargs or {}
        default_model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
        default_model_kwargs.update(model_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_id, **default_model_kwargs
        )

        if self.tokenizer.pad_token_id is None:
            # Some LLaMA tokenizers do not define a pad token; fall back to eos.
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _retrieve_contexts(self, user_query: str) -> Sequence[str]:
        """Retrieve the top-k relevant corpus entries for ``user_query``."""

        query_embedding = self.embed_model.encode(user_query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)
        return [self.corpus[hit["corpus_id"]] for hit in hits[0]]

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, user_query: str, retrieved_contexts: Sequence[str]) -> str:
        context_block = "\n".join(retrieved_contexts)
        return (
            "You are a medical assistant. Based on the medical records below, "
            "suggest top 2 possible diseases the user might have. Be concise "
            "and respond in bullet points.\n\n"
            "Always include a disclaimer that this is not a medical diagnosis "
            "and that the user should consult a doctor.\n\n"
            f"Medical Records:\n{context_block}\n\n"
            f"User Symptoms: {user_query}\n\n"
            "Your Response:"
        )

    def _generate(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                **self.generation_config.sampling_kwargs,
            )

        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Remove the prompt from the decoded text.
        return response_text[len(prompt) :].strip()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def rag_response(self, user_query: str) -> str:
        """Generate a RAG-based response for ``user_query``."""

        contexts = self._retrieve_contexts(user_query)
        prompt = self._build_prompt(user_query, contexts)
        generated = self._generate(prompt)
        if DisclaimerText not in generated:
            return f"{generated}\n\n{DisclaimerText}"
        return generated

    def chat(self) -> None:
        """Simple CLI loop for chatting with the bot."""

        print("ChatBot: I can help suggest possible diseases based on your symptoms.")
        print("Type your symptoms ('fever, cough, sore throat'), or type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print(
                    "ChatBot: Goodbye!\n Note: This is not a medical diagnosis. "
                    "Always consult a licensed physician."
                )
                break

            if not user_input:
                print("ChatBot: Please enter at least one symptom.\n")
                continue

            try:
                response = self.rag_response(user_input)
            except Exception as exc:  # pragma: no cover - CLI guardrail
                print(
                    "ChatBot: I encountered an error while generating a response. "
                    "Please verify that the required models are available."
                )
                print(f"Details: {exc}\n")
                continue

            print(f"ChatBot: {response}\n")
            print(f"{DisclaimerText}\n")


def build_chatbot(**kwargs) -> MedicalRAGChatbot:
    """Factory function that mirrors the milestone tutorial signature."""

    return MedicalRAGChatbot(**kwargs)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    bot = build_chatbot()
    bot.chat()
