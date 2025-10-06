# Milestone 3: Embeddings + RAG Checklist

This guide walks through the exact steps implemented in `rag_chatbot.py` so that you can reproduce Milestone 3 of the NSDC medical chatbot project on your own machine or in Kaggle.

## 1. Prepare the environment
- Enable a GPU runtime (e.g. *Kaggle → Settings → Accelerator → GPU*).
- Install the required libraries:
  ```bash
  pip install -q pandas sentence-transformers transformers accelerate einops
  ```
- If you use Kaggle, upload the repository files (`dataset.csv` and `rag_chatbot.py`) to your notebook session.

## 2. Load and clean the dataset
The `_load_dataset` helper executes all preprocessing steps from the milestone:
1. Read `dataset.csv`.
2. Replace underscores with spaces across all `Symptom_*` columns.
3. Merge the row symptoms into a single comma-separated string.
4. Drop duplicate symptom lists and group symptoms by the disease name.
5. Build a `Text` column with `"Disease: ... Symptoms: ..."` sentences that will feed the retriever.

If you want to perform these steps manually inside a notebook, run:
```python
from rag_chatbot import _load_dataset
prepared_df = _load_dataset("dataset.csv")
prepared_df.head()
```

## 3. Create dense embeddings
Instantiate the chatbot (or directly the `SentenceTransformer`) to encode the corpus:
```python
from rag_chatbot import MedicalRAGChatbot
bot = MedicalRAGChatbot(csv_path="dataset.csv")
```
This downloads the `all-MiniLM-L6-v2` model and converts the cleaned dataset into embedding tensors that will be used for semantic search.

## 4. Load the quantized LLaMA model
`MedicalRAGChatbot` loads the `TheBloke/Llama-2-7B-Chat-GPTQ` checkpoint with `device_map="auto"` and `torch_dtype=torch.float16`. Make sure your runtime has enough GPU memory (~16 GB). If not, swap in a smaller model such as `meta-llama/Llama-2-7b-chat-hf` or any other instruct-tuned model you have access to.

## 5. Retrieve relevant contexts
When you call `bot.rag_response("fever, cough")`, the chatbot:
1. Encodes the user query into an embedding.
2. Runs `sentence_transformers.util.semantic_search` against the corpus embeddings.
3. Picks the top `k` (default 2) disease descriptions as context for generation.

## 6. Build the prompt and generate a response
The prompt template mirrors the milestone instructions:
- Provides the retrieved medical records as context.
- Asks for the **top 2 possible diseases**.
- Requests a **concise bullet list**.
- Enforces a **medical disclaimer**.

Generation is controlled by `GenerationConfig` (max 300 tokens, temperature 0.2, top-p 0.9). Adjust these values if you need shorter or more diverse replies.

## 7. Run the chatbot interactively
To launch the CLI inside a notebook cell or terminal:
```bash
python rag_chatbot.py
```
Sample exchange:
```
ChatBot: I can help suggest possible diseases based on your symptoms.
You: fever, cough, headache
ChatBot: • Disease 1 ...
          • Disease 2 ...

Note: This is not a medical diagnosis. Always consult a licensed physician.
```

You can also reuse the class in a notebook and format the response however you like:
```python
response = bot.rag_response("sore throat, difficulty swallowing")
print(response)
```

## 8. Troubleshooting tips
- **Model download errors**: ensure that you have accepted the LLaMA-2 license on Hugging Face and that your token is configured via `huggingface-cli login`.
- **Out of memory**: lower `top_k` to 1, reduce `max_new_tokens`, or switch to a smaller causal language model.
- **Slow responses**: warm up the embedding model (`bot.embed_model`) by encoding a dummy sentence before running many queries.

With these steps you should be able to complete Milestone 3 and experiment with the RAG chatbot both programmatically and via the CLI.
