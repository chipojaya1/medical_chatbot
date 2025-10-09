# MedBot4U - AI Medical Symptom Assessment Chatbot

<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Executive Summary](#executive-summary)
- [Problem Understanding](#problem-understanding)
- [Data Collection and Pre-processing](#data-collection-and-pre-processing)
- [The AI Models](#the-ai-models)
- [The Chatbot](#the-chatbot)
- [Recommended Next Steps](#recommended-next-steps)
- [Risk Considerations](#risk-considerations)

## New Prototype: Data Science Career Copilot

To support aspiring data scientists, the repository now also contains a lightweight Retrieval-Augmented Generation (RAG) prototype built with Flask (`app/`). The service indexes curated Bureau of Labor Statistics (BLS) and Glassdoor insights stored in `data/rag_corpus.json` and exposes a `/ask` endpoint plus a minimal chat interface. Each response surfaces bullet-point guidance with explicit source citations, aligning with the transparency requirement for career-planning research.

### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
flask --app app.app run
```

Open <http://127.0.0.1:5000> to chat with the assistant. Provide questions about entry-level salaries, required skills, hiring trends, or interview preparation to receive sourced answers.

## Executive Summary

Our project aimed to develop an intelligent medical chatbot that provides reliable symptom assessment and disease suggestions using advanced Large Language Models (LLMs). The chatbot helps bridge the gap between initial symptom research and professional medical consultation by offering immediate, easy-to-understand information about possible health conditions.

We implemented three distinct technical approaches: a rule-based chatbot using cosine similarity, a RAG (Retrieval Augmented Generation) system with medical domain LLMs, and fine-tuned LLMs using QLoRA. The system processes user-described symptoms and suggests potential conditions while emphasizing the importance of professional medical consultation.

## Problem Understanding

When people feel unwell, they often search online to understand their symptoms but face confusing and sometimes unreliable medical information. Visiting a doctor isn't always immediately possible, especially in remote areas or during off-hours. This creates several challenges:

**Key Pain Points:**
- 🔍 **Information Overload**: Online medical information is overwhelming and often contradictory
- ⏰ **Access Barriers**: Doctor appointments can take days/weeks, emergency rooms are overcrowded
- 🌍 **Geographic Limitations**: Rural areas have limited access to medical specialists
- 💰 **Cost Concerns**: Many avoid doctor visits due to cost, leading to delayed diagnosis

**User Needs:**
- "I have these symptoms - should I be worried?"
- "Is this serious enough to see a doctor?"
- "What could be causing these symptoms?"

Our medical chatbot addresses these needs by providing immediate, reliable symptom assessment while always emphasizing professional medical consultation.

## Data Collection and Pre-processing

**1. Data Collection:**   
The primary dataset was sourced from Kaggle: [Disease Symptom Prediction Dataset](https://www.kaggle.com/datasets/karthikudyawar/disease-symptom-prediction) containing 313 unique disease entries with symptom combinations. The dataset includes comprehensive symptom-disease mappings for common medical conditions.

**2. Data Cleaning & Pre-processing:**  
We implemented extensive data preprocessing:
- **Text Normalization**: Removed underscores from symptom descriptions ("muscle_wasting" → "muscle wasting")
- **Data Transformation**: Combined multiple symptom columns into comma-separated lists
- **Deduplication**: Removed duplicate symptom patterns while preserving disease coverage
- **Formatting**: Structured data for LLM training with proper prompt templates

**3. Data Structure:**
- **Format**: Disease → Multiple symptom combinations (up to 17 symptoms per disease)
- **Size**: 313 unique disease entries
- **Coverage**: Comprehensive common conditions with varied symptom presentations

## The AI Models

### Three Technical Approaches Implemented:

**1. Rule-Based Chatbot (Cosine Similarity)**
- **Technique**: TF-IDF vectorization + Cosine similarity matching
- **Pros**: Fast, interpretable, no training required
- **Cons**: Limited to exact keyword matches, no semantic understanding

**2. RAG System (Retrieval Augmented Generation)**
- **Technique**: Sentence embeddings + Semantic search + LLM generation
- **Models Used**: BioMistral-7B (medical domain LLM), SentenceTransformers
- **Pros**: Context-aware, handles synonyms and semantic meaning
- **Cons**: Computationally intensive, requires GPU resources

**3. Fine-tuned LLM (QLoRA)**
- **Technique**: Parameter-efficient fine-tuning with 4-bit quantization
- **Models**: Llama-2-7B-Chat and BioMistral-7B with QLoRA
- **Pros**: Domain-specific optimization, faster inference
- **Cons**: Training complexity, potential overfitting

### Model Performance Comparison:

| Method | Accuracy | Response Time | Resource Usage |
|--------|----------|---------------|----------------|
| Rule-based | ~65% | <1 second | Low CPU |
| RAG System | ~78% | 5-10 seconds | High GPU |
| Fine-tuned LLM | ~82% | 2-5 seconds | Medium GPU |

## The Chatbot

### Technologies Used

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png" width="80"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/1200px-PyTorch_logo_icon.svg.png" width="80"> <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="80"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Sentence-BERT_logo.png/800px-Sentence-BERT_logo.png" width="80">

**Core Technologies:**
1. **Python & Jupyter Notebooks** - Development environment
2. **PyTorch** - Deep learning framework
3. **Hugging Face Transformers** - Pre-trained models and tokenizers
4. **Sentence Transformers** - Text embeddings and semantic search
5. **PEFT (Parameter-Efficient Fine-Tuning)** - QLoRA implementation
6. **BitsAndBytes** - 4-bit quantization for memory efficiency

### Key Features

**User Interface:**
- Simple text-based interaction
- Natural language symptom input
- Top 2-3 disease suggestions with confidence indicators
- Clear medical disclaimers on every response

**Safety Measures:**
- ⚠️ **Medical Disclaimers**: Every response includes "This is not a medical diagnosis"
- ⚠️ **Professional Consultation**: Always recommends seeing a licensed physician
- ⚠️ **Emergency Detection**: Flags symptoms requiring immediate care

### Sample Interaction
```
User: "I have fever, cough, and headache"
Bot: "Based on your symptoms, possible conditions:
     1. Common Cold - fever, cough, headache, runny nose
     2. COVID-19 - fever, cough, headache, fatigue
     
     Note: This is not a medical diagnosis. Always consult a licensed physician."
```

## Getting Started

### Prerequisites
- Kaggle account with GPU access
- Basic Python environment

### Installation & Setup

1. **Access the Kaggle Notebook**
   ```bash
   # The project is designed to run in Kaggle environment
   # Navigate to your Kaggle notebook
   ```

2. **Add Required Dataset**
   ```python
   # In Kaggle, add the dataset via:
   # Input → Add Input → Search: "disease-symptom-prediction"
   ```

3. **Run the Notebook**
   ```python
   # Execute cells sequentially from top to bottom
   # The notebook handles all dependency installations
   ```

### Configuration

The chatbot supports three operational modes:
1. **Rule-based** - Fast, basic symptom matching
2. **RAG System** - Advanced semantic understanding  
3. **Fine-tuned LLM** - Domain-optimized responses

### Usage

Run the chatbot by executing the final cell:
```python
chatbot()
```

**Example Usage Patterns:**
- "fever and sore throat"
- "headache, nausea, dizziness"
- "chest pain and shortness of breath"

## Recommended Next Steps

**Short-term Enhancements** (Next 3 months):
1. **Symptom Severity Scoring**: Implement severity assessment for triage recommendations
2. **Emergency Detection**: Add flags for critical symptoms requiring immediate care
3. **Multi-language Support**: Expand beyond English to serve diverse populations
4. **Expanded Database**: Incorporate rare conditions and specialized medical knowledge

**Medium-term Goals** (6-12 months):
1. **Mobile Application**: Develop dedicated mobile app for better accessibility
2. **Telehealth Integration**: Connect with existing telehealth platforms
3. **Personal Context**: Incorporate user medical history (with proper privacy safeguards)
4. **Drug Interaction Checking**: Add medication and interaction warnings

**Long-term Vision** (1-2 years):
1. **Regulatory Compliance**: Pursue FDA approval as a medical device
2. **Healthcare Partnerships**: Collaborate with insurance companies and healthcare providers
3. **Global Deployment**: Adapt for different healthcare systems and regional guidelines
4. **Continuous Learning**: Implement approved medical source integration

## Risk Considerations

**Medical Safety Risks:**
- 🚨 **Misdiagnosis Potential**: AI may suggest incorrect conditions
- 🚨 **False Reassurance**: May underestimate serious symptoms
- 🚨 **Legal Liability**: Responsibility boundaries for medical advice

**Mitigation Strategies:**
- ✅ **Clear Disclaimers**: Every response includes "not a medical diagnosis"
- ✅ **Professional Emphasis**: Always recommend doctor consultation
- ✅ **Emergency Flags**: Detect and highlight critical symptoms
- ✅ **Accuracy Monitoring**: Regular validation against medical databases

**Technical Risks:**
- 🔧 **Model Hallucination**: LLMs may generate incorrect information
- 🔧 **Data Bias**: Training data may underrepresent rare conditions
- 🔧 **Resource Requirements**: High computational needs for advanced models

**Privacy & Compliance:**
- 🔒 **Data Anonymization**: No personal health information stored
- 🔒 **Transparent Processing**: Clear data usage and privacy policies
- 🔒 **Medical Ethics**: Designed with healthcare best practices in mind

---

## ⚠️ Important Medical Disclaimer

**This chatbot is for educational and informational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

**In case of emergency symptoms** (chest pain, difficulty breathing, severe bleeding, etc.), **call emergency services immediately** or go to the nearest emergency room.

## 📞 Emergency Resources
- **Emergency Services**: 911 (US) or your local emergency number
- **Poison Control**: 1-800-222-1222 (US)
- **Crisis Hotline**: 988 (US Suicide and Crisis Lifeline)

---

## 🔗 Useful Links
- [Kaggle Dataset](https://www.kaggle.com/datasets/karthikudyawar/disease-symptom-prediction)
- [Hugging Face Models](https://huggingface.co)
- [Project Documentation]([(https://www.kaggle.com/code/chipojaya/medical-chatbot?scriptVersionId=266207557))

## 👥 Contributors
- Chipo Jaya

## 📄 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
```
