# Company Policy RAG Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about company policies and receive answers grounded in the relevant policy documents.

The project demonstrates a practical GenAI pipeline using **Python, PDF processing, text chunking, transformer embeddings, cosine similarity, and Google Gemini**.

---

## 🚀 Project Overview

Employees often need to search through multiple company policy documents to find answers to questions about benefits, vacation, expenses, remote work, and workplace conduct.

This project builds a **Company Policy RAG Assistant** that retrieves relevant information from policy documents before asking an LLM to generate the final answer.

Instead of sending the entire document collection to the LLM, the application:

1. Loads policy PDFs
2. Extracts the text
3. Splits the text into smaller chunks
4. Converts chunks into numerical embeddings
5. Finds the most relevant chunks for a user's question
6. Sends the retrieved context to Gemini
7. Generates an answer based on the retrieved policy information

### Architecture

```text
                  ┌──────────────────────┐
                  │   Company Policies   │
                  │       PDF Files      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    PDF Extraction    │
                  │        pypdf         │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      Chunking        │
                  │  Smaller text units  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │     Embeddings       │
                  │ all-MiniLM-L6-v2    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Vector Similarity   │
                  │ Cosine Similarity    │
                  └──────────┬───────────┘
                             ▲
                             │
                    User Question
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Retrieve Top Chunks  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Gemini LLM        │
                  │ Context + Question   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Grounded Answer    │
                  └──────────────────────┘
```

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with a Large Language Model (LLM).

Instead of asking an LLM to answer a question using only its general knowledge, the system first retrieves relevant information from a trusted document collection.

The retrieved information is then provided to the LLM as context.

```text
User Question
      ↓
Retrieve Relevant Information
      ↓
Provide Context to LLM
      ↓
Generate Answer
```

This approach is useful for company knowledge bases, policies, procedures, technical documentation, and internal support systems.

---

## 📚 Documents Used

The project currently uses five sample company policy documents:

* `remote_work_policy.pdf`
* `vacation_policy.pdf`
* `expense_policy.pdf`
* `benefits_policy.pdf`
* `code_of_conduct.pdf`

These documents are included as sample/demo data for the project.

---

## 🛠️ Technology Stack

| Technology            | Purpose                           |
| --------------------- | --------------------------------- |
| Python                | Application development           |
| Google Colab          | Development environment           |
| pypdf                 | PDF text extraction               |
| Sentence Transformers | Text embeddings                   |
| `all-MiniLM-L6-v2`    | Embedding model                   |
| scikit-learn          | Cosine similarity search          |
| NumPy                 | Numerical operations              |
| Google Gemini         | LLM response generation           |
| Git                   | Version control                   |
| GitHub                | Source code and portfolio hosting |

---

## 🔄 RAG Pipeline

### 1. PDF Document Loading

The application reads multiple PDF files using `pypdf`.

```python
from pypdf import PdfReader

reader = PdfReader(pdf_path)

for page in reader.pages:
    text = page.extract_text()
```

Each document is processed page by page.

Metadata such as the document name and page number is preserved so that retrieved information can be traced back to its source.

---

### 2. Text Chunking

Large documents are divided into smaller pieces called **chunks**.

Example:

```text
Policy Document
       ↓
Page Text
       ↓
Smaller Text Chunks
       ↓
Embeddings
```

Chunking makes semantic retrieval more effective because the system can retrieve a relevant section instead of an entire document.

---

### 3. Creating Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

The model converts text into numerical vectors.

For example:

```text
"Employees become eligible for benefits after 90 days."

                ↓

        [0.021, -0.184, 0.093, ...]
```

The resulting vectors contain **384 dimensions**.

Semantically similar pieces of text tend to have similar vector representations.

---

### 4. Similarity Search

When a user asks a question, the question is also converted into an embedding.

The system compares the question embedding against the document embeddings using **cosine similarity**.

Conceptually:

```text
User Question
      ↓
Question Embedding
      ↓
Compare with Document Embeddings
      ↓
Calculate Similarity Scores
      ↓
Select Top-K Relevant Chunks
```

Higher similarity means the retrieved text is more semantically relevant to the question.

---

### 5. Retrieval

The highest-scoring chunks are selected and passed to the LLM as context.

For example:

```text
Question:
"Who is eligible for company benefits?"

Retrieved Context:
"Full-time employees become eligible for benefits
after completing 90 days of employment."

        ↓

Gemini
        ↓

Answer:
"Full-time employees become eligible for benefits
after 90 days of employment."
```

---

### 6. LLM Generation

The retrieved context and user question are sent to the Gemini model.

The LLM generates a natural-language answer using the retrieved policy information.

This reduces the need for the model to rely solely on its general knowledge.

---

## 🔐 API Key Security

The Gemini API key is **not stored in the notebook or GitHub repository**.

The notebook prompts the user for the API key at runtime:

```python
import os
from getpass import getpass

os.environ["GEMINI_API_KEY"] = getpass(
    "Enter your Gemini API key: "
)
```

The `.gitignore` file also prevents environment files and other sensitive/local files from being committed.

**Never commit API keys, passwords, tokens, or other secrets to GitHub.**

---

## ▶️ How to Run the Project

### Option 1: Google Colab

The easiest way to run the project is using Google Colab.

1. Open the notebook:

```text
notebooks/company_policy_rag.ipynb
```

2. Upload/open the required policy PDFs.
3. Install the required Python packages.
4. Run the notebook cells in order.
5. Enter your Gemini API key when prompted.
6. Ask questions about the company policies.

---

### Option 2: Local Python Environment

Clone the repository:

```bash
git clone git@github.com:geetatawniya/company-policy-rag-assistant.git
```

Move into the project:

```bash
cd company-policy-rag-assistant
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```text
notebooks/company_policy_rag.ipynb
```

---

## 💬 Example Questions

The assistant can answer questions such as:

```text
What are the requirements for remote work?

How many vacation days are employees entitled to?

When do employees become eligible for benefits?

What expenses can employees submit for reimbursement?

What are the company's expectations regarding workplace conduct?
```

---

## 📊 Example RAG Flow

A question such as:

> "When do employees become eligible for benefits?"

is converted into an embedding and compared with the embeddings of the policy chunks.

The system retrieves the most relevant section from:

```text
benefits_policy.pdf
```

The retrieved context is then provided to Gemini to generate the final answer.

---

## 📁 Project Structure

```text
company-policy-rag-assistant/
│
├── .gitignore
├── requirements.txt
│
├── benefits_policy.pdf
├── code_of_conduct.pdf
├── expense_policy.pdf
├── remote_work_policy.pdf
├── vacation_policy.pdf
│
└── notebooks/
    └── company_policy_rag.ipynb
```

---

## 🎯 Skills Demonstrated

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* Generative AI
* Natural Language Processing
* Semantic Search
* Text Embeddings
* Vector Similarity
* Cosine Similarity
* Document Processing
* PDF Text Extraction
* Text Chunking
* Python
* API Integration
* Git
* GitHub
* Data Pipeline Design

---

## 🔍 Current Implementation

The current version uses:

```text
Sentence Transformers
        +
In-memory embeddings
        +
Cosine similarity
        +
Gemini
```

The embeddings are currently kept in memory during execution rather than being stored in a dedicated vector database.

---

## 🚀 Future Enhancements

Planned improvements include:

### 1. FAISS Vector Index

Replace the current in-memory similarity search with a FAISS vector index for more scalable vector retrieval.

```text
Documents
    ↓
Embeddings
    ↓
FAISS Index
    ↓
Similarity Search
    ↓
Top-K Chunks
```

### 2. Dedicated Vector Database

Evaluate vector databases such as:

* Chroma
* Pinecone
* Qdrant
* Weaviate

### 3. Better Chunking

Implement configurable:

* chunk size
* chunk overlap
* sentence-aware splitting
* paragraph-aware splitting

### 4. Source Citations

Return the source document and page number with every generated answer.

Example:

```text
Answer:
Full-time employees become eligible for benefits
after 90 days.

Source:
benefits_policy.pdf, Page 1
```

### 5. Web Application

Build a user interface using:

* Streamlit
* FastAPI
* React

### 6. Evaluation

Add RAG evaluation metrics to measure:

* Retrieval relevance
* Answer relevance
* Faithfulness
* Context precision
* Context recall

---

## 🧩 Key Learning

One of the main lessons from this project is that a RAG system is not simply an LLM application.

A complete RAG pipeline involves multiple stages:

```text
Document Ingestion
       ↓
Text Extraction
       ↓
Chunking
       ↓
Embedding Generation
       ↓
Vector Search
       ↓
Context Retrieval
       ↓
Prompt Construction
       ↓
LLM Generation
       ↓
Final Answer
```

Each stage affects the quality of the final response.

---

## 👩‍💻 About the Project

This project was created as a hands-on learning and portfolio project to demonstrate practical experience with **Data Analytics, Python, Generative AI, RAG, semantic search, and modern data/AI engineering concepts**.

It is designed to demonstrate how traditional data-processing skills can be extended into modern AI-powered applications.

---

## 📌 Repository

**GitHub:**
`geetatawniya/company-policy-rag-assistant`

---

## ⭐ If you find this project useful

Feel free to explore the notebook, experiment with different policy documents, and extend the retrieval pipeline with a vector database or web interface.
