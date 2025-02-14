# 📌 Hackathon Project: GenAI-Powered Document Intelligence

🚀 **Goal:**  
This project aims to **improve public service efficiency** by using **GenAI-powered document processing**. The system can:
- **Extract** and **clean** text/images from documents (PDFs).
- **Retrieve** relevant information using **RAG (Retrieval-Augmented Generation)**.
- **Enable voice interaction** (speech-to-text & text-to-speech).
- **Describe images** for accessibility (blind users).
- **Allow form auto-filling** based on text/voice input.

---

## ⚙️ Configuration (`config.py`)

The `config.py` file contains **all the key parameters** for the pipeline, ensuring flexibility and easy adjustments.  

### 🔹 **What Does It Contain?**
| **Variable** | **Purpose** |
|-------------|------------|
| `TEXT_EXTRACTION_MODEL`    | Specifies the **OCR model** used to extract text from PDFs. |
| `IMAGE_EXTRACTION_MODEL`   | Defines the **image extraction model** for detecting figures (if applicable). |
| `EMBEDDINGS_MODEL`         | Specifies the **embeddings model** for vector storage and retrieval (e.g., `intfloat/multilingual-e5-small`). |
| `RERANKING_MODEL`          | Defines the **reranking model** used to refine search results. |
| `CHAT_MODEL`               | Specifies the **LLM** for response generation (e.g., `AgentPublic/guillaumetell-7b`). |
| `MULTIMODAL_MODEL`         | Used for **image-to-text descriptions** (e.g., `deepseek-ai/Janus-Pro-7B`). |
| `SPEECH_TO_TEXT_MODEL`     | Defines the **ASR (automatic speech recognition) model** for voice input (e.g., `Whisper`). |
| `TEXT_TO_SPEECH_MODEL`     | Defines the **text-to-speech model** for AI-generated voice responses. |
| `CHUNK_SIZE` | Determines the **chunking strategy** (number of tokens per chunk). |
| `OVERLAP_SIZE` | Defines the **overlap between chunks** to preserve context in retrieval. |
| `FAISS_INDEX_PATH` | Specifies the **path where the FAISS vector store is saved**. |
| `ENABLE_IMAGE_RETRIEVAL` | Enables/disables **retrieval of image descriptions** in search. |
| `ENABLE_TEXT_TO_SPEECH` | Enables/disables **AI-generated voice responses**. |

### 🔹 **Example `config.py` File**
```python
TEXT_EXTRACTION_MODEL = "stepfun-ai/GOT-OCR-2.0-hf"
IMAGE_EXTRACTION_MODEL = ""
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-small"
RERANKING_MODEL = "BAAI/bge-reranker-v2-m3"
CHAT_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
MULTIMODAL_MODEL = "deepseek-ai/Janus-Pro-7B"
SPEECH_TO_TEXT_MODEL = "openai/whisper-large-v3-turbo"
TEXT_TO_SPEECH_MODEL = "hexgrad/Kokoro-82M"

CHUNK_SIZE = 512
OVERLAP_SIZE = 128

FAISS_INDEX_PATH = "vector_store/index.faiss"

ENABLE_IMAGE_RETRIEVAL = True
ENABLE_TEXT_TO_SPEECH = False
```

---

## 📂 Repository Structure

### 1️⃣ **Core Scripts (`/scripts/`)**
| **File**                  | **Purpose** |
|---------------------------|------------|
| `extractor.py`            | Extracts **text and images** from PDFs. |
| `text_cleaner.py`         | Cleans extracted text (removes artifacts, fixes spacing). |
| `doc_splitter.py`         | Chunks long documents for better embedding retrieval. |
| `embeddings_db.py`        | Stores **normalized text & image embeddings** in **FAISS**. |
| `search_engine.py`        | Performs **semantic search** in FAISS (retrieves text/images). |
| `chat_interaction.py`     | Uses **LLM (guillaumetell-7b)** to generate answers based on retrieved context. |
| `speech_to_text.py`       | Converts **voice input** to text using **Whisper**. |
| `text_to_speech.py`       | Converts AI-generated text **to speech** for accessibility. |
| `image_description.py`    | Uses **Janus-Pro-7B** to generate **textual descriptions of images** for indexing. |

---

### 2️⃣ **Pipeline Files**
| **File**         | **Purpose** |
|------------------|------------|
| `preprocess.py`  | Runs **document preprocessing** (extracts, cleans, chunks, and stores embeddings). |
| `main.py`        | Runs **inference** (retrieves text/images, generates answers, and optionally speaks responses). |
| `fine_tune.py`   | Fine-tunes embeddings or LLM if needed. |

---

### 3️⃣ **Utility Functions (`/utils/`)**
| **File**        | **Purpose** |
|-----------------|------------|
| `io_utils.py`   | Handles **file I/O** (saving/loading text, embeddings, logs). |
| `api_utils.py`  | Manages **API calls** (retry logic, error handling). |
| `text_utils.py` | Extra **text processing functions** (regex, formatting). |

---

### 4️⃣ **Directories**
| **Directory**    | **Purpose** |
|------------------|------------|
| `data/`         | Stores **raw PDFs and documents** to be processed. |
| `output/`       | Stores **processed text and descriptions** (cleaned text, extracted figures). |
| `vector_store/` | Stores **FAISS vector database** for fast semantic retrieval. |
| `notebooks/`    | Contains **Jupyter notebooks** for testing and debugging. |

---

## 🛠️ Installation

### 1️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed. Then, install the required dependencies.
```bash
pip install -r requirements.txt
```

If `FAISS` fails to install via `pip`, you can install it using `conda`:
```bash
conda install -c conda-forge faiss-cpu
```

---

## 🚀 Usage

### 1️⃣ **Preprocess Documents** (Run once to extract and store embeddings)

Run the preprocess.py file.

This step:
- Extracts text and images from PDFs.
- Cleans and chunks the text.
- Stores **text and image embeddings** in FAISS.

### 2️⃣ **Run Inference (Retrieve & Answer Questions)**
```bash
python main.py
```
This retrieves **relevant text and images** and generates an AI response.
- You can enable **text-to-speech** for spoken responses.
- Image retrieval is optional.

### 3️⃣ **Enable Form Auto-Filling (Voice or Text)**
```bash
python main.py --auto_fill
```
This feature allows users to **fill forms automatically** using either **voice input** or **typed text**.

---

## 💡 How Does the System Work?

### 🔹 **1. Document Processing**
- Extracts **text and images** from PDFs.
- Cleans and **chunks text** for efficient retrieval.
- Stores **embeddings in FAISS** for fast semantic search.

### 🔹 **2. Retrieval-Augmented Generation (RAG)**
- Finds the **most relevant document snippets** using **vector search**.
- Uses an **LLM to generate an answer** based on retrieved context.
- Can return **image descriptions** for accessibility.

### 🔹 **3. Voice & Accessibility Features**
- Supports **speech-to-text** (voice input) using **Whisper**.
- Converts AI-generated responses **to speech** for accessibility.
- Uses **multimodal AI** to describe **figures/images** for blind users.

---

## 🎯 Hackathon Focus Areas
- 📑 **Automated document understanding**
- 🔎 **AI-powered search (RAG)**
- 🗣️ **Voice-enabled interaction**
- 🖼️ **Image accessibility (textual descriptions for blind users)**

---

## 📢 Acknowledgments
Built for **[Hackathon Name]**, focused on **GenAI-powered public service improvements**.

---

**🚀 Let's build AI for efficiency and accessibility!**  

