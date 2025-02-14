
<p align="center">
  <img src="./figures/defacto_logo.png" alt="De facto" />
</p>

---

🚀 **Goal:**  
This project aims to **improve public service efficiency** by using **GenAI-powered document processing**. The system can:
- **Extract** and **clean** text/images from documents (PDFs).
- **Retrieve** relevant information using **RAG (Retrieval-Augmented Generation)**.
- **Enable voice interaction** for visually impaired users.
- **Describe images** for accessibility.

**De Facto** drastically reduces magistrates' factual review time of criminal cases, thereby enhancing the efficiency and quality of justice for our fellow citizens. Its generic backend is adaptable across the entire public sector and includes a voice interaction mode for visually impaired agents.

💡 A sovereign, trustworthy, and efficient AI, leveraging frugal open-source models! 

---

## Backend

### ⚙️ Configuration (`config.py`)

The `config.py` file contains **all the key parameters** for the pipeline, ensuring flexibility and easy adjustments.  Note that all the parameters are not used in the MVP version.

#### 🔹 **What Does It Contain?**
| **Variable**             | **Purpose**                                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------------|
| `TEXT_EXTRACTION_MODEL`  | Specifies the **OCR model** used to extract text from PDFs.                                                   |
| `IMAGE_EXTRACTION_MODEL` | Defines the **image extraction model** for detecting complex layout.                                          |
| `EMBEDDINGS_MODEL`       | Specifies the **embeddings model** for vector storage and retrieval (e.g., `intfloat/multilingual-e5-small`). |
| `RERANKING_MODEL`        | Defines the **reranking model** used to refine search results.                                                |
| `CHAT_MODEL`             | Specifies the **LLM** for response generation (e.g., `AgentPublic/guillaumetell-7b`).                         |
| `MULTIMODAL_MODEL`       | Used for **image-to-text descriptions** (e.g., `deepseek-ai/Janus-Pro-7B`).                                   |
| `SPEECH_TO_TEXT_MODEL`   | Defines the **ASR (automatic speech recognition) model** for voice input (e.g., `Whisper`).                   |
| `TEXT_TO_SPEECH_MODEL`   | Defines the **text-to-speech model** for AI-generated voice responses.                                        |
| `SERVER_URL`             | The **server URL** where the models are deployed (in addition to those served by the Albert API).             |
| `CHUNK_SIZE`             | Determines the **chunking strategy** (number of tokens per chunk).                                            |
| `OVERLAP_SIZE`           | Defines the **overlap between chunks** to preserve context in retrieval.                                      |
| `FAISS_INDEX_PATH`       | Specifies the **path where the FAISS vector store is saved**.                                                 |
| `ENABLE_IMAGE_RETRIEVAL` | Enables/disables **retrieval of image descriptions** in search.                                               |
| `ENABLE_TEXT_TO_SPEECH`  | Enables/disables **AI-generated voice responses**.                                                            |

#### 🔹 **Example `config.py` File**
```python
TEXT_EXTRACTION_MODEL = "stepfun-ai/GOT-OCR-2.0-hf"
IMAGE_EXTRACTION_MODEL = "vidore/colpali-v1.3"
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-small"
RERANKING_MODEL = "BAAI/bge-reranker-v2-m3"
CHAT_MODEL = "AgentPublic/llama3-instruct-8b"
MULTIMODAL_MODEL = "mistralai/Pixtral-12B-2409"
SPEECH_TO_TEXT_MODEL = "openai/whisper-large-v3"
TEXT_TO_SPEECH_MODEL = "hexgrad/Kokoro-82M"

SERVER_URL = "http://yourserver:port/v0/"

CHUNK_SIZE = 512
OVERLAP_SIZE = 128

FAISS_INDEX_PATH = "vector_store/index.faiss"

ENABLE_IMAGE_RETRIEVAL = True
ENABLE_TEXT_TO_SPEECH = False
```

---

### 📂 Repository Structure

#### 1️⃣ **Core Scripts (`/scripts/`)**
| **File**                  | **Purpose** |
|---------------------------|------------|
| `agent.py`       | Main AI agent in charge of **orchestrating tools** based on the user query understanding. |
| `embeddings_db.py`    | Stores **normalized text & image embeddings** in **vector store**. |
| `fine-tune.py`    | Used to **fine-tune the backend models** if needed (not implemented in MVP). |

---

#### 2️⃣ **Pipeline Files**
| **File**         | **Purpose** |
|------------------|------------|
| `preprocess.py`  | Runs **document preprocessing** (extracts, cleans, chunks, and stores embeddings). |
| `main.py`        | Runs **inference** (retrieves text/images, generates answers, and optionally speaks responses). |

---

#### 3️⃣ **Utility Functions (`/utils/`)**
| **File**        | **Purpose** |
|-----------------|------------|
| `api_utils.py`  | Manages **API calls** (retry logic, error handling). |
| `text_utils.py` | Extra **text processing functions** (regex, formatting). |

---

#### 4️⃣ **Directories**
| **Directory**    | **Purpose** |
|------------------|------------|
| `data/`         | Stores **raw PDFs and documents** to be processed. |
| `output/`       | Stores **processed text and descriptions** (cleaned text, extracted figures). |
| `vector_store/` | Stores **vector database** for fast semantic retrieval. |

---

### 🛠️ Installation

#### 1️⃣ Build images

- OCR API
```bash
cd ocr && docker build -t got:v0.1.0 .
```
- QA API
```bash
cd qa && docker build -t qa:v0.1.0 .
```

#### 2️⃣ Deploy

```bash
docker-compose up -d 
```

Export endpoints:
```bash
export QA_URL=http://localhost:7042/v0/ask
export GOT_URL=http://localhost:7041/v0/extract
```

### 2️⃣ Lauch Defacto
- Install poetry
```bash
curl -sSL https://install.python-poetry.org | python3 - 
export PATH="/root/.local/bin:$PATH"
```
- Install Defacto
```bash
poetry install
```

- Launch Defacto
```bash
cd frontend && python ui.py
```

Open `localhost:7860` in your web browser to interact with Defacto assistant.

### 3. TODO

##### Colpali: deploy the Colpali retriever
```bash
cd colpali
docker build -t colpali:v0.1.0 .
docker save colpali:v0.1.0 -o colpali_v0.1.0.tar
scp colpali_v0.1.0.tar user@your-server:/path_in_your_server
cd /path_in_your_server
docker load -i colpali_v0.1.0.tar
docker-compose up -d colpali
```
Then, in the code:
* Modify the url to embed all the dataset with `http://yourserver:port/v0/visual_embed`

* Modify the url to return the oracle page based on a query with `http://yourserver:port/v0/search`

---

### 🚀 Usage

#### **Run Inference (Document Retrieval & Questions Answering)**
```bash
python main.py
```

---

## Frontend

### 🛠️ Installation

#### 1️⃣ Install Python3 (skip if backend installed)
Ensure you have [Python 3.8+](https://docs.python-guide.org/starting/install3/linux/) installed. To install it, please run:
- `apt update`
- `apt install python3`
- `apt install python3-pip`

_Note: you may need to run above commands with root access (then prefix above commands with `sudo`)._

#### 2️⃣ Virtual environment

It is advised to use a python [virtualenv](https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv) to manage dependencies. To set it up, please run:

```bash
pip install virtualenv
```

Then you will create a virtual environment to sandbox your dependencies installation to the **defacto** project:

```bash
python3 -m venv virtualenv
```

Then you can enter this virtualenv:

```bash
source virtualenv/bin/activate
```

_Note: at this point your CLI prompt will be prefixed with `(virtualenv)` string._

#### 3️⃣ Install Dependencies

When your virtual environment is up, you can install dependencies from the **frontend** folder via:

```bash
pip install -r requirements.txt
```

The dependencies and their versions are listed in the `frontend/requirements.txt` file.

To quit the virtualenv, simply enter `deactivate` in the CLI. The CLI will not be prefixed with `(virtualenv)` anymore.

_Note: Do not commit virtualenv directory as it can be generated with requirements.txt and instructions from this readme (moreover "virtualenv" line must have been added to your `.gitignore` file)._

### 🚀 Usage

- Activate the virtual env: `source virtualenv/bin/activate`
- Run the front file: `python3 ui.py`.
- To quit the python process, you can Ctrl+C on the CLI.
- To quit the virtualenv, run `deactivate` command in the CLI. The CLI will not be prefixed with `(virtualenv)` anymore.

---

## 💡 How Does the System Work?
The features included in the MVP (Most Valuable Product) are highlighted.

### 🔹 **1. Document Processing**
- Extracts **text and images** from PDFs (either text-based or image-based PDFs). **MVP**
- Extract text through OCR. **MVP**
- Apply heuristics to split the text into consistent documents types. **MVP**
- Cleans and **chunks text** for efficient retrieval.
- Stores **embeddings in FAISS** for fast semantic search.

### 🔹 **2. Context-only Prompting**
- Summarizes the document automatically by prompting a LLM. Depending on the model used, the whole document text is passed in context or a map-reduce strategy applies. **MVP**
- Generate answers to user queries. This can be done through context-only prompting (MVP) or RAG.

### 🔹 **3. Retrieval-Augmented Generation (RAG)**
- Finds the **most relevant document snippets** using **vector search**.
- Uses an **LLM to generate an answer** based on retrieved context.
- Can return **image descriptions** for accessibility.

### 🔹 **4. Voice & Accessibility Features**
- Supports **speech-to-text** (voice input) using **Whisper**.
- Converts AI-generated responses **to speech** for accessibility. **MVP**
- Uses **multimodal AI** to describe **figures/images** for blind users.

---


## 📢 Acknowledgments
Built for **GenAI for Public Good**, focused on **GenAI-powered public service improvements**.

---

**🚀 Let's build AI for efficiency and accessibility!**  
