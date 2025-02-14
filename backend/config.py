'''Centralized configuration for models, chunking strategy, API keys'''

# Possible models (prioritize smaller ones)
# https://huggingface.co/spaces/mteb/leaderboard
# embeddings: intfloat/multilingual-e5-small, sentence-transformers/all-MiniLM-L12-v2
# reranking: BAAI/bge-reranker-v2-m3
# chat: AgentPublic/guillaumetell-7b, mistralai/Mistral-Small-24B-Instruct-2501
# multimodal: deepseek-ai/Janus-Pro-7B
# speech-to-text: openai/whisper-large-v3-turbo
# text-to-speech: hexgrad/Kokoro-82M, parler-tts/parler-tts-mini-v1, fishaudio/fish-speech-1.5, WhisperSpeech/WhisperSpeech


TEXT_EXTRACTION_MODEL = "stepfun-ai/GOT-OCR-2.0-hf"
IMAGE_EXTRACTION_MODEL = ""
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-large"
RERANKING_MODEL = "BAAI/bge-reranker-v2-m3"
CHAT_MODEL = "AgentPublic/llama3-instruct-8b"
MULTIMODAL_MODEL = ""
SPEECH_TO_TEXT_MODEL = "openai/whisper-large-v3"
TEXT_TO_SPEECH_MODEL = "hexgrad/Kokoro-82M"


SERVER_URL = "http://yourserver:port/v0/"

CHUNK_SIZE = 512
OVERLAP_SIZE = 128

FAISS_INDEX_PATH = "vector_store/index.faiss"

ENABLE_IMAGE_RETRIEVAL = True
ENABLE_TEXT_TO_SPEECH = False
ENABLE_AGENT = True

