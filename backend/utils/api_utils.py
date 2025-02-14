"""API calls."""

import base64
import io
import json
import os
import numpy as np

import requests
import soundfile as sf
from kokoro import KPipeline
from openai import OpenAI

# Adjust these imports or constants as needed:
from backend.config import (
    CHAT_MODEL,
    EMBEDDINGS_MODEL,
    MULTIMODAL_MODEL,
    RERANKING_MODEL,
    SPEECH_TO_TEXT_MODEL,
    TEXT_TO_SPEECH_MODEL,
    SERVER_URL,
)


###############################################################################
# Class 1: AlbertAPI
#  - Contains only methods Albert supports (chat, embeddings, rerank, STT).
#  - Removed text_to_speech() and describe_image(), since Albert doesn't have them.
###############################################################################
class AlbertAPI:  # noqa: D101
    def __init__(self, base_url, api_key=None):
        """Initialize the AlbertAPI client.

        :param base_url: The base URL of the Albert API.
        :param api_key: Optional API key for authentication.
        """
        self.base_url = base_url
        # Default to JSON headers, but we remove/override this for multipart (audio).
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def list_models(self):
        """Retrieve the list of available models from the Albert API.

        :return: JSON response containing the list of models.
        """
        endpoint = f"{self.base_url}/v1/models"
        response = requests.get(endpoint, headers=self.headers)
        if response.status_code == 200:
            models = response.json()
            return [{"id": model["id"], "type": model["type"]} for model in models.get("data", [])]
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    def create_chat_completion(self, messages, max_tokens=50, temperature=1.0, stream=False):
        """Generate a chat completion using the specified messages.

        :param messages: A list of message dicts, each containing 'role' and 'content'.
        :param max_tokens: The maximum number of tokens to generate.
        :param temperature: Sampling temperature.
        :param stream: Whether to stream the response (default: False).
        :return: Generated chat completion (string) or a generator if streaming.
        """
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": CHAT_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        response = requests.post(endpoint, headers=self.headers, json=payload, stream=stream)
        if stream:
            return self._handle_streaming_response(response)
        else:
            return self._extract_message_content(response)

    def create_embedding(self, input_text):
        """Generate embeddings for the given input text using the EMBEDDINGS_MODEL.

        :param input_text: The input text to embed.
        :return: Embeddings vector (JSON response).
        """
        endpoint = f"{self.base_url}/v1/embeddings"
        payload = {"model": EMBEDDINGS_MODEL, "input": input_text}
        response = requests.post(endpoint, headers=self.headers, data=json.dumps(payload))
        return self._handle_response(response)

    def rerank(self, query, documents):
        """Rerank a list of documents based on their relevance to the query.

        :param query: The search query.
        :param documents: A list of document strings to rerank.
        :return: Reranked list of documents with relevance scores (JSON response).
        """
        endpoint = f"{self.base_url}/v1/rerank"
        payload = {"model": RERANKING_MODEL, "prompt": query, "input": documents}
        response = requests.post(endpoint, headers=self.headers, data=json.dumps(payload))
        return self._handle_response(response)

    def transcribe_audio(self, audio_file_path, language="fr", response_format="json", temperature=0.5):
        """Transcribe audio using the speech-to-text model.

        :param audio_file_path: Path to the local .wav file (or another format).
        :param language: Language of the audio (default "fr").
        :param response_format: Format of the output (JSON, etc.).
        :param temperature: Sampling temperature.
        :return: Transcription response (JSON).
        """
        endpoint = f"{self.base_url}/v1/audio/transcriptions"

        # Copy headers and remove any forced "Content-Type"
        headers = dict(self.headers)
        if "Content-Type" in headers:
            del headers["Content-Type"]

        data = {
            "model": SPEECH_TO_TEXT_MODEL,
            "language": language,
            "response_format": response_format,
            "temperature": temperature,
        }

        with open(audio_file_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(audio_file_path), audio_file, "audio/wav"),
            }
            response = requests.post(endpoint, headers=headers, data=data, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    # def describe_image(self, image_path):
    #     """
    #     Generate a textual description for an image using the other provider.
    #     """
    #     endpoint = f"{self.base_url}/v1/multimodal/describe"
    #     headers = dict(self.headers)
    #     if "Content-Type" in headers:
    #         del headers["Content-Type"]
    #
    #     with open(image_path, 'rb') as image_file:
    #         files = {'file': image_file}
    #         data = {"model": MULTIMODAL_MODEL}
    #         response = requests.post(endpoint, headers=headers, data=data, files=files)
    #
    #     if response.status_code == 200:
    #         return response.json()
    #     else:
    #         raise Exception(f"OtherAPI image description failed: {response.status_code} {response.text}")

    # -------------------------------------------------------------------------
    # Internal helper methods (unchanged):
    # -------------------------------------------------------------------------
    def _handle_streaming_response(self, response):
        if response.status_code == 200:
            for chunk in response.iter_lines():
                if chunk:  # Ignore empty lines
                    try:
                        parsed_chunk = json.loads(chunk.decode("utf-8"))
                        if "choices" in parsed_chunk and len(parsed_chunk["choices"]) > 0:
                            yield parsed_chunk["choices"][0]["message"]["content"]
                    except json.JSONDecodeError:
                        continue  # Ignore invalid JSON chunks
        else:
            raise Exception(f"API streaming request failed with status code {response.status_code}: {response.text}")

    def _handle_response(self, response):
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    def _extract_message_content(self, response):
        if response.status_code == 200:
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"]
            else:
                return "No response generated."
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")


###############################################################################
# Class 2: OtherAPI
#  - Has the same interface as AlbertAPI,
#    PLUS text-to-speech and image description methods that Albert lacks.
#  - Replace endpoints below with TTS / multimodal alternative API.
###############################################################################
class OtherAPI:  # noqa: D101
    def __init__(self, base_url, api_key=None):
        """Initialize the other (non-Albert) API client.

        :param base_url: The base URL of the other provider (which may support TTS, multimodal).
        :param api_key: Optional API key for authentication.
        """
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    # -------------------------------------------------------------------------
    # Replicated Methods (similar signature)
    # -------------------------------------------------------------------------
    def list_models(self):
        """Possibly retrieve the list of available models from the other API, or return a placeholder if not supported."""
        endpoint = f"{self.base_url}/v1/models"
        response = requests.get(endpoint, headers=self.headers)
        if response.status_code == 200:
            models = response.json()
            return models.get("data", [])
        else:
            raise Exception(f"OtherAPI request failed with status {response.status_code}: {response.text}")

    def create_chat_completion(self, messages, max_tokens=50, temperature=1.0, stream=False):
        """Example chat completion in the other provider's style. Adjust the endpoint/payload as required by that provider."""
        endpoint = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": CHAT_MODEL,  # Or the other provider's chat model
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        response = requests.post(endpoint, headers=self.headers, json=payload, stream=stream)
        if response.status_code == 200:
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._extract_message_content(response)
        else:
            raise Exception(f"OtherAPI chat request failed: {response.status_code} {response.text}")

    def create_embedding(self, input_text):
        """Create embeddings using the other provider."""
        endpoint = f"{self.base_url}/v1/embeddings"
        payload = {"model": EMBEDDINGS_MODEL, "input": input_text}
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"OtherAPI embeddings failed: {response.status_code} {response.text}")

    def rerank(self, query, documents):
        """Rerank documents using the other provider."""
        endpoint = f"{self.base_url}/v1/rerank"
        payload = {"model": RERANKING_MODEL, "prompt": query, "input": documents}
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"OtherAPI rerank failed: {response.status_code} {response.text}")

    def transcribe_audio(self, audio_file_path, language="fr", response_format="json", temperature=0.5):
        """STT in the other provider, if available, or raise an exception if not supported."""
        endpoint = f"{self.base_url}/v1/audio/transcriptions"
        # Remove forced Content-Type if sending multipart:
        headers = dict(self.headers)
        if "Content-Type" in headers:
            del headers["Content-Type"]

        data = {
            "model": SPEECH_TO_TEXT_MODEL,
            "language": language,
            "response_format": response_format,
            "temperature": temperature,
        }

        with open(audio_file_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(audio_file_path), audio_file, "audio/wav"),
            }
            response = requests.post(endpoint, headers=headers, data=data, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"OtherAPI STT failed: {response.status_code} {response.text}")

    # -------------------------------------------------------------------------
    # Additional Methods (TTS and Multimodal) that Albert doesn't have
    # -------------------------------------------------------------------------
    def text_to_speech(self, text):
        """Convert text to speech using the other provider (since Albert lacks TTS). Replace with your actual TTS endpoint, model, etc."""
        endpoint = f"{self.base_url}/v1/audio/speech"
        payload = {"model": TEXT_TO_SPEECH_MODEL, "input": text}
        response = requests.post(endpoint, headers=self.headers, json=payload)
        if response.status_code == 200:
            # Possibly returns raw audio data or a URL.
            return response.json()
        else:
            raise Exception(f"OtherAPI TTS failed: {response.status_code} {response.text}")

    def describe_image(self, image_path):
        """Generate a textual description for an image using the other provider."""
        endpoint = f"{self.base_url}/v1/multimodal/describe"
        headers = dict(self.headers)
        if "Content-Type" in headers:
            del headers["Content-Type"]

        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            data = {"model": MULTIMODAL_MODEL}
            response = requests.post(endpoint, headers=headers, data=data, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"OtherAPI image description failed: {response.status_code} {response.text}")

    # -------------------------------------------------------------------------
    # Helper methods to mirror the AlbertAPI structure
    # -------------------------------------------------------------------------
    def _handle_streaming_response(self, response):
        if response.status_code == 200:
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        parsed_chunk = json.loads(chunk.decode("utf-8"))
                        if "choices" in parsed_chunk and len(parsed_chunk["choices"]) > 0:
                            yield parsed_chunk["choices"][0]["message"]["content"]
                    except json.JSONDecodeError:
                        continue
        else:
            raise Exception(f"OtherAPI streaming request failed: {response.status_code} {response.text}")

    def _extract_message_content(self, response):
        if response.status_code == 200:
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                return response_json["choices"][0]["message"]["content"]
            else:
                return "No response generated."
        else:
            raise Exception(f"OtherAPI request failed: {response.status_code} {response.text}")


###############################################################################
# Class 3: Kokoro TTS
###############################################################################
class KokoroTTS:
    """A dedicated class for local Kokoro TTS usage."""

    def __init__(self, lang_code="f"):
        """:param lang_code: Language code for Kokoro (e.g., 'a' => American English)."""
        # Initialize the Kokoro pipeline. Make sure the required packages are installed.
        self.pipeline = KPipeline(lang_code=lang_code)

    def text_to_speech(self, text, voice="ff_siwis", speed=1.0, split_pattern=r"\n+", out_prefix="kokoro_output"):
        """Converts text to speech using Kokoro locally.

        :param text: The input text to convert to speech (can contain multiple paragraphs).
        :param voice: The Kokoro voice name (e.g., 'af_heart' for American English female).
        :param speed: Playback speed factor (1.0 = normal speed).
        :param split_pattern: A regular expression for splitting the text (default splits on newlines).
        :param out_prefix: A filename prefix for the generated WAV files.

        :return: A list of filenames for the generated WAV files.
        """
        # Kokoro's pipeline(...) yields (original_text, phonemes, numpy_audio_array).
        generator = self.pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)

        output_files = []
        for i, (graphemes, phonemes, audio_data) in enumerate(generator):
            filename = f"{out_prefix}_{i}.wav"
            # Kokoro typically uses a 24kHz sample rate.
            sf.write(filename, audio_data, 24000)
            output_files.append(filename)

        return output_files

    def generate_audio_stream(self, text):
        """
        Generates speech audio from text and returns it as an in-memory stream.

        :param text: Input text to be converted to speech.
        :return: Audio stream (io.BytesIO) containing WAV data.
        """
        # Generate speech waveform from Kokoro TTS
        audio_waveform = self.pipeline(text, voice='ff_siwis')

        audio_np32 = []

        for index, (graphemes, phonemes, audio_np32_chunk) in enumerate(audio_waveform):
            # If audio_float32 is a Torch Tensor, convert it to NumPy
            if hasattr(audio_np32_chunk, "detach"):
                audio_np32 = np.concatenate((audio_np32, audio_np32_chunk.detach().cpu().numpy()))

        # Convert float32 audio (range -1 to 1) to int16
        audio_np = (audio_np32 * 32767).astype(np.int16)  # .tobytes()

        return (22050, audio_np)  # Return the sample rate + audio stream


###############################################################################
# Classe 4: Image description with Pixtral
###############################################################################
class PixtralImage:  # noqa: D101
    def __init__(self):
        """Initialize PixtralImage API client."""
        orange_api_key = "EMPTY"
        orange_api_base = ""
        self.client = OpenAI(
            api_key=orange_api_key,
            base_url=orange_api_base,
        )
        self.model = "mistralai/Pixtral-12B-2409"  # Specify the Pixtral model

    def encode_image(self, image_path):
        """Converts an image to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def describe_image(self, image_path, question="What is in this image?"):
        """Calls the Pixtral API to generate a description of an image.

        :param image_path: Path to the image file.
        :param question: The specific question to ask about the image.
        :return: Generated textual description.
        """
        base64_image = self.encode_image(image_path)

        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
        )

        return chat_response.choices[0].message.content


##############################################################################
# OCR Client
##############################################################################
class OCR:
    """OCR API client."""

    def __init__(self, url: str):
        """Init with OCR API url."""
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def extract(self, file_path: str):
        """Extract text from file_path."""
        with open(file_path, "rb") as file:
            files = {"file": file}
            data = {"output_format": "latex"}

            # Send the POST request
            response = requests.post(self.url, files=files, data=data)
            return response.json()


##############################################################################
# QA API Client
##############################################################################
class QA:
    """QA API client."""

    def __init__(self, url: str):
        """Init with QA API url."""
        self.url = url
        self.headers = {"Content-Type": "application/json"}

    def ask(self, document: str, question: str):
        """Document-based QA (streaming mode: generate token by token)."""
        data = {"document": document, "question": question}

        # Send the POST request
        with requests.post(
            self.url,
            headers=self.headers,
            json=data,
        ) as res:
            for chunk in res.iter_lines():
                chunk = json.loads(chunk.decode("utf-8"))
                yield chunk["token"]


##############################################################################
# Agent
##############################################################################
class AlbertAgentAPI:  # noqa: D101
    def __init__(self):
        """Initialize Albert API client for the agent."""
        ALBERT_API_KEY = os.getenv("ALBERT_API")
        self.headers = {"Authorization": f"Bearer {ALBERT_API_KEY}", "Content-Type": "application/json"}
        self.model = "AgentPublic/llama3-instruct-8b"  # ✅ Dedicated Model for the Agent

    def chat_completion(self, messages, max_tokens=100, temperature=0.3, stream=False):
        """Calls Albert API to generate a chat response.

        :param messages: A list of conversation messages.
        :param max_tokens: Maximum tokens for response.
        :param temperature: Controls randomness (0.0 = deterministic).
        :param stream: Whether to use streaming.
        :return: Generated text response.
        """
        ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr"
        endpoint = f"{ALBERT_BASE_URL}/v1/chat/completions"
        payload = {
            "model": self.model,  # ✅ Ensure the correct model is used
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        response = requests.post(endpoint, headers=self.headers, json=payload, stream=stream)

        if response.status_code == 200:
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]  # ✅ Extracts only the text response
        else:
            raise Exception(f"⚠️ Albert API request failed: {response.status_code} - {response.text}")


###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    # ALBERT setup
    ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr"
    ALBERT_API_KEY = os.getenv("ALBERT_API")

    # Load classes
    albert_api = AlbertAPI(base_url=ALBERT_BASE_URL, api_key=ALBERT_API_KEY)
    pixtral = PixtralImage()
    kokoro_tts = KokoroTTS(lang_code="f")  # 'a': American English, 'f': French

    # 1) List available models (Albert):
    models = albert_api.list_models()
    print("ALBERT: Available models:", models)

    # 2) Chat completion (Albert):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a fun fact about space."},
    ]
    chat_response = albert_api.create_chat_completion(messages=messages, max_tokens=50)
    print("ALBERT: Chat Completion:", chat_response)

    # 3) Transcribe audio (Albert):
    try:
        transcription = albert_api.transcribe_audio("../data/les_avantages_de_la_vaccination.wav")
        print("ALBERT: Transcription:", transcription)
    except Exception as e:
        print("ALBERT: Transcription failed:", e)

    # 4) Describe an image (Orange Pixtral)
    try:
        prompt = "Describe this image in one sentence."
        image_description = pixtral.describe_image("../data/terre.png", question="Décrit l'image en une phrase")
        print("OTHER: Image description:", image_description)
    except Exception as e:
        print("OTHER: Describe image failed:", e)

    print("-----------------------------------------------------")

    # 5) Text-to-Speech (Kokoro)

    sample_text = (
        "." * 20
        + """\
        Il s'agit d'un formulaire de demande de renouvellement de carte d'identité.
        
        [PAUSE] L'image au centre montre un exemple de photo aux normes.
        """
    )

    # Generate speech
    wav_files = kokoro_tts.text_to_speech(
        text=sample_text,
        # voice='af_heart',
        voice="ff_siwis",
        speed=1.0,
        out_prefix="my_kokoro_output",
    )

    print("Generated audio files:", wav_files)

    print("-----------------------------------------------------")

    # HUGGINGFACE setup
    OTHER_BASE_URL = "https://huggingface.co/api/modelsm"
    OTHER_API_KEY = "HF_API_KEY"

    other_api = OtherAPI(base_url=OTHER_BASE_URL, api_key=OTHER_API_KEY)

    # 1) Text-to-speech (not supported by Albert)
    try:
        tts_result = other_api.text_to_speech("Hello world, this is a TTS test.")
        print("OTHER: TTS result:", tts_result)
    except Exception as e:
        print("OTHER: TTS failed:", e)

    # 2) Describe an image (not supported by Albert)
    try:
        image_description = other_api.describe_image("../data/terre.png")
        print("OTHER: Image description:", image_description)
    except Exception as e:
        print("OTHER: Describe image failed:", e)
