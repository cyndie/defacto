'''Main agent'''
import os
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from backend.utils.api_utils import AlbertAPI, KokoroTTS, PixtralImage, AlbertAgentAPI, QA
# from scripts.text_to_speech import text_to_speech
# from scripts.speech_to_text import transcribe_audio
from backend.config import *

ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr"
ALBERT_API_KEY = os.getenv("ALBERT_API")

# Initialize API Client
api = AlbertAPI(base_url=ALBERT_BASE_URL, api_key=ALBERT_API_KEY)
pixtral = PixtralImage()
kokoro_tts = KokoroTTS()
qa = QA(SERVER_URL)
albert_agent = AlbertAgentAPI()

# Define Tools Using Existing Scripts
tools = [
    Tool(name="Question Answering", func=qa.ask, description="Retrieve relevant text."),
    Tool(name="Describe Image", func=pixtral.describe_image, description="Generate a textual description for an image."),
    Tool(name="Text-to-Speech", func=kokoro_tts.text_to_speech, description="Convert text responses into spoken audio."),
]

# Agent System Instructions (Prompt)
agent_prompt = """
You are an AI assistant that helps users process and analyze documents, images, and audio in French.

You have access to the following tools:

 **Question Answering** → Use this tool if the user asks for specific information from documents or wants to search for knowledge.  
   - Example: "What is the deadline for visa applications?"  
   - Example: "Find me the latest tax regulations."

 **Text-to-Speech** → Use this tool if the user wants a text to be vocalized.  
   - Example: "Read this response aloud."  
   - Example: "Can you vocalize this paragraph?"

 **Describe Image** → Use this tool if the user asks to describe an image.  
   - Example: "Can you describe this image?"  
   - Example: "What is shown in this picture?"

 **Rules for Choosing Tools:**
- **If the user asks for information retrieval**, use **Retrieve Documents**.
- **If the user wants a text read aloud**, use **Text-to-Speech**.
- **If the user uploads an image or asks for a description of an image**, use **Describe Image**.
- **If unsure, ask the user for clarification before selecting a tool.**

Your goal is to pick the most appropriate tool for each user request.
"""

# Initialize Agent with System Instructions
llm = ChatOpenAI(temperature=0.3)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, system_message=agent_prompt)

def run_agent(user_query):
    """
    Process the user query and let the agent decide the best strategy.
    """
    messages = [{"role": "system", "content": agent_prompt}, {"role": "user", "content": user_query}]
    return albert_agent.chat_completion(messages)

