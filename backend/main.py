'''Entry point for running the pipeline'''

from scripts.search_engine import search_documents
from scripts.chat_interaction import chat_with_llm
from scripts.text_to_speech import text_to_speech  # Import TTS function


def run_inference(query, use_voice=False, retrieve_images=False):
    """
    Runs RAG-based inference, optionally retrieving images and providing voice output.
    """
    # Step 1: Retrieve relevant text (and optionally images)
    retrieved_results = search_documents(query, retrieve_images=retrieve_images)

    # Step 2: Extract retrieved text and optional image links
    retrieved_texts = [res["description"] for res in retrieved_results]
    retrieved_images = [res["image_path"] for res in retrieved_results if "image_path" in res]

    # Step 3: Generate AI response using text retrieval
    response = chat_with_llm(query, retrieved_texts)

    print("\n💡 AI Response:", response)

    # Step 4: Convert response to speech (if enabled)
    if use_voice:
        text_to_speech(response)

    # Step 5: Show retrieved images (if any)
    if retrieve_images and retrieved_images:
        print("\n🖼️ Relevant Images:")
        for img in retrieved_images:
            print(f"- {img}")

    return response, retrieved_images


if __name__ == "__main__":
    query = "Show me the explanation of the graph about population growth."

    # Enable voice output and image retrieval as needed
    use_voice_output = True  # Set to False to disable text-to-speech
    retrieve_images = True  # Set to False to disable image retrieval

    run_inference(query, use_voice=use_voice_output, retrieve_images=retrieve_images)


# Using an agent:
# from utils.api_utils import *
# from scripts.agent import run_agent
#
# if __name__ == "__main__":
#     ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr"
#     ALBERT_API_KEY = os.getenv("ALBERT_API")
#
#     print("🤖 AI Agent is ready! Type 'exit' to stop.\n")
#
#     while True:
#         user_query = input("📝 Enter your request: ")
#
#         if user_query.lower() == "exit":
#             print("👋 Exiting agent.")
#             break
#
#         try:
#             response = run_agent(user_query)
#             print("\n💡 Agent Response:", response)
#         except Exception as e:
#             print(f"⚠️ Error: {e}")
