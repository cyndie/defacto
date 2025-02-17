"""Legal Assistant UI."""

import io
import json
import os

import gradio as gr
import requests
from backend.utils.api_utils import KokoroTTS
from bs4 import BeautifulSoup
from markdown import markdown
from pdf2image import convert_from_path

css = """
/* global part */
:root {--color-grey-700: #000091; }
@font-face {font-family: 'Marianne'; src: url('/gradio_api/file=static/fonts/Marianne/Marianne-Regular.woff') format('woff'); font-weight: normal; font-style: normal; }
* { font-family: 'Marianne', 'arial','sans-serif'; }

/* all gr.HTML containers */
.html-container.padding.svelte-phx28p { padding: 0; }
.html-container h1 {margin-bottom: 15px!important; }

/* all gr.Group containers */
.gradio-container .contain .gr-group, .gradio-container .contain div.svelte-1nguped { border-radius: 0; background-color: white; }
.gradio-container .contain .gr-group { padding: 20px; }

/* Header part */
header { display: flex; gap: 20px; border-bottom: 3px groove #fafafa; }
header h2 { font-weight: bolder; }

/* 'Import file' part */
#import_block { border: none; }
.large.unpadded_box { min-height: 30px; border: 1px solid #e4e4e7;}
button[id] { margin: 20px 20px 20px 0px; background-color: #000091; border-radius: 0; color: white; max-width: 500px; }
button[id]:hover { background-color: #1212FF; }
label.svelte-i3tvor { display:none; } /* used to hide label for file */

/* 'Recherche' part */
textarea { background-color: #fafafa; }
label.container.show_textbox_border textarea { border-radius: 0; }
.gradio-container .contain .bubble-wrap { background-color: #fafafa; }
.message-row.bubble .message.user, .message-row.bubble.message.bot { margin-top: 0px; }
span.svelte-1gfkn6j { display:none; } /* used to hide label for textbox */

"""
# fond clair : background-color: #F5F5FE
defacto_theme = gr.themes.Default(primary_hue="blue").set(
    loader_color="#000091",
    slider_color="#000091",
)

################ API(s) ################
# for extracting text from image using GOT model
GOT_URL = os.getenv("GOT_URL")
if not GOT_URL:
    raise RuntimeError("env `GOT_URL` is not specified !")

# for QA on document
QA_URL = os.getenv("QA_URL")
if not QA_URL:
    raise RuntimeError("env `QA_URL` is not specified !")

# kokoro text to speech
tts = KokoroTTS(lang_code="f")

################ Loading and Processing Legal Document ################
keywords = [
    "PROCES-VERBAL",
    "JUGEMENT CORRECTIONNEL",
    "COMPTE RENDU D'ENQUETE",
    "REQUISITION",
    "ATTESTATION DE CONFORMITE",
    "CERTIFICAT DE COMPATIBILITE A LA GARDE A VUE",
    "RELEVE DE COMPTE",
]  # keywords used to split a long document into a list of short, but relevant documents.


def upload_and_processing(file):
    """Load legal document and process it. Only .pdf accepted.

    Steps:
        + Convert .pdf to image
        + Extract .text from image
        + Split into a list of text document using delimiters specified in `keywords`
        + Generate a résumé.
    """
    # convert .pdf to image
    images = convert_from_path(file.name, thread_count=4)

    # extract .text from image and store in a list of documents
    doc_list = []  # list of documents
    page_list = []  # a doc is a list of pages
    for img in images:
        # call GOT API to extract text
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        with requests.post(GOT_URL, files={"file": buffer}, data={"output_format": "latex"}) as res:
            if res.status_code == 200:
                page_content = res.json()["text"]
                # accumulate pages in a single doc
                if any(kw in page_content for kw in keywords):
                    doc_content = "\n".join(page_list)
                    if len(doc_content.split()) > 20:  # ignore too short document (has less than 20 words)
                        doc_list.append(doc_content)
                    page_list = [page_content]
                else:
                    page_list.append(page_content)
            else:
                raise ConnectionError(f"Error: Failed to connect to {GOT_URL} with status code {res.status_code}")
    # handle last pages
    doc_content = "\n".join(page_list)
    if len(doc_content.split()) > 20:  # ignore too short document (has less than 20 words)
        doc_list.append(doc_content)

    # call QA API to generate a résumé
    response = ""
    full_document_content = ""
    for i, a_doc in enumerate(doc_list):
        full_document_content += f"===== Document {i + 1} =====\n{a_doc}\n\n"
    with requests.post(
        QA_URL,
        headers={"Content-Type": "application/json"},
        json={"document": full_document_content, "question": "faire une synthese"},
    ) as res:
        if res.status_code == 200:
            for chunk in res.iter_lines():
                token = json.loads(chunk.decode("utf-8"))
                # accumulate generated tokens
                response += token["token"]
        else:
            raise ConnectionError(f"Error: Failed to connect to {QA_URL} with status code {res.status_code}")

    return [
        file.name,
        gr.Group(visible=True),
        gr.Group(visible=True),
        gr.Group(visible=True),
        response,
        full_document_content,
    ]


################ UI ################
with gr.Blocks(title="Assistant dossier pénal", css=css, theme=defacto_theme) as demo:
    gr.HTML("<header><img id='defacto-logo' src='/gradio_api/file=defacto_logo.png'/><aside><h2>De Facto</h2><p>Mon assistant juridique</p></aside></header>")

    # IMPORTER
    with gr.Group(elem_id="import_block") as import_file:
        gr.HTML("<h1>Importer un dossier</h1>")
        gr.Markdown(
            "Importer un dossier pénal au format .pdf.<br/><br/>Le dossier peut être composé de scans, même de mauvaise qualité, de photos, de plans administratifs, de plans et tableaux."
        )
        # todo ajout barre progression upload
        u = gr.UploadButton("Ajouter un fichier 📎", file_count="single", scale=0)
        uploaded_file = gr.File()

    # SYNTHESE
    with gr.Group(visible=False) as synthese:
        gr.HTML("<h1>Résumé</h1>")
        synthese_md = gr.Markdown("## Compte rendu")  # To plug
        example_to_be_deleted = gr.HTML(
            "<hr><br/><ul><li>Victime : Madame Dupont</li>"
            + "<li>Prévenu : Monsieur Martin</li><li>Date des faits : 14/01/2025</li></ul>",
            visible=False,
        )

    # RECHERCHE
    with gr.Group(elem_id="recherche_block", visible=False) as recherche:
        gr.HTML("<h1>Recherche intelligente</h1>")
        document = gr.State()  # to track the uploaded document.
        gr.Markdown(
            "Interroger le dossier en langage naturel. La réponse est sourcée en montrant les pages utilisées par DeFacto pour établir sa réponse."
        )
        history = [
            gr.ChatMessage(role="assistant", content="Bonjour, quelle est votre question sur ce dossier pénal ?")
        ]
        recherche_chatbot = gr.Chatbot(history, type="messages")
        msg = gr.Textbox()

        def ask(message, document, chat_history):
            """Take user message and call QA API to response.

            Note: this is QA mode, not conversational mode, meaning the assistant does not retain conversation history.
            """
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": "Processing..."})
            yield "", chat_history
            response = ""
            with requests.post(
                QA_URL,
                headers={"Content-Type": "application/json"},
                json={"document": document, "question": message},
            ) as res:
                if res.status_code == 200:
                    for chunk in res.iter_lines():
                        token = json.loads(chunk.decode("utf-8"))
                        # accumulate generated tokens
                        response += token["token"]
                        chat_history[-1]["content"] = response
                        yield "", chat_history
                else:
                    raise ConnectionError(f"Error: Failed to connect to {QA_URL} with status code {res.status_code}")

        msg.submit(ask, [msg, document, recherche_chatbot], [msg, recherche_chatbot])
        gr.HTML(
            "<h3>Cas d'usages en exemple</h3>"
            + "<ul><li>Trouver une pièce spécifique d'un dossier</li>"
            + "<li>Identifier les protagonistes d'une affaire</li>"
            + "<li>Déterminer une chronologie des faits</li>"
            + "<li>Extraire une information spécifique</li></ul>"
        )

    # REQUISITOIRE
    # with gr.Group(visible=False) as requisitoire:
    #     gr.HTML("<h1>Base de synthèse</h1>")
    #     gr.HTML("<p>Documents extraits du dossier pénal</p>", visible=False)
    #     gr.DownloadButton("Official Report 20250121 📎", visible=False)
    #     gr.DownloadButton("Official Report 20250124 📎", visible=False)

    with gr.Group(visible=False) as speech:
        play_btn = gr.Button("Speak")
        speaker = gr.Audio(label="Generated Audio", type="numpy", autoplay=False)

        def speak(chat_history):
            """Speak the assistant message."""
            last_answer = chat_history[-1]["content"]
            html = markdown(last_answer)
            text = "".join(BeautifulSoup(html).findAll(text=True))
            return tts.generate_audio_stream(text)

        play_btn.click(speak, inputs=[recherche_chatbot], outputs=[speaker])

    u.upload(upload_and_processing, u, [uploaded_file, synthese, recherche, speech, synthese_md, document])


if __name__ == "__main__":
    demo.launch(favicon_path="favicon.png", allowed_paths=["."])
