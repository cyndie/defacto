from openai import OpenAI
import base64 



openai_api_key = "EMPTY"
openai_api_base = "http://localhost:7040/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_response(image_path, question):
    base64_image = encode_image(image_path)

    chat_response = client.chat.completions.create(
        model="mistralai/Pixtral-12B-2409",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }],
    )
    print(chat_response)
    return chat_response.choices[0].message.content

generate_response("form_example.png", "When was the document edited?")
