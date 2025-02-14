"""Test VLM (e.g. pixtral)."""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

image_url = "https://mobile.interieur.gouv.fr/var/miomcti/storage/images/www.interieur.gouv.fr/version-fre/actualites/l-actu-du-ministere/nouveau-permis-de-conduire-securise-le-16-septembre-2013/466172-1-fre-FR/Nouveau-permis-de-conduire-securise-le-16-septembre-2013_catcher.jpg"

vlm = ChatOpenAI(
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:7040/v1",
    model_name="mistralai/Pixtral-12B-2409",
    temperature=0.2,
)


class PermisDeConduire(BaseModel):  # noqa: D101a
    nom: str
    prenom: str
    date_de_naissance: str
    lieu_de_naissance: str
    numero_de_permis: str


message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": f"Extract information. Your response must be in JSON format. Here is the JSON schema you must adhere to: \n<schema>\n{PermisDeConduire.model_json_schema()}\n</schema>.",
        },
        {"type": "image_url", "image_url": {"url": image_url}},
    ],
)

response = vlm.invoke([message])

print(response)
"""
```json\n{\n  "nom": "Martin",\n  "prenom": "Paul",\n  "date_de_naissance": "14.07.1981",\n  "lieu_de_naissance": "Utopia city",\n  "numero_de_permis": "D1FRA13AA000026181231MARTIN<<9"\n}```
"""
