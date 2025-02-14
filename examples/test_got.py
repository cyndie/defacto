"""Test GOT API."""

import requests

url = "http://localhost:7041/v0/extract"

file_path = "test.png"


with open(file_path, "rb") as file:
    files = {"file": file}
    data = {"output_format": "latex"}

    # Send the POST request
    # curl -X POST localhost:7041/v0/extract -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@test.png" -F "output_format=latex"
    response = requests.post(url, files=files, data=data)

print(response.json())
"""
{'text': "\\title{\nCERTIFICAT DE CESSIÓN D'UN VÉHICULE D'OCCASION\n}\n\\author{\n(à remplir par l'ancien propriétaire et le nouveau propriétaire) \\\\ N° 15776*01 \\\\ Articles R322-4 et R322-9 du code de la route\n}\n
\\section*{LE VÉHICULE (a remplir par l'ancien propriétaire)}\n(A) Numéro d'immatriculation du véhicule\n(D.1 Marque)\n(C) Numéro d'identification du véhicule\n(D.2 Type, variante, version)\nKilométrage inscrit au compteur du véhicule :
\nPrésence du certificat d'immatriculation :\n\\(\\square\\) OUI - numéro de formule \\(\\underline{20}\\)\nFigure sur le 1e volet du certificat d'immatriculation de type AB-123-CD)\n
ou (I) date du certificat d'immatriculation Veuillez répéter (si ancien format d'immatriculation de type 123 AB\nExemplaire 1 destiné à l'ancien propriétaire\n(B) Date de \\(1^{\\text {st }}\\) immatriculation du véhicule\n(D.3 Denomination commerciale)\nNON - Motif d'absence de certificat d'immatriculation :"}
"""
