import argparse  # noqa: D100

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from transformers import AutoModel, AutoTokenizer

################ GOT ################
# Code from https://github.com/Ucas-HaoranWei/GOT-OCR2.0/tree/main
punctuation_dict = {
    "，": ",",
    "。": ".",
}

tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "ucaslcl/GOT-OCR2_0",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="cuda",
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id,
)
model = model.eval().cuda().to(torch.float16)

def process_image(image_file, output_format="html"):  # noqa: D103
    output = model.chat_crop(tokenizer, image_file, ocr_type="format")
    if output_format == "html":
        # process output in html
        if "\\begin{tikzpicture}" not in output:
            html_path = "content-mmd-to-html.html"
            right_num = output.count("\\right")
            left_num = output.count("\left")

            if right_num != left_num:
                output = (
                    output.replace("\left(", "(")
                    .replace("\\right)", ")")
                    .replace("\left[", "[")
                    .replace("\\right]", "]")
                    .replace("\left{", "{")
                    .replace("\\right}", "}")
                    .replace("\left|", "|")
                    .replace("\\right|", "|")
                    .replace("\left.", ".")
                    .replace("\\right.", ".")
                )

            output = output.replace('"', "`").replace("$", "")

            output_list = output.split("\n")
            gt = ""
            for out in output_list:
                gt += '"' + out.replace("\\", "\\\\") + r"\n" + '"' + "+" + "\n"

            gt = gt[:-2]

            with open(html_path) as web_f:
                lines = web_f.read()
                lines = lines.split("const text =")
                new_web = lines[0] + "const text =" + gt + lines[1]
        else:
            translation_table = str.maketrans(punctuation_dict)
            html_path = "tikz.html"
            output = output.translate(translation_table)
            outputs_list = output.split("\n")
            gt = ""
            for out in outputs_list:
                if out:
                    if "\\begin{tikzpicture}" not in out and "\\end{tikzpicture}" not in out:
                        while out[-1] == " ":
                            out = out[:-1]
                            if out is None:
                                break

                        if out:
                            if out[-1] != ";":
                                gt += out[:-1] + ";\n"
                            else:
                                gt += out + "\n"
                    else:
                        gt += out + "\n"

            with open(html_path) as web_f:
                lines = web_f.read()
                lines = lines.split("const text =")
                new_web = lines[0] + gt + lines[1]

        return new_web

    elif output_format == "latex":
        return output


################ API ################
app = FastAPI()
@app.post("/v0/extract")
async def extract(file: UploadFile = File(...), output_format: str = Form("html")):  # noqa: B008, D103
    filepath = "/tmp/" + file.filename
    try:
        # save file to tmp
        with open(filepath, "wb") as temp_file:
            temp_file.write(await file.read())    
        
        # reload and extract
        result = process_image(filepath, output_format)
        return JSONResponse({"text": result})
    except Exception as e:
        return Response(status_code=500, content=f"Failed to process the image: {str(e)}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_format", type=str, choices=["html", "latex"], default="html", help="Output format")
    args = parser.parse_args()
    result = process_image(args.image_path, args.output_format)
    print(result)
