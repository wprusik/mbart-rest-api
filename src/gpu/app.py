from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import torch

app = FastAPI()

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
_tokenizer = None
_model = None
# ustawienie urządzenia: GPU jeśli dostępne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# getter dla tokenizera z logowaniem
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        print(">> Loading tokenizer...")
        _tokenizer = MBart50Tokenizer.from_pretrained(MODEL_NAME)
        print(">> Tokenizer loaded.")
    return _tokenizer

# getter dla modelu, ładowanie na GPU
def get_model():
    global _model
    if _model is None:
        print(">> Loading model on device:", device)
        _model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model.to(device)
        print(">> Model loaded on the device.")
    return _model

class TranslateRequest(BaseModel):
    src_lang: str
    tgt_lang: str
    text: str

@app.get("/")
def root():
    print(">> GET /")
    return {"msg": "ok"}

@app.get("/languages")
def get_languages():
    print(">> GET /languages")
    tokenizer = get_tokenizer()
    return {"available_languages": list(tokenizer.lang_code_to_id.keys())}

@app.post("/translate")
def translate(req: TranslateRequest):
    print(f">> POST /translate: {req}")
    tokenizer = get_tokenizer()
    model = get_model()

    if req.tgt_lang not in tokenizer.lang_code_to_id:
        print(">> Error: unknown target language.")
        raise HTTPException(
            status_code=400,
            detail=(
                    "Unknown target language. Available languages:\n" +
                    ",\n".join(tokenizer.lang_code_to_id.keys())
            )
        )

    print(f">> Translation from {req.src_lang} to {req.tgt_lang}")
    tokenizer.src_lang = req.src_lang
    encoded = tokenizer(req.text, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[req.tgt_lang],
            max_length=200
        )
    translation = tokenizer.batch_decode(generated_tokens.cpu(), skip_special_tokens=True)[0]
    print(">> Translation finished.")
    return {"translation": translation}
