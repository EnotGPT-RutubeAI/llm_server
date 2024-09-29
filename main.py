import time
import json
import requests as req
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, WhisperForConditionalGeneration, WhisperProcessor, pipeline

device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = WhisperProcessor.from_pretrained(model_id)

pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            max_new_tokens=256,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,  # batch size for inference - set based on your device
            torch_dtype=torch_dtype,
            device=device,
        )

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(HTTPException)
async def unicorn_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": False, "error": exc.detail},
    )

class Prompt(BaseModel):
    data: str

class Audio(BaseModel):
    audio_url: str

class Context(BaseModel):
    data: list[dict]
import ollama
@app.post("/llama_ollama")
def read_root(data:Context):

    print('Получил в llama_ollama:',data.data)
    start_time = time.time()
    response = ollama.chat(model='llama3.1:8b', messages=data.data,options={'top_k':0,
        'top_p':0,
        'temperature':0,
        'repeat_penalty':1.1,})
    print(response['message']['content'])
    print('--- %s seconds ---'%(time.time() - start_time))
    return JSONResponse(status_code = 200, content={"response": response['message']['content']})


@app.post("/audio_to_text")
def read_root(audio:Audio):

    start_time = time.time()
    result = pipe(audio.audio_url, return_timestamps=True, generate_kwargs={"language": "russian"})
    print('--- %s seconds ---'%(time.time() - start_time))
    return result

