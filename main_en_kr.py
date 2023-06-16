from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import datetime

MODEL = './model_en_kr/'  # https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ko/tree/main
ENGLISH = "how are you?"

tokenizer = AutoTokenizer.from_pretrained(MODEL)  
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)  

translator = pipeline("translation", model=model, tokenizer=tokenizer)
result = translator(ENGLISH)
print(result[0]['translation_text'])
