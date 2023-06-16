from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import datetime

MODEL = './model_kr_en/'  # https://huggingface.co/Helsinki-NLP/opus-mt-ko-en/tree/main
KOREAN = "오늘은 좋은 하루 이네요"

tokenizer = AutoTokenizer.from_pretrained(MODEL)  
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)  

translator = pipeline("translation", model=model, tokenizer=tokenizer)
result = translator(KOREAN)
print(result[0]['translation_text'])
