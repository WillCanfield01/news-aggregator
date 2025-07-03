from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

model_name = "cardiffnlp/tweet-topic-21-multi"
save_path = "./models/cardiffnlp/tweet-topic-21-multi"

# Create path if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Download and save model/tokenizer locally
AutoModelForSequenceClassification.from_pretrained(model_name).save_pretrained(save_path)
AutoTokenizer.from_pretrained(model_name).save_pretrained(save_path)