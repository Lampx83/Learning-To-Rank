import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load the data
data = pd.read_csv('stepx_data.csv')

# Preprocess the data
long_profiles = data['long_profile'].tolist()
short_profiles = data['short_profile'].tolist()

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# Define the summarization function
summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def summarize(long_profile):
    # Generate the summary
    summary = summarizer(long_profile, max_length=50, min_length=10, do_sample=False)[0]['generated_text']
    return summary

# Generate the summaries
generated_summaries =summarize("Giáo sư Nguyễn Thị Hồng là một chuyên gia hàng đầu trong lĩnh vực Luật pháp. Bà đã có nhiều đóng góp quan trọng trong việc nghiên cứu, giảng dạy và áp dụng Luật pháp vào thực tiễn. Giáo sư Nguyễn Thị Hồng là tác giả của 5 cuốn sách và được trao giải thưởng vì thành tích trong nghiên cứu.")

print(generated_summaries)