import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-large')
model = AutoModelForSequenceClassification.from_pretrained('vinai/phobert-base', num_labels=2)

# Prepare data
data_df = pd.read_csv("step1_data.csv")
# Split data into train and test sets
train_df, test_df = train_test_split(data_df, test_size=0.2)

X_train = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')
y_train = torch.tensor(train_df['label'].tolist())


# Prepare data for training
y = torch.tensor(y_train).unsqueeze(1)
dataset = torch.utils.data.TensorDataset(X_train['input_ids'], X_train['attention_mask'], y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(3):
    for input_ids, attention_mask, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Predict new data
new_doc = ["Lê Quang Minh là một nhà nghiên cứu đã công bố hàng chục bài báo khoa học và tham gia nhiều dự án nghiên cứu về Trí tuệ nhân tạo. Các nghiên cứu của ông liên quan đến nhiều lĩnh vực, bao gồm học máy, xử lý ngôn ngữ tự nhiên, nhận dạng giọng nói và thị giác máy tính. "]
# Tokenize data

new_X = tokenizer(new_doc, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs = model(input_ids=new_X['input_ids'], attention_mask=new_X['attention_mask'])
    predicted_label = torch.argmax(outputs.logits)
if predicted_label == 1:
    print("Bản ghi mới là CV của chuyên gia")
else:
    print("Bản ghi mới không phải là CV của chuyên gia")

# Load test data
X_test = tokenizer(test_df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')
y_test = torch.tensor(test_df['label'].tolist())

# Predict test data
with torch.no_grad():
    outputs = model(input_ids=X_test['input_ids'], attention_mask=X_test['attention_mask'])
    predicted_labels = torch.argmax(outputs.logits, axis=1)

# Print classification report
target_names = ['non-expert', 'expert']
print(test_df['text'].tolist())
print("Actual : ",test_df['label'].tolist())
print("Predict: ",predicted_labels.tolist())
print(classification_report(y_test, predicted_labels, target_names=target_names))
