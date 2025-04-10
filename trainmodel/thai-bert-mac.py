# ================================
# STEP 1: โหลดและเตรียมข้อมูลจาก local CSV
# ================================
import pandas as pd

# โหลดไฟล์ CSV จาก local (กรณีรันใน Colab ให้ upload ก่อน)
file_path = "Updated_Occupation_Titles.csv"
df = pd.read_csv(file_path)

# แสดงตัวอย่างข้อมูล
print("ข้อมูลตัวอย่าง:", df.head())

# ใช้ชื่อคอลัมน์จริงจากไฟล์ CSV
df = df[['ชื่อตำแหน่งงาน', 'สาขาอาชีพหลัก']].dropna()

# เปลี่ยนชื่อคอลัมน์ให้เข้ากับโค้ดที่ใช้ต่อ
df = df.rename(columns={
    'ชื่อตำแหน่งงาน': 'job_title',
    'สาขาอาชีพหลัก': 'category'
})

# แปลง category เป็น label (ตัวเลข)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
num_classes = len(le.classes_)

# ================================
# STEP 2: โหลด Tokenizer และ Model
# ================================
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

# ================================
# STEP 3: เตรียม Dataset สำหรับ BERT
# ================================
from datasets import Dataset

# แปลง Pandas เป็น HuggingFace Dataset
dataset = Dataset.from_pandas(df[['job_title', 'label']])

# ฟังก์ชัน tokenize
def tokenize_function(example):
    return tokenizer(example["job_title"], truncation=True, padding="max_length", max_length=64)

# ใช้ tokenizer กับ dataset ทั้งชุด
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# แบ่งข้อมูล train/test 80/20
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# ================================
# STEP 4: เทรนโมเดล (รองรับ CPU)
# ================================
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    no_cuda=True,  # บังคับให้ใช้ CPU เท่านั้น
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)

# เริ่มเทรน!
trainer.train()

# ================================
# STEP 5: ประเมินผลลัพธ์หลังเทรน
# ================================
preds = trainer.predict(tokenized_dataset['test'])
y_true = preds.label_ids
y_pred = preds.predictions.argmax(axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ================================
# STEP 6: บันทึกโมเดลและ LabelEncoder (สำหรับใช้ใน FastAPI)
# ================================
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

import pickle
with open("labels.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ บันทึกโมเดลและ label encoder เรียบร้อยแล้ว")