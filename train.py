import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter
from pythainlp.tokenize import word_tokenize
from gensim.models import Word2Vec

# 🔹 โหลดข้อมูล
print("🔄 กำลังโหลดข้อมูล...")
file_path = "Updated_Occupation_Titles.csv"  # 🔄 แก้เป็นพาธไฟล์ของคุณ
df = pd.read_csv(file_path, encoding="utf-8")

# 🔹 ลบแถวที่มีค่า NaN และอักขระพิเศษ
df = df.dropna()
df["ชื่อตำแหน่งงาน"] = df["ชื่อตำแหน่งงาน"].str.replace(r"[^\w\s]", "", regex=True)

# 🔹 แสดงข้อมูลทั่วไปเกี่ยวกับชุดข้อมูล
print(f"จำนวนข้อมูลทั้งหมด: {len(df)}")
print(f"จำนวนสาขาอาชีพหลักทั้งหมด: {df['occupation_id'].nunique()}")
print("\nการกระจายของสาขาอาชีพหลัก:")
occupation_distribution = df['occupation_id'].value_counts().sort_values(ascending=False)
for occupation_id, count in occupation_distribution.items():
    occupation_name = df[df['occupation_id'] == occupation_id]['สาขาอาชีพหลัก'].iloc[0] 
    print(f"- รหัส {occupation_id} ({occupation_name}): {count} รายการ ({count/len(df)*100:.2f}%)")

# 🔹 แบ่งคำภาษาไทยและสร้าง word embedding
print("\n🔄 กำลังแบ่งคำและสร้าง Word Embedding...")
df['tokenized'] = df['ชื่อตำแหน่งงาน'].apply(lambda x: word_tokenize(x, engine='newmm'))

# สร้างโมเดล Word2Vec
w2v_model = Word2Vec(sentences=df['tokenized'], vector_size=100, window=5, min_count=1, workers=4)

# 🔹 สร้างเวกเตอร์จากชื่อตำแหน่งงาน
def get_document_vector(tokens, model, vector_size=100):
    vec = np.zeros(vector_size)
    count = 0
    for word in tokens:
        try:
            vec += model.wv[word]
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

# สร้างเวกเตอร์สำหรับแต่ละตำแหน่งงาน
X_w2v = np.array([get_document_vector(tokens, w2v_model) for tokens in df['tokenized']])

# 🔹 แปลงค่าหมวดหมู่เป็นตัวเลข
le_occupation = LabelEncoder()
df["occupation_id_encoded"] = le_occupation.fit_transform(df["occupation_id"])
y_main = df["occupation_id_encoded"]

# 🔹 ใช้ TF-IDF เพื่อแปลงชื่อตำแหน่งงานเป็นเวกเตอร์
print("🔄 กำลังแปลงข้อความเป็นเวกเตอร์ด้วย TF-IDF...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["ชื่อตำแหน่งงาน"])

# 🔹 แสดงการกระจายของข้อมูลก่อนปรับสมดุล
print(f"\nการกระจายของข้อมูลก่อนปรับสมดุล: {Counter(y_main)}")

# 🔹 ใช้ SMOTE เพื่อปรับสมดุลข้อมูล
print("🔄 กำลังปรับสมดุลข้อมูลด้วย SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_main)

# แสดงการกระจายของข้อมูลหลังปรับสมดุล
print(f"การกระจายของข้อมูลหลังปรับสมดุล: {Counter(y_resampled)}")

# 🔹 แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
print("\n🔄 กำลังแบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# 🔹 สร้างและฝึกสอนโมเดล RandomForest
print("🔄 กำลังสร้างและฝึกสอนโมเดล RandomForest...")
rf_main = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', random_state=42)

# 🔹 ประเมินโมเดลด้วย Cross Validation
print("🔄 กำลังประเมินโมเดลด้วย Cross Validation...")
cv_main = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score_main = cross_val_score(rf_main, X_train, y_train, cv=cv_main)
print(f"🎯 ความแม่นยำเฉลี่ยของโมเดล (Cross Validation): {np.mean(cv_score_main):.4f}")

# 🔹 ฝึกสอนโมเดลกับชุดข้อมูลฝึกสอนทั้งหมด
print("🔄 กำลังฝึกสอนโมเดลกับชุดข้อมูลฝึกสอนทั้งหมด...")
rf_main.fit(X_train, y_train)

# 🔹 ทดสอบโมเดลกับชุดข้อมูลทดสอบ
print("\n🔄 กำลังทดสอบโมเดลกับชุดข้อมูลทดสอบ...")
y_pred = rf_main.predict(X_test)
print("รายงานการจำแนกประเภท:")
print(classification_report(y_test, y_pred))

# 🔹 แสดงความสำคัญของคุณลักษณะ
print("\n🔄 กำลังวิเคราะห์ความสำคัญของคุณลักษณะ...")
feature_names = vectorizer.get_feature_names_out()
importances = rf_main.feature_importances_
indices = np.argsort(importances)[::-1]

print("20 คุณลักษณะที่สำคัญที่สุด:")
for i in range(min(20, len(indices))):
    print(f"{i+1}. {feature_names[indices[i]]} - {importances[indices[i]]:.4f}")

# 🔹 บันทึกโมเดลและส่วนประกอบสำคัญ
print("\n🔄 กำลังบันทึกโมเดลและส่วนประกอบสำคัญ...")
os.makedirs('models', exist_ok=True)
# บันทึกโมเดล RandomForest
joblib.dump(rf_main, 'models/occupation_rf_model.joblib')
# บันทึก TF-IDF Vectorizer
joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
# บันทึก LabelEncoder
joblib.dump(le_occupation, 'models/label_encoder.joblib')
# บันทึกโมเดล Word2Vec
w2v_model.save('models/word2vec_model.bin')
# บันทึกข้อมูลสำคัญจาก DataFrame
occupation_mapping = df[['occupation_id', 'สาขาอาชีพหลัก']].drop_duplicates()
occupation_mapping.to_csv('models/occupation_mapping.csv', index=False)

print("✅ บันทึกโมเดลและส่วนประกอบสำคัญเรียบร้อยแล้ว")
print("✅ การฝึกสอนโมเดลเสร็จสมบูรณ์")