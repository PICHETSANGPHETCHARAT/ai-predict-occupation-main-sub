import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 โหลดข้อมูล
file_path = "Updated_Occupation_Titles.csv"  # 🔄 แก้เป็นพาธไฟล์ของคุณ
df = pd.read_csv(file_path, encoding="utf-8")

# 🔹 ลบแถวที่มีค่า NaN และอักขระพิเศษ
df = df.dropna()
df["ชื่อตำแหน่งงาน"] = df["ชื่อตำแหน่งงาน"].str.replace(r"[^\w\s]", "", regex=True)

# 🔹 แปลงค่าหมวดหมู่เป็นตัวเลข
le_occupation = LabelEncoder()
le_sub_occupation = LabelEncoder()

df["occupation_id_encoded"] = le_occupation.fit_transform(df["occupation_id"])
df["occupation_sub_id_encoded"] = le_sub_occupation.fit_transform(df["occupation_sub_id"])

# 🔹 ใช้ TF-IDF เพื่อแปลงชื่อตำแหน่งงานเป็นเวกเตอร์
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["ชื่อตำแหน่งงาน"])
y_main = df["occupation_id_encoded"]
y_sub = df["occupation_sub_id_encoded"]

# 🔥 **โมเดลที่ 1: ทำนายตำแหน่งงานหลัก**
rf_main = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

# **ใช้ Cross Validation 5-Fold กับโมเดลตำแหน่งงานหลัก**
cv_main = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score_main = cross_val_score(rf_main, X, y_main, cv=cv_main)
rf_main.fit(X, y_main)

print(f"🎯 ความแม่นยำเฉลี่ยของตำแหน่งงานหลัก (Cross Validation): {np.mean(cv_score_main):.2f}")

# 🔥 **โมเดลที่ 2: ทำนายตำแหน่งงานรองเฉพาะในกลุ่มหลัก**
sub_category_models = {}
cv_score_sub = {}

for main_category in df["occupation_id_encoded"].unique():
    df_sub = df[df["occupation_id_encoded"] == main_category]

    X_sub = vectorizer.transform(df_sub["ชื่อตำแหน่งงาน"])
    y_sub = df_sub["occupation_sub_id_encoded"]

    rf_sub = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

    # 🔹 ใช้ Cross Validation 5-Fold กับตำแหน่งงานรองเฉพาะในกลุ่มหลัก
    cv_sub = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf_sub, X_sub, y_sub, cv=cv_sub)

    # 🔹 ฝึกโมเดลเต็มรูปแบบหลังจาก Cross Validation
    rf_sub.fit(X_sub, y_sub)

    sub_category_models[main_category] = rf_sub
    cv_score_sub[main_category] = np.mean(scores)

# 🔹 ฟังก์ชันทำนายตำแหน่งงานหลักและรอง พร้อมค้นหาชื่ออาชีพจากไฟล์
def predict_job_category(job_title):
    job_vector = vectorizer.transform([job_title])
    
    # 🔥 ทำนายตำแหน่งงานหลัก
    main_category_encoded = rf_main.predict(job_vector)[0]
    main_category = le_occupation.inverse_transform([main_category_encoded])[0]

    # 🔹 ค้นหาชื่อของสาขาอาชีพหลัก
    main_category_name = df[df["occupation_id"] == main_category]["สาขาอาชีพหลัก"].values
    main_category_name = main_category_name[0] if len(main_category_name) > 0 else "ไม่พบข้อมูล"

    # 🔥 ทำนายตำแหน่งงานรอง (เฉพาะในกลุ่มหลักที่ทำนายได้)
    if main_category_encoded in sub_category_models:
        sub_model = sub_category_models[main_category_encoded]
        sub_category_encoded = sub_model.predict(job_vector)[0]
        sub_category = le_sub_occupation.inverse_transform([sub_category_encoded])[0]

        # 🔹 ค้นหาชื่อของสาขาอาชีพรอง
        sub_category_name = df[df["occupation_sub_id"] == sub_category]["สาขาอาชีพรอง"].values
        sub_category_name = sub_category_name[0] if len(sub_category_name) > 0 else "ไม่พบข้อมูล"
    else:
        sub_category_name = "ไม่สามารถทำนายตำแหน่งงานรองได้ (ข้อมูลน้อยเกินไป)"

    return main_category, main_category_name, sub_category, sub_category_name

# 🔹 ฟังก์ชันค้นหาตำแหน่งงานที่คล้ายกันโดยใช้ Cosine Similarity
def find_similar_jobs(job_title, top_n=3):
    job_vector = vectorizer.transform([job_title])
    similarities = cosine_similarity(job_vector, X).flatten()

    # 🔎 คัดเฉพาะตำแหน่งที่คล้ายกันมากกว่า 0.2
    similar_indices = np.argsort(similarities)[::-1]
    similar_indices = [i for i in similar_indices if similarities[i] > 0.2][:top_n]

    similar_jobs = df.iloc[similar_indices][["ชื่อตำแหน่งงาน", "occupation_id", "สาขาอาชีพหลัก"]].copy()
    similar_jobs["similarity_score"] = similarities[similar_indices]

    return similar_jobs

# 🔥 วนลูปเพื่อรับค่าจาก Terminal ซ้ำได้เรื่อยๆ
while True:
    job_example = input("\n🔎 ป้อนชื่อตำแหน่งงานที่ต้องการทำนาย (พิมพ์ 'q' หรือ 'exit' เพื่อออก): ")

    if job_example.lower() in ["q", "exit"]:
        print("👋 โปรแกรมสิ้นสุดการทำงาน")
        break

    predicted_main, predicted_main_name, predicted_sub, predicted_sub_name = predict_job_category(job_example)

    print(f"\n✅ ตำแหน่ง '{job_example}' ควรอยู่ในหมวดหลัก: {predicted_main} ({predicted_main_name}) หมวดรอง: {predicted_sub} ({predicted_sub_name})")

    similar_jobs = find_similar_jobs(job_example)

    print("\n🔍 ตำแหน่งงานที่คล้ายกัน:")
    print(similar_jobs.sort_values(by="similarity_score", ascending=False).to_string(index=False))