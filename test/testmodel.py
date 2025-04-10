import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 🔹 โหลดข้อมูล
file_path = "Updated_Occupation_Titles.csv"
df = pd.read_csv(file_path)

# 🔹 ลบแถวที่มีค่า NaN และอักขระพิเศษ
df = df.dropna()
df["ชื่อตำแหน่งงาน"] = df["ชื่อตำแหน่งงาน"].str.replace(r"[^\w\s]", "", regex=True)

# 🔹 แปลงค่าหมวดหมู่เป็นตัวเลข
le_occupation = LabelEncoder()
le_sub_occupation = LabelEncoder()
df["occupation_id"] = le_occupation.fit_transform(df["occupation_id"])
df["occupation_sub_id"] = le_sub_occupation.fit_transform(df["occupation_sub_id"])

# 🔹 ใช้ TF-IDF เพื่อแปลงชื่อตำแหน่งงานเป็นเวกเตอร์
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["ชื่อตำแหน่งงาน"])

# 🔹 กำหนด Target (ค่าที่ต้องทำนาย)
y_main = df["occupation_id"]  # ตำแหน่งงานหลัก
y_sub = df["occupation_sub_id"]  # ตำแหน่งงานรอง

# 🔹 แบ่งข้อมูล Train/Test (80% - 20%)
X_train, X_test, y_train_main, y_test_main = train_test_split(X, y_main, test_size=0.2, random_state=42)
X_train, X_test, y_train_sub, y_test_sub = train_test_split(X, y_sub, test_size=0.2, random_state=42)

# 🔥 **สร้างและฝึกโมเดล Random Forest (ตำแหน่งงานหลัก)**
rf_main = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_main.fit(X_train, y_train_main)

# 🔥 **สร้างและฝึกโมเดล SVM (ตำแหน่งงานรอง)**
svm_sub = SVC(kernel="linear", C=1.0)
svm_sub.fit(X_train, y_train_sub)

# 🔹 ทดสอบความแม่นยำของโมเดล
y_pred_main = rf_main.predict(X_test)
y_pred_sub = svm_sub.predict(X_test)

accuracy_main = accuracy_score(y_test_main, y_pred_main)
accuracy_sub = accuracy_score(y_test_sub, y_pred_sub)

print(f"🎯 ความแม่นยำของโมเดลสำหรับตำแหน่งงานหลัก (Random Forest): {accuracy_main:.2f}")
print(f"🎯 ความแม่นยำของโมเดลสำหรับตำแหน่งงานรอง (SVM): {accuracy_sub:.2f}")

# 🔹 ฟังก์ชันทำนายตำแหน่งงานหลัก/รองจากชื่อตำแหน่งงาน
def predict_job_category(job_title):
    job_vector = vectorizer.transform([job_title])
    main_category = le_occupation.inverse_transform(rf_main.predict(job_vector))[0]
    sub_category = le_sub_occupation.inverse_transform(svm_sub.predict(job_vector))[0]
    # 🔹 ค้นหาชื่อของสาขาอาชีพหลัก
    main_category_name = df[df["occupation_id"] == main_category]["สาขาอาชีพหลัก"].values
    main_category_name = main_category_name[0] if len(main_category_name) > 0 else "ไม่พบข้อมูล"

    # 🔹 ค้นหาชื่อของสาขาอาชีพรอง
    sub_category_name = df[df["occupation_sub_id"] == sub_category]["สาขาอาชีพรอง"].values
    sub_category_name = sub_category_name[0] if len(sub_category_name) > 0 else "ไม่พบข้อมูล"
    
    return main_category, main_category_name, sub_category, sub_category_name

# 🔥 วนลูปเพื่อรับค่าจาก Terminal ซ้ำได้เรื่อยๆ
while True:
    # 🔥 ทดสอบทำนาย
    job_example = input("\n🔎 ป้อนชื่อตำแหน่งงานที่ต้องการทำนาย (พิมพ์ 'q' หรือ 'exit' เพื่อออก): ")

    if job_example.lower() in ["q", "exit"]:
        print("👋 โปรแกรมสิ้นสุดการทำงาน")
        break

    predicted_main, predicted_main_name, predicted_sub, predicted_sub_name = predict_job_category(job_example)

    print(f"\n✅ ตำแหน่ง '{job_example}' ควรอยู่ในหมวดหลัก: {predicted_main} ({predicted_main_name}) หมวดรอง: {predicted_sub} ({predicted_sub_name})")
