import pandas as pd
import numpy as np
import asyncio
import json
import joblib
import mysql.connector
from openai import OpenAI
from pythainlp.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# 🔹 โหลดโมเดลและส่วนประกอบที่จำเป็น
print("🔄 กำลังโหลดโมเดลและส่วนประกอบ...")
try:
    # โหลดโมเดล RandomForest
    rf_model = joblib.load('models/occupation_rf_model.joblib')
    # โหลด TF-IDF Vectorizer
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    # โหลด LabelEncoder
    le_occupation = joblib.load('models/label_encoder.joblib')
    # โหลดโมเดล Word2Vec
    w2v_model = Word2Vec.load('models/word2vec_model.bin')
    # โหลดข้อมูลหมวดหมู่อาชีพ
    occupation_mapping = pd.read_csv('models/occupation_mapping.csv')
    
    print("✅ โหลดโมเดลและส่วนประกอบเรียบร้อยแล้ว")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
    exit(1)

# 🔹 โหลดข้อมูลเพื่อใช้สำหรับ Cosine Similarity (ถ้าจำเป็น)
try:
    df = pd.read_csv("Updated_Occupation_Titles.csv", encoding="utf-8")
    df = df.dropna()
    df["ชื่อตำแหน่งงาน"] = df["ชื่อตำแหน่งงาน"].str.replace(r"[^\w\s]", "", regex=True)
    X = vectorizer.transform(df["ชื่อตำแหน่งงาน"])
    print("✅ โหลดข้อมูลสำหรับ Cosine Similarity เรียบร้อยแล้ว")
except Exception as e:
    print(f"⚠️ ไม่สามารถโหลดข้อมูลสำหรับ Cosine Similarity: {str(e)}")
    df = None
    X = None

# 🔹 ฟังก์ชันสำหรับแปลงข้อความเป็นเวกเตอร์
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

# 🔹 ฟังก์ชันเชื่อมต่อฐานข้อมูล
def get_db_connection_120other():
    return mysql.connector.connect(
        host="192.168.100.120",
        user="jobbkk_other",
        password="orerthjobk2022$",
        database="jobbkk_other",
    )

# 🔹 คลาสสำหรับแปลงข้อมูลจากฐานข้อมูลเป็นออบเจกต์
class OccupationSub:
    def __init__(self, occupation_sub_id, name):
        self.occupation_sub_id = occupation_sub_id
        self.name = name

    @classmethod
    def from_db(cls, db_record):
        return cls(
            occupation_sub_id=db_record["occupation_sub_id"], name=db_record["name"]
        )

# 🔹 ฟังก์ชันสำหรับดึงข้อมูลตำแหน่งงานรองจากฐานข้อมูล
async def get_occupation_subs(occupation_id: int):
    try:
        conn = get_db_connection_120other()
        if not conn:
            print("❌ ไม่สามารถเชื่อมต่อฐานข้อมูลได้")
            return []

        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
           SELECT 
               os.id AS occupation_sub_id,
               od.name 
           FROM 
               occupation_sub os
               INNER JOIN occupation_sub_description od ON os.id = od.occupation_sub_id 
           WHERE 
               os.occupation_id = %s 
               AND os.is_status = '1'
               AND os.is_flags = '0'
               AND os.is_online = '1'
               AND od.language_id = '1'
               ORDER BY CONVERT(od.name USING tis620) ASC;
        """,
            (occupation_id,),
        )

        occupation_subs = cursor.fetchall()
        cursor.close()
        conn.close()

        return [
            OccupationSub.from_db(sub) for sub in occupation_subs
        ]

    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล: {str(e)}")
        return []

# 🔹 ฟังก์ชันสำหรับเรียกใช้ ChatGPT API เพื่อทำนายตำแหน่งงานรอง
async def predict_sub_occupation_with_gpt(job_title, main_occupation, sub_occupations):
    try:
        # แปลงข้อมูลตำแหน่งงานรองให้อยู่ในรูปแบบที่เหมาะสม
        sub_options = [
            {"id": sub.occupation_sub_id, "name": sub.name} for sub in sub_occupations
        ]

        # สร้าง prompt สำหรับส่งไปยัง ChatGPT API
        prompt = f"""
        ชื่อตำแหน่งงาน: "{job_title}"
        สาขาอาชีพหลัก: "{main_occupation}"
        
        จากรายการสาขาอาชีพรองต่อไปนี้ โปรดเลือกสาขาที่เหมาะสมที่สุดสำหรับตำแหน่งงานนี้:
        {json.dumps(sub_options, ensure_ascii=False)}
        
        โปรดตอบกลับด้วย ID ของสาขาอาชีพรองที่เหมาะสมที่สุดเพียงค่าเดียว ในรูปแบบ JSON: {{"occupation_sub_id": ID}}
        """

        import os
        open_api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=open_api_key)

        # ใช้ OpenAI ใหม่ในการเรียก API
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "คุณคือผู้ช่วยสร้างการประกาศตำแหน่งงาน"},
                {"role": "user", "content": prompt},
            ],
        )

        # ดึงผลลัพธ์จากการตอบกลับ
        content = completion.choices[0].message.content

        # แปลงข้อความตอบกลับเป็น JSON
        try:
            response_data = json.loads(content)
            sub_id = response_data.get("occupation_sub_id")

            # หา sub_occupation ที่ตรงกับ ID
            for sub in sub_occupations:
                if sub.occupation_sub_id == sub_id:
                    return sub.occupation_sub_id, sub.name

            return None, "ไม่พบข้อมูลสาขาอาชีพรองที่ตรงกับ ID"
        except json.JSONDecodeError:
            return None, "รูปแบบข้อมูลจาก API ไม่ถูกต้อง"

    except Exception as e:
        return None, f"เกิดข้อผิดพลาด: {str(e)}"

# 🔹 ฟังก์ชันทำนายตำแหน่งงานหลักและรอง
async def predict_job_category(job_title):
    # แปลงชื่อตำแหน่งงานเป็นเวกเตอร์ด้วย TF-IDF
    job_vector = vectorizer.transform([job_title])
    
    # แบ่งคำภาษาไทยสำหรับใช้กับ Word2Vec
    tokens = word_tokenize(job_title, engine='newmm')
    
    # 🔥 ทำนายตำแหน่งงานหลักด้วย RandomForest
    main_category_encoded = rf_model.predict(job_vector)[0]
    
    # คำนวณความน่าจะเป็นของแต่ละหมวดหมู่
    probabilities = rf_model.predict_proba(job_vector)[0]
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    top_3_categories_encoded = top_3_indices
    top_3_categories = le_occupation.inverse_transform(top_3_indices)
    top_3_probs = probabilities[top_3_indices]
    
    # แสดงความน่าจะเป็น 3 อันดับแรก
    print("\nความน่าจะเป็นของสาขาอาชีพหลัก (3 อันดับแรก):")
    for cat_encoded, cat, prob in zip(top_3_categories_encoded, top_3_categories, top_3_probs):
        cat_name = occupation_mapping[occupation_mapping["occupation_id"] == cat]["สาขาอาชีพหลัก"].values
        cat_name = cat_name[0] if len(cat_name) > 0 else "ไม่พบข้อมูล"
        print(f"รหัส: {cat} ({cat_encoded}), ชื่อ: {cat_name}, ความน่าจะเป็น: {prob:.4f}")

    # แปลงให้เป็น int
    main_category = le_occupation.inverse_transform([main_category_encoded])[0]
    main_category = int(main_category)

    # 🔹 ค้นหาชื่อของสาขาอาชีพหลัก
    main_category_name = occupation_mapping[occupation_mapping["occupation_id"] == main_category]["สาขาอาชีพหลัก"].values
    main_category_name = main_category_name[0] if len(main_category_name) > 0 else "ไม่พบข้อมูล"

    # 🔥 ดึงข้อมูลตำแหน่งงานรองจากฐานข้อมูล
    sub_occupations = await get_occupation_subs(main_category)

    if sub_occupations:
        # 🔥 ใช้ ChatGPT API เพื่อทำนายตำแหน่งงานรอง
        sub_id, sub_name = await predict_sub_occupation_with_gpt(
            job_title, main_category_name, sub_occupations
        )
        return main_category, main_category_name, sub_id, sub_name
    else:
        return main_category, main_category_name, None, "ไม่พบข้อมูลตำแหน่งงานรอง"

# 🔥 ฟังก์ชันหลักสำหรับรันโปรแกรม
async def main():
    print("\n🔎 โปรแกรมทำนายสาขาอาชีพ")
    print("============================")
    
    # 🔥 วนลูปเพื่อรับค่าจาก Terminal ซ้ำได้เรื่อยๆ
    while True:
        job_example = input(
            "\n🔎 ป้อนชื่อตำแหน่งงานที่ต้องการทำนาย (พิมพ์ 'q' หรือ 'exit' เพื่อออก): "
        )

        if job_example.lower() in ["q", "exit"]:
            print("👋 โปรแกรมสิ้นสุดการทำงาน")
            break

        try:
            predicted_main, predicted_main_name, predicted_sub, predicted_sub_name = (
                await predict_job_category(job_example)
            )

            print("\n📊 ผลการทำนาย:")
            print(f"✅ ตำแหน่ง '{job_example}' ควรอยู่ในหมวดหลัก: {predicted_main} ({predicted_main_name})")
            print(f"✅ หมวดรอง: {predicted_sub} ({predicted_sub_name})")
                
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

# รันโปรแกรมด้วย asyncio
if __name__ == "__main__":
    # สร้างโฟลเดอร์ models ถ้ายังไม่มี
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
        
    asyncio.run(main())