import pandas as pd
import openai
import time
from dotenv import load_dotenv  # นำเข้า python-dotenv

# โหลด environment variables จาก .env
load_dotenv()
# ตั้งค่า API Key ของคุณ
import os
open_api_key = os.getenv("OPENAI_API_KEY_DEV")
client = OpenAI(api_key=open_api_key)

# โหลดข้อมูลจากไฟล์ CSV
file_path = "สาขาอาชีพรองพัฒนาAIสาขาอาชีพ.csv"
df = pd.read_csv(file_path).fillna("")  # แทน NaN ด้วยค่าว่าง

# ลบอักขระพิเศษ [U+200E] ที่อาจอยู่ในข้อมูล
df = df.applymap(lambda x: x.replace("\u200E", "") if isinstance(x, str) else x)

# ลบแถวที่มีค่าว่าง
df = df.replace("", pd.NA).dropna(how="any")

# แสดงชื่อคอลัมน์เพื่อเช็คว่าไม่มีปัญหา
df.columns = df.columns.str.strip()  # ลบช่องว่างที่อาจมีอยู่
print(df.columns)

# แยกชื่อตำแหน่งงานออกจากกัน
df["ชื่อตำแหน่งงาน"] = df["ชื่อตำแหน่งงาน"].astype(str).str.split(" . ")
df_expanded = df.explode("ชื่อตำแหน่งงาน")

# นับจำนวนตำแหน่งงานในแต่ละ occupation_sub_id
count_positions = df_expanded.groupby(["occupation_id", "occupation_sub_id", "name"])["ชื่อตำแหน่งงาน"].nunique().reset_index()

# ค้นหาnameที่มีตำแหน่งงานน้อยกว่า 10 และรวม "สาขาอาชีพหลัก" กลับมา
sub_occupations_need_more = count_positions[count_positions["ชื่อตำแหน่งงาน"] < 10].merge(
    df[["occupation_id", "occupation_sub_id", "สาขาอาชีพหลัก"]].drop_duplicates(),
    on="occupation_sub_id", 
    how="left"
)

# ลบ `_x` และ `_y` โดยเลือกคอลัมน์ที่ต้องการเก็บ
if "occupation_id_x" in sub_occupations_need_more.columns:
    sub_occupations_need_more["occupation_id"] = sub_occupations_need_more["occupation_id_x"].fillna(
        sub_occupations_need_more["occupation_id_y"]
    )
    sub_occupations_need_more = sub_occupations_need_more.drop(columns=["occupation_id_x", "occupation_id_y"])

print("🔍 คอลัมน์ที่มีอยู่ใน sub_occupations_need_more:", sub_occupations_need_more.columns)

def generate_job_titles(occupation_name, num_needed):
    """ใช้ ChatGPT API สร้างชื่อตำแหน่งงานใหม่"""
    prompt = (
    f"กรุณาสร้างชื่อตำแหน่งงานสำหรับสายอาชีพ '{occupation_name}' จำนวน {num_needed} ตำแหน่ง "
    f"โดยต้องไม่ซ้ำกับของเดิม โปรดตอบเป็นรายการโดยไม่มีตัวเลขนำหน้าและไม่มีข้อความเกินกว่ารายการชื่อตำแหน่งงาน"
)

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญด้านการจัดหางาน"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        response_text = response.choices[0].message.content.strip()
        new_titles = [line.strip() for line in response_text.split("\n") if line.strip() and not line[0].isdigit()]
        return [title.strip() for title in new_titles if title.strip()]
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการเรียกใช้ API: {e}")
        return []


new_data = []

# เติมตำแหน่งงานให้ครบ 10 รายการ
for _, row in sub_occupations_need_more.iterrows():
    print(f"\n🔹 กำลังเพิ่มตำแหน่งงานสำหรับ: {row['สาขาอาชีพหลัก']} - {row['name']}")
    
    occupation_id = row["occupation_id"]  # ดึงค่า occupation_id ให้ตรง
    occupation_main = row["สาขาอาชีพหลัก"]  # ดึงค่า สาขาอาชีพหลัก ให้ตรง
    sub_id = row["occupation_sub_id"]
    sub_name = row["name"]

    # ดึงตำแหน่งงานที่มีอยู่
    existing_titles = (
        df_expanded[df_expanded["occupation_sub_id"] == sub_id]["ชื่อตำแหน่งงาน"]
        .dropna()
        .unique()
        .tolist()
    )
    num_existing = len(existing_titles)
    num_needed = 10 - num_existing

    print(f"🔸 มีตำแหน่งงานอยู่แล้ว: {existing_titles}")
    print(f"🔸 ต้องเพิ่มอีก: {num_needed} ตำแหน่ง")

    # ใช้ API ของ ChatGPT สร้างชื่อตำแหน่งงานใหม่
    new_titles = generate_job_titles(sub_name, num_needed)
    
    print(f"✅ ตำแหน่งงานที่เพิ่ม: {new_titles}")

    # เพิ่มข้อมูลใหม่ในรูปแบบ DataFrame
    for title in new_titles:
        new_data.append([occupation_id, occupation_main, sub_id, sub_name, title])

    # หน่วงเวลาเล็กน้อยเพื่อป้องกันการเรียก API ถี่เกินไป
    time.sleep(0.3)

# สร้าง DataFrame ใหม่จากตำแหน่งงานที่เพิ่ม
df_new_titles = pd.DataFrame(
    new_data, columns=["occupation_id", "สาขาอาชีพหลัก", "occupation_sub_id", "name", "ชื่อตำแหน่งงาน"]
)

# รวมข้อมูลเดิมกับตำแหน่งงานที่เพิ่มใหม่
df_final = pd.concat([df_expanded, df_new_titles])

# บันทึกเป็นไฟล์ CSV ใหม่
output_path = "Updated_Occupation_Titles.csv"
df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"บันทึกข้อมูลที่อัปเดตแล้วลงในไฟล์: {output_path}")