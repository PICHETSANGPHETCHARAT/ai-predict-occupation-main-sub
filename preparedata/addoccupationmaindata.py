import pandas as pd
import openai
import time
import os
from dotenv import load_dotenv  # นำเข้า python-dotenv

# โหลด environment variables จาก .env
load_dotenv()
# API Key สำหรับ OpenAI
open_api_key = os.getenv("OPENAI_API_KEY_DEV")
client = OpenAI(api_key=open_api_key)

# โหลดข้อมูลจากไฟล์ CSV
file_path_new = "/Users/itpmac001/Documents/Python Code/Test/model_predict_occupation/สาขาอาชีพหลัก.csv"
df = pd.read_csv(file_path_new).fillna("")  # แทน NaN ด้วยค่าว่าง

# ลบอักขระพิเศษ [U+200E] ที่อาจอยู่ในข้อมูล
df = df.applymap(lambda x: x.replace("\u200E", "") if isinstance(x, str) else x)

# ตัดแถวที่มีค่า NaN ในคอลัมน์ "ชื่อตำแหน่งงาน"
df = df[df["ชื่อตำแหน่งงาน"].notna()]

# แสดงชื่อคอลัมน์เพื่อเช็คว่าไม่มีปัญหา
df.columns = df.columns.str.strip()  # ลบช่องว่างที่อาจมีอยู่
print(df.columns)

# ฟังก์ชันเพื่อสร้างตำแหน่งงานใหม่จาก ChatGPT API
def generate_job_titles(occupation_name, num_needed, existing_titles):
    prompt = (
        f"กรุณาสร้างชื่อตำแหน่งงานสำหรับสายอาชีพ '{occupation_name}' จำนวน {num_needed} ตำแหน่ง "
        f"โดยต้องไม่ซ้ำกับตำแหน่งงานเดิมต่อไปนี้: {existing_titles} "
        f"โปรดตอบเป็นรายการโดยไม่มีตัวเลขนำหน้าและไม่มีข้อความเกินกว่ารายการชื่อตำแหน่งงาน "
        f"และให้มีการผสมผสานของภาษาไทยและภาษาอังกฤษ"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "คุณคือโมเดลที่ช่วยในการสร้างชื่อตำแหน่งงาน"},
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

# ฟังก์ชันเติมตำแหน่งงานที่หายไปด้วย ChatGPT
def fill_missing_titles(row, existing_titles):
    occupation_name = row['สาขาอาชีพหลัก']
    # ใช้ ChatGPT API เพื่อเติมตำแหน่งงานที่หายไป
    new_title = generate_job_titles(occupation_name, 1, existing_titles)
    return new_title[0] if new_title else "ตำแหน่งงานไม่ระบุ"  # กรณีที่ไม่สามารถเติมได้

# กำหนดจำนวนตัวอย่างที่ต้องการเพิ่ม
num_samples_target = 100  # เป้าหมายจำนวนตัวอย่างในแต่ละสาขาอาชีพหลัก

new_data = []

# ดึงสาขาอาชีพหลักที่ไม่ซ้ำกัน
unique_occupations = df["สาขาอาชีพหลัก"].unique()

# นับจำนวนสาขาอาชีพหลักทั้งหมดก่อนการเพิ่ม
total_occupations_before = len(unique_occupations)
print(f"จำนวนสาขาอาชีพหลักทั้งหมดที่มี: {total_occupations_before} สาขา")

# เติมตำแหน่งงานให้ครบจำนวนที่ต้องการในแต่ละสาขาอาชีพหลัก
for occupation_main in unique_occupations:
    # นับจำนวนตำแหน่งงานในแต่ละสาขาอาชีพหลักที่มีอยู่
    occupation_df = df[df["สาขาอาชีพหลัก"] == occupation_main]
    existing_titles = occupation_df["ชื่อตำแหน่งงาน"].dropna().unique().tolist()
    num_existing = len(existing_titles)
    
    # แสดงจำนวนตำแหน่งงานก่อนเพิ่ม
    print(f"\n🔹 สาขาอาชีพหลัก: {occupation_main}")
    print(f"🔸 มีตำแหน่งงานอยู่แล้ว: {num_existing} ตำแหน่ง")

    # ดึง occupation_id สำหรับสาขาอาชีพนี้
    if "occupation_id" in df.columns:
        occupation_id = occupation_df["occupation_id"].iloc[0] if not occupation_df.empty else ""
    else:
        occupation_id = ""
    
    # ตรวจสอบจำนวนตำแหน่งงานในสาขาอาชีพหลักว่าเกิน 100 หรือไม่
    if num_existing >= 100:
        print(f"🔸 จำนวนตำแหน่งงานในสาขาอาชีพหลัก '{occupation_main}' เกิน 100 แล้ว, ไม่ต้องเพิ่มตำแหน่งงาน")
        existing_titles = existing_titles[:100]  # ลบตำแหน่งงานที่เกินออกให้เหลือ 100
        num_needed = 0  # ไม่ต้องเพิ่มตำแหน่งงานใหม่
        total_titles_after = 100  # จำนวนตำแหน่งงานหลังจากปรับลด
    else:
        # คำนวณจำนวนที่ต้องการเพิ่มเพื่อให้ครบ 100
        num_needed = num_samples_target - num_existing
        num_needed = min(num_needed, 100 - num_existing)

        total_titles_after = num_existing + num_needed

        print(f"🔸 ต้องเพิ่มอีก: {num_needed} ตำแหน่ง")
        print(f"🔸 ใช้ occupation_id: {occupation_id}")

        # ใช้ API ของ ChatGPT สร้างชื่อตำแหน่งงานใหม่
        new_titles = generate_job_titles(occupation_main, num_needed, existing_titles)
        
        # กรองตำแหน่งงานใหม่ที่ไม่ซ้ำกับตำแหน่งงานที่มีอยู่แล้ว
        new_titles = [title for title in new_titles if title not in existing_titles]

        # ถ้าจำนวนที่ได้จาก API มากกว่าที่ต้องการ ก็จำกัดให้เท่ากับที่ต้องการ
        new_titles = new_titles[:num_needed]

        print(f"✅ ตำแหน่งงานที่เพิ่ม: {len(new_titles)} ตำแหน่ง")

        # เพิ่มข้อมูลใหม่ในรูปแบบ DataFrame
        for title in new_titles:
            if "occupation_id" in df.columns:
                new_data.append({"occupation_id": occupation_id, "สาขาอาชีพหลัก": occupation_main, "ชื่อตำแหน่งงาน": title})
            else:
                new_data.append({"สาขาอาชีพหลัก": occupation_main, "ชื่อตำแหน่งงาน": title})

        # หน่วงเวลาเล็กน้อยเพื่อป้องกันการเรียก API ถี่เกินไป
        time.sleep(1)

    # รีเช็คค่าว่างใน "ชื่อตำแหน่งงาน" และเติมด้วย ChatGPT
    df.loc[df["ชื่อตำแหน่งงาน"].isna(), "ชื่อตำแหน่งงาน"] = df[df["ชื่อตำแหน่งงาน"].isna()].apply(fill_missing_titles, existing_titles=existing_titles, axis=1)

    # นับจำนวนตำแหน่งงานหลังจากเพิ่มตำแหน่งงานใหม่
    print(f"🔸 จำนวนตำแหน่งงานหลังเพิ่ม: {total_titles_after} ตำแหน่ง")
    print("------------------------------------------------")

# สร้าง DataFrame ใหม่จากตำแหน่งงานที่เพิ่ม
if new_data:
    df_new_titles = pd.DataFrame(new_data)
    
    # รวมข้อมูลเดิมกับตำแหน่งงานที่เพิ่มใหม่
    df_final = pd.concat([df, df_new_titles], ignore_index=True)
else:
    print("ไม่มีข้อมูลใหม่ที่ต้องเพิ่ม")
    df_final = df

# ลบเครื่องหมาย "-" ที่หน้าชื่อตำแหน่งงาน
df_final["ชื่อตำแหน่งงาน"] = df_final["ชื่อตำแหน่งงาน"].str.lstrip("-").str.strip()

# จัดเรียงข้อมูลตาม สาขาอาชีพหลัก และ ชื่อตำแหน่งงาน
df_final = df_final.sort_values(by=["สาขาอาชีพหลัก", "ชื่อตำแหน่งงาน"]).reset_index(drop=True)

# บันทึกเป็นไฟล์ CSV ใหม่
output_path = "dataforthaibert.csv"
df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nบันทึกข้อมูลที่อัปเดตแล้วลงในไฟล์: {output_path}")
print(f"ข้อมูลทั้งหมดมี {len(df_final)} แถว")

# ดูจำนวนสาขาอาชีพหลักทั้งหมด
unique_occupations_count = df_final["สาขาอาชีพหลัก"].nunique()
print(f"จำนวนสาขาอาชีพหลักทั้งหมดที่มีในไฟล์: {unique_occupations_count} สาขา")