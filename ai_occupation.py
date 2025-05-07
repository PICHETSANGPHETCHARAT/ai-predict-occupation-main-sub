"""
JobBKK AI Occupation Prediction API
------------------------------
API สำหรับทำนายตำแหน่งงานหลักและตำแหน่งงานย่อยด้วย AI
รองรับทั้งการใช้งานผ่าน AI model (ThaiBERT) และ GPT
"""

# === 1. การ import และตั้งค่าเริ่มต้น ===
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel
import torch
import pickle
import mysql.connector
import json
from openai import OpenAI
import uvicorn
from dotenv import load_dotenv
import os
import re

# โหลด environment variables จาก .env
load_dotenv()

# === OpenAI GPT Client Setup ===
open_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=open_api_key)

# === FastAPI Setup ===
app = FastAPI(
    title="JobBKK AI occupation API",
    description="API for predict occupation main and sub using AI",
    version="1.0.0",
    root_path="/ai_predict_occupation",
)
templates = Jinja2Templates(directory="templates")

# === Load ML Model & Label Encoder ===
MODEL_PATH = "./model/model_thaibert_v2"
with open(os.path.join(MODEL_PATH, "labels.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "JobBKK AI occupation API"}

# --- Pydantic Models for API Requests ---
class PredictRequestMember(BaseModel):
    """
    สำหรับรับข้อมูลการทำนายตำแหน่งงานของผู้หางาน
    """

    job_title: str


class PredictRequestEmployer(BaseModel):
    """
    สำหรับรับข้อมูลการทำนายตำแหน่งงานของนายจ้าง รวมถึงประเภทธุรกิจ
    """

    job_title: str
    bussiness_type: str


# --- Database Models ---
class OccupationSub:
    """
    โมเดลสำหรับเก็บข้อมูลตำแหน่งงานย่อย
    """

    def __init__(self, occupation_sub_id, name):
        self.occupation_sub_id = occupation_sub_id
        self.name = name

    @classmethod
    def from_db(cls, db_record):
        return cls(
            occupation_sub_id=db_record["occupation_sub_id"], name=db_record["name"]
        )


class OccupationMain:
    """
    โมเดลสำหรับเก็บข้อมูลตำแหน่งงานหลัก
    """

    def __init__(self, occupation_id, name):
        self.occupation_id = occupation_id
        self.name = name

    @classmethod
    def from_db(cls, db_record):
        return cls(occupation_id=db_record["occupation_id"], name=db_record["name"])


# === 3. Database Connection ===
def get_db_connection_120other():
    """
    สร้างการเชื่อมต่อกับฐานข้อมูล jobbkk_other

    Returns:
        mysql.connector.connection: การเชื่อมต่อกับฐานข้อมูล
    """
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


async def get_occupation_subs(occupation_id: int):
    """
    ดึงข้อมูลตำแหน่งงานย่อยตาม occupation_id

    Args:
        occupation_id: รหัสตำแหน่งงานหลัก

    Returns:
        list: รายการตำแหน่งงานย่อย
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection_120other()
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
                ORDER BY CONVERT(od.name USING tis620) ASC
            """,
            (occupation_id,),
        )
        occupation_subs = cursor.fetchall()
        return [OccupationSub.from_db(sub) for sub in occupation_subs]
    except Exception as e:
        print(f"❌ Error DB: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


async def get_occupation_main():
    """
    ดึงข้อมูลตำแหน่งงานหลักทั้งหมด

    Returns:
        list: รายการตำแหน่งงานหลัก
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection_120other()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT 
                o.id AS occupation_id,
                od.name 
            FROM 
                occupation o
                INNER JOIN occupation_description od ON o.id = od.occupation_id 
            WHERE 
                o.is_status = '1'
                AND o.is_flags = '0'
                AND o.is_online = '1'
                AND od.language_id = '1'
                ORDER BY CONVERT(od.name USING tis620) ASC
            """
        )
        occupation = cursor.fetchall()
        return [OccupationMain.from_db(main) for main in occupation]
    except Exception as e:
        print(f"❌ Error DB: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# === 4. Prediction Functions ===


# --- ThaiBERT Prediction ---
async def predict_with_thaibert(job_title: str):
    """
    ทำนายตำแหน่งงานหลักโดยใช้โมเดล ThaiBERT

    Args:
        job_title: ชื่อตำแหน่งงาน

    Returns:
        str: ชื่อตำแหน่งงานหลักที่ทำนาย
    """
    inputs = tokenizer(
        job_title, return_tensors="pt", truncation=True, padding=True, max_length=64
    )
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        main_occupation = label_encoder.inverse_transform([predicted_class_id])[0]
    return main_occupation


# --- GPT Prediction Functions ---
def generate_description_with_gpt(job_title: str) -> str:
    """
    ใช้ GPT สร้างคำอธิบายตำแหน่งงานจากชื่ออาชีพ

    Args:
        job_title (str): ชื่อตำแหน่งงาน เช่น "ช่างไม้"

    Returns:
        str: คำอธิบายตำแหน่งงาน (Job Description)
    """
    try:
        prompt = f"""
        โปรดเขียนคำอธิบายตำแหน่งงานสำหรับ "{job_title}" โดยระบุ:
        - หน้าที่รับผิดชอบหลัก
        - ทักษะที่จำเป็น
        - เครื่องมือหรือเทคโนโลยีที่ใช้บ่อย
        - ลักษณะงานโดยรวม

        ให้เขียนในรูปแบบทางการ ใช้ภาษาธรรมดา กระชับ และเข้าใจง่าย ความยาวไม่เกิน 100 คำ
        """

        completion = client.chat.completions.create(
            model="gpt-4-turbo",  # หรือ gpt-4o ก็ได้
            temperature=0.3,
            messages=[
                {"role": "system", "content": "คุณคือ HR ผู้เชี่ยวชาญด้านการอธิบายตำแหน่งงานในประเทศไทย"},
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content.strip()
        return content

    except Exception as e:
        return f"[ERROR] ไม่สามารถสร้างคำอธิบายได้: {e}"

def generate_descriptionsup_with_gpt(job_title: str, main_occupation: str) -> str:
    """
    ใช้ GPT สร้างคำอธิบายตำแหน่งงาน (ระดับละเอียด) จากชื่อตำแหน่งและสาขาอาชีพหลัก

    Args:
        job_title (str): ชื่อตำแหน่งงาน เช่น "ช่างไม้"
        main_occupation (str): หมวดอาชีพหลัก เช่น "ช่าง/ช่างเทคนิค/อิเลคโทรนิค"

    Returns:
        str: คำอธิบายตำแหน่งงานที่เน้นทักษะและลักษณะงานเฉพาะ
    """
    try:
        prompt = f"""
        คุณคือ HR ผู้เชี่ยวชาญด้านการวิเคราะห์ตำแหน่งงานในประเทศไทย

        โปรดเขียนคำอธิบายตำแหน่งงานในรูปแบบ professional สำหรับตำแหน่ง "{job_title}" 
        ซึ่งอยู่ภายใต้สาขาอาชีพหลัก "{main_occupation}" โดยอธิบายดังนี้:

        - หน้าที่รับผิดชอบหลัก
        - ทักษะที่จำเป็น
        - เครื่องมือหรือเทคโนโลยีที่ใช้บ่อย
        - ลักษณะงานโดยรวม

        **สำคัญ:** เน้นคำเฉพาะที่สอดคล้องกับหมวด "{main_occupation}" อย่างชัดเจน เช่น ถ้าเป็นงานช่าง ให้ใช้คำว่า "ลงมือปฏิบัติ", "เครื่องมือช่าง", "เทคนิคช่าง", หรือ "งานฝีมือ"

        ใช้ภาษากระชับ ชัดเจน ไม่เกิน 100 คำ
        """

        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "คุณคือ HR ผู้เชี่ยวชาญด้านการอธิบายตำแหน่งงานในประเทศไทย"},
                {"role": "user", "content": prompt}
            ]
        )

        content = completion.choices[0].message.content.strip()
        return content

    except Exception as e:
        return f"[ERROR] ไม่สามารถสร้างคำอธิบายได้: {e}"
    
async def predict_main_occupation_with_gpt(job_title, main_occupation_list):
    """
    ทำนายตำแหน่งงานหลักโดยใช้ GPT

    Args:
        job_title: ชื่อตำแหน่งงาน
        main_occupation_list: รายการตำแหน่งงานหลักทั้งหมด

    Returns:
        tuple: (occupation_id, occupation_name) หรือ (None, error_message)
    """
    try:
        main_options = [
            {"id": main.occupation_id, "name": main.name}
            for main in main_occupation_list
        ]

        main_list_text = "\n".join(
            [
                f"{i+1}. {json.dumps(option, ensure_ascii=False)}"
                for i, option in enumerate(main_options)
            ]
        )
        # STEP 1: สร้าง job description
        job_desc = generate_description_with_gpt(job_title)

        prompt = f"""
        ตำแหน่งงาน: "{job_title}"

        คำอธิบายตำแหน่งงาน:
        {job_desc}

        จากคำอธิบายด้านบน โปรดเลือกเพียง 1 สาขาอาชีพหลักที่เหมาะสมที่สุดจากรายการต่อไปนี้

        ขั้นตอนการตัดสินใจ:
        1. วิเคราะห์ลักษณะงานและทักษะที่จำเป็น
        2. พิจารณาอุตสาหกรรมที่เกี่ยวข้อง
        3. ระบุขอบเขตความรับผิดชอบของตำแหน่ง
        4. เปรียบเทียบกับคำอธิบายสาขาอาชีพในรายการ
        5. หากเป็นงานที่เกี่ยวข้องหลายสาย ให้พิจารณาหน้าที่หลักเป็นตัวตัดสิน
        6. หากตำแหน่งมีบทบาทหลักใน “การขาย” แม้จะมีความรู้ด้านเทคนิค ให้จัดอยู่ในหมวด “ขาย”
        7. หากตำแหน่งเกี่ยวข้องกับการใช้ทักษะฝีมือแรงงาน เช่น ช่างไม้ ช่างปูน ช่างแอร์ ให้จัดอยู่ในหมวด "ช่าง/ช่างเทคนิค/อิเลคโทรนิค" ไม่ใช่ "ก่อสร้าง" หรือ "การผลิต"
        8. หากตำแหน่งมีบทบาทหลักในการควบคุมงานก่อสร้างในสถานที่จริง ให้จัดอยู่ในหมวด "ก่อสร้าง" แม้จะมีการใช้ซอฟต์แวร์ออกแบบร่วมด้วย
        9. หากตำแหน่งมีบทบาทเป็นผู้บริหารระดับกลางถึงสูง เช่น ผู้จัดการ ผู้ช่วยผู้จัดการ ให้จัดอยู่ในหมวด "บริหาร/ผู้จัดการ" แม้จะดูแลสายการผลิตหรือฝ่ายอื่นๆ

        รายการสาขาอาชีพหลัก:
        {main_list_text}

        สำคัญมาก: ให้ตอบกลับเป็น JSON เท่านั้น ในรูปแบบต่อไปนี้ โดยไม่เพิ่มข้อความอื่นใด:
        {{"id": 0, "name": "ตัวอย่างตำแหน่งงานหลัก"}}
        """
        print(f"Prompt: {prompt}")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "คุณคือ HR ผู้เชี่ยวชาญด้านการวิเคราะห์ตำแหน่งงานในประเทศไทย ที่มีความเข้าใจลึกซึ้งเกี่ยวกับสาขาอาชีพและคำศัพท์เฉพาะทางในภาษาไทย คุณตอบกลับเป็น JSON เท่านั้น ไม่ต้องอธิบายเหตุผล"},
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content.strip()
        print(f"GPT Response: {content}")
        # Extract JSON from the response if it's embedded in text
        import re
        json_match = re.search(r'\{.*\}', content)
        if json_match:
            json_str = json_match.group(0)
            try:
                response_data = json.loads(json_str)
                main_id = response_data.get("id") or response_data.get("occupation_id")

                if not main_id:
                    return None, f"ไม่พบ id จาก GPT: {response_data}"

                for main in main_occupation_list:
                    if main.occupation_id == main_id:
                        return main.occupation_id, main.name

                return None, "ไม่พบสาขาหลักที่ตรง"
            except json.JSONDecodeError:
                return None, f"ไม่สามารถแปลง JSON ได้: {json_str}"
        else:
            return None, f"GPT ตอบกลับไม่ใช่ JSON: {content}"

    except Exception as e:
        return None, f"GPT ERROR: {e}"

async def predict_sub_occupation_with_gpt(job_title, main_occupation, sub_occupations):
    """
    ทำนายตำแหน่งงานย่อยโดยใช้ GPT

    Args:
        job_title: ชื่อตำแหน่งงาน
        main_occupation: ชื่อตำแหน่งงานหลัก
        sub_occupations: รายการตำแหน่งงานย่อย

    Returns:
        tuple: (occupation_sub_id, occupation_sub_name) หรือ (None, error_message)
    """
    try:
        sub_options = [
            {"id": sub.occupation_sub_id, "name": sub.name} for sub in sub_occupations
        ]

        sub_list_text = "\n".join(
            [
                f"{i+1}. {json.dumps(option, ensure_ascii=False)}"
                for i, option in enumerate(sub_options)
            ]
        )
        # STEP 1: สร้าง job description
        job_desc = generate_descriptionsup_with_gpt(job_title, main_occupation)

        prompt = f"""
        ชื่อตำแหน่งงาน: "{job_title}"
        สาขาอาชีพหลัก: "{main_occupation}"

        คำอธิบายตำแหน่งงาน:
        {job_desc}

        จากรายการตำแหน่งงานย่อยด้านล่าง โปรดเลือกเพียง 1 รายการที่เหมาะสมที่สุดสำหรับตำแหน่งงานนี้ โดยดูจากลักษณะงานจริงในคำอธิบายข้างต้น:
        หมายเหตุ: หากไม่มีตำแหน่ง "ช่างไม้" โดยตรง ให้เลือกตำแหน่งที่ใกล้เคียงที่สุด เช่น "ช่างเทคนิค"

        {sub_list_text}

        ขั้นตอนการตัดสินใจ:
        1. พิจารณาหน้าที่หลักของตำแหน่งจากคำอธิบาย
        2. เปรียบเทียบกับชื่อและขอบเขตของแต่ละตำแหน่งย่อย
        3. หากชื่อซ้ำซ้อน ให้เลือกจากลักษณะการปฏิบัติงานจริง
        4. อย่าเลือกโดยอิงจากชื่อเท่านั้น แต่ให้ดูเนื้อหาของงานเป็นหลัก

        กรุณาตอบกลับเพียง 1 บรรทัด เป็น JSON ของตำแหน่งงานย่อยที่คุณเลือก เช่น:
        {{"id": 0, "name": "ตัวอย่างตำแหน่งงานรอง"}}
        """
        
        print(f"Prompt2: {prompt}")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "คุณคือ HR ผู้เชี่ยวชาญด้านการวิเคราะห์ตำแหน่งงาน"},
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content.strip()
        print(f"GPT Response2: {content}")
        if not content.startswith("{"):
            return None, f"GPT ตอบกลับไม่ใช่ JSON: {content}"

        response_data = json.loads(content)
        sub_id = response_data.get("id") or response_data.get("occupation_sub_id")

        if not sub_id:
            return None, f"ไม่พบ id จาก GPT: {response_data}"

        for sub in sub_occupations:
            if sub.occupation_sub_id == sub_id:
                return sub.occupation_sub_id, sub.name

        return None, "ไม่พบสาขารองที่ตรง"

    except Exception as e:
        return None, f"GPT ERROR: {e}"


async def predict_main_occupation_with_gpt_with_business(
    job_title, main_occupation_list, bussiness_type
):
    """
    ทำนายตำแหน่งงานหลักโดยใช้ GPT พร้อมข้อมูลประเภทธุรกิจ

    Args:
        job_title: ชื่อตำแหน่งงาน
        main_occupation_list: รายการตำแหน่งงานหลักทั้งหมด
        bussiness_type: ประเภทธุรกิจ

    Returns:
        tuple: (occupation_id, occupation_name) หรือ (None, error_message)
    """
    try:
        main_options = [
            {"id": main.occupation_id, "name": main.name}
            for main in main_occupation_list
        ]

        main_list_text = "\n".join(
            [
                f"{i+1}. {json.dumps(option, ensure_ascii=False)}"
                for i, option in enumerate(main_options)
            ]
        )

        prompt = f"""
        ชื่อตำแหน่งงาน: "{job_title}"
        ประเภทธุรกิจ: "{bussiness_type}"
        
        จากชื่อตำแหน่งงาน โปรดเลือกเพียง 1 สาขาอาชีพหลักที่เหมาะสมที่สุดสำหรับตำแหน่งงานดังกล่าว

        ขั้นตอนการตัดสินใจ:
        1. วิเคราะห์ลักษณะงานและทักษะที่จำเป็น
        2. พิจารณาอุตสาหกรรมที่เกี่ยวข้อง
        3. ระบุขอบเขตความรับผิดชอบของตำแหน่ง
        4. เปรียบเทียบกับคำอธิบายสาขาอาชีพในรายการ
        5. หากเป็นงานที่เกี่ยวข้องหลายสาย ให้พิจารณาหน้าที่หลักเป็นตัวตัดสิน
        6. หากตำแหน่งมีบทบาทหลักใน “การขาย” แม้จะมีความรู้ด้านเทคนิค ให้จัดอยู่ในหมวด “ขาย”
        7. หากตำแหน่งเกี่ยวข้องกับการใช้ทักษะฝีมือแรงงาน เช่น ช่างไม้ ช่างปูน ช่างแอร์ ให้จัดอยู่ในหมวด "ช่าง/ช่างเทคนิค/อิเลคโทรนิค" ไม่ใช่ "ก่อสร้าง" หรือ "การผลิต"


        {main_list_text}

        กรุณาตอบกลับเพียง 1 บรรทัด โดยให้เป็น JSON ของรายการที่คุณเลือก เช่น:
        {{"id": 0, "name": "ตัวอย่างตำแหน่งงานหลัก"}}
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "คุณคือ HR ผู้เชี่ยวชาญด้านการวิเคราะห์ตำแหน่งงานและธุรกิจ",
                },
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content.strip()

        if not content.startswith("{"):
            return None, f"GPT ตอบกลับไม่ใช่ JSON: {content}"

        response_data = json.loads(content)
        main_id = response_data.get("id") or response_data.get("occupation_id")

        if not main_id:
            return None, f"ไม่พบ id จาก GPT: {response_data}"

        for main in main_occupation_list:
            if main.occupation_id == main_id:
                return main.occupation_id, main.name

        return None, "ไม่พบสาขาหลักที่ตรง"

    except Exception as e:
        return None, f"GPT ERROR: {e}"
    

async def predict_sub_occupation_with_gpt_with_business(
    job_title, main_occupation, sub_occupations, bussiness_type
):
    """
    ทำนายตำแหน่งงานย่อยโดยใช้ GPT พร้อมข้อมูลประเภทธุรกิจ

    Args:
        job_title: ชื่อตำแหน่งงาน
        main_occupation: ชื่อตำแหน่งงานหลัก
        sub_occupations: รายการตำแหน่งงานย่อย
        bussiness_type: ประเภทธุรกิจ

    Returns:
        tuple: (occupation_sub_id, occupation_sub_name) หรือ (None, error_message)
    """
    try:
        sub_options = [
            {"id": sub.occupation_sub_id, "name": sub.name} for sub in sub_occupations
        ]

        sub_list_text = "\n".join(
            [
                f"{i+1}. {json.dumps(option, ensure_ascii=False)}"
                for i, option in enumerate(sub_options)
            ]
        )

        prompt = f"""
        ชื่อตำแหน่งงาน: "{job_title}"
        ประเภทธุรกิจ: "{bussiness_type}"
        สาขาอาชีพหลัก: "{main_occupation}"

        จากรายการสาขาอาชีพรองด้านล่างนี้ โปรดเลือกเพียง 1 รายการที่เหมาะสมที่สุดสำหรับตำแหน่งงานและประเภทธุรกิจดังกล่าว โดยพิจารณาจาก ID และชื่อ:

        {sub_list_text}

        กรุณาตอบกลับเพียง 1 บรรทัด โดยให้เป็น JSON ของรายการที่คุณเลือก เช่น:
        {{"id": 0, "name": "ตัวอย่างตำแหน่งงานรอง"}}
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "คุณคือ HR ผู้เชี่ยวชาญด้านการวิเคราะห์ตำแหน่งงานและธุรกิจ",
                },
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content.strip()

        if not content.startswith("{"):
            return None, f"GPT ตอบกลับไม่ใช่ JSON: {content}"

        response_data = json.loads(content)
        sub_id = response_data.get("id") or response_data.get("occupation_sub_id")

        if not sub_id:
            return None, f"ไม่พบ id จาก GPT: {response_data}"

        for sub in sub_occupations:
            if sub.occupation_sub_id == sub_id:
                return sub.occupation_sub_id, sub.name

        return None, "ไม่พบสาขารองที่ตรง"

    except Exception as e:
        return None, f"GPT ERROR: {e}"


# --- Main prediction handlers ---
async def handle_prediction_thaibert(
    request: Request, job_title: str, render_html: bool
):
    """
    จัดการกระบวนการทำนายโดยใช้ ThaiBERT

    Args:
        request: FastAPI Request
        job_title: ชื่อตำแหน่งงาน
        render_html: True หากต้องการ render HTML, False หากต้องการ JSON

    Returns:
        HTML/JSON response
    """
    conn = None
    cursor = None
    try:
        # ทำนายตำแหน่งงานหลักด้วย ThaiBERT
        main_occupation = await predict_with_thaibert(job_title)

        # ค้นหา occupation_id จากชื่อที่ทำนายได้
        conn = get_db_connection_120other()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT od.occupation_id 
            FROM occupation_description od 
            JOIN occupation o ON od.occupation_id = o.id
            WHERE od.name = %s 
            AND od.language_id = 1 
            AND o.is_online = '1' 
            AND o.is_status = '1' 
            AND o.is_flags = '0'
            LIMIT 1
            """,
            (main_occupation,),
        )
        row = cursor.fetchone()
        if not row:
            msg = f"ไม่พบรหัส occupation_id สำหรับ '{main_occupation}'"
            if render_html:
                return templates.TemplateResponse(
                    "index.html", {"request": request, "error": msg}
                )
            return JSONResponse(content={"error": msg}, status_code=404)

        occupation_id = row["occupation_id"]

        cursor.close()
        conn.close()

        # ทำนายตำแหน่งงานย่อยด้วย GPT
        sub_occupations = await get_occupation_subs(occupation_id)
        sub_id, sub_name = await predict_sub_occupation_with_gpt(
            job_title, main_occupation, sub_occupations
        )

        # สร้างการตอบกลับตามรูปแบบที่ต้องการ
        if render_html:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "job_title": job_title,
                    "occupation_id": occupation_id,
                    "main_occupation": main_occupation,
                    "sub_occupation_id": sub_id,
                    "sub_occupation_name": sub_name,
                },
            )
        else:
            return {
                "job_title": job_title,
                "main_occupation": main_occupation,
                "occupation_id": occupation_id,
                "sub_occupation_id": sub_id,
                "sub_occupation_name": sub_name,
            }

    except Exception as e:
        if render_html:
            return templates.TemplateResponse(
                "index.html", {"request": request, "error": f"เกิดข้อผิดพลาด: {str(e)}"}
            )
        return JSONResponse(
            content={"error": f"เกิดข้อผิดพลาด: {str(e)}"}, status_code=500
        )
    finally:
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass


async def handle_prediction_gpt(
    request: Request, job_title: str, render_html: bool, bussiness_type: str = None
):
    """
    จัดการกระบวนการทำนายโดยใช้ GPT

    Args:
        request: FastAPI Request
        job_title: ชื่อตำแหน่งงาน
        render_html: True หากต้องการ render HTML, False หากต้องการ JSON
        bussiness_type: ประเภทธุรกิจ (optional)

    Returns:
        HTML/JSON response
    """
    try:
        # ดึงข้อมูลตำแหน่งงานหลักทั้งหมด
        list_main_occupations = await get_occupation_main()

        # ทำนายตำแหน่งงานหลัก
        if bussiness_type:
            main_id, main_name = await predict_main_occupation_with_gpt_with_business(
                job_title, list_main_occupations, bussiness_type
            )
        else:
            main_id, main_name = await predict_main_occupation_with_gpt(
                job_title, list_main_occupations
            )

        # ทำนายตำแหน่งงานย่อย
        sub_occupations = await get_occupation_subs(main_id)

        if bussiness_type:
            sub_id, sub_name = await predict_sub_occupation_with_gpt_with_business(
                job_title, main_name, sub_occupations, bussiness_type
            )
        else:
            sub_id, sub_name = await predict_sub_occupation_with_gpt(
                job_title, main_name, sub_occupations
            )

        # สร้างการตอบกลับตามรูปแบบที่ต้องการ
        if render_html:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "job_title": job_title,
                    "occupation_id": main_id,
                    "main_occupation": main_name,
                    "sub_occupation_id": sub_id,
                    "sub_occupation_name": sub_name,
                    "bussiness_type": bussiness_type,
                },
            )
        else:
            result = {
                "job_title": job_title,
                "occupation_id": main_id,
                "main_occupation": main_name,
                "sub_occupation_id": sub_id,
                "sub_occupation_name": sub_name,
            }
            if bussiness_type:
                result["bussiness_type"] = bussiness_type
            return result

    except Exception as e:
        if render_html:
            return templates.TemplateResponse(
                "index.html", {"request": request, "error": f"เกิดข้อผิดพลาด: {str(e)}"}
            )
        return JSONResponse(
            content={"error": f"เกิดข้อผิดพลาด: {str(e)}"}, status_code=500
        )


# === 5. API Endpoints ===


@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    """
    หน้าแบบฟอร์มสำหรับการทำนายด้วย UI
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, job_title: str = Form(...)):
    """
    API สำหรับรับข้อมูลจากฟอร์มและส่งกลับผลการทำนายในรูปแบบ HTML
    """
    return await handle_prediction_thaibert(request, job_title, render_html=True)


@app.post("/predict/member")
async def predict_member_api(data: PredictRequestMember):
    """
    API สำหรับผู้หางาน ทำนายตำแหน่งงานจากชื่อตำแหน่ง

    Args:
        data: ข้อมูลคำขอที่มีชื่อตำแหน่งงาน

    Returns:
        JSONResponse: ผลการทำนาย
    """
    job_title = data.job_title
    if not job_title:
        return JSONResponse(content={"error": "กรุณากรอกชื่อตำแหน่งงาน"}, status_code=400)

    try:
        # พยายามใช้ GPT ก่อน
        result = await handle_prediction_gpt(None, job_title, render_html=False)
        if result.get("error"):
            raise Exception(result.get("error"))
        return result
    except Exception as e:
        # ถ้า GPT มีปัญหา ให้ใช้ ThaiBERT แทน
        print(f"GPT prediction failed: {str(e)}, falling back to ThaiBERT")
        result = await handle_prediction_thaibert(None, job_title, render_html=False)
        return result


@app.post("/predict/employer")
async def predict_employer_api(data: PredictRequestEmployer):
    """
    API สำหรับนายจ้าง ทำนายตำแหน่งงานจากชื่อตำแหน่งและประเภทธุรกิจ

    Args:
        data: ข้อมูลคำขอที่มีชื่อตำแหน่งงานและประเภทธุรกิจ

    Returns:
        JSONResponse: ผลการทำนาย
    """
    job_title = data.job_title
    bussiness_type = data.bussiness_type

    # ตรวจสอบข้อมูลนำเข้า
    if not job_title:
        return JSONResponse(content={"error": "กรุณากรอกชื่อตำแหน่งงาน"}, status_code=400)
    if not bussiness_type:
        return JSONResponse(content={"error": "กรุณากรอกประเภทธุรกิจ"}, status_code=400)

    # ลบช่องว่างที่ต้นและท้ายข้อความ และลบสัญลักษณ์พิเศษจาก bussiness_type
    bussiness_type = bussiness_type.strip()  # ลบช่องว่างที่ต้นและท้าย
    bussiness_type = re.sub(r'[^\w\sก-๙]', '', bussiness_type)  # ลบสัญลักษณ์พิเศษ

    try:
        # พยายามใช้ GPT พร้อมข้อมูลประเภทธุรกิจ
        result = await handle_prediction_gpt(
            None, job_title, render_html=False, bussiness_type=bussiness_type
        )
        if result.get("error"):
            raise Exception(result.get("error"))
        return result
    except Exception as e:
        # ถ้า GPT มีปัญหา ให้ใช้ ThaiBERT แทน (ไม่ใช้ข้อมูลประเภทธุรกิจ)
        print(
            f"GPT prediction with business type failed: {str(e)}, falling back to ThaiBERT"
        )
        result = await handle_prediction_thaibert(None, job_title, render_html=False)
        return result


# === 6. Main Function ===
if __name__ == "__main__":
    uvicorn.run("ai_occupation:app", host="10.100.100.208", port=8004, reload=True)
