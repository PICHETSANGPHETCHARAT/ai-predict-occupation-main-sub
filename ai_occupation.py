"""
JobBKK AI Occupation Prediction API
------------------------------
API สำหรับทำนายตำแหน่งงานหลักและตำแหน่งงานย่อยด้วย AI
รองรับทั้งการใช้งานผ่าน AI GPT
"""

# === 1. การ import และตั้งค่าเริ่มต้น ===
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
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
# --- GPT Prediction Functions ---
async def predict_main_occupation_with_gpt(job_title, main_occupation_list):
    """
    ทำนายตำแหน่งงานหลักโดยใช้ GPT โดยไม่ใช้ post-processing
    """

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

    จากชื่อตำแหน่งงาน โปรดเลือกเพียง 1 สาขาอาชีพหลักที่เหมาะสมที่สุดจากรายการต่อไปนี้

    หมายเหตุ: ผู้จัดการโรงงาน เป็นตำแหน่งบริหารระดับองค์กร ไม่ใช่ผู้ปฏิบัติงานด้านการผลิตโดยตรง

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

    คำแนะนำเพิ่มเติม:
    - AE หมายถึงงานขายหรือบริหารการขาย
    - Chinese Speaking หมายถึงงานมนุษยศาสตร์/ล่าม
    - Internship, นักศึกษาฝึกงาน, นักศึกษาสหกิจ หมายถึง Part-time/พนักงานสัญญาจ้าง
    - ครีเอทีฟ หมายถึง โฆษณา/สื่อ
    - แคชเชียร์ หมายถึง ห้างสรรพสินค้า
    - เจ้าหน้าที่นิติบุคคล หมายถึง ธุรการ/จัดการทั่วไป
    - store หมายถึง ขนส่ง/คลังสินค้า
    - ประสานงานภาษาจีน หมายถึง มนุษยศาสตร์/ล่าม
    - พนักงานห้าง หมายถึง ห้างสรรพสินค้า
    - เร่งรัดหนี้สิน หมายถึง เจ้าหน้าที่เร่งรัดหนี้สิน
    - หากชื่อตำแหน่งงานนี้ตรงกับรายการด้านบน ให้ใช้สาขาที่ระบุโดยตรง

    ตัวอย่าง:
    - AE => ขาย / บริหารการขาย
    - Chinese Speaking => มนุษยศาสตร์ / ล่าม
    - นักศึกษาฝึกงาน => Part-time / พนักงานสัญญาจ้าง
    - Internship => Part-time / พนักงานสัญญาจ้าง
    - ครีเอทีฟ => โฆษณา / สื่อ
    - แคชเชียร์ => ห้างสรรพสินค้า
    - เจ้าหน้าที่นิติบุคคล => ธุรการ / จัดการทั่วไป
    - store => ขนส่ง / คลังสินค้า
    - ประสานงานภาษาจีน => มนุษยศาสตร์ / ล่าม
    - พนักงานห้าง => ห้างสรรพสินค้า
    - เร่งรัดหนี้สิน => เจ้าหน้าที่เร่งรัดหนี้สิน

    {main_list_text}

    กรุณาตอบกลับเฉพาะ JSON ของรายการที่เลือก เช่น:
    {{"id": 0, "name": "ตัวอย่างสาขา"}}
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "คุณคือ HR ผู้เชี่ยวชาญด้านการวิเคราะห์ตำแหน่งงาน"},
                {"role": "user", "content": prompt},
            ],
        )

        content = completion.choices[0].message.content.strip()

        if not content.startswith("{"):
            return None, f"GPT ตอบกลับไม่ใช่ JSON: {content}"

        response_data = json.loads(content)
        main_id = response_data.get("id") or response_data.get("occupation_id")
        main_name = response_data.get("name")

        if not main_id:
            return None, f"ไม่พบ id จาก GPT: {response_data}"

        for main in main_occupation_list:
            if main.occupation_id == main_id:
                return main.occupation_id, main.name

        return None, "ไม่พบสาขาหลักที่ตรง"

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
        
        จากชื่อตำแหน่งงานและประเภทธุรกิจข้างต้น โปรดเลือกเพียง 1 สาขาอาชีพหลักที่เหมาะสมที่สุดจากรายการต่อไปนี้
        หมายเหตุ: ผู้จัดการโรงงาน เป็นตำแหน่งบริหารระดับองค์กร ไม่ใช่ผู้ปฏิบัติงานด้านการผลิตโดยตรง

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

        คำแนะนำเพิ่มเติม:
        - AE หมายถึงงานขายหรือบริหารการขาย
        - Chinese Speaking หมายถึงงานมนุษยศาสตร์/ล่าม
        - Internship, นักศึกษาฝึกงาน, นักศึกษาสหกิจ หมายถึง Part-time/พนักงานสัญญาจ้าง
        - ครีเอทีฟ หมายถึง โฆษณา/สื่อ
        - แคชเชียร์ หมายถึง ห้างสรรพสินค้า
        - เจ้าหน้าที่นิติบุคคล หมายถึง ธุรการ/จัดการทั่วไป
        - store หมายถึง ขนส่ง/คลังสินค้า
        - ประสานงานภาษาจีน หมายถึง มนุษยศาสตร์/ล่าม
        - พนักงานห้าง หมายถึง ห้างสรรพสินค้า
        - เร่งรัดหนี้สิน หมายถึง เจ้าหน้าที่เร่งรัดหนี้สิน
        - หากชื่อตำแหน่งงานนี้ตรงกับรายการด้านบน ให้ใช้สาขาที่ระบุโดยตรง

        ตัวอย่าง:
        - AE => ขาย / บริหารการขาย
        - Chinese Speaking => มนุษยศาสตร์ / ล่าม
        - นักศึกษาฝึกงาน => Part-time / พนักงานสัญญาจ้าง
        - Internship => Part-time / พนักงานสัญญาจ้าง
        - ครีเอทีฟ => โฆษณา / สื่อ
        - แคชเชียร์ => ห้างสรรพสินค้า
        - เจ้าหน้าที่นิติบุคคล => ธุรการ / จัดการทั่วไป
        - store => ขนส่ง / คลังสินค้า
        - ประสานงานภาษาจีน => มนุษยศาสตร์ / ล่าม
        - พนักงานห้าง => ห้างสรรพสินค้า
        - เร่งรัดหนี้สิน => เจ้าหน้าที่เร่งรัดหนี้สิน

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

        prompt = f"""
        ชื่อตำแหน่งงาน: "{job_title}"
        สาขาอาชีพหลัก: "{main_occupation}"

        จากรายการตำแหน่งงานย่อยด้านล่าง โปรดเลือกเพียง 1 รายการที่เหมาะสมที่สุดสำหรับตำแหน่งงานและสาขาอาชีพหลัก โดยดูจากลักษณะงานจริงในคำอธิบายข้างต้น:
        หมายเหตุ: หากไม่มีตำแหน่ง "ช่างไม้" โดยตรง ให้เลือกตำแหน่งที่ใกล้เคียงที่สุด เช่น "ช่างเทคนิค"

        {sub_list_text}

        ขั้นตอนการตัดสินใจ:
        1. พิจารณาหน้าที่หลักของตำแหน่งจากคำอธิบาย
        2. เปรียบเทียบกับชื่อและขอบเขตของแต่ละตำแหน่งย่อย
        3. หากชื่อซ้ำซ้อน ให้เลือกจากลักษณะการปฏิบัติงานจริง
        4. อย่าเลือกโดยอิงจากชื่อเท่านั้น แต่ให้ดูเนื้อหาของงานเป็นหลัก

        กรุณาตอบกลับเพียง 1 บรรทัด โดยให้เป็น JSON ของรายการที่คุณเลือก เช่น:
        {{"id": 0, "name": "ตัวอย่างตำแหน่งงานรอง"}}
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "คุณคือ HR ผู้เชี่ยวชาญด้านการวิเคราะห์ตำแหน่งงาน"},
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

        จากรายการตำแหน่งงานย่อยด้านล่าง โปรดเลือกเพียง 1 รายการที่เหมาะสมที่สุดสำหรับตำแหน่งงาน,ประเภทธุรกิจและสาขาอาชีพหลัก โดยดูจากลักษณะงานจริงในคำอธิบายข้างต้น:
        หมายเหตุ: หากไม่มีตำแหน่ง "ช่างไม้" โดยตรง ให้เลือกตำแหน่งที่ใกล้เคียงที่สุด เช่น "ช่างเทคนิค"

        {sub_list_text}

        ขั้นตอนการตัดสินใจ:
        1. พิจารณาหน้าที่หลักของตำแหน่งจากคำอธิบาย
        2. เปรียบเทียบกับชื่อและขอบเขตของแต่ละตำแหน่งย่อย
        3. หากชื่อซ้ำซ้อน ให้เลือกจากลักษณะการปฏิบัติงานจริง
        4. อย่าเลือกโดยอิงจากชื่อเท่านั้น แต่ให้ดูเนื้อหาของงานเป็นหลัก

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
    return await handle_prediction_gpt(request, job_title, render_html=True)


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
        result = await handle_prediction_gpt(None, job_title, render_html=False)
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
        result = await handle_prediction_gpt(None, job_title, render_html=False)
        return result


# === 6. Main Function ===
if __name__ == "__main__":
    uvicorn.run("ai_occupation:app", host="127.0.0.1", port=8004, reload=True)
