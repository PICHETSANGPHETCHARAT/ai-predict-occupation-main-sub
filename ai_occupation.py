from fastapi import FastAPI, Request, Form, Body
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
from dotenv import load_dotenv  # นำเข้า python-dotenv
import os

# โหลด environment variables จาก .env
load_dotenv()
# === GPT Client ===
open_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=open_api_key)

# === FastAPI Setup ===
app = FastAPI(
    title="JobBKK AI occupation API",
    description="API for predict occupation main and sub using AI",
    version="1.0.0",
    root_path="/ai_predictoccupationmainsub",
)
templates = Jinja2Templates(directory="templates")

# === Load Model & Label Encoder ===
MODEL_PATH = "./model/model_thaibert_v2"  # เปลี่ยนเส้นทางเป็น model_thaibert
# โหลด LabelEncoder
with open(os.path.join(MODEL_PATH, "labels.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class PredictRequest(BaseModel):
    job_title: str


# ✅ DB Model
class OccupationSub:
    def __init__(self, occupation_sub_id, name):
        self.occupation_sub_id = occupation_sub_id
        self.name = name

    @classmethod
    def from_db(cls, db_record):
        return cls(
            occupation_sub_id=db_record["occupation_sub_id"], name=db_record["name"]
        )


class OccupationMain:
    def __init__(self, occupation_id, name):
        self.occupation_id = occupation_id
        self.name = name

    @classmethod
    def from_db(cls, db_record):
        return cls(occupation_id=db_record["occupation_id"], name=db_record["name"])


# ✅ Database Connection
def get_db_connection_120other():
    return mysql.connector.connect(
        host="192.168.100.120",
        user="jobbkk_other",
        password="orerthjobk2022$",
        database="jobbkk_other",
    )


# ✅ Get Sub Occupations
async def get_occupation_subs(occupation_id: int):
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


# ✅ Get Occupations Main
async def get_occupation_main():
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


# ✅ GPT-based Sub Occupation Prediction
async def predict_sub_occupation_with_gpt(job_title, main_occupation, sub_occupations):
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

        จากรายการสาขาอาชีพรองด้านล่างนี้ โปรดเลือกเพียง 1 รายการที่เหมาะสมที่สุดสำหรับตำแหน่งงานดังกล่าว โดยพิจารณาจาก ID และชื่อ:

        {sub_list_text}

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


# ✅ GPT-based Main Occupation Prediction
async def predict_main_occupation_with_gpt(job_title, main_occupation):
    try:
        main_options = [
            {"id": main.occupation_id, "name": main.name} for main in main_occupation
        ]

        # เปลี่ยนให้วนลูปผ่าน main_options แทน main_occupation
        main_list_text = "\n".join(
            [
                f"{i+1}. {json.dumps(option, ensure_ascii=False)}"
                for i, option in enumerate(main_options)
            ]
        )

        prompt = f"""
        ชื่อตำแหน่งงาน: "{job_title}"
        จากชื่อตำแหน่งงาน โปรดเลือกเพียง 1 สาขาอาชีพหลักที่เหมาะสมที่สุดสำหรับตำแหน่งงานดังกล่าว โดยพิจารณาจาก ID และชื่อ:

        {main_list_text}

        กรุณาตอบกลับเพียง 1 บรรทัด โดยให้เป็น JSON ของรายการที่คุณเลือก เช่น:
        {{"id": 0, "name": "ตัวอย่างตำแหน่งงานหลัก"}}
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
        main_id = response_data.get("id") or response_data.get("occupation_id")

        if not main_id:
            return None, f"ไม่พบ id จาก GPT: {response_data}"

        for main in main_occupation:
            if main.occupation_id == main_id:
                return main.occupation_id, main.name

        return None, "ไม่พบสาขาหลักที่ตรง"

    except Exception as e:
        return None, f"GPT ERROR: {e}"


# ✅ HTML Form (GET)
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ✅ HTML Form (POST แบบ reload หน้า) สำหรับเวอร์ชั่นเดิมที่ใช้ ThaiBERT ทำนายตำแหน่งงานหลัก
@app.post("/", response_class=HTMLResponse)
async def form_post(request: Request, job_title: str = Form(...)):
    return await handle_prediction(request, job_title, render_html=True)


# ✅ API JSON (POST สำหรับ JavaScript fetch) สำหรับเวอร์ชั่นใหม่
@app.post("/ai/predictoccupationmainsub")
async def predict_api(data: PredictRequest):
    job_title = data.job_title
    if not job_title:
        return JSONResponse(content={"error": "กรุณากรอกชื่อตำแหน่งงาน"}, status_code=400)

    try:
        # เรียกใช้ handle_prediction2 ซึ่งเป็นเวอร์ชั่นใหม่
        result = await handle_prediction2(None, job_title, render_html=False)
        # หากผลลัพธ์ส่งกลับไปมี error key (กรณี handle_prediction2 จับ Exception แล้วส่งกลับเป็น dict error)
        if result.get("error"):
            raise Exception(result.get("error"))
    except Exception as e:
        # เมื่อเกิดข้อผิดพลาด fallback ไปใช้ handle_prediction เวอร์ชั่นเดิม
        result = await handle_prediction(None, job_title, render_html=False)

    return result


# ✅ Core Prediction Logic เวอร์ชั่นเดิม
async def handle_prediction(request: Request, job_title: str, render_html: bool):
    conn = None
    cursor = None
    try:
        inputs = tokenizer(
            job_title, return_tensors="pt", truncation=True, padding=True, max_length=64
        )
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
            main_occupation = label_encoder.inverse_transform([predicted_class_id])[0]

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

        sub_occupations = await get_occupation_subs(occupation_id)
        sub_id, sub_name = await predict_sub_occupation_with_gpt(
            job_title, main_occupation, sub_occupations
        )

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


# ✅ Core Prediction Logic เวอร์ชั่นใหม่ (ใช้ ChatGPT API ในการเลือกทั้งสาขาหลักและรอง)
async def handle_prediction2(request: Request, job_title: str, render_html: bool):
    try:
        list_main_occupations = await get_occupation_main()
        # เรียกใช้ GPT-based main occupation prediction ด้วย await
        main_id, main_name = await predict_main_occupation_with_gpt(
            job_title, list_main_occupations
        )
        sub_occupations = await get_occupation_subs(main_id)
        sub_id, sub_name = await predict_sub_occupation_with_gpt(
            job_title, main_name, sub_occupations
        )

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
                },
            )
        else:
            return {
                "job_title": job_title,
                "occupation_id": main_id,
                "main_occupation": main_name,
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


# ✅ Run Server
if __name__ == "__main__":
    uvicorn.run("ai_occupation:app", host="10.100.100.208", port=8010, reload=True)
