import time
import pandas as pd
import mysql.connector

# 🔹 ฟังก์ชันเชื่อมต่อฐานข้อมูล
def get_db_connection_120other():
    return mysql.connector.connect(
        host="192.168.100.120",
        user="jobbkk_other",
        password="orerthjobk2022$",
        database="jobbkk_other",
    )

def get_db_connection_130():
    return mysql.connector.connect(
        host="192.168.100.130",
        user="jobbkk_job",
        password="JodsSEdfe2020$",
        database="jobbkk_job",
    )


def get_occupation_active():
    try:
        conn = get_db_connection_120other()
        if not conn:
            print("❌ ไม่สามารถเชื่อมต่อฐานข้อมูลได้")
            return []

        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT 
                os.id AS occupation_id,
                od.name 
            FROM 
                occupation os
                INNER JOIN occupation_description od ON os.id = od.occupation_id 
            WHERE 
                os.is_status = '1'
                AND os.is_flags = '0'
                AND os.is_online = '1'
                AND od.language_id = '1'
            ORDER BY CONVERT(od.name USING tis620) ASC;
        """
        )
        occupations = cursor.fetchall()
        cursor.close()
        conn.close()

        return occupations
    except Exception as e:
        print(f"❌ Error DB: {e}")
        return []
    
def get_job_post_names_by_occupation_ids(occupation_ids):
    try:
        if not occupation_ids:
            return []
            
        # แปลง list เป็น string ที่มีรูปแบบ '1,2,3'
        ids_str = ','.join(map(str, occupation_ids))
        
        conn = get_db_connection_130()
        cursor = conn.cursor(dictionary=True)
        query = f"""
            SELECT 
                j.occupation_new_id,
                j.is_online,
                jd.position
            FROM 
                jobpost_description jd
                INNER JOIN jobpost j ON j.id = jd.jobpost_id 
            WHERE 
                j.is_online = '1'
                AND j.is_flags = '0'
                AND j.is_status = '1'
                AND j.occupation_new_id IN ({ids_str})
                AND j.created_at >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
            LIMIT 500;
        """
        cursor.execute(query)
        jobs = cursor.fetchall()
        cursor.close()
        conn.close()
        return jobs
    except Exception as e:
        print(f"❌ Error fetching job titles for multiple occupation IDs: {e}")
        return []

def get_job_post_names_by_occupation_id(occupation_id):
    try:
        conn = get_db_connection_130()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT 
                jd.position
            FROM 
                jobpost_description jd
                INNER JOIN jobpost j ON j.id = jd.jobpost_id 
            WHERE 
                j.occupation_new_id = %s
                AND j.is_online = '1'
                AND j.is_flags = '0'
                AND j.is_status = '1'
                AND j.created_at >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
            LIMIT 200;
            """,
            (occupation_id,)
        )
        job_positions = cursor.fetchall()
        cursor.close()
        conn.close()
        return [job["position"] for job in job_positions]
    except Exception as e:
        print(f"❌ Error fetching job titles for occupation_id={occupation_id}: {e}")
        return []

def export_job_positions_to_csv():
    data = []
    occupations = get_occupation_active()
    total_occupations = len(occupations)
    print(f"📌 เจอสาขาอาชีพ {total_occupations} รายการ")

    # แสดงความคืบหน้า
    processed = 0
    
    for occ in occupations:
        occupation_id = occ["occupation_id"]
        occupation_name = occ["name"]
        job_titles = get_job_post_names_by_occupation_id(occupation_id)
        
        processed += 1
        print(f"⏳ กำลังดึงข้อมูล: {processed}/{total_occupations} - {occupation_name} (ID: {occupation_id}) - พบตำแหน่งงาน {len(job_titles)} รายการ")

        for title in job_titles:
            data.append({
                "occupation_id": occupation_id,
                "สาขาอาชีพหลัก": occupation_name,
                "ชื่อตำแหน่งงาน": title,
            })

        time.sleep(0.1)  # เพิ่ม delay เล็กน้อยเพื่อไม่ให้ server ทำงานหนัก

    df = pd.DataFrame(data)
    df.to_csv("job_positions_by_occupation.csv", index=False, encoding="utf-8-sig")
    print(f"✅ บันทึกไฟล์ job_positions_by_occupation.csv เรียบร้อยแล้ว จำนวน {len(data)} รายการ")

# 🔽 เรียกใช้ฟังก์ชันเพื่อเริ่มการดึงข้อมูลและบันทึกไฟล์
export_job_positions_to_csv()