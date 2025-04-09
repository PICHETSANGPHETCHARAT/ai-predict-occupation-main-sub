import pandas as pd
import openai
from dotenv import load_dotenv  # นำเข้า python-dotenv

# โหลด environment variables จาก .env
load_dotenv()
# API Key สำหรับ OpenAI
import os
open_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=open_api_key)

# Function to generate job titles using ChatGPT API
def generate_job_titles(occupation_name, existing_titles):
    prompt = (
        f"กรุณาสร้างชื่อตำแหน่งงานสำหรับสายอาชีพ '{occupation_name}'"
        f"โดยต้องไม่ซ้ำกับตำแหน่งงานเดิมต่อไปนี้: {existing_titles} "
        f"โปรดตอบเป็นรายการโดยไม่มีตัวเลขนำหน้าและไม่มีข้อความเกินกว่ารายการชื่อตำแหน่งงาน "
        f"และให้มีการผสมผสานของภาษาไทยและภาษาอังกฤษ"
    )

    try:
        # Replace with your actual OpenAI client call
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

# Load data from CSV file
file_path = '/Users/itpmac001/Documents/Python Code/Test/model_predict_occupation/dataforthaibert.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Function to fill missing job titles
def fill_missing_job_titles(data, occupation_column, title_column):
    for index, row in data[data[title_column].isnull()].iterrows():
        occupation_name = row[occupation_column]
        existing_titles = data[title_column].dropna().tolist()
        new_titles = generate_job_titles(occupation_name, existing_titles)
        
        if new_titles:
            data.at[index, title_column] = new_titles[0]  # Use the first generated title
    
    return data

# Fill missing job titles
updated_data = fill_missing_job_titles(data, 'สาขาอาชีพหลัก', 'ชื่อตำแหน่งงาน')

# Save or display the updated data
updated_data.to_csv('/Users/itpmac001/Documents/Python Code/Test/model_predict_occupation/dataforthaibert2.csv', index=False)  # Save to a new file