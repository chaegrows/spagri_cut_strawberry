import psycopg2
import os

from dotenv import load_dotenv

# schema는 직접 작성거하나 sql_script.txt를 통해 활용
schema_sql = """
  CREATE TABLE IF NOT EXISTS Type_Master (
    type_group VARCHAR(20),
    type_code VARCHAR(20),
    type_name VARCHAR(50),
    use_yn CHAR(1) DEFAULT 'Y',
    sort_order INT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )
"""

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def table_manage():
  try:
    conn = psycopg2.connect(
      host = DB_HOST,
      port = DB_PORT,
      dbname = DB_NAME,
      user = DB_USER,
      password = DB_PASSWORD
    )

    cur = conn.cursor()
    cur.execute(schema_sql)
    conn.commit()
    cur.close()
    conn.close()
    print("Table 생성 완료!")
  except Exception as e:
    print(f"테이블 생성 중 에러 발생 : {e}")

if __name__ == '__main__':
  table_manage()

