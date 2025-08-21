import psycopg2
import os
from datetime import datetime
from .db_pool import DBPool
from psycopg2.extras import RealDictCursor
from .db_config import init_db


class DBManager:
  def __init__(self):
    pass

  def insert_file_info(self, farm_code, crop_code, machine_code, file_type, file_name,
                       file_path, file_size, description, shoot_date, folder_path,
                       top_folder_path, create_user='metafarmers'):
    conn = None
    try:
      conn = DBPool.get_conn()
      cur = conn.cursor()

      cur.execute("""
        CALL insert_file_data(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
      """, (str(farm_code), str(crop_code), str(machine_code), str(file_type), 
            str(file_name), str(file_path), int(file_size), str(description), 
            shoot_date, datetime.now(), str(create_user), str(folder_path), 
            str(top_folder_path)))
      conn.commit()
      cur.close()
      print("File Metadata inserted successfully")
    
    except psycopg2.Error as e:
      print(f"[ERROR] Failed to insert metadata: {e}")
      if conn:
        conn.rollback()

    finally:
      if conn:
        DBPool.put_conn(conn)

  def select_file_info(self, file_name=None, file_path=None, folder_path=None, date_from=None,
                       date_to=None, file_type=None, crop_code=None, farm_code=None,
                       machine_code=None, top_folder_path=None):
    init_db()
    conn = None
    try:
      conn = DBPool.get_conn()
      cur = conn.cursor(cursor_factory=RealDictCursor)

      query = "SELECT * FROM file_info WHERE 1=1"
      params = []

      # condition add
      if file_name:
        query += " AND file_name = %s"
        params.append(file_name)
      if file_path:
        query += " AND file_path = %s"
        params.append(file_path)
      if folder_path:
        query += " AND folder_path = %s"
        params.append(folder_path)
      if date_from and date_to:
        query += " AND shoot_date BETWEEN %s AND %s"
        params.extend([date_from, date_to])
      elif date_from:
        query += " AND shoot_date >= %s"
        params.append(date_from)
      elif date_to:
        query += " AND shoot_date <= %s"
        params.append(date_to)
      if file_type:
        query += " AND file_type = %s"
        params.append(file_type)
      if crop_code:
        query += " AND crop_code = %s"
        params.append(crop_code)
      if farm_code:
        query += " AND farm_code = %s"
        params.append(farm_code)
      if machine_code:
        query += " AND machine_code = %s"
        params.append(machine_code)
      if top_folder_path:
        query += " AND top_folder_path = %s"
        params.append(top_folder_path)

      cur.execute(query, tuple(params))
      results = cur.fetchall()

      cur.close()
      conn.close()
      return results

    except psycopg2.Error as e:
      print(f"[ERROR] Failed to select data: {e}")
      return []
      
    finally:
      if conn:
        DBPool.put_conn(conn)