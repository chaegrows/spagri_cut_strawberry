import psycopg2
import os
from dotenv import load_dotenv
from .db_pool import DBPool

def init_db():
  load_dotenv()
  DBPool.init_pool(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
  )