from psycopg2.pool import SimpleConnectionPool

# Database Pool 관리용 class
class DBPool:
  _pool = None

  @classmethod
  def init_pool(cls, **kwargs):
    if cls._pool is None:
      cls._pool = SimpleConnectionPool(minconn=1, maxconn=20, **kwargs)

  @classmethod
  def get_conn(cls):
    if cls._pool is None:
      raise Exception("DBPool has not been initialized. Call init_pool() first")
    return cls._pool.getconn()
  
  @classmethod
  def put_conn(cls, conn):
    cls._pool.putconn(conn)
