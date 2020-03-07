import pymysql as pms


def write_dict_to_database(conn,info,is_close_conn = False):
    assert isinstance(conn,pms.Connection),'conn必须为数据库连接对象'
    assert isinstance(info,dict),'info必须为字典对象'
    sql = ''
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
    
    finally:
        if is_close_conn:
            conn.close()

