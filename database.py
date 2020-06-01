import mysql.connector

my_db = mysql.connector.connect(host="localhost", user="root", passwd="", database="attendance")
my_cursor = my_db.cursor()

def create_table(table_name):
    my_cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA='dbName'")
    my_result = my_cursor.fetchall()
    if table_name in my_result:
        print("Table already exist")
    else:
        my_cursor.execte("CREATE TABLE students")

create_table()
#my_cursor.execute("SHOW TABLES")
print(my_db)