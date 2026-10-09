import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="pythondb"
)

mycursor = conn.cursor()

mycursor.execute(
    "CREATE TABLE student (name VARCHAR(50), branch VARCHAR(50), id INT)"
)

mycursor.execute("SHOW TABLES")

for x in mycursor:
    print(x)