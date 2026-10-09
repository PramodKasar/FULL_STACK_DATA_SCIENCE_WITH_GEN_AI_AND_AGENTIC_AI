import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="pythondb"
)

mycursor = conn.cursor()

sql = 'insert into student (name, branch, id) values (%s, %s, %s)'
val = [('a', 'cse', '56'), ('b', 'IT', '78'), ('c', 'me', '80')]
# val = [('rohit', 'cse', '56'), ('kohli', 'IT', '78'), ('surya', 'me', '80')]

mycursor.execute(sql, val)
# mycursor.executemany(sql, val)
conn.commit()
print(mycursor.rowcount, "record inserted")