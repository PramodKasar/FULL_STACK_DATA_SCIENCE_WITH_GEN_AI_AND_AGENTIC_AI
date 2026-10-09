import mysql.connector

conn = mysql.connector.connect(host="localhost", user="root", password="root")

mycursor = conn.cursor()

mycursor.execute("Show databases")

for x in mycursor:
    print(x)
    
# if conn.is_connected():
#     print("Connection successful")
# print(conn)
# print(conn.is_connected())