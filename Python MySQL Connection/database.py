import mysql.connector

conn = mysql.connector.connect(host="localhost", user="root", password="root")
    
if conn.is_connected():
    print("Connection successful")
# print(conn)
# print(conn.is_connected())
mycursor = conn.cursor()

mycursor.execute("create database pythondb")
print(mycursor)

