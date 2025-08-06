import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Shariff@2003",
    port="3306",
    database='flask_app'
)

mycursor = mydb.cursor()

def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()
    return

def retrivequery1(query, values):
    mycursor.execute(query, values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data
