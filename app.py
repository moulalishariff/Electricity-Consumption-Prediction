from flask import Flask, render_template, request, redirect,session, flash
from database import executionquery, retrivequery1, retrivequery2,mycursor,mydb
from algorithms import linear_regression, rnn, lstm,load_model,gru,stacked
import numpy as np
import mysql.connector
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import base64
import random
import string
import sendgrid
from sendgrid.helpers.mail import Mail


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT upper(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT password FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password == password__data[0][0]:
                global user_email
                user_email = email
                return redirect("/home")
            else:
                return render_template('login.html', message= "Invalid Password!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')

import os
import ssl
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

app.secret_key = "123456" 
# Configure SendGrid
SENDGRID_API_KEY = "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"
SENDER_EMAIL = "moulalishariff2003@gmail.com"

# Dictionary to store OTPs temporarily (you can use Redis instead)
otp_store = {}

# 📌 Forgot Password - Step 1: Send OTP
@app.route('/forgot_password', methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form['email']

        # Check if email exists
        query = "SELECT email FROM users WHERE email = %s"
        values = (email,)
        user_data = retrivequery1(query, values)

        if not user_data:
            flash("This email is not registered!", "error")
            return redirect('/forgot_password')

        # Generate 6-digit OTP
        otp = ''.join(random.choices(string.digits, k=6))
        otp_store[email] = otp  # Store OTP temporarily

        # Send OTP via email
        send_otp_email(email, otp)

        session['reset_email'] = email  # Store email in session
        return redirect('/verify_otp')

    return render_template('forgot_password.html')


# 📌 Step 2: OTP Verification Page
@app.route('/verify_otp', methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        email = session.get('reset_email')
        entered_otp = request.form['otp']

        if email in otp_store and otp_store[email] == entered_otp:
            del otp_store[email]  # Remove OTP after successful verification
            return redirect('/reset_password')
        else:
            flash("Invalid OTP. Please try again!", "error")
            return redirect('/verify_otp')

    return render_template('verify_otp.html')


# 📌 Step 3: Reset Password
@app.route('/reset_password', methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        email = session.get('reset_email')
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect('/reset_password')

        # Update password in DB
        query = "UPDATE users SET password = %s WHERE email = %s"
        values = (new_password, email)
        executionquery(query, values)

        flash("Password reset successfully! You can now log in.", "success")
        session.pop('reset_email', None)  # Clear session
        return redirect('/login')

    return render_template('reset_password.html')


# 📌 Send OTP via SendGrid
def send_otp_email(to_email, otp):
    sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)
    subject = "Password Reset OTP"
    content = f"Your OTP for password reset is: {otp}"
    message = Mail(from_email=SENDER_EMAIL, to_emails=to_email, subject=subject, plain_text_content=content)
    response = sg.send(message)
    return response.status_code


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/algorithm', methods=['GET', 'POST'])
def algorithm():
    if request.method == "POST":
        algorithm_type = request.form['algorithm']
        if algorithm_type == 'Linear Regression':
            accuracy = linear_regression()
        elif algorithm_type == 'RNN':
            accuracy = rnn()
        elif algorithm_type == 'LSTM':
            accuracy = lstm()
        elif algorithm_type == 'Stacked LSTM':
            accuracy = stacked()
        elif algorithm_type == 'GRU':
            accuracy = gru()

        return render_template('algorithm.html', algorithm=algorithm_type, accuracy=accuracy)

    return render_template('algorithm.html', algorithm=None, accuracy=None)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Get input values from form safely
            input_features = [
                float(request.form.get("temperature", 0)),
                float(request.form.get("humidity", 0)),
                float(request.form.get("wind_speed", 0)),
                float(request.form.get("solar_radiation", 0)),
                float(request.form.get("time_of_day", 0)),
                float(request.form.get("day_of_week", 0)),
                float(request.form.get("energy_consumption", 0)),
                float(request.form.get("peak_hour_indicator", 0)),
                float(request.form.get("power_price", 0))
            ]

            # Load LSTM Model
            model = load_model('/Users/moulalishariff/Electric/lstm_model.h5')

            # Convert to NumPy array and reshape for LSTM
            input_array = np.array(input_features).reshape(1, 1, 9)

            # Predict using the model
            prediction = model.predict(input_array)
            predicted_value = float(prediction[0][0]) 

            # Store input and prediction in MySQL
            sql = """INSERT INTO energy_predictions 
                    (temperature, humidity, wind_speed, solar_radiation, time_of_day, day_of_week, 
                     energy_consumption, peak_hour_indicator, power_price, predicted_value) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            
            values = tuple(float(x) for x in input_features) + (predicted_value,)
            cursor = mydb.cursor()
            cursor.execute(sql, values)
            mydb.commit()
            
            return render_template("prediction.html", prediction_text=f'Predicted Energy Usage: {prediction[0][0]:.4f} kWh')

        except Exception as e:
            return render_template("prediction.html", prediction_text=f'Error: {str(e)}')

    return render_template("prediction.html", prediction_text="")

@app.route('/graph')
def graph():
    try:
        cursor = mydb.cursor()
        # Fetch stored data from MySQL
        cursor.execute("SELECT timestamp, predicted_value FROM energy_predictions ORDER BY timestamp DESC LIMIT 10")
        data = cursor.fetchall()

        if not data:
            return render_template('graph.html', graph_url=None, message="No data available to generate a graph.")

        timestamps = [row[0] for row in data]
        predictions = [row[1] for row in data]

        # Generate Graph
        plt.figure(figsize=(8, 5))
        plt.plot(timestamps, predictions, marker='o', linestyle='-', color='b', label='Predicted Energy')
        plt.xlabel("Timestamp")
        plt.ylabel("Predicted Energy Consumption (kWh)")
        plt.title("Energy Prediction Over Time")
        plt.legend()
        plt.xticks(rotation=0)

        # Convert Graph to Image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        return render_template('graph.html', graph_url=graph_url)

    except Exception as e:
        return render_template('graph.html', graph_url=None, message=f"Error generating graph: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)