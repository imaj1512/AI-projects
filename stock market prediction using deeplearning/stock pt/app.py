from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import matplotlib.dates as mdates  
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sqlite3

app = Flask(__name__)
app.secret_key = 'Stock-market-prediction-24'  # Secret key for session management

# Hardcoded credentials (for testing only)
USER_CREDENTIALS = {
    "admin": {"password": "password123", "email": "admin@example.com"}
}

# Load pre-trained model
model = load_model("stock_future_prediction_saved.keras")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        new_username = request.form.get("new_username")
        new_email = request.form.get("new_email")
        new_password = request.form.get("new_password")

        # Debugging: Print the new user details
        print(f"New username: {new_username}")
        print(f"New email: {new_email}")
        print(f"New password: {new_password}")

        if not new_username or not new_email or not new_password:
            flash("All fields are required.", "error")
            return redirect(url_for("register"))

        if new_username in USER_CREDENTIALS:
            flash("Username already exists.", "error")
            return redirect(url_for("register"))

        # Add new user to the dictionary
        USER_CREDENTIALS[new_username] = {"password": new_password, "email": new_email}
        print(f"Updated USER_CREDENTIALS: {USER_CREDENTIALS}")  # Debugging: Print updated credentials

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Debugging: Print the submitted username and password
        print(f"Submitted username: {username}")
        print(f"Submitted password: {password}")

        # Check if the username exists and the password matches
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username]["password"] == password:
            session["username"] = username  # Store username in session
            print(f"Session username: {session.get('username')}")  # Debugging: Print session username
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password.", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)  # Remove user from session
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
def index():
    if "username" not in session:  # Restrict access if not logged in
        return redirect(url_for("login"))

    if request.method == "POST":
        stock = request.form.get("stock_id", "").strip().upper()
        if not stock:
            flash("Please enter a valid stock ID.", "error")
            return redirect(url_for("index"))
        
        session["stock_id"] = stock  # Store stock ID in session
        return redirect(url_for("results"))  # Redirect to results page

    return render_template("index.html")

@app.route("/results")
def results():
    if "username" not in session:  # Restrict access if not logged in
        return redirect(url_for("login"))

    stock = session.get("stock_id")
    if not stock:
        return redirect(url_for("index"))

    try:
        from datetime import datetime
        end = datetime.now()
        start = datetime(end.year - 15, end.month, end.day)

        df = yf.download(stock, start, end)
        print(f"Fetched Data for {stock}: {df.tail()}") 
        
        if df.empty:
            return render_template("results.html", error_message=f"No data found for stock ID '{stock}'.")

        Close_price = df['Close']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(Close_price.values.reshape(-1, 1))
        print(f"Scaled Data for {stock}: {scaled_data[-5:]}")

        # Prepare data for model prediction
        x_data = []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
        x_data = np.array(x_data)
        print(f"Model Input Data (x_data) for {stock}: {x_data[-1]}")
        
        predicted_scaled = model.predict(x_data)
        predicted_prices = scaler.inverse_transform(predicted_scaled)
        print(f"Predicted Prices for {stock}: {predicted_prices[-5:]}")
       
        # Actual vs Predicted Prices Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index[-len(predicted_prices):], Close_price[-len(predicted_prices):], label="Actual Prices", color="blue")
        ax.plot(df.index[-len(predicted_prices):], predicted_prices, label="Model Predictions", color="orange")
        ax.set_title(f"{stock} Actual vs Predicted Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        actual_vs_predicted_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        # Next 10 Days Prediction
        last_100_days = scaled_data[-100:].reshape(1, 100, 1)
        prediction_10_days = []

        for _ in range(10):
            next_day_pred = model.predict(last_100_days)
            prediction_10_days.append(next_day_pred)
            last_100_days = np.append(last_100_days[:, 1:, :], next_day_pred.reshape(1, 1, 1), axis=1)

        prediction_10_days = np.array(prediction_10_days).reshape(-1, 1)
        prediction_10_days = scaler.inverse_transform(prediction_10_days)

        last_date = df.index[-1]
        next_10_days = pd.date_range(last_date + pd.DateOffset(days=1), periods=10)
        predictions = [{"Date": date.date(), "Predicted": float(pred)} for date, pred in zip(next_10_days, prediction_10_days)]
        
        fig, ax = plt.subplots(figsize=(10, 6))

         # Ensure the next day's predicted price is plotted as a line instead of a single dot
        ax.plot([df.index[-1], next_10_days[0]], [Close_price.iloc[-1], prediction_10_days[0]], 
        marker='o', linestyle='-', color='red', label="Next Day Prediction")

        ax.set_title(f"{stock} Next Day Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        plt.grid(True)


        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=30) 
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        next_day_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        
                
        
        # Next 10 Days Prediction Graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(next_10_days, prediction_10_days, label="Predicted Prices", color="purple")
        ax.set_title(f"{stock} Predicted Prices for Next 10 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Predicted Price")
        ax.legend()
        plt.xticks(rotation=30)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        next_10_days_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
    
        
        # Closing Price Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, Close_price, label="Closing Prices", color="blue")
        ax.set_title(f"{stock} Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        closing_prices_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

          # Moving Averages Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, Close_price.rolling(window=100).mean(), label="100-Day MA", color="orange")
        ax.plot(df.index, Close_price.rolling(window=200).mean(), label="200-Day MA", color="green")
        ax.set_title(f"{stock} Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        moving_averages_plot = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()

        return render_template(
            "results.html",
            predictions=predictions,
            plots={
                "actual_vs_predicted": actual_vs_predicted_plot,
                "next_1_day": next_day_plot,
                "next_10_days": next_10_days_plot,
                "closing_prices": closing_prices_plot,
                "moving_averages": moving_averages_plot
            }
        )
    

    except Exception as e:
        return render_template("results.html", error_message=f"An error occurred: {e}")

@app.route("/back_to_index")
def back_to_index():
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)



    
    