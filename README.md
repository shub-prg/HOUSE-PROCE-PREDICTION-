# 🏠 House Price Prediction Web App

A Machine Learning web application built using **Streamlit** that predicts house prices based on input features. This project includes a **user authentication system (Login and Registration)**, prediction history tracking, and a trained machine learning model.

This project was developed as a **B.Tech Minor Project**.

---

## 🚀 Features

* 🔐 User Login and Registration system
* 🤖 Machine Learning model using XGBoost
* 🏡 Real-time house price prediction
* 📊 Interactive web interface using Streamlit
* 📁 Prediction history saved for each user
* 💾 Stored trained model (`house_price_model.pkl`)
* 📂 User data stored using JSON
* 🚪 Logout functionality

---

## 🧠 Machine Learning Model

* Algorithm: XGBoost Regressor
* Dataset: California Housing Dataset
* Model file: `house_price_model.pkl`
* Input features:

  * Median Income
  * House Age
  * Average Rooms
  * Average Bedrooms
  * Population
  * Average Occupancy
  * Latitude
  * Longitude

---

## 📂 Project Structure

```
HOUSE-PROCE-PREDICTION/
│
├── app.py                      # Main Streamlit app
├── house_price_model.pkl      # Trained ML model
├── requirements.txt           # Dependencies
├── users.json                 # Registered users
│
├── data/                      # Dataset files
├── user_data/                 # User related data
├── user_history/              # Prediction history
│
└── README.md                  # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/shub-prg/HOUSE-PROCE-PREDICTION-.git
cd HOUSE-PROCE-PREDICTION-
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Run the application

```
streamlit run app.py
```

---

## 🖥️ Usage

1. Open the app in browser
2. Register a new account
3. Login with your credentials
4. Enter house feature values
5. Click "Predict Price"
6. View the predicted house price
7. Prediction history is saved automatically

---

## 🔐 Authentication System

This app includes:

* User Registration
* User Login
* User data storage using JSON
* Session management
* Prediction history tracking

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Matplotlib
* Seaborn
* JSON (for user data storage)

---

## 🎯 Project Purpose

This project demonstrates:

* Machine Learning model deployment
* Streamlit web app development
* User authentication system implementation
* Model integration with frontend
* Real-world ML project workflow

---

## 👨‍💻 Author

**Shubhranshu**
B.Tech CSE (Data Science)
Minor Project

GitHub: https://github.com/shub-prg

---

## 📜 License

This project is for educational purposes.
