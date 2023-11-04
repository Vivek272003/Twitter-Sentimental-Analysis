import tkinter as tk
from tkinter import messagebox
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os

# Function to train the sentiment analysis model
def train_model():
    dataset_file = dataset_entry.get()
    if dataset_file:
        if os.path.isfile(dataset_file):
            try:
                dataset = pd.read_csv("E:\Coding Languages\Python\My work\Twitter_Sentiments.csv")  # Load the dataset from a CSV file
                tweets = dataset['tweet'].tolist()
                labels = dataset['label'].tolist()

                # Split the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.1, random_state=42)

                # Train the sentiment analysis model
                sentiment_model.fit(X_train, y_train)

                # Evaluate the model on the testing set
                y_pred = sentiment_model.predict(X_test)
                report = classification_report(y_test, y_pred)

                messagebox.showinfo("Model Training Result", f"Model trained successfully!\n\nClassification Report:\n{report}")
            except FileNotFoundError:
                messagebox.showerror("File Not Found", "The dataset file was not found.")
        else:
            messagebox.showerror("Invalid File", "Please enter a valid path to the dataset file.")
    else:
        messagebox.showwarning("Input Error", "Please enter the path to the dataset file.")

# Function to analyze sentiment using the trained model
def analyze_sentiment():
    tweet = tweet_entry.get()
    if tweet:
        sentiment = sentiment_model.predict([tweet])[0]
        if sentiment==0:
            a="Positive '0'"
        elif sentiment==1:
            a="Negative '1'"
        messagebox.showinfo("Sentiment Analysis Result", f"The sentiment of the tweet is: {a}")
    else:
        messagebox.showwarning("Input Error", "Please enter a tweet.")

# Create the GUI window
window = tk.Tk()
window.title("Twitter Sentiment Analysis")
window.configure(bg='lightblue')  # Set the background color

# Configure window size and position
window.geometry('600x400')
window.geometry(f'+{int(window.winfo_screenwidth()/2 - 300)}+{int(window.winfo_screenheight()/2 - 200)}')

# Create and position the widgets
title_label = tk.Label(window, text="Twitter Sentiment Analysis", font=("Arial", 24, "bold"), fg="orange", bg="lightblue")
title_label.pack(pady=20)

dataset_frame = tk.Frame(window, bg="lightblue")
dataset_frame.pack()

dataset_label = tk.Label(dataset_frame, text="Enter the path to the training dataset file (CSV):", font=("Arial", 12), bg="lightblue", fg="darkblue")
dataset_label.pack(side="left", padx=10)

dataset_entry = tk.Entry(dataset_frame, width=40, font=("Arial", 12))
dataset_entry.pack(side="left")

train_button = tk.Button(window, text="Train Model", command=train_model, font=("Arial", 14, "bold"), bg="darkblue", fg="white")
train_button.pack(pady=10)

tweet_frame = tk.Frame(window, bg="lightblue")
tweet_frame.pack()

tweet_label = tk.Label(tweet_frame, text="Enter the tweet:", font=("Arial", 12), bg="lightblue", fg="darkgreen")
tweet_label.pack(side="left", padx=10)

tweet_entry = tk.Entry(tweet_frame, width=40, font=("Arial", 12))
tweet_entry.pack(side="left")

analyze_button = tk.Button(window, text="Analyze", command=analyze_sentiment, font=("Arial", 14, "bold"), bg="darkgreen", fg="white")
analyze_button.pack(pady=10)

# Load the sentiment analysis machine learning model
sentiment_model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

# Start the GUI event loop
window.mainloop()
