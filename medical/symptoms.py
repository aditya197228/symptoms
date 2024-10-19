import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
#file path
try:
    data = pd.read_csv('data/Disease_symptom_and_patient_profile_dataset.csv')
    print("Dataset loaded successfully.")
    
    print(f"Columns: {data.columns}")
    print(f"Data Types:\n{data.dtypes}")
    print(f"Missing Values:\n{data.isnull().sum()}")
#error handeling
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

for col in ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']:
    data[col] = data[col].map({'yes': 1, 'no': 0})

data['Blood Pressure'] = pd.to_numeric(data['Blood Pressure'], errors='coerce')
data['Cholesterol Level'] = pd.to_numeric(data['Cholesterol Level'], errors='coerce')

X = data[['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']]
y = data['Disease']

X['Gender'] = pd.get_dummies(X['Gender'], drop_first=True)

scaler = StandardScaler()
X[['Age', 'Blood Pressure', 'Cholesterol Level']] = scaler.fit_transform(X[['Age', 'Blood Pressure', 'Cholesterol Level']])
print("Features normalized.")
# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and test sets.")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully.")

def get_user_input():
    print("Welcome to the Symptom Checker!")
    try:
        fever = int(input("Do you have a fever? (yes=1 / no=0): "))
        cough = int(input("Do you have a cough? (yes=1 / no=0): "))
        fatigue = int(input("Do you feel fatigued? (yes=1 / no=0): "))
        difficulty_breathing = int(input("Do you have difficulty breathing? (yes=1 / no=0): "))
        age = float(input("What is your age? "))
        gender = input("What is your gender? (male/female): ").strip().lower()
        blood_pressure = float(input("What is your blood pressure? "))
        cholesterol = float(input("What is your cholesterol level? "))
        
        gender = 1 if gender == 'male' else 0
        user_input = [fever, cough, fatigue, difficulty_breathing, age, gender, blood_pressure, cholesterol]
        print(f"Collected user input: {user_input}")
        return user_input
    except ValueError:
        print("Invalid input. Please enter the correct data type.")
        exit(1)

def predict_disease(user_input):
    try:
        user_input[4:7] = scaler.transform([user_input[4:7]])[0]
        prediction = model.predict([user_input])
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        exit(1)

user_input = get_user_input()
print("Predicting disease...")
predicted_disease = predict_disease(user_input)
print(f"The predicted disease is: {predicted_disease}")
