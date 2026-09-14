import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv("Student Depression Dataset.csv")


df.drop(columns=["id", "City", "Profession"], inplace=True)


df.dropna(inplace=True)


categorical_cols = [
    "Gender", "Sleep Duration", "Dietary Habits",
    "Degree", "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness"
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # save each encoder


X = df.drop(columns=["Depression"])
y = df["Depression"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("feature_columns.pkl", "wb") as f:
    pickle.dump(list(df.drop(columns=["Depression"]).columns), f)


with open("processed_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

print("Preprocessing done! Files saved: scaler.pkl, label_encoders.pkl, processed_data.pkl")