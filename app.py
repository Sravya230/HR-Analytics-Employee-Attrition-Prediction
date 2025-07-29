import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")
st.title("💼 HR Analytics Dashboard")
st.markdown("Analyze attrition, salary distribution, and predict employee resignation using machine learning.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

df = load_data()

# Data cleaning
df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], inplace=True)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df_encoded = pd.get_dummies(df, drop_first=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Visualizations", "Predict Attrition"])

# Overview page
if page == "Overview":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🔢 Attrition Value Counts")
    st.write(df["Attrition"].value_counts())

    st.subheader("🧮 Basic Stats")
    st.write(df.describe())

# Visualizations
elif page == "Visualizations":
    st.subheader("🎯 Attrition Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Attrition", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("💰 Monthly Income Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["MonthlyIncome"], kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("📈 Job Role vs Attrition")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.countplot(x="JobRole", hue="Attrition", data=df, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# Prediction section
elif page == "Predict Attrition":
    st.subheader("🤖 Predict Employee Attrition")

    X = df_encoded.drop("Attrition", axis=1)
    y = df_encoded["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    st.write("**Confusion Matrix:**")
    fig4, ax4 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax4)
    st.pyplot(fig4)

    st.success("✅ Model trained and tested successfully!")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | HR Analytics by sravya")
