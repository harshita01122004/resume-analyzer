# 📄 AI Resume Analyzer & Career Assistant

## 🚀 Overview

AI Resume Analyzer is a web-based application that analyzes resumes using **Machine Learning** and **Natural Language Processing (NLP)** techniques.

The system allows users to upload their resumes (PDF format) and provides:

* Resume score
* Extracted skills
* Personalized recommendations
* Suitable job role suggestions

This project helps students and job seekers improve their resumes and increase their chances of selection.

---

## ✨ Features

* 📌 Resume upload (PDF format)
* 🧠 NLP-based resume parsing
* 🎯 Skill extraction & analysis
* 📊 Resume scoring using ML
* 💡 Personalized improvement suggestions
* 💼 Job role recommendations
* 🌐 Simple and interactive UI (Streamlit)

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:** Scikit-learn (Logistic Regression)
* **NLP:** TF-IDF, NLTK / spaCy
* **PDF Processing:** PyPDF2
* **Libraries:** Pandas, NumPy

---

## 🧠 Working Process

1. User uploads resume (PDF)
2. Text is extracted using PyPDF2
3. Text preprocessing (cleaning, stopword removal)
4. TF-IDF converts text → numerical data
5. Logistic Regression model analyzes resume
6. System generates:

   * Resume score
   * Skills
   * Suggestions
   * Job recommendations

---

## 📂 Project Structure

```
resume-analyzer/
│── app.py
│── model.pkl
│── requirements.txt
│── templates/
│── static/
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```
git clone https://github.com/harshita01122004/resume-analyzer.git
cd resume-analyzer
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Application

```
streamlit run app.py
```

---

## 📸 Output Screens

* Home Page
* Login / Register
* Dashboard
* Resume Analysis Result


---

## 🎯 Advantages

* Fast and automated resume evaluation
* Reduces manual screening effort
* Provides data-driven insights
* Helps improve resume quality

---

## 🚀 Future Enhancements

* 🔹 Advanced ML models (Random Forest, SVM, Deep Learning)
* 🔹 Real-time job API integration
* 🔹 Cloud deployment (AWS / Heroku / Render)
* 🔹 Improved UI/UX

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Harshita Sharma**
🎓 MCA Student
📍 Chandigarh University

GitHub: https://github.com/harshita01122004

---
