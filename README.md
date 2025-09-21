# 🤖 AI Chatbot  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)  
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey?logo=flask)  
![License](https://img.shields.io/badge/License-MIT-green)  

An **AI-powered chatbot** built with **Python, NLP, and Deep Learning**.  
It can understand user queries, classify intents, and generate intelligent responses through a simple web interface.  

---

## 🚀 Features
- 🎯 **Intent Classification** using a custom-trained model.  
- 🧠 **Deep Learning (TensorFlow/Keras)** for response prediction.  
- 🌐 **Flask-based Web UI** with `index.html`.  
- 🗂️ JSON dataset (`intents.json`) for training.  
- 📊 Easily extendable with new intents and responses.  
- 🔒 Secure & modular project structure.  

---

## 🛠️ Tech Stack
- **Frontend**: HTML, CSS  
- **Backend**: Python, Flask  
- **AI/ML**: TensorFlow, Keras, NLTK  
- **Database/Storage**: JSON-based intents dataset  

---

## 📂 Project Structure
AI-Chatbot/
│── app.py # Main Flask app
│── train.py # Training script
│── intents.json # Dataset for chatbot
│── models/ # Trained models
│── static/ # CSS, JS, assets
│── templates/ # HTML files (UI)
│── data/ # Extra data if any
│── README.md # Documentation

---

## ⚡ Quick Start  

### 1️⃣ Clone Repository  
```bash
git clone https://github.com/<your-username>/AI-Chatbot.git
cd AI-Chatbot
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
 When you add new data in intent.json file then run => python train.py when traning is complete .
you can run => python app.py

<img width="1920" height="1080" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/c0d0851e-2115-4f48-b3b6-4ac9a5d12b03" />
<img width="1920" height="1080" alt="Screenshot (72)" src="https://github.com/user-attachments/assets/a23cd9d1-36e7-4637-8088-8ad608e71328" />
<img width="1920" height="1080" alt="Screenshot (71)" src="https://github.com/user-attachments/assets/16f762ee-68d7-4725-b0ee-0289c15e003e" />
<img width="1920" height="1080" alt="Screenshot (70)" src="https://github.com/user-attachments/assets/fca89078-39d7-4661-adac-a7270aacc36a" />
<img width="1920" height="1080" alt="Screenshot (69)" src="https://github.com/user-attachments/assets/a24167ff-c4a6-4062-a5b2-d17435af0155" />
<img width="1920" height="1080" alt="Screenshot (65)" src="https://github.com/user-attachments/assets/dfca34d6-5e5e-42bf-bf17-b723cf84a63e" />
<img width="1920" height="1080" alt="Screenshot (63)" src="https://github.com/user-attachments/assets/4be19d5d-5757-428a-8d9d-0678f69ed6a9" />

