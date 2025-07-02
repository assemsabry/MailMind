# MailMind

**MailMind** is an intelligent email classification tool built with AI that works completely offline — no API keys, no internet connection required after model download. It's powered by a locally hosted RoBERTa language model for highly accurate classification of email content into custom categories.

---

## 🚀 Features

- 📬 Smart classification of email content
- 🧠 Powered by `roberta-base` for deep text understanding
- 🔐 100% local processing (no cloud/API dependencies)
- 🖥️ Lightweight interface using Streamlit
- 🗂️ Train on your own datasets

---

## 📂 Project Structure
```
MailMind/
├── app/
│   └── gui.py                          # Streamlit interface
├── data/
│   └── processed/emails.csv           # Your labeled email dataset
├── models/
│   └── roberta_email_model.pkl       # Trained model file
├── src/
│   ├── bert_vectorizer.py            # RoBERTa-based vectorizer class
│   ├── preprocess.py                 # Data cleaning and text preparation
│   ├── predict_with_bert.py         # Prediction pipeline using saved model
│   └── train_with_bert.py           # Training script
├── requirements.txt                 # Dependencies
├── README.md                        # Project description
└── .gitignore                       # Files/folders to ignore
```

---

## 📊 Example Dataset Format
Place your dataset at: `data/processed/emails.csv`

```csv
message,label
"Your order has been shipped!",business
"Hey! Long time no see.",personal
"Win a million dollars now!!!",spam
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/assemsabry/MailMind.git
cd MailMind
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python src/train_with_bert.py
```
This will create a `roberta_email_model.pkl` file inside the `models/` folder.

### 4. Launch the App
```bash
streamlit run app/gui.py
```

---

## 🧠 Behind the Scenes
- Texts are cleaned using NLTK stopwords and regex.
- RoBERTa (`roberta-base`) encodes the input emails into embeddings.
- A logistic regression classifier is trained on top of these embeddings.

---

## 🧪 Future Enhancements
- Multi-label classification
- Active learning from user corrections
- Email summary generation
- Integration with Gmail/Outlook via local plugins

---

## 🔒 Disclaimer
This tool runs entirely on your device. No data is sent to any server or third-party API.

---

## 📌 Author
**Developed by Assem Sabry**  
_AI Engineer & Machine Learning Developer_  
GitHub: [assemsabry](https://github.com/assemsabry)

---

## 📄 License
This project is licensed under the MIT License.
