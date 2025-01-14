# Multilingual Text Classification

## 🚀 Project Overview
The **Multilingual Text Classification** project is a language identification system capable of detecting the language of a given text input from a wide range of languages. The model can classify text into 25 different languages, including but not limited to Arabic, Chinese, English, French, Hindi, Japanese, Korean, Latin, Spanish, and more.

This project consists of:
- **Data Preprocessing:** Tokenization, padding sequences, and label encoding.
- **Model Development:** Building and training models using LSTM, GRU, and CNN architectures.
- **Evaluation:** Accuracy metrics, confusion matrix, and classification report.
- **Frontend Web Application:** A Streamlit app to provide a user-friendly interface for text classification.

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Streamlit
- TensorFlow
- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Clone the Repository
```bash
$ git clone https://github.com/your-username/multilingual-text-classification.git
$ cd multilingual-text-classification
```

### Install Dependencies
```bash
$ pip install -r requirements.txt
```

### Run the Streamlit App
```bash
$ streamlit run main.py
```

---

## 📊 Dataset
The dataset used in this project includes text samples from 25 languages, ensuring a diverse and comprehensive dataset for training the model.

**Preprocessing steps:**
- Tokenization
- Padding sequences to a maximum length of 100
- Label encoding to convert language labels into numerical format

---

## 🧪 Models Used
The project explored three different architectures:

### 1. LSTM (Long Short-Term Memory)
- **Training Accuracy:** 99.5%
- **Validation Accuracy:** 90.2%

### 2. GRU (Gated Recurrent Unit)
- **Training Accuracy:** 99.9%
- **Validation Accuracy:** 87.8%

### 3. CNN (Convolutional Neural Network)
- **Training Accuracy:** 99.9%
- **Validation Accuracy:** 92.8%

---

## 📈 Evaluation Metrics

### Confusion Matrix
Each model's confusion matrix was visualized using heatmaps to understand the classification performance across all languages.

### Classification Report
The classification reports showed the precision, recall, and f1-score for each language class.

---

## 🌐 Usage
1. **Launch the Streamlit app** by running the command:
   ```bash
   streamlit run main.py
   ```
2. **Enter text** in any language from the supported list.
3. **Click Predict** to see the language classification result.

---

## 🤖 Technologies Used
- **Python**
- **TensorFlow**
- **Scikit-learn**
- **Streamlit**

---

## 📂 Project Structure
```
.
├── main.py               # Streamlit app
├── language-classification.ipynb  # Data analysis and model development
├── requirements.txt      # List of dependencies
├── models
  ├── LC_CNN_Model.keras
  ├── LC_GRU_Model.keras
  ├── LC_LSTM_Model.keras
└── 
```

---

