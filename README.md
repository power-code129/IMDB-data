# IMDb Reviews Sentiment & Emotion Analysis

## 📌 Project Overview
This project focuses on sentiment and emotion analysis of IMDb movie reviews. The goal is to extract insights from user-generated content by applying advanced natural language processing (NLP) techniques, improving traditional sentiment classification with emotion detection and real-time analysis.

## 📊 Dataset
- Source: IMDb movie reviews dataset
- Contains user reviews with corresponding sentiment labels (positive/negative)
- Extended with emotion detection (e.g., joy, anger, sadness, surprise, etc.)
- Preprocessed for stopwords removal, tokenization, and vectorization

## 🛠️ Technologies Used
- **Python** (pandas, numpy, matplotlib, seaborn)
- **NLP & Machine Learning** (NLTK, Scikit-learn, TensorFlow, Transformers)
- **Vectorization** (TF-IDF, Word2Vec, BERT embeddings)
- **Deep Learning** (LSTMs, Transformers for emotion detection)
- **Deployment** (Flask/FastAPI for API, Streamlit for visualization)

## 📈 Methodology
### Step 1: Import Libraries
- Import `pandas` for data manipulation
- Import `NLTK` for natural language processing

### Step 2: Load Dataset
- Mount the CSV file to VSCode
- Read the dataset using pandas

### Step 3: Data Preprocessing
- Removing HTML tags
- Importing NLTK for text processing
- Removing stop words
- Text lemmatization
- Removing noise
- Adding a new column for cleaned text
- Splitting data into training and testing sets

### Step 4: Feature Extraction
- Use TF-IDF Vectorizer to transform text into numerical features

### Step 5: Model Building and Evaluation
- Train and evaluate machine learning models:
  - **Random Forest Classifier**
  - **Multinomial Naive Bayes**
- Test the model on new reviews

## 🚀 Setup & Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

## 📌 Usage
### Training Models
```bash
python train.py --model sentiment
python train.py --model emotion
```

### Running Sentiment Analysis on New Reviews
```bash
python predict.py --text "This movie was absolutely fantastic!"
```


## 📅 Future Enhancements
- Implement real-time IMDb review analysis
- Improve accuracy with advanced deep learning techniques
- Add multilingual sentiment & emotion support
- Integrate interactive dashboard for visualization

## 📝 License
This project is licensed under the MIT License. You can find the full license text in the LICENSE file of this repository.

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

---
✨ If you found this project helpful, please ⭐ the repository!

