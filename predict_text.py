import nltk
from nltk.corpus import stopwords
import string

# Preprocessing texs
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    """Fungsi untuk preprocessing teks: lowercase, hapus tanda baca, hapus stopwords"""
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Menghapus stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict(model, vectorizer, text):
    """Fungsi untuk memprediksi sentimen dari teks"""
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Vectorize the text
    vectorized_text = vectorizer.transform([preprocessed_text])
    
    # Predict sentiment (dummy prediction for illustration)
    prediction = model.predict(vectorized_text)
    
    return prediction[0]