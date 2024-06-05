import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def find_high_term_document(docs, n):
    # Menggunakan CountVectorizer untuk membuat document-term matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    # Menghitung jumlah frekuensi setiap istilah
    term_freq = np.sum(X.toarray(), axis=0)
    # Mendapatkan nama istilah
    terms = vectorizer.get_feature_names_out()
    # Mengurutkan istilah berdasarkan frekuensi dan mengambil n teratas
    top_indices = np.argsort(term_freq)[::-1][:n]
    top_terms = terms[top_indices]
    top_freq = term_freq[top_indices]
    return list(zip(top_terms, top_freq))