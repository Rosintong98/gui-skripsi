import re
import emoji
import contractions
import numpy as np
import pandas as pd
import math
import os
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import warnings
warnings.filterwarnings("ignore")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

my_stopwords = stopwords.words('indonesian')
additional_stopwords = ['nya', 'ya', 'kalo', 'aplikasi', 'pertamina', \
    'pakai', 'spbu', 'apk', 'game', 'kasih', 'thumbs', 'sih', 'bikin', \
        'orang', 'nih', 'hp', 'isi', 'mypertamina', 'si', 'g','yg', 'ga', \
            'klo', 'my', 'dan', 'ini', 'dan', 'saya']
my_stopwords.extend(additional_stopwords)

with open(f"data/stopwordbahasa.txt") as file:
    txt_stopwords = file.read().splitlines()
    my_stopwords.extend(txt_stopwords)

my_stopwords = set(my_stopwords)

normalisasi_kata = pd.read_csv(f"data/Normalisasiku.csv")
normalisasi_kata_dict = {}
# Mengiterasi setiap baris dalam DataFrame normalisasi_kata
for index, row in normalisasi_kata.iterrows():
    if row[0] not in normalisasi_kata_dict:
        normalisasi_kata_dict[row[0]] = row[1]
        
def cleansing(content):
    # Menghapus tag "RT" yang biasanya digunakan untuk retweet pada platform Twitter.
    t1 = re.sub('RT\s', '', content)
    # Menghapus username yang dimulai dengan tanda @.
    t2 = re.sub('\B@\w+', "", t1)
    # Menggantikan karakter emoji dengan teks berdasarkan Unicode.
    t3 = emoji.demojize(t2)
    # Menghapus URL yang dimulai dengan "http://" atau "https://".
    t4 = re.sub('(http|https):\/\/\S+', '', t3)
    # Menghapus karakter '#' dari hashtag.
    t5 = re.sub('#+', '', t4)
    # Mengubah semua huruf menjadi huruf kecil.
    t6 = t5.lower()
    # Menggantikan pengulangan huruf dengan hanya satu kejadian. Misalnya, 'oooooo' menjadi 'oo'.
    t7 = re.sub(r'(.)\1+', r'\1\1', t6)
    # Menggantikan pengulangan tanda baca seperti '?' atau '!' dengan hanya satu kejadian. Misalnya, '!!!!!!!!!' menjadi '!'.
    t8 = re.sub(r'[\?\.\!]+(?=[\?.\!])', '', t7)
    # Menghapus karakter alfabet dan hanya mempertahankan karakter lain seperti angka dan karakter khusus.
    t9 = re.sub(r'[^a-zA-Z\s]', ' ', t8)
    # Menggantikan kontraksi kata dengan bentuk penuhnya. Misalnya, mengubah "can't" menjadi "cannot".
    t10 = contractions.fix(t9)
    # Mengembalikan teks yang telah diubah dan dibersihkan.
    return t10

def tokenization(NewContent):
    # Memisahkan kata-kata berdasarkan karakter non-alphanumerik sebagai pemisah.
    text = re.findall(r'\b\w+\b', NewContent)
    # Mengembalikan hasil pemisahan teks dalam bentuk daftar token.
    return text

def stemmed_wrapper(term):
    return stemmer.stem(term)

def get_stemmed_term(document, term_dict = {}):
    return [term_dict[term] for term in document]

def stopwords_removal(words, list_stopwords = my_stopwords):
    return [word for word in words if word.lower() not in list_stopwords]

def normalized_term(document, normalisasi_kata_dict = normalisasi_kata_dict):
    return [normalisasi_kata_dict[term] if term in normalisasi_kata_dict else term for term in document]

def removeNonWord(doc):
    result = doc.replace({'\w*\d\w*': ''}, regex=True)
    #hapus seluruh karakter yang tidak termasuk alphabet
    result = result.replace({'[\W_]+': ' '}, regex=True)
    #remove null dari hasil final
    return result[result.notnull()]

# function to calculate idf
def idf(doc, wordBank):
    N = len(doc.index)
    #buat dataframe dengan header word dan idf
    # result = pd.DataFrame(columns=['kata_kunci', 'idf'])
    result = []
    #untuk setiap kata pada wordBank lakukan.....
    for index, word in wordBank.iterrows():
        #hitung jumlah doc yang mengandung kata word['words']
        dft = np.sum(doc.str.contains(word['kata_kunci']))
        #hitung inverse document frequency smooth
        idft = math.log(N / (dft + 1), 10)
        #tambahkan idf untuk setiap kata pada data frame
        result.append([word['kata_kunci'], idft])
    #return variable result
    return pd.DataFrame(result, columns=['kata_kunci', 'idf'])

# function to calculate tf
def tf(doc, wordBank):
    #split kata berdasarkan spasi
    wordList = doc.str.split(' ')
    #hitung jumlah kata pada setiap doc
    maxFt = [len(s) for s in wordList]
    #buat DataFrame kosong untuk menyimpan hasil perhitungan Tf
    result = []
    #untuk setiap word dalam wordbank lakukan ....
    for index, word in wordBank.iterrows():
        #hitung frekuensi kata untuk setiap doc
        ft = np.add([s.count(word['kata_kunci']) for s in wordList], 0)
        #tf log normalization
        ftd = 1 + np.log10(ft)
        result.append(pd.Series(ftd).replace(-np.inf, 0))
        #return variable result
    return pd.DataFrame(result)

# function to calculate tf-idf
def tfIdf(tf, idf):
    #buat DataFrame kosong untuk menyimpan hasil perhitungan Tf-Idf
    result = []
    #untuk setiap tf
    for i in tf:
        #tf idf untuk document term weighting tf * idf
        tfIdf = tf[i] * idf['idf']
        #tambahkan hasil perhitungan tf idf kedalam DataFrame
        result.append(pd.Series(tfIdf))
        #return variable result
    return pd.DataFrame(result)

# function to clean text
def removeNonWord(doc):
    result = doc.replace({'\w*\d\w*': ''}, regex=True)
    #hapus seluruh karakter yang tidak termasuk alphabet
    result = result.replace({'[\W_]+': ' '}, regex=True)
    #remove null dari hasil final
    return result[result.notnull()]

def extract_meaningful_bigrams(text, stop_words=my_stopwords):
    # Memeriksa apakah teks adalah string
    if isinstance(text, str):
        # Tokenisasi teks
        tokens = word_tokenize(text.lower())
        
        # Menghapus tanda baca dan stopwords
        filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        
        # Membuat bigram
        bigram = ngrams(filtered_tokens, 2)
        
        # Menggabungkan bigram dengan underscore di antara kata-kata
        meaningful_bigrams = ['_'.join(bi) for bi in bigram]
        
        return meaningful_bigrams
    else:
        # Jika teks bukan string, kembalikan list kosong
        return []

def save_meaningful_bigrams(df, output_file):
    # Membuat DataFrame baru untuk menyimpan hasil
    results_df = pd.DataFrame(columns=['Index', 'Text', 'Rating', 'Meaningful_Bigrams'])
    
    # Melakukan iterasi untuk setiap baris dalam DataFrame
    for index, row in df.iterrows():
        # Menambahkan variabel 'index' dan 'text'
        text = row['Review']
        meaningful_bigrams = extract_meaningful_bigrams(text)
        
        # Menambahkan hasil ke DataFrame baru
        results_df = results_df.append({
            'Index': index,  # Menambahkan index dari iterasi
            'Text': text,  # Menambahkan teks review
            'Rating': row['Rating'],
            'Meaningful_Bigrams': meaningful_bigrams
        }, ignore_index=True)