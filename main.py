# Penerapan Algoritma K-Means dan DBSCAN untuk clustering review pengguna aplikasi mobile mypertamina pada google playstore
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import swifter
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics import silhouette_score
from utils.preprocess import (
    cleansing, 
    tokenization,
    stemmed_wrapper,
    get_stemmed_term,
    stopwords_removal,
    normalized_term,
    extract_meaningful_bigrams,
    idf,
    tf,
    tfIdf
)
from utils.eda import find_high_term_document
c = st.container()

with st.sidebar:
    
    st.image("https://www.usd.ac.id/logo/usd.png")
    
    st.markdown("---")
    st.markdown("ROSINTONG AMANDA SIHOTANG")
    st.markdown("205314080")
    st.markdown("---")
    
    st.markdown("# Menu")
    uploaded = st.file_uploader("Upload file", key="data", type=["csv"])
    
    st.markdown("## Terms")
    terms = st.selectbox("Choose", key="terms", options=[10, 20])
    
    st.markdown("## K-Means")
    k_cluster = st.number_input("K Cluster", key="k_cluster",min_value=2, max_value=16, value=3)
    
    st.markdown("## DBSCAN")
    epsilon = st.slider("Epsilon", key="epsilon", min_value=0.5, max_value=0.9, value=0.5)
    min_samples = st.number_input("Min Samples", key="min_samples", min_value=2, max_value=20, value=5)
    

c.markdown("# PENERAPAN ALGORITMA K-MEANS DAN DBSCAN UNTUK CLUSTERING REVIEW PENGGUNA APLIKASI MOBILE MYPERTAMINA PADA GOOGLE PLAY STORE")
c.markdown("---")

if uploaded is not None:
    uploaded_file = st.session_state.get("data")
    df = pd.read_csv(uploaded_file) if uploaded_file is not None else None
    c.markdown("## Data")
    c.dataframe(df, hide_index=True)
    
    rating = c.selectbox("fitur untuk 'Rating'", options=df.columns)
    st.session_state.rating_key = rating
    review = c.selectbox("fitur untuk 'Review'", options=df.columns)
    st.session_state.review_key = review
    c.button("Process", key="process_to_preprocessing")

# Preprocessing Data
expander_processing_data = c.expander("Preprocessing Data")

if st.session_state.get('process_to_preprocessing') or 'df' in st.session_state:
    bar_processing_data = c.progress(0, text="Operation in progress. Please wait...")
    if 'df' not in st.session_state:
        df = df[[rating, review]]
        df['CLEANING'] = df[review].apply(lambda x: cleansing(x))
        bar_processing_data.progress(20, text="Cleaning done...")
        df['TOKENIZATION'] = df['CLEANING'].apply(lambda x: tokenization(x))
        bar_processing_data.progress(40, text="Tokenizing done...")
        
        term_dict = {}
        for document in df['TOKENIZATION']:
            for term in document:
                if term not in term_dict:
                    term_dict[term] = ' '
                    
        for term in term_dict:
            term_dict[term] = stemmed_wrapper(term)

        df['STEMMED'] = df['TOKENIZATION'].swifter.apply(lambda x: get_stemmed_term(x, term_dict=term_dict))
        bar_processing_data.progress(70, text="Stemming done...")
        
        df['STOPWORDS_REMOVAL'] = df['STEMMED'].apply(lambda x: stopwords_removal(x))
        bar_processing_data.progress(80, text="Stopwords removal done...")
        
        df['NORMALIZED'] = df['STOPWORDS_REMOVAL'].apply(lambda x: normalized_term(x)).str.join(" ")
        bar_processing_data.progress(90, text="Normalization done...")
        
        df["Bigrams"] = df['NORMALIZED'].apply(lambda x: extract_meaningful_bigrams(x))
        bar_processing_data.progress(100, text="Bigrams done...")
        st.session_state.df = df
    else:
        df = st.session_state.df
        
    bar_processing_data.empty()
    expander_processing_data.dataframe(df, hide_index=True)
    expander_processing_data.button("Lanjutkan", key="process_to_clustering_rating")

# Clustering Rating
expander_clustering_rating = c.expander("Clustering Rating")

if st.session_state.get('process_to_clustering_rating') or 'df_grouped' in st.session_state:
    
    if 'df_grouped' not in st.session_state:
        st.session_state.df_grouped = df_grouped = df.groupby(rating)
    else:
        df_grouped = st.session_state.df_grouped
        
    expander_clustering_rating.markdown("## Distribusi Rating Review")
    fig, ax = plt.subplots()
    ax = sns.barplot(x=df_grouped.size().index, y=df_grouped.size().values, palette='viridis')
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 9), textcoords = 'offset points')
        
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    expander_clustering_rating.pyplot(fig)
    
    df[review] = df[review].fillna('')
    docs = df[review].tolist()
    # Memanggil fungsi dan menyimpan hasilnya dalam 'high_terms'
    high_terms = find_high_term_document(docs, terms)

    # Konversi list 'high_terms' ke DataFrame
    st.session_state.high_terms_df = high_terms_df = pd.DataFrame(high_terms, columns=['Term', 'Frequency'])

    # Menghitung jumlah total dokumen
    total_documents = len(docs)

    # Menambahkan kolom 'Percentage' ke DataFrame
    high_terms_df['Percentage'] = (high_terms_df['Frequency'] / total_documents) * 100

    # Menambahkan kolom kumulatif persentase
    high_terms_df['Cumulative Percentage'] = high_terms_df['Percentage'].cumsum()

    # Format kolom 'Percentage' dengan string
    high_terms_df['Percentage'] = high_terms_df['Percentage'].apply(
        lambda x: f'{x:.2f}% dari {total_documents} total dokumen'
    )

    expander_clustering_rating.markdown("## Tabel Top Terms")
    # Print output dalam bentuk DataFrame
    expander_clustering_rating.dataframe(high_terms_df, hide_index=True)
    
    expander_clustering_rating.markdown("## Pareto Chart of Top Terms")
    # Membuat diagram Pareto
    fig, ax1 = plt.subplots(figsize=(14, 10))

    # Membuat diagram batang
    ax1.bar(high_terms_df['Term'], high_terms_df['Frequency'], color='teal')
    ax1.set_xlabel('Term')
    ax1.set_ylabel('Frequency', color='C0')

    # Membuat garis kumulatif persentase
    ax2 = ax1.twinx()
    ax2.plot(high_terms_df['Term'], high_terms_df['Cumulative Percentage'], color='C1', marker='D', ms=5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax2.set_ylabel('Cumulative Percentage', color='C1')

    # Menambahkan angka pada garis kumulatif persentase
    for i in range(len(high_terms_df)):
        ax2.annotate(f'{high_terms_df["Cumulative Percentage"].iloc[i]:.2f}%',
                    (high_terms_df['Term'].iloc[i], high_terms_df['Cumulative Percentage'].iloc[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    fontsize=8,
                    color='C1')

    plt.title('Pareto Chart of Top Terms')
    st.session_state.high_terms = pd.DataFrame({'kata_kunci': high_terms_df['Term']})
    expander_clustering_rating.pyplot(fig)
    
    expander_clustering_rating.button("Lanjutkan ke TF-IDF", key="process_to_tfidf")
    
expander_tfidf = c.expander("TF-IDF")

if st.session_state.get("process_to_tfidf") or 'tfidf' in st.session_state:
    high_terms = st.session_state.high_terms
    df = st.session_state.df
    review = st.session_state.review_key
    resultIDF = idf(df[review], high_terms)
    resultTF = tf(df[review], high_terms)
    resultTFIDF = tfIdf(resultTF, resultIDF)
    
    st.session_state.tfidf = resultTFIDF
    expander_tfidf.button("Lanjutkan ke K-Means", key="process_to_kmeans")
    expander_tfidf.button("Lanjutkan ke DBSCAN", key="process_to_dbscan")
    

# KMEANS
expander_kmeans = c.expander("K-Means", expanded=True)
if st.session_state.get("process_to_kmeans") or 'tfidf' in st.session_state:
    data = st.session_state.tfidf
    df = st.session_state.df
    k_cluster = st.session_state.k_cluster
    results = []

    for k in range(2, 17):  # Mengubah range sesuai kebutuhan
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10, max_iter=100)
        kmeans.fit(data)
        labels = kmeans.labels_
        silhouette_score = round(metrics.silhouette_score(data, labels), 4)  # Memformat skor dengan 4 angka di belakang koma
        results.append({'n_clusters': k, 'silhouette_coefficient': silhouette_score})

    # Membuat DataFrame dari hasil
    results_df = pd.DataFrame(results)

    # Mencari nilai n_clusters dengan silhouette_score tertinggi
    optimal_n = results_df.loc[results_df['silhouette_coefficient'].idxmax()]

    # Membuat plot
    fig = plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_clusters'], results_df['silhouette_coefficient'], marker='o', linestyle='-')
    plt.title('Silhouette Coefficient untuk Berbagai Jumlah Kluster')
    plt.xlabel('Jumlah Kluster (n_clusters)')
    plt.ylabel('Silhouette Coefficient')
    plt.axvline(x=optimal_n['n_clusters'], color='red', linestyle='--', label=f'Optimal (n_clusters={optimal_n["n_clusters"]})')
    plt.legend()
    plt.grid(True)
    
    expander_kmeans.markdown("## Silhouette Coefficient")
    expander_kmeans.pyplot(fig)
    
    kmeans = KMeans(n_clusters=int(k_cluster), random_state=0, n_init=10, max_iter=100)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    expander_kmeans.markdown('Estimated number of clusters: %d' % n_clusters_)
    expander_kmeans.markdown("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))
    
    if len(labels) == len(df):
        # Setelah memverifikasi panjangnya, tambahkan kolom hasil clustering ke dataset
        df['Cluster'] = labels
    else:
        # Jika panjangnya tidak sama, cetak pesan kesalahan atau lakukan penanganan lain
        if len(labels) > len(df):
            print("Panjang labels lebih besar dari jumlah baris di DataFrame. Menghapus label terakhir.")
            labels = labels[:len(df)]  # Menghapus label terakhir agar panjangnya sesuai
            df['Cluster'] = labels
        else:
            print("Panjang labels lebih kecil dari jumlah baris di DataFrame. Menambah label 0 ke akhir.")
            diff = len(df) - len(labels)  # Menghitung selisih panjang labels dengan jumlah baris df
            labels = np.append(labels, np.zeros(diff))  # Menambahkan label 0 ke akhir labels
            df['Cluster'] = labels

    df['Cluster'] = df['Cluster'].astype(int)
    df['Bigrams'] = df['Bigrams'].fillna('')

    # Mengumpulkan teks untuk setiap klaster
    cluster_texts = {}
    for cluster_id in df['Cluster'].unique():
        df_text = df[df['Cluster'] == cluster_id]['Bigrams'].apply(lambda x: ' '.join(x))
        texts = ' '.join(df_text.to_numpy())
        cluster_texts[int(cluster_id)] = texts  # Mengubah cluster_id menjadi integer
    
    expander_kmeans.markdown("## Word Cloud")
    for cluster_id, text in cluster_texts.items():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig = plt.figure(figsize=(8, 4))  # Tidak perlu membuat figur untuk ditampilkan
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Cluster {cluster_id} Word Cloud')
        expander_kmeans.pyplot(fig)

        # Menampilkan kata-kata yang paling dominan
        words = text.split()
        word_freq = Counter(words)
        dominant_words = word_freq.most_common(5)  # Mengambil kata paling dominan
        expander_kmeans.dataframe(pd.DataFrame(dominant_words, columns=['Word', 'Frequency']), hide_index=True)
    
expander_dbscan = c.expander("DBSCAN", expanded=True)
if st.session_state.get("process_to_dbscan") or 'tfidf' in st.session_state:
    data = st.session_state.tfidf
    df = st.session_state.df
    results = []

    # Perulangan untuk berbagai nilai eps dan min_samples
    # for eps in np.arange(0.1, 1.0, 0.1):  # Mengubah range dan step sesuai kebutuhan
    #     for min_samples in range(2, 20):  # Mengubah range sesuai kebutuhan
    for eps in np.arange(0.5, 1.0, 0.1):  # Mengubah range dan step sesuai kebutuhan
        for min_samples in range(2, 20):  # Mengubah range sesuai kebutuhan
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
            labels = db.labels_
            
            # Menghitung jumlah kluster dan titik noise
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            
            # Menghitung skor silhouette, hanya jika ada lebih dari satu kluster dan kurang dari jumlah data
            if n_clusters_ > 1 and n_clusters_ < len(data):
                silhouette_score = round(metrics.silhouette_score(data, labels), 4)
            else:
                silhouette_score = 'N/A'  # Tidak dapat dihitung
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters_,
                'n_noise': n_noise_,
                'silhouette_score': silhouette_score
            })

    # Membuat DataFrame dari hasil
    results_df = pd.DataFrame(results)
    
    expander_dbscan.markdown("## Silhouette Coefficient")
    results_df['silhouette_score'] = pd.to_numeric(results_df['silhouette_score'], errors='coerce')

    # Membuat pivot table untuk heatmap
    pivot_table = results_df.pivot(index='min_samples', columns='eps', values='silhouette_score')

    # Membuat heatmap
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='YlGnBu', cbar_kws={'label': 'Silhouette Coefficient'})
    plt.title('Heatmap of Silhouette Coefficient for Different eps and min_samples')
    plt.xlabel('eps')
    plt.ylabel('min_samples')
    
    expander_dbscan.pyplot(fig)
    
    eps = st.session_state.epsilon
    min_samples = st.session_state.min_samples
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    expander_dbscan.markdown('Estimated number of clusters: %d' % n_clusters_)
    expander_dbscan.markdown('Estimated number of noise points: %d' % n_noise_)
    expander_dbscan.markdown("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))
    
    if len(labels) == len(df):
        # Setelah memverifikasi panjangnya, tambahkan kolom hasil clustering ke dataset
        df['Cluster'] = labels
    else:
        # Jika panjangnya tidak sama, cetak pesan kesalahan atau lakukan penanganan lain
        if len(labels) > len(df):
            print("Panjang labels lebih besar dari jumlah baris di DataFrame. Menghapus label terakhir.")
            labels = labels[:len(df)]  # Menghapus label terakhir agar panjangnya sesuai
            df['Cluster'] = labels
        else:
            print("Panjang labels lebih kecil dari jumlah baris di DataFrame. Menambah label 0 ke akhir.")
            diff = len(df) - len(labels)  # Menghitung selisih panjang labels dengan jumlah baris df
            labels = np.append(labels, np.zeros(diff))  # Menambahkan label 0 ke akhir labels
            df['Cluster'] = labels

    # Mengubah tipe data kolom 'Cluster' menjadi integer
    df['Cluster'] = df['Cluster'].astype(int)
    
    # Mengganti nilai NaN dengan string kosong ('')
    df['Bigrams'] = df['Bigrams'].fillna('')

    # Mengumpulkan teks untuk setiap klaster
    cluster_texts = {}
    for cluster_id in df['Cluster'].unique():
        df_text = df[df['Cluster'] == cluster_id]['Bigrams'].apply(lambda x: ' '.join(x))
        texts = ' '.join(df_text.to_numpy())
        cluster_texts[int(cluster_id)] = texts  # Mengubah cluster_id menjadi integer

    # Membuat word cloud untuk setiap klaster dan menyimpannya ke file
    for cluster_id, text in cluster_texts.items():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig = plt.figure(figsize=(8, 4))  # Tidak perlu membuat figur untuk ditampilkan
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Cluster {cluster_id} Word Cloud')
        
        expander_dbscan.pyplot(fig)

        # Menampilkan kata-kata yang paling dominan
        words = text.split()
        word_freq = Counter(words)
        dominant_words = word_freq.most_common(5)  # Mengambil kata paling dominan
        expander_dbscan.dataframe(pd.DataFrame(dominant_words, columns=['Word', 'Frequency']), hide_index=True)