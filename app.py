import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import numpy as np
from joblib import load, dump
import os
import io

st.title("Customer Segmentation using K-Means Clustering")

st.header("Dataset")
DATA_PATH = "data/Country-data.csv"
MODEL_PATH = "models/kmeans_model.pkl"

if os.path.exists(DATA_PATH):
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    st.write("### Raw Data")
    st.write(df.head())

    # Missing Values Check
    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # Visualisasi Distribusi Data
    st.header("Visualisasi Distribusi Data")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f"Distribusi {column}")
        st.pyplot(fig)

    # Normalisasi Data
    st.header("Normalisasi Data")
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    st.write("### Data Setelah Normalisasi")
    st.write(df.head())

    # Select features for clustering
    st.header("Clustering")
    selected_features = st.multiselect(
        "Select Features for Clustering",
        numeric_columns,
        default=numeric_columns
    )
    if selected_features:
        data = df[selected_features]

        # Perform Clustering Analysis
        st.subheader("Elbow Method untuk Menentukan Jumlah Cluster")
        distortions = []
        K_range = range(2, 11)
        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(data)
            distortions.append(model.inertia_)

        # Elbow Plot
        fig, ax = plt.subplots()
        ax.plot(K_range, distortions, marker='o', color='blue')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        # Select K
        k = st.number_input("Pilih Jumlah Cluster (k):", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(data)
        df['Cluster'] = clusters
        st.write("### Hasil Clustering")
        st.write(df.head())

        # Jumlah Data Tiap Cluster
        st.subheader("Jumlah Data pada Tiap Cluster")
        cluster_counts = df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        st.write(cluster_counts)

        # Statistik Tiap Cluster
        st.subheader("Statistik Tiap Cluster")
        cluster_stats = df.groupby('Cluster').mean(numeric_only=True)
        st.write(cluster_stats)

        # Silhouette Score
        st.subheader("Silhouette Score")
        silhouette_avg = silhouette_score(data, clusters)
        st.write(f"Silhouette Score: {silhouette_avg:.4f}")

        # PCA Visualization
        st.subheader("PCA Visualization")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = clusters

        fig, ax = plt.subplots()
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='tab10', ax=ax)
        ax.set_title("PCA Visualization of Clusters")
        st.pyplot(fig)

        # Save Model Automatically
        os.makedirs("models", exist_ok=True)
        dump(kmeans, MODEL_PATH)
        st.success(f"Model K-Means telah disimpan di {MODEL_PATH}")

st.header("Prediksi Cluster Menggunakan Model K-Means")
if os.path.exists(MODEL_PATH):
    model = load(MODEL_PATH)

    # Input data manual menggunakan slider
    st.subheader("Masukkan Data untuk Prediksi Menggunakan Slider")
    input_data = []
    for feature in numeric_columns:
        value = st.slider(
            f"{feature}",
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].mean()),
            step=0.01
        )
        input_data.append(value)

    # Tombol untuk prediksi
    if st.button("Prediksi Cluster"):
        # Membuat DataFrame dari input slider
        input_df = pd.DataFrame([input_data], columns=numeric_columns)

        # Normalisasi input sesuai skala MinMaxScaler
        scaler = MinMaxScaler()
        scaled_input = scaler.fit_transform(input_df)

        # Prediksi cluster
        prediction = model.predict(scaled_input)
        st.success(f"Data yang dimasukkan diprediksi masuk ke Cluster: {prediction[0]}")

        # Tampilkan Data Input dan Hasil Prediksi
        st.write("### Data Input yang Dimaksud")
        st.write(input_df)

        # Tambahkan kolom hasil prediksi ke DataFrame
        input_df['Predicted_Cluster'] = prediction[0]
        st.write("### Hasil Prediksi dengan Data Input")
        st.write(input_df)
else:
    st.warning("Model belum tersedia. Silakan jalankan clustering terlebih dahulu.")

