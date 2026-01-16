import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Diabetes Clustering Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tema merah hati dan coklat muda
st.markdown("""
<style>
    .main {
        background-color: #FFF5F5;
    }
    .stApp {
        background: linear-gradient(135deg, #FFF5F5 0%, #F8F0E3 100%);
    }
    .css-18e3th9 {
        background-color: #FFF5F5;
    }
    h1, h2, h3 {
        color: #B03A2E;
        font-family: 'Arial', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8F0E3;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #B03A2E;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #B03A2E;
        color: white !important;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #B03A2E;
        margin: 10px 0;
    }
    .highlight {
        background-color: #FFE4E1;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #B03A2E;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('diabetes_aggregated.csv')
    return data

# Preprocessing function
def preprocess_data(data):
    df = data.copy()
    
    # 1. Handle missing values in Country column
    df['Country'] = df['Country'].fillna('Unknown')
    
    # 2. Remove extreme outliers in diabetes_mean
    Q1 = df['diabetes_mean'].quantile(0.25)
    Q3 = df['diabetes_mean'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['diabetes_mean'] >= lower_bound) & (df['diabetes_mean'] <= upper_bound)]
    
    # 3. Log transformation for skewed variables
    skewed_cols = ['diabetes_mean', 'diabetes_min', 'diabetes_max']
    for col in skewed_cols:
        if df[col].min() > 0:  # Only apply log if all values are positive
            df[f'log_{col}'] = np.log1p(df[col])
    
    # 4. Encode categorical variable
    le = LabelEncoder()
    df['Country_encoded'] = le.fit_transform(df['Country'])
    
    # 5. Feature scaling
    features_for_clustering = ['diabetes_mean', 'diabetes_min', 'diabetes_max', 'Country_encoded']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features_for_clustering])
    
    return df, df_scaled, features_for_clustering, scaler

# Function to perform clustering
def perform_clustering(X, method='kmeans', n_clusters=3):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        return labels, model
    
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = model.fit_predict(X)
        return labels, model
    
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        return labels, model
    
    elif method == 'meanshift':
        bandwidth = estimate_bandwidth(X, quantile=0.2)
        model = MeanShift(bandwidth=bandwidth)
        labels = model.fit_predict(X)
        return labels, model
    
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)
        return labels, model
    
    return None, None

# Main app
def main():
    st.title("❤️ Diabetes Clustering Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    data = load_data()
    
    # Create tabs
    tabs = st.tabs(["📋 About", "📊 Dataset", "⚙️ Preprocessing", "🤖 Machine Learning", 
                   "📈 Analysis (K-Means)", "🎨 Visualisasi", "🔮 Prediksi", "📞 Contact"])
    
    # Tab 1: About
    with tabs[0]:
        st.header("📋 Tentang Project")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 🎯 Pendahuluan Project")
            st.markdown("""
            Project ini merupakan implementasi dari mata kuliah **Machine Learning** yang fokus pada analisis clustering 
            data diabetes menggunakan berbagai metode algoritma unsupervised learning. Tujuan utama project ini adalah 
            untuk mengelompokkan negara-negara berdasarkan karakteristik data diabetesnya sehingga dapat memberikan 
            insights untuk intervensi kesehatan yang lebih terfokus.
            """)
            
            st.markdown("### 📚 Tentang Dataset")
            st.markdown("""
            **Sumber Data**: World Health Organization (WHO) Global Health Observatory  
            **Variabel yang tersedia**:
            - `SpatialDimensionValueCode`: Kode negara/wilayah
            - `diabetes_mean`: Rata-rata prevalensi diabetes
            - `diabetes_min`: Nilai minimum prevalensi diabetes
            - `diabetes_max`: Nilai maksimum prevalensi diabetes
            - `Country`: Nama wilayah/wilayah WHO
            
            **Jumlah Data**: Dataset ini terdiri dari **195 entri** yang mencakup berbagai negara dan wilayah di dunia.
            """)
            
            st.markdown("### 🎯 Tujuan Analisis")
            st.markdown("""
            1. **Mengidentifikasi pola** dalam data diabetes secara global
            2. **Mengelompokkan negara** berdasarkan karakteristik diabetes yang serupa
            3. **Memberikan rekomendasi** untuk intervensi kesehatan yang tepat sasaran
            4. **Membandingkan performa** berbagai algoritma clustering
            5. **Mengembangkan model prediktif** untuk klasifikasi data baru
            """)
            
        with col2:
            st.image("https://img.icons8.com/color/200/000000/heart-health.png", 
                    caption="Analisis Kesehatan Diabetes")
            
        st.markdown("### 🔬 Metodologi Algoritma Clustering")
        
        methods_info = {
            "K-Means": {
                "description": "Algoritma partitional clustering yang membagi data menjadi K cluster berdasarkan jarak ke centroid",
                "formula": "$$J = \\sum_{i=1}^{n} \\sum_{k=1}^{K} w_{ik} ||x_i - \\mu_k||^2$$",
                "strength": "Efisien untuk dataset besar, mudah diimplementasi"
            },
            "Gaussian Mixture Model (GMM)": {
                "description": "Model probabilistik yang mengasumsikan data berasal dari campuran beberapa distribusi Gaussian",
                "formula": "$$p(x) = \\sum_{k=1}^{K} \\pi_k \\mathcal{N}(x|\\mu_k, \\Sigma_k)$$",
                "strength": "Dapat menangani cluster dengan bentuk berbeda"
            },
            "Hierarchical Clustering": {
                "description": "Membangun hierarki cluster dengan pendekatan agglomerative atau divisive",
                "formula": "$$d(C_i, C_j) = \\min_{x \\in C_i, y \\in C_j} d(x, y)$$",
                "strength": "Tidak perlu menentukan jumlah cluster awal"
            },
            "Mean Shift": {
                "description": "Algoritma non-parametric yang menemukan moda dari density function",
                "formula": "$$m(x) = \\frac{\\sum_{x_i \\in N(x)} K(x_i - x)x_i}{\\sum_{x_i \\in N(x)} K(x_i - x)}$$",
                "strength": "Menentukan jumlah cluster secara otomatis"
            },
            "Spectral Clustering": {
                "description": "Menggunakan eigenvector dari matriks similarity untuk melakukan clustering",
                "formula": "$$L = D - W$$",
                "strength": "Efektif untuk data dengan struktur non-convex"
            }
        }
        
        for method, info in methods_info.items():
            with st.expander(f"**{method}**"):
                st.markdown(f"**Deskripsi**: {info['description']}")
                st.markdown(f"**Rumus**: {info['formula']}")
                st.markdown(f"**Kelebihan**: {info['strength']}")
    
    # Tab 2: Dataset
    with tabs[1]:
        st.header("📊 Dataset Preview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(data.head(20), use_container_width=True)
            
        with col2:
            st.markdown("### 📋 Informasi Dataset")
            st.metric("Jumlah Baris", f"{len(data)}")
            st.metric("Jumlah Kolom", f"{len(data.columns)}")
            st.metric("Missing Values", f"{data.isnull().sum().sum()}")
            
            st.markdown("### 📝 Deskripsi Variabel")
            var_info = {
                "SpatialDimensionValueCode": "Kode unik untuk negara/wilayah (3 huruf)",
                "diabetes_mean": "Rata-rata prevalensi diabetes (kontinu)",
                "diabetes_min": "Nilai minimum prevalensi diabetes (kontinu)",
                "diabetes_max": "Nilai maksimum prevalensi diabetes (kontinu)",
                "Country": "Region WHO atau nama negara (kategorikal)"
            }
            
            for var, desc in var_info.items():
                with st.expander(var):
                    st.write(desc)
    
    # Tab 3: Preprocessing
    with tabs[2]:
        st.header("⚙️ Preprocessing Data")
        
        st.markdown("""
        ### 🔧 Langkah-langkah Preprocessing yang Dilakukan:
        
        1. **Penanganan Missing Values**:
           - Kolom 'Country' yang memiliki nilai NaN diisi dengan 'Unknown'
           
        2. **Deteksi dan Penanganan Outliers**:
           - Menggunakan metode IQR (Interquartile Range) untuk mendeteksi outliers
           - Data yang berada di luar 1.5 * IQR dari Q1 dan Q3 dianggap outlier
           
        3. **Transformasi Data**:
           - Melakukan log transformation pada variabel yang skewed
           - Menggunakan np.log1p() untuk menghindari log(0)
           
        4. **Encoding Variabel Kategorikal**:
           - Menggunakan Label Encoding untuk kolom 'Country'
           - Mengubah nilai kategorikal menjadi numerik
           
        5. **Feature Scaling**:
           - Menggunakan StandardScaler untuk menormalisasi data
           - Mean = 0, Standard Deviation = 1
           
        6. **Pemilihan Fitur**:
           - Memilih 4 fitur utama untuk clustering:
             * diabetes_mean
             * diabetes_min
             * diabetes_max
             * Country_encoded
        """)
        
        # Show preprocessing results
        st.subheader("🔍 Hasil Preprocessing")
        
        df_processed, X_scaled, features, scaler = preprocess_data(data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data setelah preprocessing:**")
            st.dataframe(df_processed.head(10), use_container_width=True)
            
        with col2:
            st.markdown("**Statistik Deskriptif:**")
            st.dataframe(df_processed[features].describe(), use_container_width=True)
    
    # Tab 4: Machine Learning
    with tabs[3]:
        st.header("🤖 Analisis Clustering")
        
        # Get preprocessed data
        df_processed, X_scaled, features, scaler = preprocess_data(data)
        
        # Slider for number of clusters
        n_clusters = st.slider("Pilih jumlah cluster:", 2, 6, 3)
        
        # Perform clustering with all methods
        methods = ['kmeans', 'gmm', 'hierarchical', 'meanshift', 'spectral']
        results = []
        
        st.subheader("📊 Performa Metode Clustering")
        
        cols = st.columns(5)
        for idx, (col, method) in enumerate(zip(cols, methods)):
            with col:
                labels, model = perform_clustering(X_scaled, method, n_clusters)
                if len(np.unique(labels)) > 1:  # Check if we have more than 1 cluster
                    score = silhouette_score(X_scaled, labels)
                else:
                    score = 0
                
                # Display score
                st.metric(
                    label=method.upper(),
                    value=f"{score:.4f}",
                    delta="Terbaik" if score == max([r[1] for r in results if len(results) > 0] + [0]) else None
                )
                
                results.append((method, score, labels))
        
        # Find best method
        best_method = max(results, key=lambda x: x[1])
        
        st.markdown("---")
        st.markdown(f"### 🏆 **Metode Terbaik: {best_method[0].upper()}**")
        st.markdown(f"**Silhouette Score: {best_method[1]:.4f}**")
        
        # Visualization of clustering results
        st.subheader("📈 Visualisasi Hasil Clustering")
        
        # Create scatter plot for each method
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[m.upper() for m in methods],
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}],
                   [{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        for i, (method, score, labels) in enumerate(results):
            row = i // 3 + 1
            col = i % 3 + 1
            
            # Create 3D scatter plot
            scatter = go.Scatter3d(
                x=X_scaled[:, 0],
                y=X_scaled[:, 1],
                z=X_scaled[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=labels,
                    colorscale='RdYlBu',
                    showscale=True if i == 0 else False
                ),
                text=df_processed['SpatialDimensionValueCode'],
                name=method.upper()
            )
            
            fig.add_trace(scatter, row=row, col=col)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Hasil Clustering dengan Berbagai Metode"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Analysis (K-Means)
    with tabs[4]:
        st.header("📈 Analisis Detail dengan K-Means")
        
        st.markdown("### 📋 Langkah-langkah Algoritma K-Means:")
        
        steps = [
            "1. **Inisialisasi**: Pilih K titik acak sebagai centroid awal",
            "2. **Assignment**: Hitung jarak setiap titik data ke semua centroid, kelompokkan ke centroid terdekat",
            "3. **Update**: Hitung centroid baru sebagai mean dari semua titik dalam cluster",
            "4. **Iterasi**: Ulangi langkah 2-3 hingga konvergen atau mencapai iterasi maksimum",
            "5. **Konvergensi**: Algoritma berhenti ketika centroid tidak berubah signifikan"
        ]
        
        for step in steps:
            st.markdown(step)
        
        # Perform K-Means with elbow method
        df_processed, X_scaled, features, scaler = preprocess_data(data)
        
        st.subheader("🔍 Menentukan Jumlah Cluster Optimal (Elbow Method)")
        
        # Calculate inertia for different k values
        inertias = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='#B03A2E', width=3)
        ))
        
        fig_elbow.update_layout(
            title='Elbow Method untuk Menentukan K Optimal',
            xaxis_title='Jumlah Cluster (K)',
            yaxis_title='Inertia',
            height=500
        )
        
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Let user select optimal k
        optimal_k = st.slider("Pilih jumlah cluster optimal berdasarkan elbow method:", 2, 10, 3)
        
        # Perform K-Means with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to dataframe
        df_processed['Cluster'] = labels
        
        st.subheader("📊 Karakteristik Cluster")
        
        # Show cluster statistics
        cluster_stats = df_processed.groupby('Cluster').agg({
            'diabetes_mean': ['mean', 'std', 'min', 'max'],
            'diabetes_min': ['mean', 'std'],
            'diabetes_max': ['mean', 'std'],
            'SpatialDimensionValueCode': 'count'
        }).round(2)
        
        st.dataframe(cluster_stats, use_container_width=True)
    
    # Tab 6: Visualisasi
    with tabs[5]:
        st.header("🎨 Visualisasi Hasil Clustering")
        
        df_processed, X_scaled, features, scaler = preprocess_data(data)
        
        # Perform K-Means
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df_processed['Cluster'] = labels
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # 2D Scatter Plot
            fig_2d = px.scatter(
                df_processed,
                x='diabetes_mean',
                y='diabetes_max',
                color='Cluster',
                hover_data=['SpatialDimensionValueCode', 'Country'],
                title='2D Scatter Plot: Mean vs Max Diabetes',
                color_continuous_scale='RdYlBu'
            )
            st.plotly_chart(fig_2d, use_container_width=True)
            
        with col2:
            # Box Plot
            fig_box = px.box(
                df_processed,
                x='Cluster',
                y='diabetes_mean',
                color='Cluster',
                title='Distribusi Diabetes Mean per Cluster',
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # 3D Scatter Plot
        fig_3d = px.scatter_3d(
            df_processed,
            x='diabetes_mean',
            y='diabetes_min',
            z='diabetes_max',
            color='Cluster',
            hover_data=['SpatialDimensionValueCode', 'Country'],
            title='3D Visualization of Clusters',
            color_continuous_scale='RdYlBu'
        )
        
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Cluster Interpretation
        st.subheader("📝 Interpretasi Cluster")
        
        cluster_interpretation = {
            0: {
                "title": "Cluster 0: Negara dengan Prevalensi Diabetes Rendah",
                "characteristics": "Rata-rata prevalensi diabetes di bawah standar global",
                "recommendation": "Pertahankan program pencegahan dan edukasi kesehatan"
            },
            1: {
                "title": "Cluster 1: Negara dengan Prevalensi Diabetes Sedang",
                "characteristics": "Prevalensi diabetes dalam kisaran menengah",
                "recommendation": "Perkuat screening dan program manajemen diabetes"
            },
            2: {
                "title": "Cluster 2: Negara dengan Prevalensi Diabetes Tinggi",
                "characteristics": "Prevalensi diabetes signifikan di atas rata-rata global",
                "recommendation": "Diperlukan intervensi intensif dan kebijakan kesehatan khusus"
            }
        }
        
        for cluster_num, info in cluster_interpretation.items():
            with st.expander(f"**{info['title']}**"):
                st.markdown(f"**Karakteristik**: {info['characteristics']}")
                st.markdown(f"**Rekomendasi**: {info['recommendation']}")
                
                # Show countries in this cluster
                countries_in_cluster = df_processed[df_processed['Cluster'] == cluster_num]['SpatialDimensionValueCode'].tolist()
                st.markdown(f"**Contoh Negara**: {', '.join(countries_in_cluster[:10])}{'...' if len(countries_in_cluster) > 10 else ''}")
    
    # Tab 7: Prediksi
    with tabs[6]:
        st.header("🔮 Prediksi Cluster untuk Data Baru")
        
        st.markdown("### 📤 Upload Data Baru")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload file CSV dengan data diabetes baru:", type=['csv'])
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.success(f"✅ File berhasil diupload! ({len(new_data)} baris)")
                
                # Show preview
                st.dataframe(new_data.head(), use_container_width=True)
                
                # Select clustering method
                method = st.selectbox(
                    "Pilih metode clustering untuk prediksi:",
                    ["K-Means", "GMM", "Hierarchical", "Mean Shift", "Spectral"]
                )
                
                if st.button("🔍 Prediksi Cluster", type="primary"):
                    # Preprocess new data
                    df_processed_original, X_scaled_original, features, scaler = preprocess_data(data)
                    
                    # Prepare new data
                    new_data_processed = new_data.copy()
                    
                    # Apply same preprocessing
                    if 'Country' in new_data_processed.columns:
                        new_data_processed['Country'] = new_data_processed['Country'].fillna('Unknown')
                    
                    # Scale the data using original scaler
                    X_new_scaled = scaler.transform(new_data_processed[features])
                    
                    # Perform prediction based on selected method
                    method_lower = method.lower().replace(' ', '')
                    
                    if method_lower == 'kmeans':
                        model = KMeans(n_clusters=3, random_state=42, n_init=10)
                        model.fit(X_scaled_original)
                        
                    elif method_lower == 'gmm':
                        model = GaussianMixture(n_components=3, random_state=42)
                        model.fit(X_scaled_original)
                        
                    elif method_lower == 'hierarchical':
                        model = AgglomerativeClustering(n_clusters=3)
                        model.fit(X_scaled_original)
                        
                    elif method_lower == 'meanshift':
                        bandwidth = estimate_bandwidth(X_scaled_original, quantile=0.2)
                        model = MeanShift(bandwidth=bandwidth)
                        model.fit(X_scaled_original)
                        
                    elif method_lower == 'spectral':
                        model = SpectralClustering(n_clusters=3, random_state=42)
                        model.fit(X_scaled_original)
                    
                    # Predict clusters
                    predictions = model.fit_predict(X_new_scaled)
                    new_data_processed['Predicted_Cluster'] = predictions
                    
                    st.success("✅ Prediksi berhasil!")
                    
                    # Show results
                    st.subheader("📊 Hasil Prediksi")
                    st.dataframe(new_data_processed, use_container_width=True)
                    
                    # Visualization
                    fig_pred = px.scatter(
                        new_data_processed,
                        x='diabetes_mean' if 'diabetes_mean' in new_data_processed.columns else new_data_processed.columns[1],
                        y='diabetes_max' if 'diabetes_max' in new_data_processed.columns else new_data_processed.columns[2],
                        color='Predicted_Cluster',
                        title=f'Hasil Prediksi menggunakan {method}',
                        color_continuous_scale='RdYlBu'
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Interpretation
                    st.subheader("📝 Interpretasi Hasil")
                    
                    cluster_counts = new_data_processed['Predicted_Cluster'].value_counts().sort_index()
                    
                    for cluster, count in cluster_counts.items():
                        percentage = (count / len(new_data_processed)) * 100
                        st.markdown(f"""
                        **Cluster {cluster}**: {count} data ({percentage:.1f}%)
                        - Negara dalam cluster ini memiliki karakteristik diabetes yang serupa
                        - Cocok untuk intervensi kesehatan yang spesifik
                        """)
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Pastikan file CSV memiliki format yang sama dengan dataset asli!")
        else:
            st.info("👆 Silakan upload file CSV untuk melakukan prediksi")
            
            # Show expected format
            st.markdown("### 📝 Format Data yang Diharapkan")
            expected_format = pd.DataFrame({
                'SpatialDimensionValueCode': ['XXX', 'YYY'],
                'diabetes_mean': [10000, 20000],
                'diabetes_min': [500, 1000],
                'diabetes_max': [50000, 100000],
                'Country': ['Region', 'Region']
            })
            st.dataframe(expected_format, use_container_width=True)
    
    # Tab 8: Contact
    with tabs[7]:
        st.header("📞 Kontak dan Informasi")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 👤 Identitas")
            st.markdown("""
            **Nama**: Nafida Amalia A  
            **No. WhatsApp**: 082136548627  
            **Instagram**: @nfdnsa_  
            **Email**: amalianafida2@gmail.com  
            **Status**: Mahasiswa Sains Data  
            **Universitas**: Universitas Muhammadiyah Semarang
            """)
            
            st.markdown("### 📚 Tentang Saya")
            st.markdown("""
            Saya adalah mahasiswa Sains Data yang tertarik pada bidang Machine Learning, 
            Data Analysis, dan Health Informatics. Project ini merupakan bagian dari 
            pembelajaran dan pengembangan keterampilan dalam analisis data kesehatan.
            """)
            
        with col2:
            # You can add a profile picture or other graphics here
            st.markdown("### 🎓")
            st.image("https://img.icons8.com/color/200/000000/graduation-cap.png", 
                    caption="Mahasiswa Sains Data")
 

if __name__ == "__main__":
    main()