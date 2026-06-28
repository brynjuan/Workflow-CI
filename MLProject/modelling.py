import os
import shutil
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Mengabaikan pesan warning agar log terminal tetap bersih
warnings.filterwarnings("ignore")

def eval_metrics(actual, pred):
    """Fungsi untuk menghitung metrik evaluasi model."""
    acc = accuracy_score(actual, pred)
    prec = precision_score(actual, pred)
    rec = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return acc, prec, rec, f1

if __name__ == "__main__":
    print("=== MEMULAI PROSES TRAINING ===")
    
    # 1. Load Dataset
    # Path disesuaikan karena folder dataset_preprocessing berada di dalam folder MLProject
    data_path = "dataset_preprocessing/bank_cleaned.csv"
    
    try:
        df = pd.read_csv(data_path)
        print(f"-> Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom.")
    except Exception as e:
        print(f"Error memuat data: {e}")
        print(f"Pastikan file berada di path: {data_path}")
        exit(1)

    # 2. Pisahkan Fitur (X) dan Target (y)
    X = df.drop('deposit', axis=1)
    y = df['deposit']

    # 3. Split Data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Inisiasi MLflow Run    
    with mlflow.start_run():
        # Hyperparameter Model
        n_estimators = 100
        max_depth = 10
        
        # 5. Latih Model
        print("-> Melatih model Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)

        # 6. Prediksi dan Evaluasi
        y_pred = model.predict(X_test)
        acc, prec, rec, f1 = eval_metrics(y_test, y_pred)
        
        print("\n=== HASIL EVALUASI ===")
        print(f"Akurasi   : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}\n")

        # 7. Logging ke MLflow Server/DagsHub
        print("-> Logging parameter dan metrik ke MLflow...")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model secara remote sebagai artefak MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # 8. Simpan Model Secara Lokal (Wajib untuk Target Advance Kriteria 3)
        # Menghapus folder lama jika sudah ada agar tidak error saat menimpa file
        local_model_path = "model_output"
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        
        # Menyimpan model di root directory untuk dieksekusi oleh mlflow build-docker
        mlflow.sklearn.save_model(model, local_model_path)
        print(f"=== TRAINING SELESAI. Model lokal disimpan di folder '{local_model_path}' ===")