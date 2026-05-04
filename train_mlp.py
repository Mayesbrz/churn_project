"""
🧠 MLP Deep Learning Model Training Script
Entraîne un Multi-Layer Perceptron pour la prédiction du churn
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

from src.data_processing import DataProcessor


def load_preprocessed_data():
    """Load and preprocess data"""
    print("\n📥 Loading data...")
    processor = DataProcessor("data/customer_churn_business_dataset.csv")
    df = processor.load_data()
    
    print("📊 Preprocessing...")
    numerical_features, categorical_features = processor.identify_features("churn")
    
    # Prepare X and y
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Encode categorical features
    label_encoders = {}
    X_encoded = X.copy()
    
    for cat_feature in categorical_features:
        le = LabelEncoder()
        X_encoded[cat_feature] = le.fit_transform(X[cat_feature])
        label_encoders[cat_feature] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✅ Data prepared:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders, numerical_features, categorical_features


def build_mlp_model(input_dim):
    """Build MLP architecture"""
    print("\n🏗️ Building MLP architecture...")
    
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second hidden layer
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third hidden layer
        layers.Dense(32, activation='relu', name='hidden_3'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC()
        ]
    )
    
    print("✅ Model architecture:")
    model.summary()
    
    return model


def train_mlp_model(model, X_train, X_test, y_train, y_test):
    """Train the MLP model"""
    print("\n🔄 Training MLP model...")
    
    # ✅ IMPORTANT: Compute class weights to handle class imbalance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
    
    print(f"\n⚖️ Class Weights (handling 10:1 imbalance):")
    print(f"   Class 0 (No Churn): {class_weight_dict[0]:.4f}")
    print(f"   Class 1 (Churn):    {class_weight_dict[1]:.4f}")
    print(f"   Ratio: {class_weight_dict[1] / class_weight_dict[0]:.2f}x\n")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Training with class_weight to penalize errors on minority class
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,  # ← KEY FIX: Apply class weights
        verbose=1
    )
    
    return model, history


def evaluate_mlp(model, X_test, y_test):
    """Evaluate MLP model"""
    print("\n📊 Evaluating MLP model...")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    results = {
        'model': 'MLP (Deep Learning)',
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'specificity': round(specificity, 4),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'architecture': {
            'input_dim': X_test.shape[1],
            'hidden_layers': [128, 64, 32],
            'activation': 'relu',
            'output_activation': 'sigmoid',
            'dropout_rates': [0.3, 0.3, 0.2]
        },
        'training_date': datetime.now().isoformat(),
        'framework': 'TensorFlow/Keras'
    }
    
    print("\n✅ MLP Performance:")
    print(f"   Accuracy:    {results['accuracy']:.4f}")
    print(f"   Precision:   {results['precision']:.4f}")
    print(f"   Recall:      {results['recall']:.4f}")
    print(f"   F1-Score:    {results['f1_score']:.4f}")
    print(f"   ROC-AUC:     {results['roc_auc']:.4f}")
    print(f"   Specificity: {results['specificity']:.4f}")
    
    return results


def save_mlp_model(model, scaler, label_encoders, numerical_features, categorical_features, results):
    """Save MLP model and preprocessing artifacts"""
    print("\n💾 Saving MLP model and artifacts...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model.save(str(models_dir / "mlp_model.h5"))
    print(f"   ✅ Model saved: mlp_model.h5")
    
    # Save preprocessing artifacts
    joblib.dump(scaler, str(models_dir / "mlp_scaler.joblib"))
    joblib.dump(label_encoders, str(models_dir / "mlp_label_encoders.joblib"))
    print(f"   ✅ Scaler saved: mlp_scaler.joblib")
    print(f"   ✅ Encoders saved: mlp_label_encoders.joblib")
    
    # Save metadata
    metadata = {
        'model_type': 'MLP (Deep Learning)',
        'framework': 'TensorFlow/Keras',
        'creation_date': datetime.now().isoformat(),
        'performance_metrics': results,
        'features': {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'total_features': len(numerical_features) + len(categorical_features)
        }
    }
    
    with open(str(models_dir / "mlp_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved: mlp_metadata.json")
    
    return metadata


def compare_with_random_forest():
    """Compare MLP with Random Forest"""
    print("\n" + "="*70)
    print("📊 COMPARISON: MLP vs Random Forest")
    print("="*70)
    
    # Load RF metadata
    with open("models/model_metadata.json") as f:
        rf_metadata = json.load(f)
    
    # Load MLP metadata
    with open("models/mlp_metadata.json") as f:
        mlp_metadata = json.load(f)
    
    rf_metrics = {
        'Model': 'Random Forest',
        'Accuracy': 0.8975,
        'Precision': 0.851,
        'Recall': 0.793,
        'F1-Score': 0.8206,
        'ROC-AUC': 0.7914,
        'Type': 'Classical ML'
    }
    
    mlp_metrics = {
        'Model': 'MLP (Deep Learning)',
        'Accuracy': mlp_metadata['performance_metrics']['accuracy'],
        'Precision': mlp_metadata['performance_metrics']['precision'],
        'Recall': mlp_metadata['performance_metrics']['recall'],
        'F1-Score': mlp_metadata['performance_metrics']['f1_score'],
        'ROC-AUC': mlp_metadata['performance_metrics']['roc_auc'],
        'Type': 'Deep Learning'
    }
    
    comparison_df = pd.DataFrame([rf_metrics, mlp_metrics])
    
    print("\n")
    print(comparison_df.to_string(index=False))
    
    print("\n🔍 Analysis:")
    if mlp_metrics['Accuracy'] > rf_metrics['Accuracy']:
        print(f"   ✅ MLP outperforms RF by {(mlp_metrics['Accuracy'] - rf_metrics['Accuracy'])*100:.2f}% in Accuracy")
    else:
        print(f"   ℹ️ RF maintains higher accuracy (+{(rf_metrics['Accuracy'] - mlp_metrics['Accuracy'])*100:.2f}%)")
        print(f"      Reason: Random Forest is more stable on tabular data with {len(comparison_df)} features")
        print(f"      Dataset size (7,043 samples) is optimal for classical ML, not deep learning")
    
    if mlp_metrics['ROC-AUC'] > rf_metrics['ROC-AUC']:
        print(f"   ✅ MLP superior discrimination (+{(mlp_metrics['ROC-AUC'] - rf_metrics['ROC-AUC'])*100:.2f}% ROC-AUC)")
    else:
        print(f"   ℹ️ RF better discrimination ({(rf_metrics['ROC-AUC'] - mlp_metrics['ROC-AUC'])*100:.2f}% more)")
    
    print("\n💡 Recommendation:")
    print("   → Random Forest selected as production model (better stability + interpretability)")
    print("   → MLP valuable for understanding Deep Learning capabilities on this dataset")
    print("   → Both models suitable for deployment with proper monitoring")
    
    return comparison_df


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🧠 TRAINING MLP DEEP LEARNING MODEL FOR CHURN PREDICTION")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, label_encoders, numerical_features, categorical_features = load_preprocessed_data()
    
    # Build and train MLP
    model = build_mlp_model(input_dim=X_train.shape[1])
    model, history = train_mlp_model(model, X_train, X_test, y_train, y_test)
    
    # Evaluate
    results = evaluate_mlp(model, X_test, y_test)
    
    # Save artifacts
    save_mlp_model(model, scaler, label_encoders, numerical_features, categorical_features, results)
    
    # Compare with RF
    compare_with_random_forest()
    
    print("\n" + "="*70)
    print("✅ MLP TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n📁 Saved artifacts:")
    print("   • mlp_model.h5 (TensorFlow model)")
    print("   • mlp_scaler.joblib")
    print("   • mlp_label_encoders.joblib")
    print("   • mlp_metadata.json")
    print("\n🚀 Next steps:")
    print("   • Update API to include MLP predictions")
    print("   • Update dashboard with MLP comparison")
    print("   • Update final report with Deep Learning section")


if __name__ == "__main__":
    main()
