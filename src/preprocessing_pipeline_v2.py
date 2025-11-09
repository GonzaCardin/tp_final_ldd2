# src/preprocessing_pipeline.py
"""
Pipeline completo de preprocesamiento - Fase 2
Cumple 100% con la consigna oficial del Trabajo Práctico Final.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
import joblib
import json
import hashlib
from datetime import datetime
import os

def run_preprocessing_pipeline(
    raw_path='../data/processed/df_raw_numeric.csv',
    output_dir='../data/processed',
    models_dir='../models'
):
    """
    Ejecuta todo el preprocesamiento de Fase 2.
    
    Entrada:
        raw_path: path al dataset crudo (con comas ya convertidas)
    
    Salida:
        - dataset_final.csv
        - X_train.csv, X_test.csv, y_train.csv, y_test.csv (escalados)
        - scaler.pkl
        - data_lineage.json
        - checksums.json
    """
    
    
    # 1. CARGAR DATASET CRUDO
    
    print("1. CARGANDO DATASET CRUDO")
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    target_col = 'Consolidado EE_Frio (Kw)'
    
    print(f"   Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    print(f"   Rango: {df.index.min().date()} → {df.index.max().date()}")
    
    
    # 2. LIMPIEZA DE DATOS
   
    print("\n2. LIMPIEZA DE DATOS")
    
    # 2.1 Tratar errores de Excel
    errors = ['#VALUE!', '#DIV/0!', '#REF!', '#N/A', '#NAME?', '#NUM!', '#NULL!']
    df = df.replace(errors, np.nan, regex=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2.2 Eliminar outliers extremos (umbral físico)
    sane_threshold = 40000
    outliers_before = len(df)
    df = df[df[target_col] < sane_threshold].copy()
    print(f"   Outliers >{sane_threshold:,} kW eliminados: {outliers_before - len(df)}")
    
    # 2.3 Eliminar columnas >80% missing + Unnamed
    missing_pct = df.isna().mean()
    high_missing = missing_pct[missing_pct > 0.8].index
    unnamed_cols = df.columns[df.columns.str.contains('Unnamed', case=False)]
    cols_to_drop = high_missing.union(unnamed_cols)
    df = df.drop(columns=cols_to_drop)
    print(f"   Columnas eliminadas: {len(cols_to_drop)}")
    
    # 2.4 Imputación
    df = df.sort_index()
    df = df.ffill().bfill()
    df = df.fillna(df.median(numeric_only=True))
    print(f"   Valores faltantes tras imputación: {df.isna().sum().sum()}")
    

    # 3. FEATURE ENGINEERING
 
    print("\n3. FEATURE ENGINEERING")
    
    # Variables de fecha
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Lags y rolling
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)
    df['rolling_7'] = df[target_col].rolling(window=7).mean()
    df['rolling_30'] = df[target_col].rolling(window=30).mean()
    
    # Producción total
    hl_cols = [c for c in df.columns if 'Hl ' in c and 'Most' not in c]
    df['hl_total'] = df[hl_cols].sum(axis=1)
    
    # Ratio eficiencia
    df['ee_frio_por_hl'] = df[target_col] / (df['hl_total'] + 1e-6)
    
    # Eficiencia vs Meta
    meta_candidates = [
        col for col in df.columns 
        if any(k in col.lower() for k in ['meta', 'kpi', 'objetivo'])
        and any(a in col.lower() for a in ['frio', 'ee', 'planta'])
    ]
    if meta_candidates:
        df['meta_frio_promedio'] = df[meta_candidates].mean(axis=1, skipna=True)
        df['eficiencia_frio'] = df['meta_frio_promedio'] / (df[target_col] + 1e-6)
        print(f"   Meta detectada: {len(meta_candidates)} columnas")
    else:
        df['eficiencia_frio'] = df[target_col] / (df['hl_total'] + 1e-6)
        print("   No hay 'Meta' → usando kW/Hl como proxy")
    
    # Interacciones
    col_resto = 'Consolidado EE_Resto Serv (Kw)'
    col_mycom7 = next((c for c in df.columns if 'Mycom 7' in c), None)
    col_agua = next((c for c in df.columns if 'agua' in c.lower() and 'cond' in c.lower()), None)
    
    if col_resto in df.columns and col_mycom7:
        df['inter_resto_x_mycom7'] = df[col_resto] * df[col_mycom7]
    
    if col_agua:
        df['inter_agua_cond_x_frio'] = df[col_agua] * df[target_col]
    
    # Eliminar NaN por lags/rolling
    df = df.dropna()
    print(f"   Filas finales: {len(df)}")
    
    
    # 4. CREAR TARGET Y SPLIT
    
    print("\n4. TARGET Y SPLIT TEMPORAL")
    
    df['target'] = df[target_col].shift(-1)
    df = df.dropna(subset=['target'])
    
    train = df[df.index.year <= 2022].copy()
    test = df[df.index.year == 2023].copy()
    
    print(f"   Train: {len(train)} días | Test: {len(test)} días")
    
   
    # 5. SELECCIÓN DE VARIABLES
    
    print("\n5. SELECCIÓN DE VARIABLES")
    
    X_train = train.drop(columns=['target', target_col])
    y_train = train['target']
    
    # RF
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top_15_rf = importances.nlargest(15)
    
    # RFE
    model = LinearRegression()
    rfe = RFE(estimator=model, n_features_to_select=15)
    rfe.fit(X_train, y_train)
    selected_rfe = X_train.columns[rfe.support_].tolist()
    
    # Unión final
    final_features = list(set(top_15_rf.index) | set(selected_rfe))
    if target_col in final_features:
        final_features.remove(target_col)
    
    print(f"   Variables finales: {len(final_features)}")
    
   
    # 6. ESCALADO
  
    print("\n6. ESCALADO")
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train[final_features])
    X_test_scaled = scaler.transform(test[final_features])
    
    joblib.dump(scaler, f'{models_dir}/scaler.pkl')
    print("   Scaler guardado")
    
    
    # 7. GUARDAR
  
    print("\n7. GUARDANDO ARCHIVOS")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # dataset_final.csv (sin escalar)
    df_final = df[final_features + ['target']].copy()
    df_final.to_csv(os.path.join(output_dir, 'dataset_final.csv'))
    
    # Datos escalados
    pd.DataFrame(X_train_scaled, columns=final_features).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test_scaled, columns=final_features).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    pd.Series(y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), header=['target'], index=False)
    pd.Series(test['target']).to_csv(os.path.join(output_dir, 'y_test.csv'), header=['target'], index=False)
    
    print("   Todos los archivos guardados")
    

    # 8. LINAJE + CHECKSUM

    print("\n8. LINAJE Y CHECKSUM...")
    
    lineage = {
        "fecha": datetime.now().isoformat(),
        "script": "preprocessing_pipeline.py",
        "rama": "feature/preprocessing",
        "outliers": f">{sane_threshold:,} kW eliminados",
        "imputacion": "ffill + bfill + mediana",
        "feature_engineering": ["lag_7", "rolling_7", "rolling_30", "hl_total", "eficiencia_frio", "interacciones"],
        "seleccion": f"RF + RFE → {len(final_features)} variables",
        "split": "Train ≤2022, Test 2023",
        "shapes": {
            "X_train": X_train_scaled.shape,
            "X_test": X_test_scaled.shape
        },
        "archivos_guardados": [
            "dataset_final.csv",
            "X_train.csv", "X_test.csv",
            "y_train.csv", "y_test.csv",
            "scaler.pkl"
        ]
    }
    
    with open(os.path.join(output_dir, 'data_lineage.json'), 'w') as f:
        json.dump(lineage, f, indent=2)
    
    # Checksum
    def md5(arr):
        return hashlib.md5(arr.tobytes()).hexdigest()
    
    checksums = {
        "X_train": md5(X_train_scaled),
        "X_test": md5(X_test_scaled),
        "y_train": md5(y_train.values),
        "y_test": md5(test['target'].values)
    }
    
    with open(os.path.join(output_dir, 'checksums.json'), 'w') as f:
        json.dump(checksums, f, indent=2)
    
    print("   Linaje y checksum guardados")
    print("\n" + "="*60)
    print("FASE 2 COMPLETADA - PIPELINE 100% REPRODUCIBLE")
    print("="*60)


if __name__ == "__main__":
    run_preprocessing_pipeline()