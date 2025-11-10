import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from datetime import datetime

def load_model(version='latest'):
 
    registry_path = '../models/model_registry.json'
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"No existe {registry_path}")
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    if version == 'latest':
        versions = [k for k in registry.keys() if k.startswith('v')]
        if not versions:
            raise ValueError("No hay versiones en registry")
        version = max(versions, key=lambda x: [int(n) for n in x[1:].split('.')])
    
    if version not in registry:
        raise ValueError(f"Versión {version} no encontrada")
    
    model_info = registry[version]
    model_path = model_info['path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    model = joblib.load(model_path)
    print(f"Modelo cargado: {version} - {model_info.get('model', 'Unknown')}")
    return model, model_info

def load_and_preprocess(filepath):
    """
    Aplica todo el preprocesamiento del pipeline de entrenamiento
    """
    print(f"Cargando archivo: {filepath}")
    df = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')  # todas las hojas
    
    # === FILTRO 23:59 (TOTAL DIARIO) ===
    all_dfs = []
    for sheet_name, sheet_df in df.items():
        if 'HORA' not in sheet_df.columns or 'DIA' not in sheet_df.columns:
            continue
        sheet_df['HORA'] = sheet_df['HORA'].astype(str).str.strip()
        sheet_df = sheet_df[sheet_df['HORA'].str.contains('23:59', na=False)]
        if sheet_df.empty:
            continue
        sheet_df['Date'] = pd.to_datetime(sheet_df['DIA']).dt.date
        sheet_df = sheet_df.drop(columns=['HORA', 'DIA'], errors='ignore')
        sheet_df = sheet_df.set_index('Date')
        sheet_df = sheet_df.add_prefix(f"{sheet_name}_")
        all_dfs.append(sheet_df)
    
    if not all_dfs:
        raise ValueError("No se encontraron datos a las 23:59")
    
    df_daily = pd.concat(all_dfs, axis=1)
    df_daily = df_daily[~df_daily.index.duplicated(keep='last')]
    df_daily = df_daily.sort_index()
    
    # limpieza de datos
    errors = ['#VALUE!', '#DIV/0!', '#REF!', '#N/A']
    df_daily = df_daily.replace(errors, np.nan, regex=True)
    for col in df_daily.columns:
        if df_daily[col].dtype == 'object':
            df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
    
    target_col = 'Consolidado EE_Frio (Kw)'
    if target_col in df_daily.columns:
        df_daily = df_daily[df_daily[target_col] < 40000]
    
    missing_pct = df_daily.isna().mean()
    high_missing = missing_pct[missing_pct > 0.8].index
    unnamed = df_daily.columns[df_daily.columns.str.contains('Unnamed')]
    df_daily = df_daily.drop(columns=high_missing.union(unnamed))
    
    df_daily = df_daily.ffill().bfill().fillna(df_daily.median(numeric_only=True))
    
    # feature engineering
    df_daily['day_of_week'] = df_daily.index.dayofweek
    df_daily['month'] = df_daily.index.month
    df_daily['is_weekend'] = df_daily['day_of_week'].isin([5, 6]).astype(int)
    
    df_daily['lag_1'] = df_daily[target_col].shift(1)
    df_daily['lag_7'] = df_daily[target_col].shift(7)
    df_daily['rolling_7'] = df_daily[target_col].rolling(7).mean()
    df_daily['rolling_30'] = df_daily[target_col].rolling(30).mean()
    
    hl_cols = [c for c in df_daily.columns if 'Hl ' in c]
    df_daily['hl_total'] = df_daily[hl_cols].sum(axis=1)
    df_daily['ee_frio_por_hl'] = df_daily[target_col] / (df_daily['hl_total'] + 1e-6)
    
    meta_candidates = [c for c in df_daily.columns if any(k in c.lower() for k in ['meta', 'kpi'])]
    if meta_candidates:
        df_daily['meta_frio_promedio'] = df_daily[meta_candidates].mean(axis=1, skipna=True)
        df_daily['eficiencia_frio'] = df_daily['meta_frio_promedio'] / (df_daily[target_col] + 1e-6)
    else:
        df_daily['eficiencia_frio'] = df_daily[target_col] / (df_daily['hl_total'] + 1e-6)
    
    col_resto = 'Consolidado EE_Resto Serv (Kw)'
    col_mycom7 = next((c for c in df_daily.columns if 'Mycom 7' in c), None)
    col_agua = next((c for c in df_daily.columns if 'agua' in c.lower() and 'cond' in c.lower()), None)
    
    if col_resto in df_daily.columns and col_mycom7:
        df_daily['inter_resto_x_mycom7'] = df_daily[col_resto] * df_daily[col_mycom7]
    if col_agua:
        df_daily['inter_agua_cond_x_frio'] = df_daily[col_agua] * df_daily[target_col]
    
    df_daily = df_daily.dropna()
    
    # selección de variables (mismas que entrenamiento)
    final_features = [
        'Consolidado EE_KW Gral Planta', 'Consolidado EE_Planta (Kw)', 'Consolidado EE_Restos Planta (Kw)',
        'Consolidado EE_Sala Maq (Kw)', 'Consolidado EE_Servicios (Kw)', 'Consolidado KPI_Agua Elab / Hl',
        'Consolidado KPI_Agua Planta / Hl', 'Consolidado KPI_Aire Elaboracion / Hl', 'Consolidado KPI_Aire Planta / Hl',
        'Consolidado KPI_Aire Servicios / Hl', 'Consolidado KPI_CO 2 Linea 4 / Hl', 'Consolidado KPI_CO 2 linea 3 / Hl',
        'Consolidado KPI_EE Aire / Hl', 'Consolidado KPI_EE Bodega / Hl', 'Consolidado KPI_EE CO2 / Hl',
        'Consolidado KPI_EE Eflu / Hl', 'Consolidado KPI_EE Elaboracion / Hl', 'Consolidado KPI_EE Planta / Hl',
        'Consolidado KPI_EE Servicios / Hl', 'Consolidado KPI_ET Bodega/Hl', 'Servicios_total',
        'Totalizadores Energia_KW Bba Glicol Bod', 'Totalizadores Energia_KW Cond 5. 6 y 9',
        'Totalizadores Energia_KW Laboratorio', 'eficiencia_frio', 'inter_agua_cond_x_frio',
        'lag_7', 'rolling_30', 'rolling_7'
    ]
    
    missing_features = [f for f in final_features if f not in df_daily.columns]
    if missing_features:
        print(f"Advertencia: {len(missing_features)} features faltantes → se llenan con 0")
        for f in missing_features:
            df_daily[f] = 0
    
    X = df_daily[final_features].copy()
    
    return X, df_daily.index, ['23:59'] * len(df_daily)

def predict_consumption(filepath):
    """
    Función principal de predicción
    """
    model, _ = load_model()
    X, dates, hours = load_and_preprocess(filepath)
    
    scaler = joblib.load('../models/scaler.pkl')
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    
    results = pd.DataFrame({
        'fecha': dates,
        'hora': hours,
        'prediccion_frio_kw': predictions
    })
    
    output_path = '../results/predicciones.csv'
    os.makedirs('../results', exist_ok=True)
    results.to_csv(output_path, index=False)
    
    print(f"Predicciones generadas: {output_path}")
    print(results.head())
    
    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python src/predict.py <archivo_excel>")
        sys.exit(1)
    
    predict_consumption(sys.argv[1])