import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.preprocessing import StandardScaler

target_col='Frio (Kw)'
scaler_path='../models/scaler.joblib'

# Variables globales

# Viarables ORO
FEATURES_PROD = [
    'Hl de Mosto_Produccion', 'Hl Cerveza Cocina_Produccion', 
    'Hl Producido Bodega_Produccion', 'Hl Cerveza Filtrada_Produccion',
    'Hl Cerveza Envasada_Produccion', 'Hl Cerveza L2_Produccion',
    'Hl Cerveza L3_Produccion', 'Hl Cerveza L4_Produccion',
    'Hl Cerveza L5_Produccion', 'Cocimientos Diarios_Produccion',
    'Hl de Mosto Copia_Produccion'
]
FEATURES_GAS = [
    'Conversion Kg/Mj_GasVapor', 'Gas Planta (Mj)_GasVapor', 
    'Vapor Elaboracion (Kg)_GasVapor', 'Vapor Cocina (Kg)_GasVapor',
    'Vapor Envasado (Kg)_GasVapor', 'Vapor Servicio (Kg)_GasVapor',
    'ET Elaboracion (Mj)_GasVapor', 'ET Envasado (Mj)_GasVapor',
    'ET Servicios (Mj)_GasVapor', 'Tot_Vapor_L3_L4_GasVapor',
    'VAPOR DE LINEA 1 Y 2 KG_GasVapor', 'VAPOR DE LINEA 4 KG_GasVapor',
    'Vapor_L5 (KG)_GasVapor', 'Tot_Vapor_CIP_Bodega_GasVapor',
    'Vapor L3_GasVapor', 'Tot Vap Paste L3 / Hora_GasVapor',
    'Tot Vap Lav L3 / Hora_GasVapor', 'Medicion Gas Planta (M3)_GasVapor'
]

# Variables TRAMPA
FEATURES_TRAMPA_EE = [
    'Planta (Kw)', 'Elaboracion (Kw)', 'Bodega (Kw)', 'Cocina (Kw)', 
    'Linea 2 (Kw)', 'Linea 3 (Kw)', 'Linea 4 (Kw)', 'Linea 5 (Kw)', 
    'Servicios (Kw)', 'Sala Maq (Kw)', 'Aire (Kw)', 'Efluentes (Kw)', 
    'CO2 (Kw)', 'EE Planta / Hl', 'EE Elaboracion / Hl', 'EE Bodega / Hl', 
    'EE Cocina / Hl', 'EE L2 / Hl', 'EE L3 / Hl', 'EE L4 / Hl', 'EE L5 / Hl', 
    'EE Servicios / Hl', 'EE Sala Maq / Hl', 'EE Aire / Hl', 
    'EE Efluentes / Hl', 'EE CO2 / Hl', 'EE Frio / Hl'
]

# Top 15 features finales seleccionadas por el RF
FINAL_FEATURES = [
    'Frio_Kw_roll_7',
    'Frio_Kw_lag_1',
    'Frio_Kw_lag_7',
    'dia_del_anio',
    'Conversion Kg/Mj_GasVapor',
    'Hl Cerveza Filtrada_Produccion',
    'ET Servicios (Mj)_GasVapor',
    'VAPOR DE LINEA 1 Y 2 KG_GasVapor',
    'VAPOR DE LINEA 4 KG_GasVapor',
    'Hl Cerveza L4_Produccion',
    'Vapor Envasado (Kg)_GasVapor',
    'Vapor_L5 (KG)_GasVapor',
    'Tot Vap Paste L3 / Hora_GasVapor',
    'Hl Cerveza L2_Produccion',
    'dia_semana'
]

# Funciones del pipeline

def load_and_unify_raw_data(filepath):
    sheets_to_load = {
        'Consolidado EE': '',
        'Consolidado Produccion': '_Produccion',
        'Consolidado Gas Vapor': '_GasVapor',
        'Consolidado Agua': '_Agua'
    }
    
    df_hourly_merge = pd.DataFrame()
    timestamp_col_name = None
    for sheet_name, suffix in sheets_to_load.items():
        try:
            df_sheet = pd.read_excel(filepath, sheet_name=sheet_name)
            if timestamp_col_name is None:
                timestamp_col_name = df_sheet.columns[0]
            df_sheet = df_sheet.rename(columns={timestamp_col_name: 'Timestamp'})
            
            df_sheet['Timestamp'] = pd.to_datetime(df_sheet['Timestamp'], errors='coerce')
            df_sheet = df_sheet.dropna(subset=['Timestamp'])
            df_sheet = df_sheet.sort_values(by='Timestamp').drop_duplicates(subset=['Timestamp'], keep='last')
            
            if sheet_name == 'Consolidado EE':
                df_hourly_merged = df_sheet
            else:
                merge_cols = {col: f"{col}{suffix}" for col in df_sheet.columns if col != 'Timestamp'}
                df_to_merge = df_sheet.rename(columns=merge_cols)
                
                df_hourly_merged = pd.merge(df_hourly_merged, df_to_merge, on='Timestamp', how='outer')
                
        except Exception as e:
            print(f"Error loading sheet {sheet_name}: {e}")
    return df_hourly_merged.sort_values(by='Timestamp')

def aggregate_to_daily(df_hourly):
    if 'Timestamp' not in df_hourly.columns or df_hourly.empty:
        return pd.DataFrame()
    df_hourly['fecha_dia'] = df_hourly['Timestamp'].dt.date
    idx_last_hour = df_hourly.groupby('fecha_dia')['Timestamp'].idxmax()
    df_daily = df_hourly.loc[idx_last_hour].copy()
    
    df_daily = df_daily.set_index(pd.to_datetime(df_daily['fecha_dia']))
    df_daily = df_daily.sort_index()
    
    cols_to_drop = [col for col in df_daily.columns if 'Timestamp' in str(col) or 'fecha_dia' in str(col)]
    df_daily = df_daily.drop(columns=cols_to_drop)
    return df_daily



def apply_cleaning(df_daily, target_col, threshold=40000):
    """
    Limpieza total de los datos:
    1. Convierte todo a numérico (maneja errores '#VALUE!').
    2. Elimina outliers extremos (basado en el target).
    3. Elimina columnas "Trampa" (EE y Agua) y "Basura" (NaN > 50%).
    4. Imputa los NaN restantes con Cero (0).
    """
    for col in df_daily.columns:
        df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')
    
    df_cleaned = df_daily[df_daily[target_col] <= threshold].copy()
    
    percent_missing = df_cleaned.isnull().mean()
    cols_basura = percent_missing[percent_missing > 0.5].index.tolist()
    cols_trampa_agua = [col for col in df_cleaned.columns if 'Agua' in col]
    
    cols_trampa_ee_final = [col for col in FEATURES_TRAMPA_EE if col in df_cleaned.columns]
    
    all_cols_to_drop = list(set(cols_basura + cols_trampa_agua + cols_trampa_ee_final))
    df_cleaned = df_cleaned.drop(columns=all_cols_to_drop)

    df_cleaned.fillna(0, inplace=True)
    return df_cleaned

def create_features(df_cleaned, target_col):
    """
    Feature Engineering:
    1. Crea las variables de tiempo.
    2. Crea las variables de Lag y Rolling.
    3. Crea las variables de interacción.
    4. Limpia los NaNs generados por los lags.
    """
    df_fe = df_cleaned.copy()
    
    df_fe['mes'] = df_fe.index.month
    df_fe['dia_semana'] = df_fe.index.dayofweek
    df_fe['dia_del_anio'] = df_fe.index.dayofyear
    df_fe['es_fin_de_semana'] = df_fe['dia_semana'].isin([5, 6]).astype(int)
    
    df_fe['Frio_Kw_lag_1'] = df_fe[target_col].shift(1)
    df_fe['Frio_Kw_lag_7'] = df_fe[target_col].shift(7)
    df_fe['Frio_Kw_roll_7'] = df_fe[target_col].shift(1).rolling(window=7).mean()
    
    gas_col = 'Medicion Gas Planta (M3)_GasVapor'
    mosto_col = 'Hl de Mosto Copia_Produccion'
    if gas_col in df_fe.columns and mosto_col in df_fe.columns:
        df_fe['Gas_x_Mosto'] = df_fe[gas_col] * df_fe[mosto_col]
    
    df_fe.dropna(inplace=True)
    return df_fe

def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

def run_preprocessing_pipeline(filepath,scaler_path, is_training=False):
    """
    Función principal que ejecuta todo el pipeline de preprocesamiento
    sobre un archivo Excel crudo.
    
    Si 'is_training' es True, crea y guarda el target.
    Si 'is_training' es False (predicción), no crea el target.
    """
    
    print('Inicio de la pipeline de preprocesamiento ')
    
    # 1. Cargar y unifica datos
    df_hourly = load_and_unify_raw_data(filepath)
    if df_hourly.empty:
        print("No se cargaron datos del archivo.")
        return None, None
    print('Datos cargados y unificados correctamente.')
    
    # 2. Agregar a nivel diario
    df_daily = aggregate_to_daily(df_hourly)
    if df_daily.empty:
        print("No se pudieron agregar los datos a nivel diario.")
        return None, None
    print('Datos agregados a nivel diario correctamente.')
    
    # 3. Limpieza de datos
    df_cleaned = apply_cleaning(df_daily, target_col)
    print('Datos limpiados correctamente.')
    
    # 4. Creación de características
    df_fe = create_features(df_cleaned, target_col)
    print('Características creadas correctamente.')
    
    # 5. Creación de Target (y) y limpieza del mismo
    target_col_name = 'target_Frio_Kw_next_day'
    if is_training:
        df_fe[target_col_name] = df_fe['Frio (Kw)'].shift(-1)
        df_fe.dropna(subset=[target_col_name], inplace=True) # Limpia el NaN del shift
        y = df_fe[target_col_name]
    else:
        y = None
    
    if 'Frio (Kw)' in df_fe.columns:
        df_fe = df_fe.drop(columns=['Frio (Kw)'])
    
    # 6. Selección de características finales
    X = df_fe.reindex(columns=FINAL_FEATURES, fill_value=0)
    
    # 7. Escalado de características
    scaler = load_scaler(scaler_path)
    if scaler:
        X_scaled_array = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
        print("Datos escalados listos.")
        return X_scaled, y
    else:
        print("No se pudo cargar el scaler. Abortando.")
        return None, None


if __name__ == "__main__":
    print("Pipeline de preprocesamiento - Ejecución directa")
    DATA_PATH = '../data'
    excel_files = sorted(glob.glob(os.path.join(DATA_PATH, 'Totalizadores*.xlsx')))
    
    # Unificación de todos los archivos
    df_hourly_list = [load_and_unify_raw_data(f) for f in excel_files]
    df_hourly_full = pd.concat(df_hourly_list)
    df_hourly_full = df_hourly_full.sort_values(by='Timestamp').drop_duplicates(subset=['Timestamp'], keep='last')
    
    # Agregación a nivel diario
    df_daily_full = aggregate_to_daily(df_hourly_full)
    
    # Limpieza de datos
    df_cleaned = apply_cleaning(df_daily_full)
    
    # Creación de características
    df_fe = create_features(df_cleaned)
    
    # Creación del Target
    target_col_name_next = 'target_Frio_Kw_next_day'
    df_fe[target_col_name_next] = df_fe['Frio (Kw)'].shift(-1)
    df_fe.dropna(subset=[target_col_name_next], inplace=True)
    
    if 'Frio (Kw)' in df_fe.columns:
        df_fe = df_fe.drop(columns=['Frio (Kw)'])
    
    # Selección de características finales
    X = df_fe.reindex(columns=FINAL_FEATURES, fill_value=0)
    y = df_fe[target_col_name_next]
    
    # Escalado de características
    train_size = int(len(X) * 0.70)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    
    # Guardado de artegactos
    MODELS_PATH = '../models'
    os.makedirs(MODELS_PATH, exist_ok=True)
    scaler_path = os.path.join(MODELS_PATH, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler guardado en: {scaler_path}")
    
    # Datos Finales
    train_data = X_train_scaled_df.join(y_train.rename('target'))
    test_data = X_test_scaled_df.join(y_test.rename('target'))
    dataset_final = pd.concat([train_data, test_data])
    
    PROCESSED_PATH = '../data/processed'
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    final_csv_path = os.path.join(PROCESSED_PATH, 'dataset_final.csv')
    dataset_final.to_csv(final_csv_path, index=True)
    print(f"Dataset final procesado guardado en: {final_csv_path}")
    
    print('Preprocesamiento completado.')