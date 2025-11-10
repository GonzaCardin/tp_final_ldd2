# Predicción de Consumo Energético Industrial - Planta de Cerveza

*Trabajo Práctico Final - LDD2*  
*Autor:* Cardin Gonzalo y Palkovic Micaela
*Fecha:* 10 de noviembre de 2025  

---

## Objetivo General

Desarrollar un *pipeline de machine learning completo y reproducible* para predecir el **consumo energético del sistema de refrigeración Frio (kW) del día siguiente** en una planta cervecera mexicana.

Se implementan buenas prácticas de MLOps para garantizar trazabilidad, colaboración y gestión del ciclo de vida del modelo.

---

## Dataset

Se proporcionan archivos Excel con registros horarios de métricas operacionales de la planta desde *2021 hasta 2024*:

- Totalizadores Planta de Cerveza 2020_2022.xlsx
- Totalizadores Planta de Cerveza - 2021_2023.xlsx
- Totalizadores Planta de Cerveza 2022_2023.xlsx
- Totalizadores Planta de Cerveza 2024_2025.xlsx → *Test*

> *Variable Objetivo (Target):* Frio (kW)  
> *Predicción:* Consumo del día siguiente (D+1)

---

## Naturaleza de los Datos

> *IMPORTANTE:* Los datos son totalizadores acumulados horarios.

- Cada valor es acumulativo desde las 00:00
- El valor de 23:59 = TOTAL del día
- Para obtener consumo diario: filtrar 23:59
- Para predecir D+1: **usar shift(-1)**

---

## Estructura del Proyecto

```bash
.
├── data/
│   ├── raw/                     ← archivos Excel
│   └── processed/               ← datos limpios y versionados
├── models/                      ← modelos, scaler, registry
├── notebooks/                   ← EDA + preprocesamiento + modelado
├── src/
│   ├── preprocessing_pipeline.py
│   ├── train_model.py
│   └── predict.py               ← Fase cuatro
├── results/
│   ├── experiment_logs.csv
│   └── predicciones.csv
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md