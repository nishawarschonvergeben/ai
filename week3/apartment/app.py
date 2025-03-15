import gradio as gr
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import pickle

# Laden des gespeicherten Modells
model_filename = "random_forest_regression.pkl"

try:
    with open(model_filename, 'rb') as f:
        random_forest_model = pickle.load(f)
except FileNotFoundError:
    random_forest_model = RandomForestRegressor()

# Laden der Daten
df_bfs_data = pd.read_csv('bfs_municipality_and_tax_data.csv', sep=',', encoding='utf-8')
df_bfs_data['tax_income'] = df_bfs_data['tax_income'].astype(str).str.replace("'", "").astype(float)

df_water_distance = pd.read_csv('bfs_with_water_distance.csv', sep=',', encoding='utf-8')
df_bfs_data = df_bfs_data.merge(df_water_distance[['bfs_name', 'nearest_water_distance']], on='bfs_name', how='left')

# Fehlende Werte ersetzen
df_bfs_data['nearest_water_distance'].fillna(df_bfs_data['nearest_water_distance'].median(), inplace=True)

# Features für das Modell
features = ['pop', 'pop_dens', 'frg_pct', 'emp', 'nearest_water_distance']
df_bfs_data = df_bfs_data[features + ['tax_income']].dropna()

X = df_bfs_data[features]
y = df_bfs_data['tax_income']

# Falls das Modell nicht existiert, neu trainieren
if random_forest_model.n_features_in_ != len(features):
    random_forest_model = RandomForestRegressor()
    random_forest_model.fit(X, y)

    with open("random_forest_regression.pkl", "wb") as f:
        pickle.dump(random_forest_model, f)

# Vorhersagefunktion für Gradio
def predict_tax_income(pop, pop_dens, frg_pct, emp, nearest_water_distance):
    input_data = pd.DataFrame({
        'pop': [pop],
        'pop_dens': [pop_dens],
        'frg_pct': [frg_pct],
        'emp': [emp],
        'nearest_water_distance': [nearest_water_distance]
    })
    prediction = random_forest_model.predict(input_data)
    return np.round(prediction[0], 2)

# Gradio Interface
iface = gr.Interface(
    fn=predict_tax_income,
    inputs=[
        gr.Number(label="Bevölkerung"),
        gr.Number(label="Bevölkerungsdichte"),
        gr.Number(label="Anteil Ausländer (%)"),
        gr.Number(label="Anzahl Beschäftigte"),
        gr.Number(label="Entfernung zum nächsten Gewässer (m)")
    ],
    outputs=gr.Textbox(label="Vorhersage Steueraufkommen"),
    examples=[
        [10000, 1500, 20, 5000, 300],
        [20000, 2500, 35, 7000, 1000]
    ]
)

if __name__ == "__main__":
    iface.launch()