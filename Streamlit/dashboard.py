from datetime import datetime
###################################################################################################
#Per Eseguire il codice con facilità bisogna installare mediante pip install i seguenti pacchetti:
#matplotlib
#seaborn
#sklearn
#scikit-learn
#openpyxl
###################################################################################################
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Funzione per caricare i dati
@st.cache_resource
def load_data():
    data = pd.read_excel("temp_humid_data.xlsx", sheet_name='Sheet3')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Funzione per addestrare il modello di regressione
def train_model(data):
    X = data[['temperature_mean', 'relativehumidity_mean']]
    y = data['no. of Adult males']
    model = LinearRegression()
    model.fit(X, y)
    return model

def main():
    st.title("Analisi e Predizione: Temperatura, Umidità e Adulti Maschi")
    # Aggiungi un sottotitolo
    st.subheader("Esplora i dati e predici il numero di adulti maschi in base a temperatura e umidità per Sheet3 - temp_humid_data")
    st.image("intestazione.jpg", use_column_width=True)

    data = load_data()
    model = train_model(data)

    st.subheader("Osservazione giornaliera")
    data2 = load_data()
    data2['Date'] = pd.to_datetime(data['Date'])
    date_time = st.slider(
        "Selezionare il giorno per valutare le variazioni degli attributi",
        value=(datetime(2023, 6, 15)),
        format = "YYYY-MM-DD",
        min_value=(data2['Date'].min()),
        max_value=(data2['Date'].max())
    )

    # Filter data for the selected date
    fil_data = data[data2['Date'] == date_time]

    col1, col2, col3 = st.columns(3)

    if not fil_data.empty:
        temperature_mean = data['temperature_mean'].mean()
        humidity_mean = data['relativehumidity_mean'].mean()
        adult_males_mean = data['no. of Adult males'].mean()

        # Calculate the variations
        temperature_variation = fil_data['temperature_mean'].values[0] - temperature_mean
        humidity_variation = fil_data['relativehumidity_mean'].values[0] - humidity_mean
        adult_males_variation = fil_data['no. of Adult males'].values[0] - adult_males_mean

        col1.metric("Temperature", f"{fil_data['temperature_mean'].values[0]} °C",
                    f"{temperature_variation:.2f} °C")
        col2.metric("Humidity", f"{fil_data['relativehumidity_mean'].values[0]}%",
                    f"{humidity_variation:.2f}%")
        col3.metric("No. of Adult Males", f"{fil_data['no. of Adult males'].values[0]}",
                    f"{adult_males_variation:.2f}")
    else:
        st.warning("Nessun dato disponibile per la data selezionata.")

    # Sidebar per filtri
    st.sidebar.subheader("Filtri Visualizzazione")
    start_date = st.sidebar.date_input("Data Iniziale", data['Date'].min().date())
    end_date = st.sidebar.date_input("Data Finale", data['Date'].max().date())
    # Filtra i dati in base all'intervallo di date selezionato
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    # Visualizzazione line chart
    st.subheader("Visualizzazione dati")
    st.write("Questo grafico mostra l'andamento dei dati presenti nello Sheet3 nel periodo selezionato.")
    # Widget per selezionare le serie temporali da visualizzare
    selected_data = st.multiselect('Selezionare gli attributi da visualizzare', filtered_data.columns[1:], default=filtered_data.columns[1:].tolist())
    # Filtrare il DataFrame in base alla selezione dell'utente
    filtered_data_selected = filtered_data[['Date'] + selected_data]
    # Creazione del grafico con line_chart di Streamlit
    st.line_chart(filtered_data_selected.set_index('Date'))


    st.subheader("Trend del Numero di Adulti Maschi")
    st.write("Grafico di dispersione 3D degli attributi")


    # Creazione di un grafico di dispersione 3D
    scatter_3d = go.Figure()
    scatter_3d.add_trace(go.Scatter3d(
        x=filtered_data['temperature_mean'],
        y=filtered_data['relativehumidity_mean'],
        z=filtered_data['no. of Adult males'],
        mode='markers',
        marker=dict(size=10, color=filtered_data['no. of Adult males'], opacity=0.9),
        text=filtered_data['no. of Adult males'],
    ))
    scatter_3d.update_layout(scene=dict(
        xaxis_title='Temperature Mean',
        yaxis_title='Relative Humidity Mean',
        zaxis_title='Number of Adult Males',
    ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    st.plotly_chart(scatter_3d)


    # Creazione dei dati per il grafico a barre
    plot_data = filtered_data.iloc[::1, :]  # Mostra solo ogni quinta data
    bar_chart_data = plot_data[['Date', 'no. of Adult males']].set_index('Date')
    filtered_plot_data = plot_data[plot_data['no. of Adult males'] > 0]
    st.write("Focus sui giorni con presenza di Adult Males")
    # Visualizza solo i dati filtrati
    st.write(filtered_plot_data)
    # Creazione del grafico a barre con st.bar_chart di Streamlit
    st.write("Visualizzazione del numero di adulti maschi nel periodo selezionato.")
    st.bar_chart(bar_chart_data, use_container_width=True)



    # Statistiche descrittive
    st.subheader("Statistiche Descrittive")
    st.write("Statistiche descrittive delle variabili nel dataset.")
    st.write(filtered_data.describe())

    # Mappa di calore per le correlazioni
    st.subheader("Mappa di Calore delle Correlazioni")
    st.write("Questa mappa di calore mostra le correlazioni tra temperatura, umidità e numero di adulti maschi.")
    corr = filtered_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    # Sezione di previsione
    st.sidebar.subheader("Predizione di Adulti Maschi")
    temp_input = st.sidebar.slider("Inserisci la Temperatura Media", min_value=10.0, max_value=50.0)
    humidity_input = st.sidebar.slider("Inserisci l'Umidità Relativa Media", min_value=20.0, max_value=90.0)

    if st.sidebar.button("Prevedi"):
        prediction = model.predict([[temp_input, humidity_input]])[0]
        prediction_rounded = round(prediction)
        st.sidebar.write(f"Numero previsto di adulti maschi: {prediction_rounded:.2f}")

if __name__ == "__main__":
    main()
