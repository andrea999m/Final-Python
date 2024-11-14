#Códigos para crear el enlace (usamos github y render)

import yfinance as yf
import datetime
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
from dash import Dash
import plotly.express as px
import dash_bootstrap_components as dbc
from scipy.stats import kurtosis, skew

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import matplotlib.pyplot as plt
import seaborn as sns

#lista de acciones a importar
stocks = ["DJT", "INTC", "F", "PFE", "SHOP", "AAL"]

#rango de fechas a importar
end_date=datetime.datetime.now()
start_date=end_date - datetime.timedelta(days=3*365) #3 años

#crear un dataframe vacio para almacenar los datos
historical_data=yf.download(stocks,start=start_date, end=end_date)["Adj Close"]
#usamos precios de cierre ajustados


#dataframe de los precios
data = historical_data.reset_index()
data_1=pd.melt(data, id_vars=["Date"], var_name="Company", value_name="Precio")
data_1 = pd.melt(data, id_vars=["Date"], var_name="Company", value_name="Precio")


#dataframe de los retornos
returns = data[stocks].pct_change()
returns["Date"] = data["Date"]
returns_1=pd.melt(returns, id_vars=["Date"], var_name="Company", value_name="Retorno")
returns_1 = pd.melt(returns, id_vars=["Date"], var_name="Company", value_name="Retorno")

#Dataframe final - unimos ambos dataframes para facilitar la creación del dashboard
data_final = pd.merge(data_1, returns_1, on=["Date", "Company"])
df = data_final
#verificamos que esté en formato correcto
df["Date"] = pd.to_datetime(df["Date"])  

#rango de fechas para el slider
min_date = df["Date"].min()
max_date = df["Date"].max()


returns1 = pd.DataFrame({
    "Portafolio Max Sharpe": np.random.randn(len(df))
}, index=df["Date"])

returns2 = pd.DataFrame({
    "Portafolio Min Vol": np.random.randn(len(df))
}, index=df["Date"])

#unir los dos DataFrames de los portafolios
data_final2 = pd.merge(returns1[["Portafolio Max Sharpe"]], returns2[["Portafolio Min Vol"]],
                       left_index=True, right_index=True)
data_final2_clean = data_final2.dropna(subset=["Portafolio Max Sharpe", "Portafolio Min Vol"])


#iniciamos dash
app = Dash(__name__)

#layout app
app.layout = html.Div([
    html.H4("Dashboard Acciones"),
    
    #dropdown para elegir las empresas
    html.Div([
        html.Div([
            dcc.Dropdown(
                id="Company",
                value=["DJT", "INTC", "F", "PFE", "SHOP", "AAL"],
                clearable=False,
                multi=True,
                options=[{"label": x, "value": x} for x in sorted(df.Company.unique())]
            ),
        ], className="six columns", style={"width": "50%"}),

        #dropdown para elegir los valores (precio o retorno)
        html.P("Values:"),
        dcc.Dropdown(
            id="Values",
            options=[{"label": "Precio", "value": "Precio"}, {"label": "Retorno", "value": "Retorno"}],
            value="Precio",  
            clearable=False
        )
    ], className="row"),

#usamos range slider para seleccionar el rango para las fechas
    html.Div([
        dcc.RangeSlider(
            id="fechas",
            min=0,
            max=(max_date - min_date).days,  # Número de días entre la fecha más temprana y la más tardía
            step=1,
            marks={i: (min_date + pd.Timedelta(i, "D")).strftime("%Y-%m-%d") for i in range(0, (max_date - min_date).days, 365)}, 
            value=[0, (max_date - min_date).days],  
        ),
    ], className="row"),
    
    #gráfica html
    html.Div([dcc.Graph(id="graph", figure={})]),

    html.Div([dcc.Graph(id="graph2", figure={})])
    
])

#callback gráfica
@app.callback(
    Output("graph", "figure"),
    [Input("Company", "value"), Input("Values", "value"),  Input("fechas", "value")]
)


def display_value(selected_stock, selected_numeric, date_range):
    start_date = min_date + pd.Timedelta(date_range[0], "D")
    end_date = min_date + pd.Timedelta(date_range[1], "D")

    dfv_fltrd = df[df["Company"].isin(selected_stock)]
    
    dfv_fltrd = dfv_fltrd[(dfv_fltrd["Date"] >= start_date) & (dfv_fltrd["Date"] <= end_date)]

    
    #crear la gráfica de líneas
    fig = px.line(dfv_fltrd, x="Date", y=selected_numeric, color="Company", markers=True,
                  width=950, height=450)
   
    fig.update_layout(title=f"{selected_numeric} de {selected_stock}",
                        xaxis_title="Date",)

    fig.update_traces(line=dict(width=2))
    
    #figura
    return fig

@app.callback(
    Output("graph2", "figure"),
    [Input("fechas", "value")]
)
def display_portfolio_comparison(date_range):
    start_date = min_date + pd.Timedelta(date_range[0], "D")
    end_date = min_date + pd.Timedelta(date_range[1], "D")

    # Filtrar los datos de los portafolios por el rango de fechas
    data_final2_clean_filtered = data_final2_clean[(data_final2_clean.index >= start_date) & 
                                                   (data_final2_clean.index <= end_date)]

    # Crear la gráfica de comparación
    fig2 = px.line(data_final2_clean_filtered, x=data_final2_clean_filtered.index, 
                   y=["Portafolio Max Sharpe", "Portafolio Min Vol"],
                   line_shape="linear",
                   title="Comparación Retorno de los Portafolios")

    return fig2
    

#agregar host
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=10000)
    
