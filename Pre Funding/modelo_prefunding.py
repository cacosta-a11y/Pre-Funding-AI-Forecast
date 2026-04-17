import pandas as pd
import numpy as np
from prophet import Prophet
import redshift_connector
import calendar
from datetime import date

# ── 1. CONEXIÓN A REDSHIFT ────────────────────────────────────────────────────
conn = redshift_connector.connect(
    host     = "tu-cluster.redshift.amazonaws.com",
    database = "tu_base_de_datos",
    user     = "tu_usuario",
    password = "tu_contraseña",
    port     = 5439
)

query = """
    SELECT
        wt.dat_transaction,
        wt.amt_transaction * -1  AS amount_trx,
        wt.cod_provider,
        wt.des_dest_ent_country,
        wt.des_transaction_type
    FROM process_data.wallet_transaction wt
    WHERE wt.is_cashout     = '1'
      AND wt.dat_transaction >= '2025-01-01'
      AND wt.dat_transaction <  CURRENT_DATE
"""

fondeo_data = pd.read_sql(query, conn)
conn.close()

# ── 2. LIMPIEZA Y PREPARACIÓN ─────────────────────────────────────────────────
fondeo_data['ds'] = pd.to_datetime(fondeo_data['dat_transaction']).dt.normalize()

# ── 3. SERIE DEL MODELO ───────────────────────────────────────────────────────
def es_fin_mes(fecha):
    dia        = fecha.day
    ultimo_dia = calendar.monthrange(fecha.year, fecha.month)[1]
    return int(dia >= (ultimo_dia - 2) or dia <= 3)

serie_modelo = (
    fondeo_data
    .query("cod_provider in ['MANUAL', 'BAMBOO', 'COBRE'] and des_dest_ent_country == 'COLOMBIA'")
    .groupby('ds')['amount_trx']
    .sum()
    .reset_index()
    .rename(columns={'amount_trx': 'y'})
    .sort_values('ds')
    .reset_index(drop=True)
)

serie_modelo['fin_mes'] = serie_modelo['ds'].apply(es_fin_mes)

# Verificar duplicados
assert serie_modelo['ds'].nunique() == len(serie_modelo), "Hay fechas duplicadas"

# ── 4. MODELO PROPHET ─────────────────────────────────────────────────────────
modelo = Prophet(
    weekly_seasonality      = True,
    yearly_seasonality      = False,
    daily_seasonality       = False,
    changepoint_prior_scale = 0.03
)

modelo.add_country_holidays(country_name='CO')
modelo.add_regressor('fin_mes')
modelo.fit(serie_modelo)

# ── 5. FORECAST ───────────────────────────────────────────────────────────────
future = modelo.make_future_dataframe(periods=30, freq='D')
future['fin_mes'] = future['ds'].apply(es_fin_mes)

forecast = modelo.predict(future)

forecast['yhat']       = forecast['yhat'].clip(lower=0)
forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

# ── 6. REGLA DEL VIERNES ──────────────────────────────────────────────────────
forecast_operativo = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_operativo['dia_num'] = forecast_operativo['ds'].dt.dayofweek  # 0=lunes, 4=viernes

for i in range(len(forecast_operativo) - 2):
    if forecast_operativo.iloc[i]['dia_num'] == 4:  # viernes
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            idx = forecast_operativo.index[i]
            forecast_operativo.at[idx, col] += (
                forecast_operativo.iloc[i + 1][col] +
                forecast_operativo.iloc[i + 2][col]
            )

forecast_operativo['nota'] = (
    forecast_operativo['dia_num']
    .map({4: 'incluye sab + dom', 5: 'cubierto por viernes', 6: 'cubierto por viernes'})
    .fillna('')
)

# ── 7. RESULTADOS ─────────────────────────────────────────────────────────────
resultado = (
    forecast_operativo[forecast_operativo['ds'].dt.date >= date.today()]
    [['ds', 'dia_num', 'nota', 'yhat', 'yhat_upper']]
    .head(14)
)

print(resultado.to_string(index=False))
