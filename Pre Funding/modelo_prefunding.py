import pandas as pd
import numpy as np
from prophet import Prophet
import redshift_connector
import calendar
from datetime import date
from dotenv import load_dotenv
import os

# ── 1. CONEXIÓN A REDSHIFT ────────────────────────────────────────────────────
load_dotenv()

conn = redshift_connector.connect(
    host     = os.getenv("REDSHIFT_HOST"),
    database = os.getenv("REDSHIFT_DB"),
    user     = os.getenv("REDSHIFT_USER"),
    password = os.getenv("REDSHIFT_PASSWORD"),
    port     = int(os.getenv("REDSHIFT_PORT", 5439))
)

query = """
    SELECT
        wt.dat_transaction,
        wt.amt_transaction * -1  AS amount_trx,
        wt.cod_provider,
        wt.des_dest_ent_country,
        wt.des_transaction_type,
        wt.des_pay_flow,
        wt.cod_dest_acc_currency
    FROM process_data.wallet_transaction wt
    WHERE wt.is_cashout     = '1'
      AND wt.dat_transaction >= '2025-01-01'
      AND wt.dat_transaction <  CURRENT_DATE
"""

fondeo_data = pd.read_sql(query, conn)
conn.close()

# ── 2. LIMPIEZA Y PREPARACIÓN ─────────────────────────────────────────────────
fondeo_data['ds'] = pd.to_datetime(fondeo_data['dat_transaction']).dt.normalize()

# ── 3. FUNCIONES AUXILIARES ───────────────────────────────────────────────────
def es_fin_mes(fecha):
    dia        = fecha.day
    ultimo_dia = calendar.monthrange(fecha.year, fecha.month)[1]
    return int(dia >= (ultimo_dia - 2) or dia <= 3)

def preparar_serie(df_filtrado):
    serie = (
        df_filtrado
        .groupby('ds')['amount_trx']
        .sum()
        .reset_index()
        .rename(columns={'amount_trx': 'y'})
        .sort_values('ds')
        .reset_index(drop=True)
    )
    serie['fin_mes'] = serie['ds'].apply(es_fin_mes)
    assert serie['ds'].nunique() == len(serie), "Hay fechas duplicadas"
    return serie

def entrenar_y_predecir(serie, nombre):
    if len(serie) < 30:
        print(f"[{nombre}] Datos insuficientes ({len(serie)} dias), se omite.")
        return None

    modelo = Prophet(
        weekly_seasonality      = True,
        yearly_seasonality      = False,
        daily_seasonality       = False,
        changepoint_prior_scale = 0.03
    )
    modelo.add_country_holidays(country_name='CO')
    modelo.add_regressor('fin_mes')
    modelo.fit(serie)

    future = modelo.make_future_dataframe(periods=30, freq='D')
    future['fin_mes'] = future['ds'].apply(es_fin_mes)

    forecast = modelo.predict(future)
    forecast['yhat']       = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    return forecast

def aplicar_regla_viernes(forecast_df):
    fo = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    fo['dia_num'] = fo['ds'].dt.dayofweek  # 0=lunes, 4=viernes

    for i in range(len(fo) - 2):
        if fo.iloc[i]['dia_num'] == 4:
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                idx = fo.index[i]
                fo.at[idx, col] += fo.iloc[i + 1][col] + fo.iloc[i + 2][col]

    fo['nota'] = (
        fo['dia_num']
        .map({4: 'incluye sab + dom', 5: 'cubierto por viernes', 6: 'cubierto por viernes'})
        .fillna('')
    )
    return fo

# ── 4. FILTROS POR PROVEEDOR ──────────────────────────────────────────────────
def f_bamboo_cobre_col_mex(df):
    return df[
        df['cod_provider'].isin(['BAMBOO', 'COBRE']) &
        (df['des_transaction_type'] == 'SENT_TO_BANK') &
        df['des_dest_ent_country'].isin(['COLOMBIA', 'MEXICO'])
    ]

def f_bamboo_brasil(df):
    return df[
        (df['cod_provider'] == 'BAMBOO') &
        (df['des_transaction_type'] == 'SENT_TO_BANK') &
        (df['des_dest_ent_country'] == 'BRAZIL')
    ]

def f_bamboo_peru_uru_usd(df):
    return df[
        (df['cod_provider'] == 'BAMBOO') &
        (df['des_transaction_type'] == 'SENT_TO_BANK') &
        df['des_dest_ent_country'].isin(['PERU', 'URUGUAY']) &
        (df['cod_dest_acc_currency'] == 'USD')
    ]

def f_bamboo_peru_local(df):
    return df[
        df['cod_provider'].isin(['BAMBOO', 'MANUAL']) &
        (df['des_transaction_type'] == 'SENT_TO_BANK') &
        df['des_dest_ent_country'].isin(['PERU', 'PARAGUAY', 'NICARAGUA']) &
        df['cod_dest_acc_currency'].isin(['PEN', 'PYG', 'NIO'])
    ]

def f_zamp(df):
    return df[
        (df['cod_provider'] == 'ZAMP') &
        (df['des_transaction_type'] == 'SENT_TO_CRYPTO')
    ]

def f_bvnk(df):
    return df[
        (df['cod_provider'] == 'BVNK') &
        (df['des_transaction_type'] == 'SENT_TO_CRYPTO')
    ]

def f_zinli(df):
    return df[df['des_transaction_type'] == 'SENT_TO_ZINLI']

def f_payoneer(df):
    return df[df['des_transaction_type'] == 'SENT_TO_PAYONEER']

def f_paysend(df):
    return df[df['des_transaction_type'] == 'SENT_TO_EXTERNAL_CARD']

def f_astropay_usd(df):
    return df[df['des_transaction_type'] == 'SENT_TO_ASTROPAY']

def f_local_payment(df):
    cond1 = (
        (df['cod_provider'] == 'LOCAL_PAYMENTS') &
        (df['des_transaction_type'] == 'SENT_TO_BANK')
    )
    cond2 = (
        (df['cod_provider'] == 'MANUAL') &
        (df['des_dest_ent_country'] == 'GUATEMALA') &
        (df['cod_dest_acc_currency'] == 'GTQ')
    )
    return df[cond1 | cond2]

def f_thunes(df):
    cond1 = (
        (df['cod_provider'] == 'THUNES') &
        (df['des_transaction_type'] == 'SENT_TO_BANK')
    )
    cond2 = (
        (df['cod_provider'] == 'MANUAL') &
        (df['des_pay_flow'] == 'THUNES')
    )
    return df[cond1 | cond2]

def f_wise(df):
    return df[
        (df['cod_provider'] == 'MANUAL') &
        (df['des_transaction_type'] == 'SENT_TO_BANK') &
        (df['des_pay_flow'] == 'WISE')
    ]

def f_dlocal(df):
    cond1 = (
        (df['cod_provider'] == 'DLOCAL') &
        (df['des_transaction_type'] == 'SENT_TO_BANK')
    )
    cond2 = (
        (df['cod_provider'] == 'MANUAL') &
        (df['des_pay_flow'] == 'DLOCAL')
    )
    return df[cond1 | cond2]

def f_transfermate(df):
    return df[
        (df['cod_provider'] == 'MANUAL') &
        (df['des_transaction_type'] == 'SENT_TO_BANK') &
        (df['des_pay_flow'] == 'TRANSFERMATE')
    ]

# ── 5. REGISTRO DE PROVEEDORES ────────────────────────────────────────────────
PROVEEDORES = {
    "bamboo_cobre_col_mex" : ("Bamboo/Cobre - Colombia + Mexico", f_bamboo_cobre_col_mex),
    "bamboo_brasil"        : ("Bamboo - Brasil",                  f_bamboo_brasil),
    "bamboo_peru_uru_usd"  : ("Bamboo - Peru + Uruguay USD",      f_bamboo_peru_uru_usd),
    "bamboo_peru_local"    : ("Bamboo - Peru Local",              f_bamboo_peru_local),
    "zamp"                 : ("ZAMP",                             f_zamp),
    "bvnk"                 : ("BVNK",                             f_bvnk),
    "zinli"                : ("Zinli",                            f_zinli),
    "payoneer"             : ("Payoneer",                         f_payoneer),
    "paysend"              : ("Paysend",                          f_paysend),
    "astropay_usd"         : ("Astropay USD",                     f_astropay_usd),
    "local_payment"        : ("Local Payment",                    f_local_payment),
    "thunes"               : ("Thunes",                           f_thunes),
    "wise"                 : ("Wise",                             f_wise),
    "dlocal"               : ("DLocal",                           f_dlocal),
    "transfermate"         : ("Transfermate",                     f_transfermate),
}

# ── 6. EJECUCIÓN POR PROVEEDOR ────────────────────────────────────────────────
resultados = {}

for key, (nombre, filtro_fn) in PROVEEDORES.items():
    print(f"\n{'='*50}")
    print(f"Procesando: {nombre}")

    df_filtrado = filtro_fn(fondeo_data)

    if df_filtrado.empty:
        print(f"  Sin datos, se omite.")
        continue

    serie   = preparar_serie(df_filtrado)
    forecast = entrenar_y_predecir(serie, nombre)

    if forecast is None:
        continue

    forecast_op = aplicar_regla_viernes(forecast)

    resultado = (
        forecast_op[forecast_op['ds'].dt.date >= date.today()]
        [['ds', 'dia_num', 'nota', 'yhat', 'yhat_upper']]
        .head(14)
    )

    resultados[key] = resultado

    resultado_fmt = resultado.copy()
    resultado_fmt['yhat']       = resultado_fmt['yhat'].apply(lambda x: f"{x:>15,.0f}")
    resultado_fmt['yhat_upper'] = resultado_fmt['yhat_upper'].apply(lambda x: f"{x:>15,.0f}")
    resultado_fmt['ds']         = resultado_fmt['ds'].dt.strftime('%Y-%m-%d')
    resultado_fmt.columns       = ['Fecha', 'Dia', 'Nota', 'Prediccion', 'Escenario Alto']
    print(resultado_fmt.to_string(index=False))

print(f"\n{'='*50}")
print(f"Completado: {len(resultados)} proveedores procesados.")

# ── 7. GUARDAR RESULTADOS EN EXCEL ────────────────────────────────────────────
nombre_archivo = f"forecast_{date.today().strftime('%Y-%m-%d')}.xlsx"

with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
    for key, (nombre, _) in PROVEEDORES.items():
        if key not in resultados:
            continue
        df_excel = resultados[key].copy()
        df_excel['ds']          = df_excel['ds'].dt.strftime('%Y-%m-%d')
        df_excel['yhat']        = df_excel['yhat'].round(0)
        df_excel['yhat_upper']  = df_excel['yhat_upper'].round(0)
        df_excel.columns        = ['Fecha', 'Dia', 'Nota', 'Prediccion', 'Escenario Alto']
        nombre_hoja = nombre[:31].replace('/', '-').replace('\\', '-').replace('*', '').replace('?', '').replace('[', '').replace(']', '')
        df_excel.to_excel(writer, sheet_name=nombre_hoja, index=False)

print(f"Resultados guardados en: {nombre_archivo}")
