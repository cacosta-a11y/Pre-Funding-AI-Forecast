library(tidyverse)
library(prophet)
library(lubridate)

# ── 1. CARGA DE DATOS ──────────────────────────────────────────────────────────
fondeo_data <- read_csv("C:/Users/Cata-/OneDrive/Escritorio/OnTop/R/Pre Funding/Fondeo_data.csv")

fondeo_data$dat_transaction <- as.POSIXct(
  fondeo_data$dat_transaction,
  format = "%Y-%m-%d %H:%M:%S",
  tz = "UTC"
)

# ── 2. LIMPIEZA Y PREPARACIÓN ─────────────────────────────────────────────────
fondeo_data$ds <- as.Date(fondeo_data$dat_transaction)

fondeo_data <- fondeo_data %>%
  filter(des_transaction_type %in% c("SENT_TO_BANK", "SENT_TO_CRYPTO"))

# Consolidar por día (evita duplicados por proveedor)
fondeo_diario <- fondeo_data %>%
  group_by(ds, cod_provider, des_dest_ent_country) %>%
  summarise(
    y                 = sum(amount_trx, na.rm = TRUE),
    num_transacciones = n(),
    .groups           = "drop"
  )

# ── 3. SERIE DEL MODELO ───────────────────────────────────────────────────────
es_fin_mes <- function(fecha) {
  dia        <- day(fecha)
  ultimo_dia <- days_in_month(fecha)
  as.integer(dia >= (ultimo_dia - 2) | dia <= 3)
}

serie_modelo <- fondeo_diario %>%
  filter(
    cod_provider %in% c("MANUAL", "BAMBOO", "COBRE"),
    des_dest_ent_country == "COLOMBIA"
  ) %>%
  group_by(ds) %>%
  summarise(y = sum(y), .groups = "drop") %>%
  arrange(ds) %>%
  mutate(fin_mes = es_fin_mes(ds))

serie_modelo$ds <- as.Date(as.character(serie_modelo$ds))

# Verificar que no hay fechas duplicadas
stopifnot(nrow(serie_modelo %>% count(ds) %>% filter(n > 1)) == 0)

# ── 4. MODELO PROPHET ─────────────────────────────────────────────────────────
modelo <- prophet(
  weekly.seasonality      = TRUE,
  yearly.seasonality      = FALSE,
  daily.seasonality       = FALSE,
  changepoint.prior.scale = 0.03
)

modelo <- add_country_holidays(modelo, country_name = "CO")
modelo <- add_regressor(modelo, "fin_mes")

modelo <- fit.prophet(modelo, serie_modelo)

# ── 5. FORECAST ───────────────────────────────────────────────────────────────
future <- make_future_dataframe(modelo, periods = 30, freq = "day") %>%
  mutate(fin_mes = es_fin_mes(as.Date(ds)))

forecast <- predict(modelo, future)

forecast$yhat       <- pmax(forecast$yhat, 0)
forecast$yhat_lower <- pmax(forecast$yhat_lower, 0)
forecast$yhat_upper <- pmax(forecast$yhat_upper, 0)

# ── 6. REGLA DEL VIERNES ──────────────────────────────────────────────────────
forecast_operativo <- forecast %>%
  select(ds, yhat, yhat_lower, yhat_upper) %>%
  mutate(
    yhat       = pmax(yhat, 0),
    yhat_lower = pmax(yhat_lower, 0),
    yhat_upper = pmax(yhat_upper, 0),
    dia_num    = wday(ds, week_start = 1)  # 1=lunes, 5=viernes, 6=sabado, 7=domingo
  )

for (i in 1:(nrow(forecast_operativo) - 2)) {
  if (forecast_operativo$dia_num[i] == 5) {
    forecast_operativo$yhat[i]       <- forecast_operativo$yhat[i] +
      forecast_operativo$yhat[i + 1] +
      forecast_operativo$yhat[i + 2]
    forecast_operativo$yhat_upper[i] <- forecast_operativo$yhat_upper[i] +
      forecast_operativo$yhat_upper[i + 1] +
      forecast_operativo$yhat_upper[i + 2]
    forecast_operativo$yhat_lower[i] <- forecast_operativo$yhat_lower[i] +
      forecast_operativo$yhat_lower[i + 1] +
      forecast_operativo$yhat_lower[i + 2]
  }
}

forecast_operativo <- forecast_operativo %>%
  mutate(
    nota = case_when(
      dia_num == 5        ~ "incluye sab + dom",
      dia_num %in% c(6,7) ~ "cubierto por viernes",
      TRUE                ~ ""
    )
  )

# ── 7. RESULTADOS ─────────────────────────────────────────────────────────────
# Forecast crudo (ultimos 30 dias)
tail(forecast[, c("ds", "yhat", "yhat_lower", "yhat_upper")], 30)

# Forecast operativo (proximos 14 dias)
forecast_operativo %>%
  filter(as.Date(ds) >= Sys.Date()) %>%
  select(ds, dia_num, nota, yhat, yhat_upper) %>%
  head(14) %>%
  print()

# Graficos
plot(modelo, forecast)
prophet_plot_components(modelo, forecast)

# ── 8. VALIDACION: CROSS-VALIDATION MANUAL ───────────────────────────────────
cutoff_dates <- seq(as.Date("2025-07-01"), as.Date("2026-01-01"), by = "30 days")
resultados_cv <- list()

for (corte in as.list(cutoff_dates)) {
  corte <- as.Date(corte)

  train <- serie_modelo %>% filter(ds <= corte)
  test  <- serie_modelo %>% filter(ds > corte, ds <= corte + 30)

  if (nrow(test) == 0) next

  m_temp <- prophet(
    weekly.seasonality      = TRUE,
    yearly.seasonality      = FALSE,
    daily.seasonality       = FALSE,
    changepoint.prior.scale = 0.03
  )
  m_temp <- add_country_holidays(m_temp, country_name = "CO")
  m_temp <- add_regressor(m_temp, "fin_mes")
  m_temp <- fit.prophet(m_temp, train)

  fut_temp <- make_future_dataframe(m_temp, periods = 30, freq = "day") %>%
    mutate(fin_mes = es_fin_mes(as.Date(ds)))

  pred_temp <- predict(m_temp, fut_temp) %>%
    mutate(ds = as.Date(ds)) %>%
    filter(ds > corte, ds <= corte + 30) %>%
    select(ds, yhat)

  comp <- test %>%
    inner_join(pred_temp, by = "ds") %>%
    mutate(
      cutoff    = corte,
      error_pct = abs(y - yhat) / y * 100,
      error_abs = abs(y - yhat)
    )

  resultados_cv[[length(resultados_cv) + 1]] <- comp
}

cv_final <- bind_rows(resultados_cv)

cat("--- Metricas Cross-Validation ---\n")
cat("MAE: ", round(mean(cv_final$error_abs, na.rm = TRUE), 0), "\n")
cat("MAPE:", round(mean(cv_final$error_pct, na.rm = TRUE), 1), "%\n")
cat("RMSE:", round(sqrt(mean((cv_final$y - cv_final$yhat)^2, na.rm = TRUE)), 0), "\n")
cat("Dias evaluados:", nrow(cv_final), "\n")
