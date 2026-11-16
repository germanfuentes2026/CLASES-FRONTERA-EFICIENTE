"""
Frontera Eficiente de Markowitz + CML
-------------------------------------
App de Streamlit que:
  * Descarga precios (Yahoo Finance) o lee un CSV propio.
  * Calcula retornos ANUALES (precio de cierre de cada año) o anualizados desde diarios.
  * Construye la frontera eficiente (sin ventas en corto, con peso mínimo por activo configurable).
  * Dibuja la Capital Market Line (CML).
  * Marca 3 portafolios: Mínima Varianza, Máximo Retorno y Máximo Sharpe (tangente).

Ejecutar local:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Frontera Eficiente de Markowitz", layout="wide")


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def descargar_precios(tickers, inicio, fin):
    import yfinance as yf

    data = yf.download(tickers, start=inicio, end=fin, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        precios = data["Close"]
    else:  # un solo ticker
        precios = data[["Close"]].rename(columns={"Close": tickers[0]})
    return precios.dropna(how="all")


def leer_csv(archivo):
    df = pd.read_csv(archivo)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0]).sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def calcular_retornos(precios, metodo):
    """Devuelve (retornos, mu_anual, cov_anual)."""
    precios = precios.dropna()
    if metodo == "Anuales (cierre de cada año)":
        anual = precios.resample("YE").last()
        rets = anual.pct_change().dropna()
        mu = rets.mean()
        cov = rets.cov()
    else:  # anualizados desde diarios
        rets = precios.pct_change().dropna()
        mu = rets.mean() * 252
        cov = rets.cov() * 252
    return rets, mu, cov


# ----------------------------------------------------------------------------
# Optimización
# ----------------------------------------------------------------------------
def stats_portafolio(w, mu, cov, rf):
    r = float(w @ mu)
    s = float(np.sqrt(w @ cov @ w))
    sharpe = (r - rf) / s if s > 0 else np.nan
    return r, s, sharpe


def _base(n, w_min=0.0):
    w0 = np.repeat(1 / n, n)
    bounds = tuple((w_min, 1.0) for _ in range(n))
    suma_1 = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    return w0, bounds, suma_1


def min_varianza(mu, cov, w_min=0.0):
    n = len(mu)
    w0, bounds, suma_1 = _base(n, w_min)
    res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                   bounds=bounds, constraints=[suma_1])
    return res.x


def max_sharpe(mu, cov, rf, w_min=0.0):
    n = len(mu)
    w0, bounds, suma_1 = _base(n, w_min)

    def neg_sharpe(w):
        r = w @ mu
        s = np.sqrt(w @ cov @ w)
        return -(r - rf) / s

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=[suma_1])
    return res.x


def max_retorno(mu, cov, w_min=0.0):
    # Cada activo recibe el mínimo exigido y todo el resto va al activo
    # de mayor retorno esperado (solución exacta de este problema lineal).
    n = len(mu)
    w = np.repeat(w_min, n)
    w[int(np.argmax(mu))] += 1.0 - n * w_min
    return w


def frontera_eficiente(mu, cov, r_min, r_max, w_min=0.0, puntos=60):
    n = len(mu)
    w0, bounds, suma_1 = _base(n, w_min)
    rets_obj = np.linspace(r_min, r_max, puntos)
    vols, rets_ok = [], []
    for r_obj in rets_obj:
        cons = [suma_1, {"type": "eq", "fun": lambda w, r=r_obj: w @ mu - r}]
        res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if res.success:
            vols.append(np.sqrt(res.fun))
            rets_ok.append(r_obj)
            w0 = res.x
    return np.array(vols), np.array(rets_ok)


def portafolios_aleatorios(mu, cov, rf, w_min=0.0, n_port=3000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mu)
    W = w_min + (1 - n * w_min) * rng.dirichlet(np.ones(n), n_port)
    r = W @ mu
    s = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    return r, s, (r - rf) / s


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
st.title("📈 Frontera Eficiente de Markowitz + CML")

with st.sidebar:
    st.header("Parámetros")
    fuente = st.radio("Fuente de datos", ["Yahoo Finance", "Subir CSV"])

    if fuente == "Yahoo Finance":
        tickers_txt = st.text_input("Tickers (separados por coma)",
                                    "AAPL, MSFT, AMZN, JPM, KO, XOM, GLD")
        c1, c2 = st.columns(2)
        inicio = c1.date_input("Desde", pd.Timestamp("2010-01-01"))
        fin = c2.date_input("Hasta", pd.Timestamp.today())
        archivo = None
    else:
        st.caption("CSV: 1ª columna = fecha, resto = precios de cada activo.")
        archivo = st.file_uploader("Archivo CSV", type="csv")

    metodo = st.selectbox("Tipo de retorno",
                          ["Anuales (cierre de cada año)",
                           "Anualizados desde diarios"])
    rf = st.number_input("Tasa libre de riesgo anual (%)", 0.0, 50.0, 4.0, 0.25) / 100
    w_min = st.number_input(
        "Peso mínimo por activo (%)", 0.0, 50.0, 5.0, 0.5,
        help="Cada activo tendrá al menos este porcentaje en todos los portafolios. "
             "0% = el modelo puede excluir activos.") / 100
    mostrar_mc = st.checkbox("Mostrar portafolios aleatorios", True)
    calcular = st.button("Calcular", type="primary", use_container_width=True)

if not calcular:
    st.info("Configurá los parámetros en la barra lateral y apretá **Calcular**.")
    st.stop()

# --- Carga de datos
try:
    if fuente == "Yahoo Finance":
        tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
        if len(tickers) < 2:
            st.error("Ingresá al menos 2 tickers.")
            st.stop()
        with st.spinner("Descargando precios..."):
            precios = descargar_precios(tickers, inicio, fin)
    else:
        if archivo is None:
            st.error("Subí un archivo CSV.")
            st.stop()
        precios = leer_csv(archivo)
except Exception as e:
    st.error(f"Error al obtener los datos: {e}")
    st.stop()

precios = precios.dropna(axis=1, how="all").dropna()
if precios.shape[1] < 2 or precios.empty:
    st.error("No hay datos suficientes (se necesitan al menos 2 activos con precios).")
    st.stop()

rets, mu, cov = calcular_retornos(precios, metodo)
if len(rets) < 3:
    st.error(f"Solo hay {len(rets)} observaciones de retorno. Ampliá el rango de fechas.")
    st.stop()
if metodo.startswith("Anuales") and len(rets) <= len(mu):
    st.warning(f"Hay {len(rets)} años y {len(mu)} activos: la matriz de covarianzas "
               "puede ser poco confiable. Conviene ampliar el período o usar menos activos.")

activos = list(mu.index)
mu_v, cov_v = mu.values, cov.values

n_act = len(activos)
if n_act * w_min > 1 + 1e-9:
    st.error(f"Con {n_act} activos el peso mínimo no puede superar "
             f"{100 / n_act:.2f}% (si no, la suma pasa del 100%). "
             f"Ingresaste {w_min:.2%}.")
    st.stop()
if abs(n_act * w_min - 1) < 1e-9:
    st.warning("Con ese mínimo el único portafolio posible es el equiponderado.")
st.caption(f"Peso mínimo por activo: **{w_min:.2%}** · "
           f"Peso máximo posible en un activo: **{1 - (n_act - 1) * w_min:.2%}**")

# --- Portafolios clave
w_mv = min_varianza(mu_v, cov_v, w_min)
w_ms = max_sharpe(mu_v, cov_v, rf, w_min)
w_mr = max_retorno(mu_v, cov_v, w_min)

r_mv, s_mv, sh_mv = stats_portafolio(w_mv, mu_v, cov_v, rf)
r_ms, s_ms, sh_ms = stats_portafolio(w_ms, mu_v, cov_v, rf)
r_mr, s_mr, sh_mr = stats_portafolio(w_mr, mu_v, cov_v, rf)

# --- Frontera y CML
vols_f, rets_f = frontera_eficiente(mu_v, cov_v, r_mv, r_mr, w_min)
if len(vols_f) == 0:  # caso equiponderado forzado
    vols_f, rets_f = np.array([s_mv]), np.array([r_mv])
s_max = max(vols_f.max(), np.sqrt(np.diag(cov_v)).max()) * 1.15
x_cml = np.linspace(0, s_max, 100)
y_cml = rf + sh_ms * x_cml

# --- Gráfico
fig = go.Figure()

if mostrar_mc:
    r_mc, s_mc, sh_mc = portafolios_aleatorios(mu_v, cov_v, rf, w_min)
    fig.add_trace(go.Scatter(
        x=s_mc, y=r_mc, mode="markers", name="Portafolios aleatorios",
        marker=dict(size=4, color=sh_mc, colorscale="Viridis", opacity=0.45,
                    colorbar=dict(title="Sharpe")),
        hovertemplate="σ=%{x:.2%}<br>E[r]=%{y:.2%}<extra></extra>"))

fig.add_trace(go.Scatter(
    x=vols_f, y=rets_f, mode="lines", name="Frontera eficiente",
    line=dict(color="#1f3b73", width=3),
    hovertemplate="σ=%{x:.2%}<br>E[r]=%{y:.2%}<extra></extra>"))

fig.add_trace(go.Scatter(
    x=x_cml, y=y_cml, mode="lines", name="CML",
    line=dict(color="#d62728", width=2, dash="dash"),
    hovertemplate="σ=%{x:.2%}<br>E[r]=%{y:.2%}<extra></extra>"))

# Activos individuales
fig.add_trace(go.Scatter(
    x=np.sqrt(np.diag(cov_v)), y=mu_v, mode="markers+text", name="Activos",
    text=activos, textposition="top center",
    marker=dict(size=9, color="gray", symbol="circle-open", line=dict(width=2)),
    hovertemplate="%{text}<br>σ=%{x:.2%}<br>E[r]=%{y:.2%}<extra></extra>"))

# Tasa libre de riesgo
fig.add_trace(go.Scatter(
    x=[0], y=[rf], mode="markers", name=f"Rf ({rf:.2%})",
    marker=dict(size=10, color="black", symbol="diamond")))

for nombre, s, r, color, simbolo in [
    ("Mínima Varianza", s_mv, r_mv, "#2ca02c", "star"),
    ("Máximo Sharpe (tangente)", s_ms, r_ms, "#ff7f0e", "star"),
    ("Máximo Retorno", s_mr, r_mr, "#9467bd", "star"),
]:
    fig.add_trace(go.Scatter(
        x=[s], y=[r], mode="markers", name=nombre,
        marker=dict(size=20, color=color, symbol=simbolo,
                    line=dict(color="black", width=1)),
        hovertemplate=f"{nombre}<br>σ=%{{x:.2%}}<br>E[r]=%{{y:.2%}}<extra></extra>"))

fig.update_layout(
    height=620,
    xaxis_title="Riesgo (desvío estándar anual)",
    yaxis_title="Retorno esperado anual",
    xaxis=dict(tickformat=".0%", range=[0, s_max]),
    yaxis=dict(tickformat=".0%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# --- Métricas de los 3 portafolios
st.subheader("Portafolios destacados")
c1, c2, c3 = st.columns(3)
for col, nombre, r, s, sh in [
    (c1, "🟢 Mínima Varianza", r_mv, s_mv, sh_mv),
    (c2, "🟠 Máximo Sharpe", r_ms, s_ms, sh_ms),
    (c3, "🟣 Máximo Retorno", r_mr, s_mr, sh_mr),
]:
    with col:
        st.markdown(f"**{nombre}**")
        st.metric("Retorno esperado", f"{r:.2%}")
        st.metric("Volatilidad", f"{s:.2%}")
        st.metric("Sharpe", f"{sh:.3f}")

# --- Ponderaciones
st.subheader("Ponderaciones")
pesos = pd.DataFrame({
    "Mínima Varianza": w_mv,
    "Máximo Sharpe": w_ms,
    "Máximo Retorno": w_mr,
}, index=activos)
pesos = pesos.clip(lower=w_min).div(pesos.clip(lower=w_min).sum(), axis=1)

fig_w = go.Figure()
for col, color in zip(pesos.columns, ["#2ca02c", "#ff7f0e", "#9467bd"]):
    fig_w.add_trace(go.Bar(x=pesos.index, y=pesos[col], name=col, marker_color=color))
fig_w.update_layout(barmode="group", height=380, yaxis=dict(tickformat=".0%"),
                    margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_w, use_container_width=True)
st.dataframe(pesos.style.format("{:.2%}"), use_container_width=True)

# --- Datos de entrada
with st.expander("Retornos esperados y matriz de covarianzas"):
    resumen = pd.DataFrame({"Retorno anual": mu, "Volatilidad anual": np.sqrt(np.diag(cov))})
    st.dataframe(resumen.style.format("{:.2%}"), use_container_width=True)
    st.markdown("**Matriz de covarianzas (anual)**")
    st.dataframe(cov.style.format("{:.4f}"), use_container_width=True)
    st.markdown("**Matriz de correlaciones**")
    st.dataframe(rets.corr().style.format("{:.2f}").background_gradient(cmap="RdYlGn"),
                 use_container_width=True)

with st.expander("Retornos utilizados"):
    st.dataframe(rets.style.format("{:.2%}"), use_container_width=True)

st.download_button("Descargar ponderaciones (CSV)",
                   pesos.to_csv().encode("utf-8"),
                   "ponderaciones.csv", "text/csv")
