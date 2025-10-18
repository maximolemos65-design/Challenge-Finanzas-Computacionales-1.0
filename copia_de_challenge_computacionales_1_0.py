import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
import math
import pandas as pd
from datetime import date, timedelta

# ==========================
# Encabezado
# ==========================
st.title("📊 Calculadora Financiera")
st.caption("UADE • Challenge Computacionales")

st.markdown(
    "Esta app calcula estadísticas de retornos, volatilidad anualizada y precios de opciones con **Black-Scholes**. "
    "Cargá los parámetros y apretá **Calcular**."
)

# ==========================
# Inicializamos variables de estado
# ==========================
if "calculado" not in st.session_state:
    st.session_state.calculado = False

# ==========================
# 1. Entradas del usuario
# ==========================
st.write("💡 Para acciones argentinas agregar '.BA' al final del ticker. Ejemplo: 'GGAL.BA'")

ticker = st.text_input("📌 Ingrese el ticker (ej: AAPL o GGAL.BA):", value="AAPL").strip().upper()
start_date = st.date_input("🗓️ Fecha inicio", value=date.today() - timedelta(days=180))
end_date = st.date_input("🗓️ Fecha fin", value=date.today())
interval = st.selectbox("⏱️ Periodicidad de datos:", ["1d", "1wk", "1mo"], index=0)

r = float(st.number_input("📉 Tasa libre de riesgo (ej: 0.05 para 5%): ", value=0.05))
K = float(st.number_input("💰 Precio strike: ", value=100.0))
meses = int(st.number_input("⏳ Tiempo al vencimiento (meses): ", value=6))
T = meses / 12

# ==========================
# 2. Botón Calcular
# ==========================
if st.button("Calcular"):
    st.session_state.calculado = True

# ==========================
# 3. Cálculos (solo si el botón fue presionado)
# ==========================
if st.session_state.calculado:
    # Todo el código de cálculos y gráficos

    # ==========================
    # 2. Descargar datos
    # ==========================
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    # Retornos logarítmicos
    data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
    returns = data['Return'].dropna()
    
    # ==========================
    # 4. Estadísticas básicas
    # ==========================
    mean_return = returns.mean()
    std_return  = returns.std()
    
    st.subheader(f"\n📊 Retornos y distribución de {ticker}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Promedio retorno", f"{mean_return*100:.2f}%")
    with col2:
        st.metric("📉 Desvío retorno", f"{std_return*100:.2f}%")
    
   # ==========================
    # 5. Histograma con campana normal
    # ==========================
    
    st.markdown("##### 📊 Distribución de Retornos con Campana Normal")
    
    # Verificar que existan datos de retornos
    if "returns" in locals() and not returns.empty:
    
        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(8, 5))
    
        # Histograma de retornos (densidad)
        count, bins, _ = ax.hist(returns, bins=50, density=True, edgecolor='black', alpha=0.7, label="Histograma")
    
        # Rango para la curva normal
        x = np.linspace(min(returns), max(returns), 1000)
        pdf = norm.pdf(x, mean_return, std_return)
    
        # Graficar campana normal teórica
        ax.plot(x, pdf, 'r-', linewidth=2, label="Normal teórica")
    
        # Líneas de media y desviaciones estándar
        ax.axvline(mean_return, color='blue', linestyle='dashed', linewidth=2, label=f"Media: {mean_return*100:.2f}%")
        ax.axvline(mean_return + std_return, color='green', linestyle='dashed', linewidth=2, label=f"+1σ: {(mean_return+std_return)*100:.2f}%")
        ax.axvline(mean_return - std_return, color='green', linestyle='dashed', linewidth=2, label=f"-1σ: {(mean_return-std_return)*100:.2f}%")
        ax.axvline(mean_return + 2*std_return, color='green', linestyle='dashed', linewidth=2, label=f"+2σ: {(mean_return+2*std_return)*100:.2f}%")
        ax.axvline(mean_return - 2*std_return, color='green', linestyle='dashed', linewidth=2, label=f"-2σ: {(mean_return-2*std_return)*100:.2f}%")
    
        # Estética del gráfico
        ax.set_title(f"Distribución de Retornos - {ticker}")
        ax.set_xlabel("Retorno logarítmico")
        ax.set_ylabel("Densidad")
        ax.legend()
        ax.grid(alpha=0.3)
    
        # Mostrar en Streamlit
        st.pyplot(fig)
    
    else:
        st.warning("⚠️ No hay datos de retornos disponibles. Calculá los retornos primero.")


    # ==========================
    # 6. Asimetría y curtosis
    # ==========================
    asimetria = skew(returns)
    curtosis_val = kurtosis(returns, fisher=False)  # fisher=True → curtosis "exceso" (0 = normal)
    curtosis_total = kurtosis(returns, fisher=True)
    
    st.markdown("##### 📊 Características de la serie de retornos")

    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("📐 Asimetría", f"{asimetria:.4f}")
    
    with col_b:
        st.metric("🏔️ Curtosis", f"{curtosis_val:.4f}")
    
    # ==========================
    # 7. Volatilidad histórica y ajuste por intervalo
    # ==========================
    if interval == "1d":
        factor = 252
        Dt = 1/252
    elif interval == "1wk":
        factor = 52
        Dt = 1/52
    elif interval == "1mo":
        factor = 12
        Dt = 1/12
    else:
        factor = 252
        Dt = 1/252
    
    vol_daily = std_return
    vol_annual = vol_daily * np.sqrt(factor)
    
    with col_c:
        st.metric("🌪️ Volatilidad anualizada", f"{vol_annual*100:.2f}%")
    
    # ==========================
    # 8. Black-Scholes
    # ==========================
    sigma = float(vol_annual)
    S = float(data['Close'].iloc[-1])  # Último precio spot
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    
    st.markdown("##### 📊 Black-Scholes")
    
    st.write(f"d1 = {d1:.4f}")
    st.write(f"d2 = {d2:.4f}")
    st.write(f"Nd1 = {Nd1:.4f}")
    st.write(f"Nd2 = {Nd2:.4f}")
    
    # ==========================
    # 9. Precios de la opción
    # ==========================
    call_price = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    put_price  = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # ==========================
    # 10. Valor intrínseco y extrínseco
    # ==========================
    call_intrinsic = max(S - K, 0)
    put_intrinsic  = max(K - S, 0)
    
    call_extrinsic = call_price - call_intrinsic
    put_extrinsic  = put_price - put_intrinsic
    
    st.write(f"\n💰 Precio Call (BS): {call_price:.2f}")
    st.write(f"   - Intrínseco: {call_intrinsic:.2f}")
    st.write(f"   - Extrínseco: {call_extrinsic:.2f}")
    
    st.write(f"\n💰 Precio Put  (BS): {put_price:.2f}")
    st.write(f"   - Intrínseco: {put_intrinsic:.2f}")
    st.write(f"   - Extrínseco: {put_extrinsic:.2f}")
    
    st.subheader("🌪️ Volatilidad")
    
    # ==========================
    # 2.1 Calcular Z-scores
    # ==========================
    z_scores = (returns - mean_return) / std_return
    
    # Media y desvío de los Z-scores
    mean_z = z_scores.mean()
    std_z  = z_scores.std()
   
    # ==========================
    # 2.3 Histograma de Z-Scores
    # ==========================
    
    st.markdown("##### 📉 Distribución de Z-Scores")
    st.write()
    st.write(f"Media z-scores: {mean_z:.6f}")
    st.write(f"Desvío z-scores: {std_z:.6f}")
    
    # Verificar que existan los z-scores
    if "z_scores" in locals() and not isinstance(z_scores, type(None)) and len(z_scores) > 0:
    
        # Crear figura y ejes
        fig, ax = plt.subplots(figsize=(8, 5))
    
        # Histograma de los z-scores
        ax.hist(z_scores, bins=40, edgecolor='black', alpha=0.7, density=True, label="Z-Scores observados")
    
        # Curva normal estándar teórica
        x = np.linspace(-4, 4, 200)
        ax.plot(x, norm.pdf(x, 0, 1), 'r-', lw=2, label="N(0,1) teórica")
    
        # Líneas de referencia
        ax.axvline(0, color='blue', linestyle='dashed', linewidth=2, label=f"Media=0: {mean_z:.4f}")
        ax.axvline(1, color='green', linestyle='dashed', linewidth=1, label=f"+1σ: {mean_z+std_z:.4f}")
        ax.axvline(-1, color='green', linestyle='dashed', linewidth=1, label=f"-1σ: {mean_z-std_z:.4f}")
    
        # Texto con resumen de parámetros
        ax.text(
            2.5, 0.35,
            f"μ_ret = {mean_return:.5f}\nσ_ret = {std_return:.5f}\nμ_z = {mean_z:.5f}\nσ_z = {std_z:.5f}",
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
        # Estética
        ax.set_title(f"Distribución de Z-Scores - {ticker}")
        ax.set_xlabel("Z-Score (zt)")
        ax.set_ylabel("Densidad")
        ax.legend()
        ax.grid(alpha=0.3)
    
        # Mostrar gráfico en Streamlit
        st.pyplot(fig)
    
    else:
        st.warning("⚠️ No hay datos de Z-Scores disponibles. Calculá los retornos primero.")

    
    # ==========================
    # 2.4 Asimetría y curtosis de Z-Scores
    # ==========================
    asimetriaz = skew(z_scores)
    curtosisz_val = kurtosis(z_scores, fisher=False)  # fisher=True → curtosis "exceso" (0 = normal)
    
    st.write("\n📉 Asimetría y Curtosis de la serie de Z-Scores")
    st.write(f"Asimetría: {asimetria:.4f}")
    st.write(f"Curtosis: {curtosis_val:.4f}")

    # ==========================
    #  Volatilidad móvil
    # ==========================
    
    st.markdown("##### 📉 Volatilidad Móvil de los Retornos")
    
    # Verificar que existan los retornos en el DataFrame
    if "Return" in data.columns and not data["Return"].dropna().empty:
    
        # --- Desvíos móviles (ventanas) calculados sobre data['Return'] ---
        std_20 = data['Return'].rolling(window=20).std()
        std_250 = data['Return'].rolling(window=250).std()
    
        # --- Anualizar ---
        std_20_ann = std_20 * np.sqrt(factor)
        std_250_ann = std_250 * np.sqrt(factor)
    
        # --- Gráfico ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(std_20.index, std_20_ann, '--', label='Std20 (anualizada)', color='orange')
        ax.plot(std_250.index, std_250_ann, '--', label='Std250 (anualizada)', color='red')
        ax.axhline(y=vol_annual, color='green', linestyle='--', linewidth=1.5, label='Std constante')
    
        ax.set_title(f"Evolución del Desvío Estándar Móvil de Retornos - {ticker}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Desvío estándar (anualizado)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
    
        # Mostrar gráfico en la app
        st.pyplot(fig)
    
        # --- Cálculo de valores recientes ---
        last_std20 = std_20.dropna().iloc[-1] if std_20.dropna().size > 0 else np.nan
        last_std250 = std_250.dropna().iloc[-1] if std_250.dropna().size > 0 else np.nan
    
        # --- Mostrar estadísticas ---
        st.markdown("##### 📉 Estadísticas de Volatilidades Anualizadas Recientes")
        st.write(f"**Último Std 20:** {(last_std20 * np.sqrt(factor) * 100):.4f}%")
        st.write(f"**Último Std 250:** {(last_std250 * np.sqrt(factor) * 100):.4f}%")
        st.write(f"**Volatilidad constante (anualizada):** {(vol_annual * 100):.4f}%")
    
    else:
        st.warning("⚠️ No se encontraron datos de retornos para calcular la volatilidad móvil.")

    # ==========================
    # 14. Value at Risk (VaR empírico) + Conditional VaR con sombreado
    # ==========================

    st.subheader("💥 Value at Risk (VaR) Empírico")
    
    # 1️⃣ Input de confianza
    conf_input = st.text_input("📌 Ingrese el nivel de confianza (ej: 0.95 para 95%):", value="0.95")
    
    try:
        conf = float(conf_input)
        if 0 < conf < 1:
            alpha = 1 - conf
    
            # 2️⃣ Calcular VaR y CVaR
            VaR_empirico = np.percentile(returns, alpha * 100)
            mean_emp = returns.mean()
            cola = returns[returns <= VaR_empirico]
            CVaR_empirico = cola.mean()
    
            # 3️⃣ Mostrar resultados numéricos
            st.success(f"🔹 Nivel de confianza: **{conf*100:.1f}%**")
            st.write(f"📉 **VaR empírico ({alpha*100:.1f}%): `{VaR_empirico*100:.2f}%`** → En el {alpha*100:.2f}% de los casos, suponiendo una volatilidad constante y bajo condiciones normales de mercado, {ticker} tendrá un rendimiento menor o igual a {VaR_empirico*100:.5f}%.")
            st.write(f"📊 **CVaR (Expected Shortfall): `{CVaR_empirico*100:.2f}%`** → ¿Qué podemos esperar si se rompe el VaR? Para saberlo es útil hacer uso del VaR Condicional (CVaR), que es el promedio de los retornos más allá de esa barrera. Para este caso el CVaR es {CVaR_empirico*100:.5f}%.")
            st.write(f"Observaciones en la cola: {len(cola)}")
    
            # 4️⃣ Gráfico con sombreado
            fig, ax = plt.subplots(figsize=(10, 6))
    
            # Histograma
            counts, bins, patches = ax.hist(
                returns,
                bins=50,
                density=True,
                edgecolor='black',
                alpha=0.6,
                label="Distribución empírica"
            )
    
            # Sombrear cola (roja)
            for patch, bin_left in zip(patches, bins[:-1]):
                if bin_left <= VaR_empirico:
                    patch.set_facecolor('red')
                    patch.set_alpha(0.4)
    
            # Líneas de referencia
            ax.axvline(VaR_empirico, color="red", linestyle="--", linewidth=2, label=f"VaR {conf*100:.1f}% = {VaR_empirico*100:.2f}%")
            ax.axvline(CVaR_empirico, color="darkred", linestyle=":", linewidth=2, label=f"CVaR = {CVaR_empirico*100:.2f}%")
            ax.axvline(mean_emp, color="blue", linestyle="--", linewidth=2, label=f"Media = {mean_emp*100:.2f}%")
    
            # Títulos y leyenda
            ax.set_title(f"Distribución de Retornos y Value at Risk ({conf*100:.1f}%) - {ticker}")
            ax.set_xlabel("Retornos")
            ax.set_ylabel("Densidad")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
    
        else:
            st.warning("⚠️ Ingrese un valor entre 0 y 1 (por ejemplo, 0.95).")
    
    except ValueError:
        st.warning("⚠️ Ingrese un número válido (por ejemplo, 0.95).")

    # =====================================================
    # 📈 Simulación de Montecarlo
    # =====================================================
    st.subheader("🎲 Simulación de Montecarlo")
    
    # --- Gráfico combinado: histórico + simulaciones ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # 1. Gráfico histórico
    ax1.plot(data.index, data['Close'], label=f"Precio histórico {ticker}", color="blue", linewidth=2)
    
    # 2. Determinar frecuencia según intervalo elegido
    if interval == "1d":
        freq = "B"
        N = 252
    elif interval == "1wk":
        freq = "W"
        N = 52
    elif interval == "1mo":
        freq = "M"
        N = 12
    else:
        freq = "B"
        N = 252
    
    # 3. Simulaciones Monte Carlo
    n_simulaciones = 500
    simulaciones = np.zeros((N+1, n_simulaciones))   # incluye S0
    S0 = float(data['Close'].iloc[-1])               # ✅ escalar
    mu = mean_return
    dt = Dt                                          # ya definido según el intervalo
    
    for j in range(n_simulaciones):
        prices = [S0]
        for t in range(1, N+1):
            z = np.random.normal()
            St = float(prices[-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*z))
            prices.append(St)
        simulaciones[:, j] = np.array(prices).flatten()
    
        # Fechas futuras (arranca después del último dato real)
        future_dates = pd.date_range(start=data.index[-1], periods=N+1, freq=freq)[1:]
        ax1.plot(future_dates, prices[1:], linewidth=1, alpha=0.2, color="orange")
    
    # 4. Línea promedio de todas las simulaciones
    mean_path = simulaciones.mean(axis=1)[1:]
    ax1.plot(future_dates, mean_path, color="black", linewidth=2, label="Media de simulaciones")
    
    # 5. Último precio como referencia
    ax1.scatter(data.index[-1], S0, color="black", zorder=5, label=f"Último precio: {S0:.2f}")
    
    # 6. Strike
    ax1.axhline(y=K, color="red", linestyle="--", linewidth=1.5, label=f"Strike = {K}")
    
    # 7. Formato del gráfico
    ax1.set_title(f"Trayectoria histórica y simulaciones Monte Carlo - {ticker}")
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel("Precio")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)
    
    
    # =====================================================
    # 📊 Histograma de precios finales
    # =====================================================
    st.markdown("##### 🎲 Histograma de Simulación de Precios")
    
    final_prices = simulaciones[-1, :]  # últimos precios de cada simulación
    
    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.hist(final_prices, bins=10, edgecolor='black', alpha=0.7)
    
    # Línea en el precio inicial
    ax2.axvline(S0, color="blue", linestyle="--", linewidth=2, label=f"Precio inicial: {S0:.2f}")
    
    # Línea en el strike
    ax2.axvline(K, color="red", linestyle="--", linewidth=2, label=f"Strike = {K}")
    
    # Estadísticas
    mean_final = np.mean(final_prices)
    std_final  = np.std(final_prices)
    ax2.axvline(mean_final, color="green", linestyle="--", linewidth=2, label=f"Media final: {mean_final:.2f}")
    
    ax2.set_title(f"Distribución de precios finales - Monte Carlo ({ticker})")
    ax2.set_xlabel("Precio al vencimiento (1 año)")
    ax2.set_ylabel("Frecuencia")
    ax2.legend()
    st.pyplot(fig2)
    
    # ==========================
    # 📊 Resultados Monte Carlo
    # ==========================
    
    st.markdown("##### 🎲 Resultados de la Simulación Montecarlo")
    
    # 1️⃣ Cálculos básicos
    precio_inicial = S0
    precio_medio = mean_final
    desvio_precios = std_final
    
    # 2️⃣ Escenarios positivos / negativos
    positivos = np.sum(final_prices > precio_inicial)
    negativos = np.sum(final_prices < precio_inicial)
    total = len(final_prices)
    
    pct_positivos = (positivos / total) * 100
    pct_negativos = (negativos / total) * 100
    
    # 3️⃣ Mostrar métricas
    col1, col2, col3 = st.columns(3)
    
    col1.metric("💵 Precio Inicial", f"${precio_inicial:,.2f}")
    col2.metric(
        "📈 Precio Medio Simulado (1 año)",
        f"${precio_medio:,.2f}",
        delta=f"{(precio_medio - precio_inicial)/precio_inicial*100:.2f}%",
    )
    col3.metric("📊 Desvío Estándar", f"${desvio_precios:,.2f}")
    
    # Nueva métrica de escenarios (como texto)
    st.markdown("#### ⚖️ Escenarios Positivos / Negativos")
    
    st.markdown(f"""
    **📈 Escenarios Positivos:** `{pct_positivos:.1f}%`  
    **📉 Escenarios Negativos:** `{pct_negativos:.1f}%`  
    
    > 🔍 *Indica el porcentaje de simulaciones donde el precio final supera o cae respecto al inicial.*
    """)

           
    # ==========================
    # 1. Preparar datos
    # ==========================
    # returns ya lo tenés calculado antes
    # Calcular cambio porcentual en volumen
    
    st.subheader("📊Volumen")
    
    # Verificar que existan los datos
    if "data" in locals() and not data.empty:
    
        # --- 1. Preparar datos ---
        volumen = data["Volume"].squeeze()
        vol_media = volumen.mean()
        vol_ratio = volumen / vol_media if vol_media != 0 else pd.Series(np.nan, index=volumen.index)
    
        df_aux = pd.DataFrame({
            "Return": returns,
            "Vol_Ratio": vol_ratio
        }).dropna()
    
        # --- 2. Estadísticas ---
        vol_desvio = volumen.std()
        vol_max = volumen.max()
        vol_min = volumen.min()
    
        # Mostrar métricas en columnas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📈 Promedio Volumen", f"{vol_media/1000000:,.3f}M")
        col2.metric("📉 Desvío", f"{vol_desvio/1000000:,.3f}M")
        col3.metric("🔝 Máximo", f"{vol_max/1000000:,.3f}M")
        col4.metric("🔻 Mínimo", f"{vol_min/1000000:,.3f}M")
    
        # --- 3. Gráfico de dispersión ---
        st.markdown("##### 🔹 Dispersión: Retornos vs Proporción del Volumen Promedio")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(df_aux["Return"], df_aux["Vol_Ratio"], alpha=0.5, color="purple")
        ax1.set_title(f"Relación entre Retornos y Volumen Promedio - {ticker}")
        ax1.set_xlabel("Retornos")
        ax1.set_ylabel("Volumen observado / Promedio")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
    
        # --- 4. Gráfico de barras del volumen ---
        st.markdown("##### 🔹 Volumen de negociación a lo largo del tiempo")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.bar(volumen.index, volumen.values, width=1, color="purple", alpha=0.7)
        ax2.set_title(f"Volumen de negociación - {ticker}")
        ax2.set_xlabel("Fecha")
        ax2.set_ylabel("Volumen")
        ax2.grid(axis="y", alpha=0.5)
        # Ajustar etiquetas de fechas (si hay muchas)
        if len(volumen) > 10:
            ax2.set_xticks(volumen.index[::len(volumen)//10])
            ax2.set_xticklabels(volumen.index[::len(volumen)//10].strftime("%Y-%m-%d"), rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
    
    else:
        st.warning("⚠️ Aún no se cargaron datos. Presioná *Calcular* primero.")
    
    # ==========================
    # 8. Diagrama de Secuencias (Lag Chains) - CORREGIDO con P_negativo directo
    # ==========================
    st.subheader("🔄 Análisis de Secuencias de Momentum (Lag Chains)")
    st.write("El objetivo es analizar la probabilidad del próximo movimiento de la acción a partir de secuencias históricas.")

    # DataFrame base (usa 'returns' que ya tenés)
    momentum_data = pd.DataFrame({
        'Return': returns,
        'Return_class': np.where(returns > 0, 'Positivo',
                                 np.where(returns < 0, 'Negativo', 'Sin variación'))
    })

    def calcular_probabilidades_lags(df, max_lag=5):
        resultados = []
        for lag in range(1, max_lag + 1):
            df_lag = df.copy()
            # crear columnas Lag1..LagN (Lag1 = día anterior, etc.)
            for i in range(1, lag + 1):
                df_lag[f'Lag{i}'] = df_lag['Return_class'].shift(i)
            df_lag.dropna(inplace=True)

            lag_cols = [f'Lag{i}' for i in range(lag, 0, -1)]  # orden LagN ... Lag1 para clave legible
            grupos = df_lag.groupby(lag_cols)

            for secuencia, grupo in grupos:
                # convertir secuencia a lista para poder inspeccionarla
                if isinstance(secuencia, str):
                    seq_list = [secuencia]
                else:
                    seq_list = list(secuencia)

                # Omitir secuencias que incluyan 'Sin variación' en los días previos (no queremos mostrarlas)
                if any(s == 'Sin variación' for s in seq_list):
                    continue

                # Conteos directos para el día siguiente (la columna 'Return_class' del grupo corresponde al "hoy" dado los lags)
                total = len(grupo)
                cnt_pos = (grupo['Return_class'] == 'Positivo').sum()
                cnt_neg = (grupo['Return_class'] == 'Negativo').sum()
                cnt_neu = (grupo['Return_class'] == 'Sin variación').sum()

                # Probabilidades directas
                p_pos = cnt_pos / total if total > 0 else 0.0
                p_neg = cnt_neg / total if total > 0 else 0.0
                p_neu = cnt_neu / total if total > 0 else 0.0

                # Retorno esperado (del "hoy" dado la secuencia) y rango 5%-95% (expresados en %)
                retorno_esperado = grupo['Return'].mean() * 100
                ret_min, ret_max = grupo['Return'].quantile([0.05, 0.95]).values * 100
                rango_str = f"{retorno_esperado:.2f}% ({ret_min:.2f}% ; {ret_max:.2f}%)"

                # Secuencia como emojis (solo Positivo/Negativo, porque ya descartamos secuencias con neutros)
                emojis = ''.join('🟢' if s == 'Positivo' else '🔴' for s in seq_list)

                resultados.append({
                    '🧩 Secuencia': emojis,
                    'Días previos': lag,
                    'Probabilidad subida (%)': f"{p_pos*100:.2f}%",
                    'Probabilidad caída (%)': f"{p_neg*100:.2f}%",
                    'Probabilidad neutra (%)': f"{p_neu*100:.2f}%",
                    'Retorno esperado [5%-95%]': rango_str,
                    'Observaciones (n)': total
                })

        return pd.DataFrame(resultados)

    # Cálculo (lags 1 a 5)
    resultados = calcular_probabilidades_lags(momentum_data, max_lag=5)

    # Orden: por Días previos ascendente y por Probabilidad subida descendente
    if not resultados.empty:
        resultados['Prob subida (num)'] = resultados['Probabilidad subida (%)'].str.replace('%', '').astype(float)
        resultados = resultados.sort_values(by=['Días previos', 'Prob subida (num)'], ascending=[True, False])
        resultados = resultados.drop(columns=['Prob subida (num)'])

    # Mostrar en Streamlit (columnas seleccionadas)
    mostrar_cols = [
        'Días previos',
        '🧩 Secuencia',
        'Probabilidad subida (%)',
        'Probabilidad caída (%)',
        'Probabilidad neutra (%)',
        'Retorno esperado [5%-95%]',
        'Observaciones (n)'
    ]
    st.dataframe(resultados[mostrar_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

    # ==========================
    # 9. Últimos 5 movimientos (contexto actual)
    # ==========================
    ultimos = returns.tail(5).tolist()

    def ret_to_emoji(x):
        if x > 0:
            return "🟢"
        elif x < 0:
            return "🔴"
        else:
            return "⚪"

    secuencia_actual = ''.join(ret_to_emoji(x) for x in ultimos)

    st.markdown("##### 📈 Últimos 5 movimientos del activo")
    st.markdown(
        f"Secuencia reciente: **{secuencia_actual}**  "
        f"*(más antiguo → más reciente)*"
    )

    st.markdown("""
    ##### 🧠 Nota sobre las probabilidades
    - Las probabilidades `subida`, `caída` y `neutra` se calculan directamente a partir de la frecuencia observada del **día siguiente** tras cada secuencia histórica.
    - No se asume `P(caída) = 1 - P(subida)`: la probabilidad `neutra` también forma parte del universo y se muestra explícitamente.
    - `Observaciones (n)` indica cuántos casos históricos contribuyeron a cada estimación (útil para evaluar robustez).
    
    ##### 🧠 Cómo leer el resultado
    - Cada **secuencia de emojis** representa los últimos días observados (🟢 = positivo, 🔴 = negativo).  
    - Las secuencias que incluyen días sin variación **no se muestran**, aunque sí se consideran en los cálculos.  
    - **Días previos** indica cuántos días consecutivos se analizaron antes del movimiento actual.  
    - **Probabilidad +** muestra la chance de que el próximo día también sea positivo.  
    - **Retorno esperado [5%-95%]** combina el retorno promedio con su rango de confianza.  
    - La **Interpretación** resume el patrón observado en lenguaje natural.
    """)

        # --- Selector de Estrategias ---
    st.subheader("🔎 Selector de Estrategias con Opciones")
    
    # ==========================
    # 1. Crear DataFrame
    # ==========================
    df_estrategias = pd.DataFrame({
        "Objetivo": [
            "Cobertura","Acompañar tendencia","Volatilidad",
            "Cobertura","Acompañar tendencia","Volatilidad",
            "Cobertura","Acompañar tendencia","Volatilidad",
            "Cobertura","Acompañar tendencia","Volatilidad",
            "Cobertura","Acompañar tendencia","Volatilidad",
            "Cobertura","Acompañar tendencia","Volatilidad"
        ],
        "Tendencia": [
            "Alcista","Alcista","Alcista",
            "Bajista","Bajista","Bajista",
            "Lateral","Lateral","Lateral",
            "Alcista","Alcista","Alcista",
            "Bajista","Bajista","Bajista",
            "Lateral","Lateral","Lateral"
        ],
        "Volatilidad": [
            "Alta","Alta","Alta",
            "Alta","Alta","Alta",
            "Alta","Alta","Alta",
            "Baja","Baja","Baja",
            "Baja","Baja","Baja",
            "Baja","Baja","Baja"
        ],
        "Estrategia": [
            "Compra CALL",
            "Bull spread con calls",
            "Cono comprado",
            "Compra PUT",
            "Bear spread con puts",
            "Cuna comprada",
            "Collar",
            "Iron condor vendido",
            "Iron condor vendido",
            "Venta PUT",
            "Ratio call spread",
            "Mariposa vendida",
            "Venta CALL",
            "Venta sintético",
            "Cuna vendida",
            "Venta CALL, compra tasa",
            "Ratio put spread",
            "Mariposa comprada"
        ]
    })

    # ==========================
    # 2. Entradas del usuario
    # ==========================
    st.write("Seleccione las condiciones de mercado y su objetivo:")
    
    obj = st.selectbox("🎯 Objetivo:", df_estrategias["Objetivo"].unique())
    tend = st.selectbox("📈 Tendencia:", df_estrategias["Tendencia"].unique())
    vol = st.selectbox("🌪️ Volatilidad:", df_estrategias["Volatilidad"].unique())

    # ==========================
    # 3. Botón y resultado
    # ==========================
    if st.button("Buscar Estrategia"):
        resultado = df_estrategias[
            (df_estrategias["Objetivo"].str.lower() == obj.lower()) &
            (df_estrategias["Tendencia"].str.lower() == tend.lower()) &
            (df_estrategias["Volatilidad"].str.lower() == vol.lower())
        ]

        if not resultado.empty:
            recommended_strategy = resultado["Estrategia"].values[0]
            st.success(f"✅ Estrategia recomendada: **{recommended_strategy}**")

            # --- Estrategia específica: Compra CALL ---
            if recommended_strategy == "Compra CALL":
                st.write(
                    "Comprar un **call (opción de compra)** te da el **derecho, pero no la obligación**, "
                    "de comprar el activo subyacente a un precio determinado (strike) hasta la fecha de vencimiento."
                )

                # ==========================
                # Black-Scholes Call
                # ==========================
                K = S * 1.02  # Strike del call (ligeramente OTM)

                def black_scholes_call(S, K, T, r, sigma):
                    """Calcula el precio de un call europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                    return call_price

                # ==========================
                # Cálculo prima
                # ==========================
                prima = black_scholes_call(S, K, T, r, sigma)

                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.7, S * 1.3, 100)
                payoff_call = np.maximum(S_range - K, 0) - prima  # beneficio neto

                # Punto de equilibrio
                breakeven = K + prima

                # ==========================
                # Ejemplo y descripción
                # ==========================
                st.markdown(
                    f"""
                    **Ejemplo práctico**
                    
                    Compra de un call de `{ticker}` con base **`{K:.2f}`**, vencimiento en **`{T*12:.0f}` meses**,  
                    y una prima de **`${prima:.2f}`** tendría el siguiente resultado al vencimiento:
                    """
                )

                # ==========================
                # Gráfico de payoff
                # ==========================
                plt.figure(figsize=(10, 6))
                plt.plot(S_range, payoff_call, label="Payoff Call", color="blue", linewidth=2)

                # Líneas de referencia
                plt.axhline(0, color="black", linestyle="--", linewidth=1)
                plt.axvline(K, color="red", linestyle="--", linewidth=1, label=f"Strike = {K:.2f}")
                plt.axvline(S, color="green", linestyle="--", linewidth=1, label=f"S actual = {S:.2f}")
                plt.axvline(breakeven, color="orange", linestyle="--", linewidth=1.5, label=f"Breakeven = {breakeven:.2f}")

                # Estética
                plt.title("Payoff de un Call Europeo al Vencimiento")
                plt.xlabel("Precio del subyacente al vencimiento")
                plt.ylabel("Beneficio / Pérdida")
                plt.legend()
                plt.grid(alpha=0.3)

                st.pyplot(plt)
                plt.close()

                # ==========================
                # Información resumen
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima del call: `${prima:.2f}`** 
                - **Costo total: `${prima:.2f}`** 
                - **Pérdida máxima: `${prima:.2f}`** (si **`S < {K:.2f}`**) 
                - **Ganancia máxima:** Ilimitada 🚀 
                - **Breakeven: `{breakeven:.2f}`**  →  Variación necesaria del subyacente: **{(breakeven/S - 1)*100:.2f}%**
                """)

            # --- Estrategia específica: Bull spread con calls ---
            elif recommended_strategy == "Bull spread con calls":         
                st.write("""
                Esta estrategia se basa en **comprar un call** con una determinada base (strike) y **vender un call**
                con una base mayor a la comprada.  
                👉 La venta financia parcialmente la compra, generando una **posición alcista con pérdidas y ganancias limitadas.**
                """)
            
                # ==========================
                # Strikes
                # ==========================
                K_compra = S * 0.95   # Strike del call comprado (más bajo)
                K_venta = S * 1.05    # Strike del call vendido (más alto)
            
                # ==========================
                # Función Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                    return call_price
            
                # ==========================
                # Primas y costo total
                # ==========================
                prima_call_compra = black_scholes_call(S, K_compra, T, r, sigma) * 1.01
                prima_call_venta = black_scholes_call(S, K_venta, T, r, sigma) * 0.99
                costo_total = prima_call_compra - prima_call_venta  # débito neto
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.7, S * 1.3, 200)
                payoff_call_compra = np.maximum(S_range - K_compra, 0) - prima_call_compra
                payoff_call_venta = -np.maximum(S_range - K_venta, 0) + prima_call_venta
                payoff_bull_spread = payoff_call_compra + payoff_call_venta
            
                # ==========================
                # Breakeven y métricas
                # ==========================
                BE = K_compra + costo_total
                ganancia_max = (K_venta - K_compra) - costo_total
                perdida_max = costo_total
            
                # ==========================
                # Ejemplo
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**
                
                Venta de un call de `{ticker}` base **`{K_venta:.2f}`**  
                con una prima de **`${prima_call_venta:.2f}`**, y compra de un call base **`{K_compra:.2f}`**  
                con prima **`${prima_call_compra:.2f}`**, ambos con vencimiento en **`{T*12:.0f}` meses**, tendría el siguiente resultado al vencimiento:
                """)

            
                # ==========================
                # Gráfico del payoff
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff_bull_spread, label="Bull Spread (Calls)", color="green", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K_compra, color="blue", linestyle="--", linewidth=1, label=f"Strike Call comprado = {K_compra:.2f}")
                ax.axvline(K_venta, color="red", linestyle="--", linewidth=1, label=f"Strike Call vendido = {K_venta:.2f}")
                ax.axvline(BE, color="orange", linestyle="--", linewidth=1.5, label=f"Breakeven = {BE:.2f}")
            
                ax.set_title("Estrategia Bull Spread con Calls")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
                # ==========================
                # Resumen numérico
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima call comprado: `${prima_call_compra:.2f}`**
                - **Prima call vendido: `${prima_call_venta:.2f}`**
                - **Costo neto total (prima total): `${costo_total:.2f}`**
                - **Breakeven: `{BE:.2f}`**  → Variación necesaria del subyacente: **{(BE/S - 1)*100:.2f}%**
                - **Ganancia Máxima: `${ganancia_max:.2f}`**
                - **Pérdida Máxima: `${perdida_max:.2f}`**
                """)
            
                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Cono comprado":
                st.write("""
                Esta estrategia consiste en **comprar un call y un put sobre el mismo subyacente**, 
                con **la misma base y mismo vencimiento**.  
                Permite beneficiarse ante **movimientos fuertes del precio**, ya sea al alza o a la baja, 
                dado que se tienen derechos en ambas direcciones.
                """)
            
                # ==========================
                # Parámetros
                # ==========================
                K = S * 1.02  # Strike común para call y put
            
                # ==========================
                # Funciones Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    """Calcula el precio de un call europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            
                def black_scholes_put(S, K, T, r, sigma):
                    """Calcula el precio de un put europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
                # ==========================
                # Cálculo de primas
                # ==========================
                call_price = black_scholes_call(S, K, T, r, sigma)
                put_price = black_scholes_put(S, K, T, r, sigma)
                prima_total = call_price + put_price
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.5, S * 1.5, 200)
                payoff_straddle = np.maximum(S_range - K, 0) + np.maximum(K - S_range, 0) - prima_total
            
                # ==========================
                # Breakeven points
                # ==========================
                BE_lower = K - prima_total
                BE_upper = K + prima_total
            
                # ==========================
                # Ejemplo descriptivo (formato Solución 1)
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
                
                Compra de un **call** de **{ticker}** a **`${call_price:.2f}`** y un **put** a **`${put_price:.2f}`**,  
                ambos con base **`{K:.2f}`** y vencimiento en **`{T*12:.0f}` meses**,  
                tendría el siguiente resultado:
                """)

            
                # ==========================
                # Gráfico del payoff
                # ==========================
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(S_range, payoff_straddle, label="Cono Comprado (Long Straddle)", color="blue", linewidth=2)
            
                # Líneas de referencia
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K, color="red", linestyle="--", linewidth=1, label=f"Strike = {K:.2f}")
                ax.axvline(BE_lower, color="orange", linestyle="--", linewidth=1.5, label=f"BE inferior = {BE_lower:.2f}")
                ax.axvline(BE_upper, color="orange", linestyle="--", linewidth=1.5, label=f"BE superior = {BE_upper:.2f}")
            
                # Estética
                ax.set_title("📈 Estrategia de Cono Comprado (Long Straddle)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Información resumen
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima call: `${call_price:.2f}`**
                - **Prima put: `${put_price:.2f}`**
                - **Costo total (prima total): `${prima_total:.2f}`**
                - **Pérdida máxima: `${prima_total:.2f}`** (si **`S ≈ {K:.2f}`**)
                - **Ganancia máxima:** Ilimitada 🚀
                - **Breakeven inferior: `{BE_lower:.2f}`**  → Variación necesaria del subyacente: **{(BE_lower/S - 1)*100:.2f}%**
                - **Breakeven superior: `{BE_upper:.2f}`**  → Variación necesaria del subyacente: **{(BE_upper/S - 1)*100:.2f}%**
                """)

            elif recommended_strategy == "Compra PUT":
                st.write("""
                Comprar un **put** (opción de venta) te da el derecho pero no la obligación de vender el activo subyacente del contrato a un precio determinado hasta la fecha de vencimiento.
                """)
            
                # ==========================
                # Black-Scholes Put
                # ==========================
                K = S * 0.98
            
                def black_scholes_put(S, K, T, r, sigma):
                    """Calcula el precio de un put europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                    return put_price
            
                # ==========================
                # Cálculo prima
                # ==========================
                prima = black_scholes_put(S, K, T, r, sigma)
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.7, S * 1.3, 200)
                payoff_put = np.maximum(K - S_range, 0) - prima
            
                # Punto de equilibrio
                breakeven = K - prima
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Compra de un **put** de `{ticker}` con base **`${K:.2f}`**, vencimiento en **`{T*12:.0f}` meses**,  
                y una prima de **`${prima:.2f}`**, tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(S_range, payoff_put, label="Compra de Put", color="blue", linewidth=2)
            
                # Líneas de referencia
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K, color="red", linestyle="--", linewidth=1, label=f"Strike = {K:.2f}")
                ax.axvline(S, color="green", linestyle="--", linewidth=1, label=f"S = {S:.2f}")
                ax.axvline(breakeven, color="orange", linestyle="--", linewidth=1.5, label=f"Breakeven = {breakeven:.2f}")
            
                # Estética
                ax.set_title("Payoff de un Put Europeo al Vencimiento")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Información resumen
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**
                
                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima put: `${prima:.2f}`**
                - **Costo total de la estrategia: `${prima:.2f}`**
                - **Pérdida máxima: `${prima:.2f}`** (si **`S > {K:.2f}`**)
                - **Ganancia máxima:** Ilimitada 🚀
                - **Breakeven: `{breakeven:.2f}`** → Variación necesaria del subyacente: **{(breakeven/S - 1)*100:.2f}%**
                """)

            elif recommended_strategy == "Bear spread con puts":
                st.write("""
                Esta estrategia se basa en **comprar un put** con una determinada base y **vender un put** de una base menor.  
                Así, la venta va a financiar una parte de la compra y el inversor va a tener una **estrategia bajista** cuya pérdida va a estar limitada por la parte de la compra que la venta no logra financiar 
                y la ganancia estará limitada a que el subyacente alcance la base comprada**.  
                """)
            
                # ==========================
                # Parámetros
                # ==========================
                K_compra = S * 1.05   # Strike del put comprado (más alto)
                K_venta = S * 0.95    # Strike del put vendido (más bajo)
            
                # ==========================
                # Función Black-Scholes
                # ==========================
                def black_scholes_put(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                    return put_price
            
                # ==========================
                # Primas
                # ==========================
                prima_put_compra = black_scholes_put(S, K_compra, T, r, sigma) * 1.02
                prima_put_venta = black_scholes_put(S, K_venta, T, r, sigma) * 0.98
                costo_total = prima_put_compra - prima_put_venta  # costo neto (débito)
            
                # ==========================
                # Payoff
                # ==========================
                S_range = np.linspace(S * 0.7, S * 1.3, 200)
                payoff_put_compra = np.maximum(K_compra - S_range, 0) - prima_put_compra
                payoff_put_venta = -np.maximum(K_venta - S_range, 0) + prima_put_venta
                payoff_bear_spread = payoff_put_compra + payoff_put_venta
            
                # ==========================
                # Breakeven y resultados
                # ==========================
                BE = K_compra - costo_total
                ganancia_max = (K_compra - K_venta) - costo_total
                perdida_max = costo_total
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Venta de un **put** de `{ticker}` con base **`{K_venta:.2f}`** a **`${prima_put_venta:.2f}`**  
                y compra de un **put** con base **`{K_compra:.2f}`** a **`${prima_put_compra:.2f}`**  
                con vencimiento en **`{T*12:.0f}` meses**, tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(S_range, payoff_bear_spread, label="Bear Spread (Puts)", color="darkred", linewidth=2)
            
                # Líneas de referencia
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K_compra, color="blue", linestyle="--", linewidth=1, label=f"Strike Put comprado = {K_compra:.2f}")
                ax.axvline(K_venta, color="red", linestyle="--", linewidth=1, label=f"Strike Put vendido = {K_venta:.2f}")
                ax.axvline(BE, color="orange", linestyle="--", linewidth=1.5, label=f"Breakeven = {BE:.2f}")
            
                ax.set_title("Estrategia Bear Spread con Puts")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Resumen numérico
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima put comprado: `${prima_put_compra:.2f}`**
                - **Prima put vendido: `${prima_put_venta:.2f}`**
                - **Costo neto (prima total): `${costo_total:.2f}`**
                - **Breakeven: `{BE:.2f}`** → Variación necesaria del subyacente: **{(BE/S - 1)*100:.2f}%**
                - **Ganancia máxima: `${ganancia_max:.2f}`**
                - **Pérdida máxima: `${perdida_max:.2f}`**
                """)
        
                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Cuna comprada":
                st.write("""
                Esta estrategia consiste en comprar un **call OTM** sobre un subyacente con una base superior al precio actual y comprar, a su vez, un **put OTM** sobre el mismo subyacente con una base inferior al precio actual.  
                Esto nos permite estar bien posicionados si el activo baja considerablemente, dado que tendremos el derecho a venderlo a un precio superior.  
                A su vez, también nos cubre al alza, ya que si el precio sube significativamente, contamos con una opción de compra.
                """)
            
                # ==========================
                # Parámetros y strikes
                # ==========================
                K_call = S * 1.05
                K_put = S * 0.96
            
                # ==========================
                # Funciones Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    """Calcula el precio de un call europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            
                def black_scholes_put(S, K, T, r, sigma):
                    """Calcula el precio de un put europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
                # ==========================
                # Primas pagadas
                # ==========================
                prima_put = black_scholes_put(S, K_put, T, r, sigma)
                prima_call = black_scholes_call(S, K_call, T, r, sigma)
                prima_total = prima_put + prima_call  # costo total
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.6, S * 1.4, 200)
                payoff_strangle = np.maximum(K_put - S_range, 0) + np.maximum(S_range - K_call, 0) - prima_total
            
                # ==========================
                # Breakeven points
                # ==========================
                BE_lower = K_put - prima_total
                BE_upper = K_call + prima_total
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Compra de un **put** de `{ticker}` con base **`{K_put:.2f}`** a **`${prima_put:.2f}`**,  
                y un **call** de `{ticker}` con base **`{K_call:.2f}`** a **`${prima_call:.2f}`**,  
                ambos con vencimiento en **`{T*12:.0f}` meses**,  
                tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff_strangle, label="Cuna Comprada (Long Strangle)", color="blue", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K_put, color="red", linestyle="--", linewidth=1, label=f"Strike Put = {K_put:.2f}")
                ax.axvline(K_call, color="green", linestyle="--", linewidth=1, label=f"Strike Call = {K_call:.2f}")
                ax.axvline(BE_lower, color="orange", linestyle="--", linewidth=1.5, label=f"BE inferior = {BE_lower:.2f}")
                ax.axvline(BE_upper, color="orange", linestyle="--", linewidth=1.5, label=f"BE superior = {BE_upper:.2f}")
                ax.set_title("Estrategia de Cuna Comprada (Long Strangle)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Resultados informativos
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima put comprada: `${prima_put:.2f}`**
                - **Prima call comprada: `${prima_call:.2f}`**
                - **Pérdida máxima: `${prima_total:.2f}`** (si **`{K_put:.2f} < S < {K_call:.2f}`**)
                - **Ganancia máxima:** Ilimitada 🚀
                - **Breakeven inferior: `{BE_lower:.2f}`** → Variación necesaria del subyacente: **{(BE_lower/S - 1)*100:.2f}%**
                - **Breakeven superior: `{BE_upper:.2f}`** → Variación necesaria del subyacente: **{(BE_upper/S - 1)*100:.2f}%**
                """)

            elif recommended_strategy == "Collar":    
                st.write("""
                Esta estrategia se basa en la **compra de acciones del subyacente**, la **compra de puts OTM (o ATM)** y la **venta de calls OTM**.  
                De esta manera, la prima cobrada por los calls financia la compra del activo y de los puts.  
                Si la volatilidad juega en contra del precio, el **put** nos protege;  
                si el precio sube significativamente, el ejercicio del **call** estará cubierto con las acciones compradas.
                """)
            
                # ==========================
                # Strikes
                # ==========================
                K_put = S * 0.95   # Protección
                K_call = S * 1.05  # Techo
            
                # ==========================
                # Funciones Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    """Calcula el precio de un call europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            
                def black_scholes_put(S, K, T, r, sigma):
                    """Calcula el precio de un put europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
                # ==========================
                # Primas y costo total
                # ==========================
                prima_put = black_scholes_put(S, K_put, T, r, sigma)
                prima_call = black_scholes_call(S, K_call, T, r, sigma)
                costo_total = S + prima_put - prima_call
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.6, S * 1.4, 200)
            
                payoff_accion = S_range - S
                payoff_put = np.maximum(K_put - S_range, 0) - prima_put
                payoff_call = -np.maximum(S_range - K_call, 0) + prima_call
                payoff_total = payoff_accion + payoff_put + payoff_call
            
                # ==========================
                # Breakevens
                # ==========================
                BE_lower = S - (prima_call - prima_put) - (S - K_put)
                BE_upper = K_call + (prima_put - prima_call)
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Compra de `{ticker}` a **`${S:.2f}`**,  
                venta de un **call** de `{ticker}` con base **`{K_call:.2f}`** a **`{T*12:.0f}` meses** por **`${prima_call:.2f}`**,  
                y compra de un **put** de `{ticker}` con base **`{K_put:.2f}`** a **`{T*12:.0f}` meses** por **`${prima_put:.2f}`**,  
                tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff_total, label="Payoff Collar", color="blue", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K_put, color="red", linestyle="--", linewidth=1, label=f"Strike Put = {K_put:.2f}")
                ax.axvline(K_call, color="green", linestyle="--", linewidth=1, label=f"Strike Call = {K_call:.2f}")
                ax.set_title("Estrategia Collar (Acción + Put comprado + Call vendido)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Resultados informativos
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima put: `${prima_put:.2f}`**
                - **Prima call: `${prima_call:.2f}`**
                - **Costo total neto: `${costo_total:.2f}`**
                - **Breakeven inferior: `{K_put:.2f}`** → Variación necesaria del subyacente: **{(K_put/S - 1)*100:.2f}%** (pérdida limitada más allá de este punto)
                - **Breakeven superior: `{K_call:.2f}`** → Variación necesaria del subyacente: **{(K_call/S - 1)*100:.2f}%** (ganancia limitada más allá de este punto)
                """)

            elif recommended_strategy == "Iron condor vendido":
                st.write("""
                Esta estrategia consiste en **vender un call** de determinada base OTM (o ATM) y **comprar otro call** de base superior.  
                A su vez, implica **vender un put** de una base inferior (OTM o ATM) y **comprar otro put** con una base aún menor.  
                En resumen, es la suma de un **bear spread con calls** y un **bull spread con puts**.  
                """)
            
                # ==========================
                # Strikes
                # ==========================
                K1 = S * 0.90  # Put comprado (protección inferior)
                K2 = S * 0.95  # Put vendido
                K3 = S * 1.05  # Call vendido
                K4 = S * 1.10  # Call comprado (protección superior)
            
                # ==========================
                # Funciones Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            
                def black_scholes_put(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
                # ==========================
                # Primas individuales
                # ==========================
                prima_put_long = black_scholes_put(S, K1, T, r, sigma)
                prima_put_short = black_scholes_put(S, K2, T, r, sigma)
                prima_call_short = black_scholes_call(S, K3, T, r, sigma)
                prima_call_long = black_scholes_call(S, K4, T, r, sigma)
            
                # ==========================
                # Crédito neto recibido
                # ==========================
                credito_neto = (prima_put_short - prima_put_long) + (prima_call_short - prima_call_long)
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.7, S * 1.3, 300)
            
                payoff_put_long = np.maximum(K1 - S_range, 0) - prima_put_long
                payoff_put_short = -np.maximum(K2 - S_range, 0) + prima_put_short
                payoff_call_short = -np.maximum(S_range - K3, 0) + prima_call_short
                payoff_call_long = np.maximum(S_range - K4, 0) - prima_call_long
            
                payoff_condor = payoff_put_long + payoff_put_short + payoff_call_short + payoff_call_long
            
                # ==========================
                # Breakevens
                # ==========================
                BE_lower = K2 - credito_neto
                BE_upper = K3 + credito_neto
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                1️⃣ Venta de un **put** de `{ticker}` con base **`{K2:.2f}`** a **`{K2:.2f}`**.
                2️⃣ Compra de un **put** de `{ticker}` con base **`${K1:.2f}`** a **`{K1:.2f}`**. 
                3️⃣ Venta de un **call** de `{ticker}` con base **`{K3:.2f}`** a **`{K3:.2f}`**.
                4️⃣ Compra de un **call** de `{ticker}` con base **`{K4:.2f}`** a **`{K4:.2f}`**.
                Todos con vencimiento a **`{T*12:.0f}` meses**  
                Resultaría en el siguiente payoff:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff_condor, label="Iron Condor Vendida", color="red", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K1, color="gray", linestyle="--", linewidth=1, label=f"K1 = {K1:.2f}")
                ax.axvline(K2, color="blue", linestyle="--", linewidth=1, label=f"K2 = {K2:.2f}")
                ax.axvline(K3, color="green", linestyle="--", linewidth=1, label=f"K3 = {K3:.2f}")
                ax.axvline(K4, color="gray", linestyle="--", linewidth=1, label=f"K4 = {K4:.2f}")
                ax.axvline(BE_lower, color="orange", linestyle="--", linewidth=1.5, label=f"BE inferior = {BE_lower:.2f}")
                ax.axvline(BE_upper, color="orange", linestyle="--", linewidth=1.5, label=f"BE superior = {BE_upper:.2f}")
                ax.set_title("Estrategia Iron Condor Vendida (Short Iron Condor)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Información final
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima put comprado (`{K1:.2f}`): `${prima_put_long:.2f}`**
                - **Prima put vendido (`{K2:.2f}`): `${prima_put_short:.2f}`**
                - **Prima call vendido (`{K3:.2f}`): `${prima_call_short:.2f}`**
                - **Prima call comprado (`{K4:.2f}`): `${prima_call_long:.2f}`**
                - **Crédito neto recibido: `${credito_neto:.2f}`**
                - **Ganancia máxima: `${credito_neto:.2f}`** (si **`{K2:.2f} < S < {K3:.2f}`**)
                - **Pérdida máxima: `${min(K2 - K1 - credito_neto, K4 - K3 - credito_neto):.2f}`** (limitada por los spreads)
                - **Breakeven inferior: `{BE_lower:.2f}`** → Variación necesaria del subyacente: **{(BE_lower/S-1)*100:.2f}%**
                - **Breakeven superior: `{BE_upper:.2f}`** → Variación necesaria del subyacente: **{(BE_upper/S-1)*100:.2f}%**
                """)

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Venta PUT":
                st.write(""" 
                Al vender un **put (opción de venta)** se cobra la prima que abona el comprador.  
                Si se espera **baja volatilidad** y una **tendencia alcista**, es probable que el put no se ejerza y el lanzador conserve la prima.  
                En caso contrario, si el precio cae, el vendedor tiene la obligación de **comprar el activo subyacente** al precio pactado.
                """)
            
                # ==========================
                # Parámetros y función BS
                # ==========================
                K = S * 0.98
            
                def black_scholes_put(S, K, T, r, sigma):
                    """Calcula el precio de un put europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                    return put_price
            
                prima = black_scholes_put(S, K, T, r, sigma)
            
                # ==========================
                # Payoff del vendedor
                # ==========================
                S_range = np.linspace(S*0.7, S*1.3, 200)
                payoff_put_seller = prima - np.maximum(K - S_range, 0)
                breakeven = K - prima
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
                
                Venta de un **put** de `{ticker}` con strike **`{K:.2f}`**,  
                vencimiento en **`{T*12:.0f}` meses** y prima **`${prima:.2f}`**,  
                tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(S_range, payoff_put_seller, label="Payoff Vendedor Put", color="crimson", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K, color="red", linestyle="--", linewidth=1, label=f"Strike = {K:.2f}")
                ax.axvline(S, color="green", linestyle="--", linewidth=1, label=f"S = {S:.2f}")
                ax.axvline(breakeven, color="orange", linestyle="--", linewidth=1.5, label=f"Breakeven = {breakeven:.2f}")
            
                ax.set_title("Payoff de un Put Europeo (Vendedor) al Vencimiento")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Información final
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:** 
                
                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima del put: `${prima:.2f}`**  
                - **Costo total: `0`**  
                - **Ganancia máxima: `${prima:.2f}`** (si **`S > {K:.2f}`**)  
                - **Pérdida máxima: Ilimitada ⚠️**  
                - **Punto de equilibrio (Breakeven): `{breakeven:.2f}`** → Variación necesaria del subyacente: **{(breakeven/S-1)*100:.2f}%**  
                """)

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Ratio call spread":
                st.write("""     
                Esta estrategia se basa en la **compra de un call** de una determinada base y la **venta de dos calls** de una base superior.  
                El objetivo es **financiar la compra** del call largo con las primas recibidas por los calls vendidos.  
                Es ideal cuando se espera una **suba moderada** del subyacente pero se desea estar **cubierto a la baja**.
                """)
            
                # ==========================
                # Parámetros
                # ==========================
                K1 = S * 0.98  # Strike del call comprado
                K2 = S * 1.02  # Strike de los calls vendidos
            
                # ==========================
                # Función Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    """Calcula el precio de un call europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                    return call_price
            
                # ==========================
                # Primas
                # ==========================
                prima_call_long = black_scholes_call(S, K1, T, r, sigma)
                prima_call_short = black_scholes_call(S, K2, T, r, sigma)
                prima_neta = prima_call_long - 2 * prima_call_short  # costo neto (puede ser negativo o pequeño)
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S*0.6, S*1.6, 200)
                payoff_long_call = np.maximum(S_range - K1, 0) - prima_call_long
                payoff_short_calls = -2 * (np.maximum(S_range - K2, 0) - prima_call_short)
                payoff_ratio_call = payoff_long_call + payoff_short_calls
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Compra de **1 call** de `{ticker}` con base **`{K1:.2f}`** a **`{T*12:.0f}` meses** por **`${prima_call_long:.2f}`**,  
                y venta de **2 calls** de `{ticker}` con base **`{K2:.2f}`** a **`{T*12:.0f}` meses** por **`${prima_call_short:.2f}`** cada uno,  
                tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(S_range, payoff_ratio_call, label="Ratio Call Spread (1:-2)", color="purple", linewidth=2)
            
                # Líneas de referencia
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K1, color="blue", linestyle="--", linewidth=1, label=f"Strike largo = {K1:.2f}")
                ax.axvline(K2, color="red", linestyle="--", linewidth=1, label=f"Strike corto = {K2:.2f}")
                ax.axvline(K2 + prima_call_long, color="darkblue", linestyle="--", linewidth=1, label=f"Breakeven = {K2 + prima_call_long:.2f}")
            
                # Estética
                ax.set_title("Estrategia Ratio Call Spread (Compra 1 Call, Venta 2 Calls)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Información final
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**  

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima del call comprado: `${prima_call_long:.2f}`**  
                - **Prima de cada call vendido: `${prima_call_short:.2f}`**  
                - **Prima neta total: `${-prima_neta:.2f}`**  
                - **Ganancia máxima: `${max(payoff_ratio_call):.2f}`** (si **`S = {K2:.2f}`**)  
                - **Pérdida potencial: Ilimitada ⚠️** (si **`S > {K2 + prima_call_long:.2f}`**)  
                - **Breakeven: `{K2 + prima_call_long:.2f}`** → Variación necesaria del subyacente: **{((K2 + prima_call_long)/S - 1)*100:.2f}%** 
                """)

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Mariposa vendida":
                st.write("""
                Esta estrategia consiste en **comprar 2 calls ATM** (de base central)  
                y **vender un call con base inferior** junto con **otro con base superior**.  
                Limita las ganancias, pero reduce el costo del armado.
                """)
            
                # ==========================
                # Strikes
                # ==========================
                K1 = S * 0.95   # strike inferior
                K2 = S * 1.00   # strike central (ATM)
                K3 = S * 1.05   # strike superior
            
                # ==========================
                # Funciones Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                    return call_price
            
                # ==========================
                # Primas (costos de la estrategia)
                # ==========================
                prima_K1 = black_scholes_call(S, K1, T, r, sigma)*0.95
                prima_K2 = black_scholes_call(S, K2, T, r, sigma)
                prima_K3 = black_scholes_call(S, K3, T, r, sigma)*0.95
            
                # Comprar 1 call K1, vender 2 calls K2, comprar 1 call K3
                prima_total = prima_K1 - 2*prima_K2 + prima_K3
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S*0.7, S*1.3, 200)
            
                payoff = (
                    -np.maximum(S_range - K1, 0)   # short call K1
                    + 2*np.maximum(S_range - K2, 0)  # long 2 calls K2
                    - np.maximum(S_range - K3, 0)    # short call K3
                    - prima_total                   # prima neta
                )
            
                # ==========================
                # Breakeven points
                # ==========================
                BE_lower = K1 - prima_total
                BE_upper = K3 + prima_total
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                1️⃣ Venta de un **call** de `{ticker}` base **`${K1:.2f}`** con prima **`${prima_K1:.2f}`**.  
                2️⃣ Compra de **dos calls** de `{ticker}` base **`${K2:.2f}`** con prima **`${prima_K2:.2f}`**.  
                3️⃣ Venta de un **call** de `{ticker}` base **`${K3:.2f}`** con prima **`${prima_K3:.2f}`**.  
                Todos con vencimiento en **`{T*12:.0f}` meses**  
                Tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(S_range, payoff, label="Mariposa Vendida (Short Butterfly)", color="red", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K1, color="blue", linestyle="--", linewidth=1, label=f"K1 = {K1:.2f}")
                ax.axvline(K2, color="green", linestyle="--", linewidth=1, label=f"K2 = {K2:.2f}")
                ax.axvline(K3, color="purple", linestyle="--", linewidth=1, label=f"K3 = {K3:.2f}")
                ax.axvline(BE_lower, color="orange", linestyle="--", linewidth=1.5, label=f"BE inferior = {BE_lower:.2f}")
                ax.axvline(BE_upper, color="orange", linestyle="--", linewidth=1.5, label=f"BE superior = {BE_upper:.2f}")
                ax.set_title("Estrategia Mariposa Vendida (Short Butterfly)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Info
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:** 

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima call lanzado base `{K1:.2f}`: `${prima_K1:.2f}`**  
                - **Prima call comprado base `{K2:.2f}` (individual): `${prima_K2:.2f}`**  
                - **Prima call lanzadoo base `{K3:.2f}`: `${prima_K3:.2f}`**
                - **Prima neta pagada/recibida: `${-prima_total:.2f}`**  
                - **Ganancia máxima: `${np.max(payoff):.2f}`**  
                - **Pérdida máxima: `${np.min(payoff):.2f}`**  
                - **Breakeven inferior: `${BE_lower:.2f}`** → Variación necesaria del subyacente: **{(BE_lower/S-1)*100:.2f}%**  
                - **Breakeven superior: `${BE_upper:.2f}`** → Variación necesaria del subyacente: **{(BE_upper/S-1)*100:.2f}%**
                """)
            
                # ==========================
                # Recomendación
                # ==========================
                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Venta CALL":
                st.write("""
                Al vender un **call (opción de compra)** se cobra la **prima** que abona el comprador.  
                Si se espera **baja volatilidad con tendencia bajista**, es probable que el call no se ejerza,  
                permitiendo al lanzador quedarse con la prima.  
                Si se ejerce, el lanzador tiene la obligación de vender el activo subyacente al precio pactado.
                """)
            
                # ==========================
                # Black-Scholes Call
                # ==========================
                K = S * 1.02
                def black_scholes_call(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                    return call_price
            
                # ==========================
                # Parámetros
                # ==========================
                prima = black_scholes_call(S, K, T, r, sigma)
            
                # ==========================
                # Payoff al vencimiento (Vendedor de Call)
                # ==========================
                S_range = np.linspace(S*0.7, S*1.5, 100)
                payoff_call_seller = prima - np.maximum(S_range - K, 0)
            
                # Punto de equilibrio
                breakeven = K + prima
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Venta de un **call** de `{ticker}` con base **`{K:.2f}`**  
                y vencimiento en **`{T*12:.0f}` meses**,  
                cobrando una prima de **`${prima:.2f}`**,  
                tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10,6))
                ax.plot(S_range, payoff_call_seller, label="Payoff Vendedor Call", color="purple", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K, color="red", linestyle="--", linewidth=1, label=f"Strike = {K:.2f}")
                ax.axvline(S, color="green", linestyle="--", linewidth=1, label=f"S = {S:.2f}")
                ax.axvline(breakeven, color="orange", linestyle="--", linewidth=1.5, label=f"Breakeven = {breakeven:.2f}")
                ax.set_title("Payoff de un Call Europeo (Vendedor) al Vencimiento")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Info
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:** 

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima call: `${prima:.2f}`**  
                - **Costo total de la estrategia: `0`**  
                - **Pérdida máxima: Ilimitada ⚠️**
                - **Ganancia máxima: `${prima:.2f}`** (si **`S < {K:.2f}`**)  
                - **Breakeven: `{breakeven:.2f}`** → Variación necesaria del subyacente: **{(breakeven/S-1)*100:.2f}%**
                """)
            
                # ==========================
                # Recomendación
                # ==========================
                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Venta sintético":
                st.write("""
                Esta estrategia consiste en comprar un **put** y vender un **call** de misma base.  
                El objetivo es replicar el resultado de la **venta de una acción en corto**, pero con una inversión significativamente menor (es decir, con apalancamiento).  
                
                La importancia de la **venta sintética** radica en que en algunos mercados —como el argentino— está prohibida la venta en descubierto de acciones.  
                """)
            
                K = S * 1.02
            
                # ==========================
                # Funciones Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    """Calcula el precio de un call europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                    return call_price
            
                def black_scholes_put(S, K, T, r, sigma):
                    """Calcula el precio de un put europeo con Black-Scholes"""
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                    return put_price
            
                # ==========================
                # Primas
                # ==========================
                prima_call = black_scholes_call(S, K, T, r, sigma)
                prima_put = black_scholes_put(S, K, T, r, sigma)
                prima_neta = prima_call - prima_put  # ingreso neto de la posición
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.6, S * 1.4, 200)
            
                payoff_put = np.maximum(K - S_range, 0) - prima_put     # long put
                payoff_call = -np.maximum(S_range - K, 0) + prima_call  # short call
                payoff_sintetico = payoff_put + payoff_call
            
                # ==========================
                # Payoff de venta directa
                # ==========================
                payoff_short_stock = S - S_range  # vender el subyacente directamente
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Venta sintética de `{ticker}`:  
                Compra de un **put** a **`${prima_put:.2f}`** y venta de un **call** a **`${prima_call:.2f}`**,  
                ambos con base **`{K:.2f}`** y vencimiento en **`{T*12:.0f}` meses**,  
                replicando la **venta del subyacente en corto** con menor capital.
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff_sintetico, label="Venta sintética (Put + Call)", color="red", linewidth=2)
                ax.plot(S_range, payoff_short_stock, label="Venta directa del subyacente", color="gray", linestyle="--", linewidth=1.5)
            
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K, color="blue", linestyle="--", linewidth=1, label=f"Strike = {K:.2f}")
            
                ax.set_title("Estrategia de Venta Sintética (Synthetic Short)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Info
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima put comprado: `${prima_put:.2f}`**  
                - **Prima call vendido: `${prima_call:.2f}`**  
                - **Prima neta recibida: `${prima_neta:.2f}`**  
            
                La posición replica el payoff de una **venta del subyacente**.
                """)
                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Cuna vendida":
                st.write("""
                Esta estrategia consiste en **vender un call OTM** con una base superior al precio actual  
                y **vender un put OTM** con una base inferior al precio actual del mismo subyacente.  
                
                Esto permite obtener un ingreso por la **venta de primas** que, según la expectativa del inversor, **no serán ejercidas**.  
                Si el precio del activo se mueve demasiado hacia arriba o hacia abajo, el inversor estará obligado a vender o comprar el subyacente.  
                """)
            
                K_call = S * 1.05
                K_put = S * 0.95
            
                # ==========================
                # Funciones Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            
                def black_scholes_put(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
                # ==========================
                # Primas cobradas
                # ==========================
                prima_put = black_scholes_put(S, K_put, T, r, sigma) * 1.05
                prima_call = black_scholes_call(S, K_call, T, r, sigma) * 1.05
                prima_total = prima_put + prima_call
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.6, S * 1.4, 200)
                payoff_strangle = -np.maximum(K_put - S_range, 0) - np.maximum(S_range - K_call, 0) + prima_total
            
                # ==========================
                # Breakeven points
                # ==========================
                BE_lower = K_put - prima_total
                BE_upper = K_call + prima_total
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Venta de un **put** de `{ticker}` a **`${prima_put:.2f}`** con base **`{K_put:.2f}`**  
                y de un **call** a **`${prima_call:.2f}`** con base **`{K_call:.2f}`**,  
                ambos con vencimiento en **`{T*12:.0f}` meses**,
                tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff_strangle, label="Cuna Vendida (Short Strangle)", color="red", linewidth=2)
            
                # Líneas de referencia
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K_put, color="blue", linestyle="--", linewidth=1, label=f"Strike Put = {K_put:.2f}")
                ax.axvline(K_call, color="green", linestyle="--", linewidth=1, label=f"Strike Call = {K_call:.2f}")
                ax.axvline(BE_lower, color="orange", linestyle="--", linewidth=1.5, label=f"BE inferior = {BE_lower:.2f}")
                ax.axvline(BE_upper, color="orange", linestyle="--", linewidth=1.5, label=f"BE superior = {BE_upper:.2f}")
            
                ax.set_title("Estrategia de Cuna Vendida (Short Strangle)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Info
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima put vendido: `${prima_put:.2f}`**  
                - **Prima call vendido: `${prima_call:.2f}`**  
                - **Ganancia máxima:`${prima_total:.2f}`** (si **`{K_put:.2f}< S < {K_call:.2f}`**)  
                - **Pérdida máxima: Ilimitada ⚠️**    
                - **Breakeven inferior: `{BE_lower:.2f}`**  → Variación necesaria del subyacente: **{(BE_lower/S - 1)*100:.2f}%**  
                - **Breakeven superior: `{BE_upper:.2f}`**  → Variación necesaria del subyacente: **{(BE_upper/S - 1)*100:.2f}%**
                """)

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Venta CALL, compra tasa":
                st.write("""
                Si se espera que el activo se mantenga estable con **volatilidad baja**,  
                se puede optar por **vender opciones CALL** para cobrar la prima  
                y **reinvertir ese dinero** en la **tasa libre de riesgo** si ésta fuera atractiva.
                """)
            
                # ==========================
                # Parámetros
                # ==========================
                K = S * 1.02  # strike levemente superior
            
                # ==========================
                # Black-Scholes Call
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                    return call_price
            
                prima = black_scholes_call(S, K, T, r, sigma)
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.7, S * 1.5, 100)
                payoff_call_seller = prima * (1 + r) ** T - np.maximum(S_range - K, 0)
            
                # ==========================
                # Punto de equilibrio
                # ==========================
                breakeven = K + prima * (1 + r) ** T
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Venta de un **call** de `{ticker}` a **`${prima:.2f}`**, con base **`{K:.2f}`**,  
                vencimiento en **`{T*12:.0f}` meses** e inversión de la prima al **`{r*100:.2f}%` anual**,  
                generaría el siguiente resultado al vencimiento:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff_call_seller, label="Payoff Vendedor Call", color="purple", linewidth=2)
            
                # Líneas de referencia
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K, color="red", linestyle="--", linewidth=1, label=f"Strike = {K:.2f}")
                ax.axvline(S, color="green", linestyle="--", linewidth=1, label=f"S = {S:.2f}")
                ax.axvline(breakeven, color="orange", linestyle="--", linewidth=1.5, label=f"Breakeven = {breakeven:.2f}")
            
                # Estética
                ax.set_title("Payoff de un Call Europeo (Vendedor) al Vencimiento")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Información adicional
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima call:`${prima:.2f}`**  
                - **Costo total: `0`**  
                - **Ganancia máxima: `${prima*(1+r)**T:.2f}`** (si **`S < {K:.2f}`**)  
                - **Pérdida máxima: Ilimitada ⚠️**    
                - **Breakeven: `{breakeven:.2f}`** → Variación necesaria del subyacente: **{(breakeven/S - 1)*100:.2f}%**
                """)
            
                # ==========================
                # Recomendación
                # ==========================
                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Mariposa comprada":
                st.write("""
                Para implementar esta estrategia debemos **vender 2 calls** con una determinada base y **comprar un call** de base inferior y **otro de base superior**.  
                Estas compras van a cumplir la función de limitar nuestras pérdidas pero, por supuesto, a costa de obtener un ingreso neto menor.  
                """)
            
                # ==========================
                # Strikes
                # ==========================
                K1 = S * 0.95   # strike inferior
                K2 = S * 1.00   # strike central (ATM)
                K3 = S * 1.05   # strike superior
            
                # ==========================
                # Función Black-Scholes
                # ==========================
                def black_scholes_call(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
                    return call_price
            
                # ==========================
                # Primas
                # ==========================
                prima_K1 = black_scholes_call(S, K1, T, r, sigma)*1.03
                prima_K2 = black_scholes_call(S, K2, T, r, sigma)
                prima_K3 = black_scholes_call(S, K3, T, r, sigma)*1.03
            
                # Comprar 1 call K1, vender 2 calls K2, comprar 1 call K3
                prima_total = prima_K1 - 2*prima_K2 + prima_K3
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S*0.7, S*1.3, 200)
            
                payoff = (
                    np.maximum(S_range - K1, 0)
                    - 2*np.maximum(S_range - K2, 0)
                    + np.maximum(S_range - K3, 0)
                    - prima_total
                )
            
                # ==========================
                # Breakeven points
                # ==========================
                BE_lower = K1 + prima_total
                BE_upper = K3 - prima_total
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                1️⃣ Compra de un **call** de `{ticker}` base **`{K1:.2f}`** a **`${prima_K1:.2f}`**.  
                2️⃣ Venta de **dos calls** de `{ticker}` base **`{K2:.2f}`** a **`${prima_K2:.2f}`**.  
                3️⃣ Compra de un **call** de `{ticker}` base **`{K3:.2f}`** a **`${prima_K3:.2f}`**.  
                Todos con vencimiento en **`{T*12:.0f}` meses**.
                Tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff, label="Long Butterfly (con calls)", color="red", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K1, color="blue", linestyle="--", linewidth=1, label=f"K1 = {K1:.2f}")
                ax.axvline(K2, color="green", linestyle="--", linewidth=1, label=f"K2 = {K2:.2f}")
                ax.axvline(K3, color="purple", linestyle="--", linewidth=1, label=f"K3 = {K3:.2f}")
                ax.axvline(BE_lower, color="orange", linestyle="--", linewidth=1.5, label=f"BE inf = {BE_lower:.2f}")
                ax.axvline(BE_upper, color="orange", linestyle="--", linewidth=1.5, label=f"BE sup = {BE_upper:.2f}")
                ax.set_title("Estrategia Long Butterfly (con calls)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Información final
                # ==========================
                st.markdown(f"""
                **Detalles de la estrategia:** 

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima call comprado base `{K1:.2f}`: `${prima_K1:.2f}`**  
                - **Prima call lanzado base `{K2:.2f}`: `${prima_K2:.2f}`**  
                - **Prima call comprado base `{K3:.2f}`: `${prima_K3:.2f}`**
                - **Prima neta pagada/recibida: `${-prima_total:.2f}`**  
                - **Ganancia máxima: `${np.max(payoff):.2f}`** (si **`S = {K2:.2f}`**)  
                - **Pérdida máxima (en extremos): `${prima_total:.2f}`**  
                - **Breakeven inferior: `{BE_lower:.2f}`** → Variación necesaria del subyacente: **{(BE_lower/S-1)*100:.2f}%**  
                - **Breakeven superior: `{BE_upper:.2f}`** → Variación necesaria del subyacente: **{(BE_upper/S-1)*100:.2f}%**  
                """)

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Ratio put spread":
                st.write("""
                Esta estrategia se basa en **comprar 1 put** de una determinada base y **vender 2 puts** de una base inferior,  
                lo que nos permite financiar la compra del put.  
                Es importante destacar que la **ganancia máxima** proviene de las primas netas cobradas.  
                """)
            
                # ==========================
                # Strikes
                # ==========================
                K1 = S * 1.02  # Strike del put comprado
                K2 = S * 0.98  # Strike de los puts vendidos
            
                # ==========================
                # Función Black-Scholes
                # ==========================
                def black_scholes_put(S, K, T, r, sigma):
                    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
                    d2 = d1 - sigma * math.sqrt(T)
                    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                    return put_price
            
                # ==========================
                # Cálculo de primas
                # ==========================
                prima_put_long = black_scholes_put(S, K1, T, r, sigma)
                prima_put_short = black_scholes_put(S, K2, T, r, sigma)
                prima_neta = prima_put_long - 2 * prima_put_short
            
                # ==========================
                # Payoff al vencimiento
                # ==========================
                S_range = np.linspace(S * 0.5, S * 1.5, 200)
                payoff_long_put = np.maximum(K1 - S_range, 0) - prima_put_long
                payoff_short_puts = -2 * (np.maximum(K2 - S_range, 0) - prima_put_short)
                payoff_ratio_put = payoff_long_put + payoff_short_puts
            
                # ==========================
                # Breakeven estimado
                # ==========================
                breakeven = 2*K2 - K1 + prima_neta
            
                # ==========================
                # Ejemplo práctico
                # ==========================
                st.markdown(f"""
                **Ejemplo práctico**  
            
                Compra de un **put** de `{ticker}` base **`{K1:.2f}`** a **`${prima_put_long:.2f}`**  
                y venta de **dos puts** base **`{K2:.2f}`** a **`${prima_put_short:.2f}`**,  
                ambos con vencimiento en **`{T*12:.0f}` meses**,  
                tendría el siguiente resultado:
                """)
            
                # ==========================
                # Gráfico
                # ==========================
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(S_range, payoff_ratio_put, label="Ratio Put Spread (1:-2)", color="purple", linewidth=2)
                ax.axhline(0, color="black", linestyle="--", linewidth=1)
                ax.axvline(K1, color="blue", linestyle="--", linewidth=1, label=f"Strike largo = {K1:.2f}")
                ax.axvline(K2, color="red", linestyle="--", linewidth=1, label=f"Strike corto = {K2:.2f}")
                ax.axvline(breakeven, color="orange", linestyle="--", linewidth=1.5, label=f"Breakeven = {breakeven:.2f}")
                ax.set_title("Estrategia Ratio Put Spread (Compra 1 Put, Venta 2 Puts)")
                ax.set_xlabel("Precio del subyacente al vencimiento")
                ax.set_ylabel("Beneficio / Pérdida")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
                # ==========================
                # Información resumen
                # ==========================
                st.markdown(f"""
                 **Detalles de la estrategia:**

                - **Último precio observado del subyacente: `{S:.2f}`**
                - **Prima put comprado: `${prima_put_long:.2f}`**
                - **Prima puts vendidos (individual): `${prima_put_short:.2f}`**
                - **Prima neta total: `${-prima_neta:.2f}`**
                - **Ganancia máxima: {max(payoff_ratio_put):.2f}** (si **`S = {K2:.2f}`**)
                - **Breakeven: `{breakeven:.2f}`** → Variación necesaria del subyacente: **{(breakeven/S-1)*100:.2f}%**
                """)

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")
                
        else:
            st.warning("⚠️ No se encontró una estrategia que cumpla esas condiciones.")



    
