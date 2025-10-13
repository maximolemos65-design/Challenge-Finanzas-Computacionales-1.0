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
    st.write(f"\nPromedio retorno: {mean_return*100:.2f}%")
    st.write(f"Desvío retorno:   {std_return*100:.2f}%")
    st.write()
    
    
   # ==========================
    # 5. Histograma con campana normal
    # ==========================
    
    st.markdown("#### 📈 Distribución de Retornos con Campana Normal")
    
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
        ax.axvline(mean_return, color='blue', linestyle='dashed', linewidth=2, label=f"Media: {mean_return:.4f}")
        ax.axvline(mean_return + std_return, color='green', linestyle='dashed', linewidth=2, label=f"+1σ: {mean_return+std_return:.4f}")
        ax.axvline(mean_return - std_return, color='green', linestyle='dashed', linewidth=2, label=f"-1σ: {mean_return-std_return:.4f}")
        ax.axvline(mean_return + 2*std_return, color='green', linestyle='dashed', linewidth=2, label=f"+2σ: {mean_return+2*std_return:.4f}")
        ax.axvline(mean_return - 2*std_return, color='green', linestyle='dashed', linewidth=2, label=f"-2σ: {mean_return-2*std_return:.4f}")
    
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
    
    st.caption("\n📊 Asimetría y Curtosis de la serie de retornos")
    st.write(f"Asimetría: {asimetria:.4f}")
    st.write(f"Curtosis (total): {curtosis_val:.4f}")
    st.write(f"Curtosis (exceso): {curtosis_total:.4f}")
    
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
    
    st.write(f"\nVolatilidad anualizada: {vol_annual*100:.3f}%")
    
    # ==========================
    # 8. Black-Scholes
    # ==========================
    sigma = float(vol_annual)
    S = float(data['Close'].iloc[-1])  # Último precio spot
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    
    st.caption(f"\n📊 Black-Scholes")
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
    
    st.markdown("#### 📈 Volatilidad")
    
    st.write("— Cantidad de desvíos —")
    
    # ==========================
    # 2.1 Calcular Z-scores
    # ==========================
    z_scores = (returns - mean_return) / std_return
    
    # Media y desvío de los Z-scores
    mean_z = z_scores.mean()
    std_z  = z_scores.std()
    
    st.write()
    st.write(f"Media z-scores: {mean_z:.6f}")
    st.write(f"Desvío z-scores: {std_z:.6f}")
    
    # ==========================
    # 2.3 Histograma de Z-Scores
    # ==========================
    
    st.markdown("### 🧮 Distribución de Z-Scores")
    
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
    
    st.write("\n📊 Asimetría y Curtosis de la serie de Z-Scores")
    st.write(f"Asimetría: {asimetria:.4f}")
    st.write(f"Curtosis: {curtosis_val:.4f}")
    
    # ==========================
    # 2.5 Medias móviles
    # ==========================
    
    # ==========================
    #  Volatilidad móvil
    # ==========================
    
    st.markdown("### 📉 Volatilidad Móvil de los Retornos")
    
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
        st.markdown("#### 📊 Estadísticas de Volatilidades Anualizadas Recientes")
        st.write(f"**Último Std 20:** {(last_std20 * np.sqrt(factor) * 100):.4f}%")
        st.write(f"**Último Std 250:** {(last_std250 * np.sqrt(factor) * 100):.4f}%")
        st.write(f"**Volatilidad constante (anualizada):** {(vol_annual * 100):.4f}%")
    
    else:
        st.warning("⚠️ No se encontraron datos de retornos para calcular la volatilidad móvil.")

    
    """Value at Risk"""
    
    # ==========================
    # 14. Value at Risk (VaR empírico) + Conditional VaR con sombreado
    # ==========================
    
    # 1. Preguntar nivel de confianza
    conf = st.number_input(st.text_input("📌 Ingrese el nivel de confianza (ej: 0.95 para 95%): "))
    alpha = 1 - conf
    
    # 2. Calcular percentil empírico (VaR)
    VaR_empirico = np.percentile(returns, alpha*100)  # percentil en base a la muestra
    mean_emp = returns.mean()
    
    # 3. Filtrar retornos en la cola (<= VaR)
    cola = returns[returns <= VaR_empirico]
    
    # Suma, conteo y CVaR
    suma_cola = cola.sum()
    conteo_cola = cola.count()
    CVaR_empirico = cola.mean()
    
    st.write(f"\n🔹 Nivel de confianza (una cola, izquierda): {conf*100:.1f}%")
    st.write(f"   ➤ VaR empírico ({alpha*100:.1f}%) = {VaR_empirico:.5f}")
    st.write(f"   ➤ CVaR empírico (Expected Shortfall) = {CVaR_empirico:.5f}")
    st.write(f"   ➤ Suma retornos cola = {suma_cola:.5f}")
    st.write(f"   ➤ Conteo retornos cola = {conteo_cola}")
    
    # 4. Graficar histograma con VaR y sombreado de la cola
    plt.figure(figsize=(10,6))
    
    # Histograma completo
    counts, bins, patches = plt.hist(
        returns,
        bins=50,
        density=True,
        edgecolor='black',
        alpha=0.6,
        label="Distribución empírica"
    )
    
    # Sombrear las barras que están a la izquierda del VaR
    for patch, bin_left in zip(patches, bins[:-1]):
        if bin_left <= VaR_empirico:
            patch.set_facecolor('red')
            patch.set_alpha(0.4)
    
    # Línea en el VaR
    plt.axvline(VaR_empirico, color="red", linestyle="--", linewidth=2, label=f"VaR {conf*100:.1f}%: {VaR_empirico:.4f}")
    
    # Línea en la media
    plt.axvline(mean_emp, color="blue", linestyle="--", linewidth=2, label=f"Media: {mean_emp:.4f}")
    
    plt.title(f"VaR y CVaR empírico de {ticker} al {conf*100:.1f}% de confianza")
    plt.xlabel("Retornos")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # 5. Mostrar resultados finales
    st.write(f"\n📊 Resultados VaR y CVaR empírico")
    st.write(f"Nivel de confianza: {conf*100:.1f}%")
    st.write(f"En el {alpha*100:.2f}% de los casos, suponiendo una distribución normal de los retornos y bajo condiciones normales de mercado, {ticker} tendrá un rendimiento menor o igual a {VaR_empirico*100:.5f}%.")
    st.write(f"¿Qué podemos esperar si se rompe el VaR? Para saberlo es útil hacer uso del VaR Condicional (CVaR), que es el promedio de los retornos más allá de esa barrera. Para este caso el CVaR es {CVaR_empirico*100:.5f}%.")
    
    """Simulación de Montecarlo"""
    
    # ==========================
    # Gráfico combinado: histórico + Monte Carlo
    # ==========================
    
    plt.figure(figsize=(12,6))
    
    # 1. Gráfico histórico
    plt.plot(data.index, data['Close'], label=f"Precio histórico {ticker}", color="blue", linewidth=2)
    
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
    S0 = float(data['Close'].iloc[-1])   # ✅ escalar
    mu = mean_return
    dt = Dt   # ya lo definiste según intervalo
    
    for j in range(n_simulaciones):
        prices = [S0]
        for t in range(1, N+1):
            z = np.random.normal()
            St = float(prices[-1] * np.exp((mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*z))  # ✅ float
            prices.append(St)
        simulaciones[:, j] = np.array(prices).flatten()   # ✅ vector 1D
    
        # Fechas futuras (arranca después del último dato real)
        future_dates = pd.date_range(start=data.index[-1], periods=N+1, freq=freq)[1:]
        plt.plot(future_dates, prices[1:], linewidth=1, alpha=0.2, color="orange")
    
    # 4. Línea promedio de todas las simulaciones
    mean_path = simulaciones.mean(axis=1)[1:]
    plt.plot(future_dates, mean_path, color="black", linewidth=2, label="Media de simulaciones")
    
    # 5. Último precio como referencia
    plt.scatter(data.index[-1], S0, color="black", zorder=5, label=f"Último precio: {S0:.2f}")
    
    # 6. Strike
    plt.axhline(y=K, color="red", linestyle="--", linewidth=1.5, label=f"Strike = {K}")
    
    # 7. Formato
    plt.title(f"Trayectoria histórica y simulaciones Monte Carlo - {ticker}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # ==========================
    # 8. Histograma de precios finales
    # ==========================
    final_prices = simulaciones[-1, :]  # últimos precios de cada simulación
    
    plt.figure(figsize=(8,5))
    plt.hist(final_prices, bins=10, edgecolor='black', alpha=0.7)
    
    # Línea en el precio inicial
    plt.axvline(S0, color="blue", linestyle="--", linewidth=2, label=f"Precio inicial: {S0:.2f}")
    
    # Línea en el strike
    plt.axvline(K, color="red", linestyle="--", linewidth=2, label=f"Strike = {K}")
    
    # Estadísticas
    mean_final = np.mean(final_prices)
    std_final  = np.std(final_prices)
    
    st.write()
    plt.axvline(mean_final, color="green", linestyle="--", linewidth=2, label=f"Media final: {mean_final:.2f}")
    plt.title(f"Distribución de precios finales - Monte Carlo ({ticker})")
    plt.xlabel("Precio al vencimiento (1 año)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()
    
    st.write("\n📊 Resultados Monte Carlo")
    st.write(f"Precio inicial: {S0:.2f}")
    st.write(f"Precio medio simulado a 1 año: {mean_final:.2f}")
    st.write(f"Desvío de precios finales: {std_final:.2f}")
   
    # ==========================
    # 1. Preparar datos
    # ==========================
    # returns ya lo tenés calculado antes
    # Calcular cambio porcentual en volumen
    
    st.caption("📊Volumen")
    
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
        st.caption("🔹 Dispersión: Retornos vs Proporción del Volumen Promedio")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(df_aux["Return"], df_aux["Vol_Ratio"], alpha=0.5, color="purple")
        ax1.set_title(f"Relación entre Retornos y Volumen Promedio - {ticker}")
        ax1.set_xlabel("Retornos")
        ax1.set_ylabel("Volumen observado / Promedio")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
    
        # --- 4. Gráfico de barras del volumen ---
        st.caption("🔹 Volumen de negociación a lo largo del tiempo")
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

    
    """Momentum - Variaciones"""
    
    import numpy as np
    import pandas as pd
    
    # Create a new DataFrame for the momentum variations
    momentum_data = pd.DataFrame(index=data.index)
    
    # Paso 1: clasificar retorno actual y agregar a momentum_data
    condiciones = [
        data['Return'] > 0,
        data['Return'] < 0,
        data['Return'] == 0
    ]
    valores = ['Positivo', 'Negativo', 'Sin variación']
    momentum_data['Return_class'] = np.select(condiciones, valores, default='Sin variación')
    
    # Paso 2: clasificación del retorno anterior y agregar a momentum_data
    momentum_data['Return_lag1'] = momentum_data['Return_class'].shift(1)
    momentum_data['Return_lag1'] = momentum_data['Return_lag1'].fillna('Sin dato previo')
    
    # Paso 3: retornos de días previos y agregar a momentum_data
    momentum_data['Return_lag2'] = momentum_data['Return_class'].shift(2)
    momentum_data['Return_lag2'] = momentum_data['Return_lag2'].fillna('Sin dato previo')
    momentum_data['Return_lag3'] = momentum_data['Return_class'].shift(3)
    momentum_data['Return_lag3'] = momentum_data['Return_lag3'].fillna('Sin dato previo')
    momentum_data['Return_lag4'] = momentum_data['Return_class'].shift(4)
    momentum_data['Return_lag4'] = momentum_data['Return_lag4'].fillna('Sin dato previo')
    
    # ------------------------------------------------------------------------LAG1-----------------------------------------------------------------------------------------------
    
    # Filter momentum_data where 'Return_class' is 'Positivo'
    filtered_momentum_data_positivo = momentum_data[momentum_data['Return_class'] == 'Positivo']
    
    # Count positive and negative values in Return_lag1 for the filtered data (Positivo)
    count_pp = filtered_momentum_data_positivo['Return_lag1'].value_counts().get('Positivo', 0)
    count_pn = filtered_momentum_data_positivo['Return_lag1'].value_counts().get('Negativo', 0)
    count_svarp1 = filtered_momentum_data_positivo['Return_lag1'].value_counts().get('Sin variación', 0)
    total_lag1_positivo = count_pp + count_pn + count_svarp1
    prob_pp = count_pp / total_lag1_positivo
    prob_pn = count_pn / total_lag1_positivo
    
    st.write(f"Probabilidad positivo-positivo: {prob_pp*100:.2f}%")
    st.write(f"Probabilidad positivo-negativo: {prob_pn*100:.2f}%")
    
    # Filter momentum_data where 'Return_class' is 'Negativo'
    filtered_momentum_data_negativo = momentum_data[momentum_data['Return_class'] == 'Negativo']
    
    # Count positive and negative values in Return_lag1 for the filtered data (Negativo)
    count_np = filtered_momentum_data_negativo['Return_lag1'].value_counts().get('Positivo', 0)
    count_nn = filtered_momentum_data_negativo['Return_lag1'].value_counts().get('Negativo', 0)
    count_svarn1 = filtered_momentum_data_negativo['Return_lag1'].value_counts().get('Sin variación', 0)
    total_lag1_negativo = count_np + count_nn + count_svarn1
    prob_nn = count_nn / total_lag1_negativo
    prob_np = count_np / total_lag1_negativo
    
    st.write(f"Probabilidad negativo-negativo: {prob_nn*100:.2f}%")
    st.write(f"Probabilidad negativo-positivo: {prob_np*100:.2f}%")
    
    # ------------------------------------------------------------------------LAG2-----------------------------------------------------------------------------------------------
    
    # Filter momentum_data where 'Return_class' and 'Return_lag1' are 'Positivo'
    filtered_momentum_data_pp = momentum_data[(momentum_data['Return_class'] == 'Positivo') & (momentum_data['Return_lag1'] == 'Positivo')]
    
    # Count positive and negative values in Return_lag2 for the filtered data (Positivo, Positivo)
    count_ppp = filtered_momentum_data_pp['Return_lag2'].value_counts().get('Positivo', 0)
    count_ppn = filtered_momentum_data_pp['Return_lag2'].value_counts().get('Negativo', 0)
    count_svarpp2 = filtered_momentum_data_pp['Return_lag2'].value_counts().get('Sin variación', 0)
    total_lag2_positivo = count_ppp + count_ppn + count_svarpp2
    prob_ppp = count_ppp / total_lag2_positivo
    prob_ppn = count_ppn / total_lag2_positivo
    
    st.write(f"Probabilidad positivo-positivo-positivo: {prob_ppp*100:.2f}%")
    st.write(f"Probabilidad positivo-positivo-negativo: {prob_ppn*100:.2f}%")
    
    # Filter momentum_data where 'Return_class' and 'Return_lag1' are 'Negativo'
    filtered_momentum_data_nn = momentum_data[(momentum_data['Return_class'] == 'Negativo') & (momentum_data['Return_lag1'] == 'Negativo')]
    
    # Count positive and negative values in Return_lag1 for the filtered data (Negativo)
    count_nnp = filtered_momentum_data_nn['Return_lag2'].value_counts().get('Positivo', 0)
    count_nnn = filtered_momentum_data_nn['Return_lag2'].value_counts().get('Negativo', 0)
    count_svarnn2 = filtered_momentum_data_nn['Return_lag2'].value_counts().get('Sin variación', 0)
    total_lag2_negativo = count_nnp + count_nnn + count_svarnn2
    prob_nnp = count_nnp / total_lag2_negativo
    prob_nnn = count_nnn / total_lag2_negativo
    
    st.write(f"Probabilidad negativo-negativo-positivo: {prob_nnp*100:.2f}%")
    st.write(f"Probabilidad negativo-negativo-negativo: {prob_nnn*100:.2f}%")
    
    # ------------------------------------------------------------------------LAG3-----------------------------------------------------------------------------------------------
    
    # Filter momentum_data where 'Return_class' and 'Return_lag1' are 'Positivo'
    filtered_momentum_data_ppp = momentum_data[(momentum_data['Return_class'] == 'Positivo') & (momentum_data['Return_lag1'] == 'Positivo') & (momentum_data['Return_lag2'] == 'Positivo')]
    
    # Count positive and negative values in Return_lag2 for the filtered data (Positivo, Positivo)
    count_pppp = filtered_momentum_data_ppp['Return_lag3'].value_counts().get('Positivo', 0)
    count_pppn = filtered_momentum_data_ppp['Return_lag3'].value_counts().get('Negativo', 0)
    count_svarpp3 = filtered_momentum_data_ppp['Return_lag3'].value_counts().get('Sin variación', 0)
    total_lag3_positivo = count_pppp + count_pppn + count_svarpp3
    prob_pppp = count_pppp / total_lag3_positivo
    prob_pppn = count_pppn / total_lag3_positivo
    
    st.write(f"Probabilidad positivo-positivo-positivo-positivo: {prob_pppp*100:.2f}%")
    st.write(f"Probabilidad positivo-positivo-positivo-negativo: {prob_pppn*100:.2f}%")
    
    # Filter momentum_data where 'Return_class' and 'Return_lag1' are 'Negativo'
    filtered_momentum_data_nnn = momentum_data[(momentum_data['Return_class'] == 'Negativo') & (momentum_data['Return_lag1'] == 'Negativo') & (momentum_data['Return_lag2'] == 'Negativo')]
    
    # Count positive and negative values in Return_lag1 for the filtered data (Negativo)
    count_nnnp = filtered_momentum_data_nnn['Return_lag3'].value_counts().get('Positivo', 0)
    count_nnnn = filtered_momentum_data_nnn['Return_lag3'].value_counts().get('Negativo', 0)
    count_svarnn3 = filtered_momentum_data_nnn['Return_lag3'].value_counts().get('Sin variación', 0)
    total_lag3_negativo = count_nnnp + count_nnnn + count_svarnn3
    prob_nnnp = count_nnnp / total_lag3_negativo
    prob_nnnn = count_nnnn / total_lag3_negativo
    
    st.write(f"Probabilidad negativo-negativo-negativo-positivo: {prob_nnnp*100:.2f}%")
    st.write(f"Probabilidad negativo-negativo-negativo-negativo: {prob_nnnn*100:.2f}%")
    
    # ------------------------------------------------------------------------LAG4-----------------------------------------------------------------------------------------------
    
    # Filter momentum_data where 'Return_class' and 'Return_lag1' are 'Positivo'
    filtered_momentum_data_pppp = momentum_data[(momentum_data['Return_class'] == 'Positivo') & (momentum_data['Return_lag1'] == 'Positivo') & (momentum_data['Return_lag2'] == 'Positivo') & (momentum_data['Return_lag3'] == 'Positivo')]
    
    # Count positive and negative values in Return_lag2 for the filtered data (Positivo, Positivo)
    count_ppppp = filtered_momentum_data_pppp['Return_lag4'].value_counts().get('Positivo', 0)
    count_ppppn = filtered_momentum_data_pppp['Return_lag4'].value_counts().get('Negativo', 0)
    count_svarppp3 = filtered_momentum_data_pppp['Return_lag4'].value_counts().get('Sin variación', 0)
    total_lag4_positivo = count_ppppp + count_ppppn + count_svarppp3
    prob_ppppp = count_ppppp / total_lag4_positivo
    prob_ppppn = count_ppppn / total_lag4_positivo
    
    st.write(f"Probabilidad positivo-positivo-positivo-positivo-positivo: {prob_ppppp*100:.2f}%")
    st.write(f"Probabilidad positivo-positivo-positivo-positivo-negativo: {prob_ppppn*100:.2f}%")
    
    # Filter momentum_data where 'Return_class' and 'Return_lag1' are 'Negativo'
    filtered_momentum_data_nnnn = momentum_data[(momentum_data['Return_class'] == 'Negativo') & (momentum_data['Return_lag1'] == 'Negativo') & (momentum_data['Return_lag2'] == 'Negativo') & (momentum_data['Return_lag3'] == 'Negativo')]
    
    # Count positive and negative values in Return_lag1 for the filtered data (Negativo)
    count_nnnnp = filtered_momentum_data_nn['Return_lag4'].value_counts().get('Positivo', 0)
    count_nnnnn = filtered_momentum_data_nn['Return_lag4'].value_counts().get('Negativo', 0)
    count_svarnnn3 = filtered_momentum_data_nn['Return_lag4'].value_counts().get('Sin variación', 0)
    total_lag4_negativo = count_nnnnp + count_nnnnn + count_svarnnn3
    prob_nnnnp = count_nnnnp / total_lag4_negativo
    prob_nnnnn = count_nnnnn / total_lag4_negativo
    
    st.write(f"Probabilidad negativo-negativo-negativo-negativo-positivo: {prob_nnnnp*100:.2f}%")
    st.write(f"Probabilidad negativo-negativo-negativo-negativo-negativo: {prob_nnnnn*100:.2f}%")
    
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
                st.subheader("📘 Estrategia: Compra de CALL")

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
                    
                    Compra de un call sobre **{ticker}** con base **{K:.2f}**, vencimiento en **{T*12:.0f} meses**,  
                    y una prima de **${prima:.2f}** tendría el siguiente resultado al vencimiento:
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
                st.markdown("### 📊 Resumen de la Estrategia")
                st.write(f"**Prima del call:** ${prima:.2f}")
                st.write(f"**Costo total:** ${prima:.2f}")
                st.write(f"**Pérdida máxima:** ${prima:.2f} (si S < {K:.2f})")
                st.write("**Ganancia máxima:** Ilimitada 🚀")
                st.write(f"**Breakeven:** {breakeven:.2f}  →  variación necesaria: {(breakeven/S - 1)*100:.2f}%")

                st.success("💡 Una compra de CALL es ideal para escenarios con expectativa **alcista** y volatilidad **moderada o creciente**.")
        else:
            st.warning("⚠️ No se encontró una estrategia que cumpla esas condiciones.")



    
