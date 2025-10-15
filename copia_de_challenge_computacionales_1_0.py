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

    st.markdown("### 💥 Value at Risk (VaR) Empírico")
    
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
            st.write(f"📉 **VaR empírico ({alpha*100:.1f}%):** {VaR_empirico*100:.2f}%")
            st.write(f"📊 **CVaR (Expected Shortfall):** {CVaR_empirico*100:.2f}%")
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
                    
                    Compra de un call sobre **{ticker}** con base **`{K:.2f}`**, vencimiento en **`{T*12:.0f}` meses**,  
                    y una prima de **$`{prima:.2f}`** tendría el siguiente resultado al vencimiento:
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
                st.write(f"**Breakeven:** {breakeven:.2f}  →  Variación necesaria del subyacente: {(breakeven/S - 1)*100:.2f}%")

                st.success("💡 Una compra de CALL es ideal para escenarios con expectativa **alcista** y volatilidad **moderada o creciente**.")

            # --- Estrategia específica: Bull spread con calls ---
            elif recommended_strategy == "Bull spread con calls":
                st.subheader("📘 Estrategia: Bull Spread con Calls")
            
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
                
                Venta de un call de **{ticker}** base **`{K_venta:.2f}`**, vencimiento en **`{T*12:.0f}` meses**,  
                con una prima de **`${prima_call_venta:.2f}`**, y compra de un call base **`{K_compra:.2f}`**  
                con prima **`${prima_call_compra:.2f}`** tendría el siguiente resultado:
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
                st.markdown("### 📊 Resumen de la Estrategia")
                st.write("Strike Call Comprado:", f"{K_compra:.2f}")
                st.write("Strike Call Vendido:", f"{K_venta:.2f}")
                st.write("Costo Neto (prima total):", f"{costo_total:.2f}")
                st.write("Breakeven:", f"{BE:.2f}  → Variación necesaria del subyacente: {(BE/S - 1)*100:.2f}%")
                st.write("Ganancia Máxima:", f"{ganancia_max:.2f}")
                st.write("Pérdida Máxima:", f"{perdida_max:.2f}")
            
                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Cono comprado":
                st.subheader("🎯 Estrategia: Cono Comprado (Long Straddle)")
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
                
                Compra de un **call** de `{ticker}` a **`${call_price:.2f}`** y un **put** a **`${put_price:.2f}`**,  
                ambos con base **`${K:.2f}`** y vencimiento en **`{T*12:.0f}` meses**,  
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
                st.subheader("📊 Resumen numérico de la estrategia")
                st.write(f"**Prima call:** ${call_price:.2f}")
                st.write(f"**Prima put:** ${put_price:.2f}")
                st.write(f"**Costo total (prima total):** ${prima_total:.2f}")
                st.write(f"**Pérdida máxima:** ${prima_total:.2f} (si S ≈ {K:.2f})")
                st.write(f"**Ganancia máxima:** Ilimitada 🚀")
                st.write(f"**Breakeven inferior:** ${BE_lower:.2f}  → Variación necesaria del subyacente: {(BE_lower/S - 1)*100:.2f}%")
                st.write(f"**Breakeven superior:** ${BE_upper:.2f}  → Variación necesaria del subyacente: {(BE_upper/S - 1)*100:.2f}%")

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
                st.write(f"**Prima put:** ${prima:.2f}")
                st.write(f"**Costo total de la estrategia:** ${prima:.2f}")
                st.write(f"**Pérdida máxima:** ${prima:.2f} (si S > {K:.2f})")
                st.write("**Ganancia máxima:** Ilimitada 🚀")
                st.write(f"**Breakeven:** ${breakeven:.2f} → Variación necesaria del subyacente: {(breakeven/S - 1)*100:.2f}%")

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
            
                Venta de un **put** de `{ticker}` con base **`${K_venta:.2f}`** (prima **`${prima_put_venta:.2f}`**)  
                y compra de un **put** con base **`${K_compra:.2f}`** (prima **`${prima_put_compra:.2f}`**)  
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
                st.write(f"**Put comprado (strike alto):** ${K_compra:.2f}")
                st.write(f"**Put vendido (strike bajo):** ${K_venta:.2f}")
                st.write(f"**Costo neto (prima total):** ${costo_total:.2f}")
                st.write(f"**Breakeven:** ${BE:.2f} → Variación necesaria del subyacente: {(BE/S - 1)*100:.2f}%")
                st.write(f"**Ganancia máxima:** ${ganancia_max:.2f}")
                st.write(f"**Pérdida máxima:** ${perdida_max:.2f}")
        
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
            
                Compra de un **put** de `{ticker}` con base **`${K_put:.2f}`** a **`${prima_put:.2f}`**,  
                y un **call** de `{ticker}` con base **`${K_call:.2f}`** a **`${prima_call:.2f}`**,  
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
                st.write(f"**Prima put comprada:** ${prima_put:.2f}")
                st.write(f"**Prima call comprada:** ${prima_call:.2f}")
                st.write(f"**Pérdida máxima:** ${prima_total:.2f} (si {K_put:.2f} < S < {K_call:.2f})")
                st.write("**Ganancia máxima:** Ilimitada 🚀")
                st.write(f"**Breakeven inferior:** ${BE_lower:.2f} → Variación necesaria del subyacente: {(BE_lower/S - 1)*100:.2f}%")
                st.write(f"**Breakeven superior:** ${BE_upper:.2f} → Variación necesaria del subyacente: {(BE_upper/S - 1)*100:.2f}%")

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
                venta de un **call** de `{ticker}` con base **`${K_call:.2f}`** a **`{T*12:.0f}` meses** por **`${prima_call:.2f}`**,  
                y compra de un **put** de `{ticker}` con base **`${K_put:.2f}`** a **`{T*12:.0f}` meses** por **`${prima_put:.2f}`**,  
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
                st.write(f"**Precio del subyacente:** ${S:.2f}")
                st.write(f"**Strike Put (protección):** ${K_put:.2f}")
                st.write(f"**Strike Call (tope):** ${K_call:.2f}")
                st.write(f"**Prima put:** ${prima_put:.2f}")
                st.write(f"**Prima call:** ${prima_call:.2f}")
                st.write(f"**Costo total neto:** ${costo_total:.2f}")
                st.write(f"**Breakeven inferior:** ${K_put:.2f} → Variación necesaria del subyacente: {(K_put/S - 1)*100:.2f}% (pérdida limitada más allá de este punto)")
                st.write(f"**Breakeven superior:** ${K_call:.2f} → Variación necesaria del subyacente: {(K_call/S - 1)*100:.2f}% (ganancia limitada más allá de este punto)")

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
            
                Venta de un **put** de `{ticker}` con base **`${K2:.2f}`** y compra de un **put** con base **`${K1:.2f}`**,  
                junto con la venta de un **call** de `{ticker}` con base **`${K3:.2f}`** y compra de un **call** con base **`${K4:.2f}`**,  
                todos con vencimiento a **`{T*12:.0f}` meses**,  
                resultaría en el siguiente payoff:
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
                st.write(f"**Prima put comprada (`{K1:.2f}`):** `${prima_put_long:.2f}`")
                st.write(f"**Prima put vendida (`{K2:.2f}`):** `${prima_put_short:.2f}`")
                st.write(f"**Prima call vendida (`{K3:.2f}`):** `${prima_call_short:.2f}`")
                st.write(f"**Prima call comprada (`{K4:.2f}`):** `${prima_call_long:.2f}`")
                st.write(f"**Crédito neto recibido:** `${credito_neto:.2f}`")
                st.write(f"**Ganancia máxima:** `${credito_neto:.2f}` (si `{K2:.2f}` < S < `{K3:.2f}`)")
                st.write(f"**Pérdida máxima:** `${min(K2 - K1 - credito_neto, K4 - K3 - credito_neto):.2f}` (limitada por los spreads)")
                st.write(f"**Breakeven inferior:** `${BE_lower:.2f}` → Variación necesaria del subyacente: {(BE_lower/S-1)*100:.2f}%")
                st.write(f"**Breakeven superior:** `${BE_upper:.2f}` → Variación necesaria del subyacente: {(BE_upper/S-1)*100:.2f}%")

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Venta PUT":
                st.markdown("""
                **💡 Estrategia: Venta de PUT**
                
                Al vender un **put (opción de venta)** se cobra la prima que abona el comprador.  
                Si se espera **baja volatilidad** y una **tendencia alcista**, es probable que el put no se ejerza y el lanzador conserve la prima.  
                En caso contrario, si el precio cae, el vendedor tiene la obligación de **comprar el activo subyacente** al precio pactado.
            
                ⚠️ **Recomendación:** Consultar los **requerimientos de garantía** con su agente de bolsa por el lanzamiento de las opciones.
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
                
                Venta de un **put** de `{ticker}` con strike **`${K:.2f}`**,  
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
                **Detalles de la estrategia**  
            
                - Prima del put: **`${prima:.2f}`**  
                - Costo total: **`0`**  
                - Ganancia máxima: **`${prima:.2f}`** (si el precio `{S:.2f}` > strike `{K:.2f}`)  
                - Pérdida máxima: **Ilimitada ⚠️**  
                - Punto de equilibrio (Breakeven): **`${breakeven:.2f}`** → Variación necesaria del subyacente: **`{(breakeven/S-1)*100:.2f}%`**  
                """)

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

            elif recommended_strategy == "Ratio call spread":
                st.markdown("""
                **💡 Estrategia: Ratio Call Spread**
                
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
            
                Compra de **1 call** de `{ticker}` con strike **`${K1:.2f}`** a **`{T*12:.0f}` meses** por **`${prima_call_long:.2f}`**,  
                y venta de **2 calls** de `{ticker}` con strike **`${K2:.2f}`** a **`{T*12:.0f}` meses** por **`${prima_call_short:.2f}`** cada uno,  
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
                **Detalles de la estrategia**  
            
                - Prima del call comprado (strike `{K1:.2f}`): **`${prima_call_long:.2f}`**  
                - Prima de cada call vendido (strike `{K2:.2f}`): **`${prima_call_short:.2f}`**  
                - Prima neta total: **`${prima_neta:.2f}`**  
                - Ganancia máxima: **Limitada entre los strikes `{K1:.2f}` y `{K2:.2f}`**  
                - Pérdida potencial: **Ilimitada** si el subyacente supera **`${K2 + prima_call_long:.2f}`**  
                  (Variación necesaria: **`{((K2 + prima_call_long)/S - 1)*100:.2f}%`**)  
                """)

                st.info("💡 **Recomendación:** Consultar requerimientos de garantía con su agente de bolsa por el lanzamiento de las opciones.")

        else:
            st.warning("⚠️ No se encontró una estrategia que cumpla esas condiciones.")



    
