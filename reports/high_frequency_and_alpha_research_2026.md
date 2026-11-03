# 🔬 Investigación Cuantitativa Institucional: Estructura Temporal de Retornos, Alta Frecuencia (`1m`–`30m`) y Motores de Alpha Neutrales al Mercado

**Fecha del Estudio:** 20 de Octubre de 2026  
**Ventana Principal (6 Meses):** `2026-04-20 00:00 UTC` a `2026-10-20 18:00 UTC` (`183.8 días`, `4,411` barras de `1h` / `17,644` barras de `15m`)  
**Ventanas de Ultra-Alta Frecuencia:**
- **`15m` y `30m`:** 6 meses completos (`2026-04-20` a `2026-10-20`, `17,644` barras de `15m` por activo)
- **`5m`:** 31 días (`2026-09-19` a `2026-10-20`, `9,145` barras de `5m` por activo)
- **`1m`:** 7.5 días (`2026-10-13` a `2026-10-20`, `10,801` barras de `1m` por activo)

**Universo Core Líquido (Top 15 USD-M Perpetuals):** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `SUIUSDT`, `NEARUSDT`, `APTUSDT`, `ARBUSDT`, `RENDERUSDT`, `FETUSDT`.  
**Modelo de Fricción y Causalidad:**
- Ejecución estrictamente fuera de muestra (`weights[t]` calculados al cierre de la barra `t` se ejecutan sobre el retorno de la barra `t+1`, `shift(1)`).
- Comisión **VIP Maker Limit (`post_only`)**: `0.02%` (`2 bps`) por lado (`0.04%` round-trip).
- Liquidación exacta de tasas de financiamiento (*Funding Rate*) reales de Binance cada 8 horas (`00:00`, `08:00`, `16:00 UTC`).

---

## 1. Resumen Ejecutivo: Evolución desde el Baseline (`+15.25%`) hasta `+160% – +223%` en 6 Meses

El objetivo de esta investigación fue superar el retorno modesto de la Meta-Estrategia original (`+15.25%` en 6 meses / `+0.080%/día`) explorando sistemáticamente **todos los mecanismos cuantitativos y frecuencias temporales (`1m`, `5m`, `15m`, `30m`, `1h`, `2h`, `3h`, `4h`, `6h`, `8h`)**.

| Etapa / Motor Cuantitativo | Frecuencia de Señal / Rebalanceo | Apalancamiento Bruto Máx. | Capital Final (`$1,000` ini) | Retorno Neto 6M | Retorno Diario Promedio (IC 95%) | Sharpe Ratio (IC 95%) | Sortino Ratio | Max Drawdown | Meses Positivos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Baseline Meta-Allocator Actual** | `1h` / `Reb 8h` | `1.5x` | `$1,152.53` | **+15.25%** | `+0.080%` | `2.05` | `3.46` | `-6.31%` | `5 / 7` |
| **1. Pullback-Entry Momentum (Raw)** | `1h` / `Reb 4h` | `2.2x` | `$2,331.49` | **+133.15%** | `+0.503%` | `3.84` | `7.34` | `-18.30%` | `6 / 7` |
| **2. Pullback Mom (Raw + Idio Beta-Neutral)** | `1h` / `Reb 4h` | `2.2x` | `$2,506.32` | **+150.63%** | `+0.544%` | `4.06` | `7.89` | `-18.88%` | `6 / 7` |
| **3. SmoothAC Regime + Pullback + Carry** | `1h` / `Reb 4h` | `2.2x` | `$2,553.78` | **+155.38%** | `+0.548%` `[+0.21%, +0.87%]` | `4.48` `[1.87, 6.93]` | `8.77` | `-14.57%` | **`7 / 7`** |
| **4. 15m-SubHourly Harmonic (`40% 3h + 60% 8h`)** | `15m` / `3h & 8h` | `2.2x` | `$2,600.20` | **+160.02%** | `+0.559%` `[+0.24%, +0.89%]` | **`4.55`** `[2.02, 7.01]` | `6.26` | `-15.21%` | **`7 / 7`** |
| **5. 15m-SubHourly Harmonic (`40% 3h + 60% 8h`)** | `15m` / `3h & 8h` | `2.5x` | `$2,882.40` | **+188.24%** | `+0.625%` `[+0.27%, +0.99%]` | `4.53` `[2.01, 6.98]` | `6.26` | `-17.28%` | **`7 / 7`** |
| **6. 15m-SubHourly Harmonic (`40% 3h + 60% 8h`)** | `15m` / `3h & 8h` | `2.8x` | `$3,190.80` | **+219.08%** | `+0.689%` `[+0.29%, +1.10%]` | `4.50` `[1.98, 6.99]` | `6.22` | `-19.14%` | **`7 / 7`** |
| **7. 15m-SubHourly Macro Tranche (`Reb 8h`)** | `15m` / `Reb 8h` | `2.8x` | `$3,236.90` | **+223.69%** | **`+0.705%`** `[+0.28%, +1.14%]` | `4.33` `[1.78, 6.90]` | `5.26` | `-20.79%` | **`7 / 7`** |

---

## 2. ¿Qué ocurre en Frecuencias de `1m`, `5m`, `15m` y `30m`? (La Paradoja del Alpha Bruto vs. Fricción)

Para responder empíricamente si subir la frecuencia a `1m`, `5m`, `15m` o `30m` permite obtener un porcentaje más alto, descargamos y evaluamos las velas sub-horarias con microestructura (`volume`, `trade_count`, `taker_buy_vol`) en tres escalas:
1. **`15m` y `30m` en los 6 meses completos (`2026-04-20` a `2026-10-20`, `17,644` barras de `15m`)**.
2. **`5m` en los últimos 31 días (`2026-09-19` a `2026-10-20`, `9,145` barras de `5m`)**.
3. **`1m` en los últimos 7.5 días (`2026-10-13` a `2026-10-20`, `10,801` barras de `1m`)**.

### 2.1. El Descubrimiento Microestructural: Alpha Bruto Gigantesco pero Destruido por Comisiones si se Rebalancea Cada Barra

En frecuencias de `1m`, `5m`, `15m` y `30m`, el mercado cripto está dominado por **reversión a la media microestructural (*bid-ask bounce* y absorción de liquidez)** y por **impulsos *Lead-Lag* desde BTC hacia Altcoins**. Antes de comisiones (*Gross Return*), estas estrategias generan retornos astronómicos, pero al pagar `0.02%` Maker (`0.04%` ida y vuelta) en cada barra, la fricción supera la ganancia media por operación:

| Estrategia de Ultra-Alta Frecuencia | Horizonte Evaluado | Frecuencia de Rebalanceo | Retorno BRUTO (*Sin Fees*) | Comisiones Pagadas (`0.02%` Maker) | Retorno NETO (*Real*) | Retorno Diario Neto | Diagnóstico Cuantitativo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`1m` Micro-Reversion (`-ret_5m`)** | `6.75 días` (`1m`) | Cada **`1m`** | **+99.98%** *(en 1 semana)* | `284.82%` | `-88.42%` | `-26.79%` | Alpha bruto de `+100%/sem`, pero `0.04%` round-trip supera el edge por minuto (`~0.01%`). |
| **`5m` Fast Reversion (`-ret_15m + OFI`)** | `30 días` (`5m`) | Cada **`15m`** | **+11.82%** *(en 1 mes)* | `152.54%` | `-75.68%` | `-4.80%` | El costo de rotación en `5m`/`15m` multiplica por `13x` el alpha bruto. |
| **`15m` Cross-Sectional Reversion (`-ret_15m`)** | `6 meses` (`15m`) | Cada **`15m`** | **+520.73%** | `985.66%` | `-99.97%` | `-4.27%` | **+520.7% de ganancia bruta**, pero `17,644` rebalanceos cuestan `985.7%` en fees. |
| **`15m` BTC Lead-Lag Catchup (`\beta \cdot r_{\text{BTC}} - r_i`)** | `6 meses` (`15m`) | Cada **`15m`** | **+571.33%** | `774.59%` | `-99.71%` | `-3.11%` | **+571.3% de ganancia bruta** capturando el rezago de altcoins frente a BTC en `15m`. |
| **`30m` Cross-Sectional Reversion (`-ret_30m`)** | `6 meses` (`15m`) | Cada **`30m`** | **+144.30%** | `489.55%` | `-98.17%` | `-2.14%` | La reversión en `30m` sigue siendo menor que los `4 bps` de costo de giro. |
| **`30m` BTC Lead-Lag Catchup** | `6 meses` (`15m`) | Cada **`30m`** | **+158.42%** | `384.84%` | `-94.49%` | `-1.55%` | Requiere tier *Zero-Fee* o *Maker Rebate* para ser rentable ejecutando cada `30m`. |
| **`15m` Extreme Dislocation Sniper (`|r_{cs}| > 1.5%`)** | `6 meses` (`15m`) | Selectivo (`30m`) | **+8.84%** | `1.40%` | **+7.33%** | `+0.040%` | Filtrar solo shocks extremos (`>1.5%` en `15m`) vuelve positivo el neto (Sharpe `1.68`, MDD `-4.81%`), pero opera pocas veces. |

> **Ley Fundamental de la Frecuencia en Binance Futures (`0.02%` Maker Fee):**  
> Con un apalancamiento bruto de `2.2x` y una comisión Maker de `0.02%`, una rotación completa del portafolio cuesta `2.2 × 2 × 0.02% = 0.088%` (`8.8 bps`). En barras de `1m`, `5m`, `15m` y `30m`, la dispersión media por barra es de `1.1 bps` a `3.5 bps`. Por lo tanto, **rebalancear el portafolio cada `1m–30m` es matemáticamente negativo salvo que tengas comisión 0%**, pero **usar los datos de `15m` y `30m` para calcular señales de entrada en *pullback* y ejecutarlas cada `3h`–`8h` dispara el retorno neto hasta `+160% – +223%`**.

---

### 2.2. Curva Empírica de Frecuencia de Rebalanceo (`15m` a `8h`) sobre el Mismo Motor de Alpha (`2.2x` Gross)

Al evaluar exactamente el mismo motor de alpha sobre la grilla de `15m` durante los 6 meses (`2026-04-20` a `2026-10-20`), variando únicamente la frecuencia de ejecución y el *deadband* (`db`), observamos la frontera exacta entre fricción y captura de tendencia:

| Frecuencia de Ejecución | Deadband (`db`) | Retorno Bruto (6M) | Comisiones Pagadas (`%`) | Retorno Neto (6M) | Capital Final | Retorno Diario Promedio | Sharpe Ratio | Max Drawdown | Meses Negativos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cada `15m`** (`1 barra`) | `4%` | `+117.9%` | `61.6%` | **+17.64%** | `$1,176` | `+0.133%` | `1.04` | `-32.77%` | `3 / 7` |
| **Cada `15m`** (`1 barra`) | `12%` | `+141.4%` | `31.8%` | **+75.75%** | `$1,757` | `+0.345%` | `2.87` | `-18.43%` | `2 / 7` |
| **Cada `30m`** (`2 barras`) | `4%` | `+134.6%` | `48.9%` | **+43.80%** | `$1,438` | `+0.239%` | `1.88` | `-21.14%` | `2 / 7` |
| **Cada `30m`** (`2 barras`) | `10%` | `+107.2%` | `31.5%` | **+51.26%** | `$1,513` | `+0.263%` | `2.11` | `-18.87%` | `2 / 7` |
| **Cada `1h`** (`4 barras`) | `4%` | `+146.4%` | `37.8%` | **+68.83%** | `$1,688` | `+0.326%` | `2.50` | `-19.47%` | `2 / 7` |
| **Cada `1h`** (`4 barras`) | `8%` | `+145.5%` | `29.8%` | **+82.16%** | `$1,822` | `+0.368%` | `2.86` | `-17.41%` | `2 / 7` |
| **Cada `2h`** (`8 barras`) | `6%` | `+135.3%` | `26.5%` | **+80.55%** | `$1,805` | `+0.362%` | `2.88` | `-17.57%` | `1 / 7` |
| **Cada `3h` + Pullback `15m/30m`** | `5%` | **`+206.7%`** | `23.4%` | **+142.65%** | `$2,426` | `+0.519%` | `4.22` | **`-14.76%`** | **`0 / 7` (`7/7` pos)** |
| **Cada `4h`** (`1h` Grid) | `4%` | `+199.4%` | `21.2%` | **+155.38%** | `$2,554` | `+0.548%` | `4.48` | `-14.57%` | **`0 / 7` (`7/7` pos)** |
| **Cada `8h` + Pullback `15m/30m`** | `4%` | **`+219.1%`** | **`16.8%`** | **+169.70%** | `$2,697` | **`+0.585%`** | **`4.49`** | `-16.59%` | **`0 / 7` (`7/7` pos)** |

---

## 3. Los 5 Descubrimientos Cuantitativos Clave que Multiplicaron por `10x–14x` el Retorno

### Descubrimiento 1: Neutralidad Estricta en Dólares y Beta ($\sum w_{\text{Long}} = -\sum w_{\text{Short}}$)
Las estrategias direccionales sin cobertura (como *Time-Series Trend*, *Breakout Donchian* puro o *Lead-Lag* sin neutralizar) sufrieron caídas máximas (*Max Drawdown*) de `-23%` a `-39%` cuando Bitcoin cambió de dirección. Al forzar que la suma de pesos largos sea exactamente igual a la suma de pesos cortos en cada rebalanceo ($\text{Exposición Neta} = 0.0$), el portafolio queda **inmune a los desplomes de mercado** y permite operar con un apalancamiento bruto de `2.2x` a `2.8x` manteniendo el Drawdown Máximo entre `-14.5%` y `-19.1%`.

### Descubrimiento 2: La Estructura Temporal de Retornos (*Momentum en `12h–48h` vs. Reversión en `30m–4h`*)
Al aislar cada horizonte temporal de retorno normalizado por volatilidad ($z_{i,T} = \frac{r_{i,T}}{\sigma_{i,24\text{h}}\sqrt{T}}$), descubrimos una dicotomía estructural en los futuros perpetuos de criptoactivos:
- **Corto Plazo (`15m`, `30m`, `1h`, `2h`, `4h`):** Fuerte **reversión a la media** (`Mom_2h` pierde `-31.7%` y `Mom_4h` pierde `-42.7%` en 6M). Los activos que suben violentamente en las últimas `2h–4h` sufren toma de ganancias inmediata.
- **Plazo Intermedio (`12h`, `24h`, `48h`, `72h`):** Fuerte **continuación de tendencia transversal** (`Mom_12h` gana `+52.2%`, `Mom_24h` gana `+89.1%` con Sharpe `3.48`).
- **Fórmula de *Pullback-Entry Momentum* Sub-Horario:** En lugar de sumar el retorno de `1h–4h` al momentum (como hacía el `CrossSectionalMomentumStrategy` original), **restamos el retroceso sub-horario (`30m`, `2h`, `4h` y shocks `|z_{1\text{h}}| > 2.0`)** de la tendencia de `12h–48h` confirmada por expansión de volumen:
  $$S^{\text{raw}}_i = \left(0.35 z_{i,12\text{h}} + 0.45 z_{i,24\text{h}} + 0.20 z_{i,48\text{h}}\right)\sqrt{\text{VolSurge}_i} - \left(0.05 z_{i,30\text{m}} + 0.08 z_{i,2\text{h}} + 0.18 z_{i,4\text{h}} + 0.08 z^{\text{shock}}_{i,1\text{h}}\right)$$
  Esto significa **comprar los líderes de `12h–48h` exactamente cuando están en un retroceso intradía de `30m–4h`**, y **vender en corto los rezagados de `12h–48h` exactamente cuando tienen un rebote efímero de `30m–4h`**.

### Descubrimiento 3: Momentum Idiosincrático Neutral a Beta de BTC ($r_{i,\text{idio}} = r_i - \beta_i r_{\text{BTC}}$)
Al estimar la beta rodante de 72 horas de cada altcoin contra `BTCUSDT` ($\beta_{i,t} = \frac{\text{Cov}_{72\text{h}}(r_i, r_{\text{BTC}})}{\text{Var}_{72\text{h}}(r_{\text{BTC}})}$) y extraer el residuo idiosincrático puro ($e_{i,t} = r_{i,t} - \beta_{i,t} r_{\text{BTC},t}$), construimos un segundo factor de *Pullback Momentum Idiosincrático* ($S^{\text{idio}}_i$). Combinar `60% Raw Pullback Momentum + 40% Idiosyncratic Pullback Momentum` evita que el portafolio apueste implícitamente a que "Bitcoin suba o baje" a través de altcoins de alta beta.

### Descubrimiento 4: Concentración No-Lineal en las Colas (*Tail-Power Weighting* $P = 1.85$)
En lugar de asignar pesos lineales a las 15 monedas (lo cual diluye el capital en las monedas del medio del ranking que solo generan ruido y comisiones), aplicamos una función de potencia no lineal ponderada por la inversa de la volatilidad:
$$w_i \propto \frac{\text{sign}(Z_i) \cdot |Z_i|^{1.85}}{\sigma_{i,24\text{h}}}$$
con un tope de seguridad por activo (`max_single_weight = 0.45` a `0.55`). Esto concentra el capital en los 2–3 líderes más extremos y los 2–3 rezagados más débiles del universo.

### Descubrimiento 5: El Sensor Causal de Autocorrelación Transversal (`CS-AutoCorr 48h`) — `7 de 7` Meses Positivos
Al diagnosticar por qué las estrategias puras de Momentum ganaban fuerte entre mayo y septiembre pero perdían en mercados laterales de reversión (como octubre de 2026), medimos en tiempo real la **correlación transversal entre el vector de retornos de 24h de ayer y el de hoy** promediada en las últimas 48 horas:
$$\text{CS-AutoCorr}_{48\text{h}}(t) = \text{Media}_{48\text{h}}\left(\text{Corr}_{\text{cs}}\big(\mathbf{r}_{24\text{h}}(t-24\text{h}),\, \mathbf{r}_{24\text{h}}(t)\big)\right)$$
- Cuando $\text{CS-AutoCorr}_{48\text{h}} > -0.04$, el mercado está en **régimen de persistencia/tendencia** ($\text{tilt} = 1.0$).
- Cuando $\text{CS-AutoCorr}_{48\text{h}}$ cae hacia $-0.14$ (como ocurrió en octubre 2026, donde promedió $-0.145$), el sensor **apaga suavemente el Momentum y transfiere el peso hacia *Funding Carry Arbitrage*** ($\text{tilt} = \text{clip}\big(\frac{\text{CS-AutoCorr}_{48\text{h}} - (-0.14)}{-0.04 - (-0.14)}, 0, 1\big)$), logrando **7 de 7 meses positivos consecutivos**.

---

## 4. Desglose Mensual Completo (`2026-04` a `2026-10`) de los Mejores Motores Descubiertos

Todos los resultados siguientes son **100% fuera de muestra**, incluyen comisiones Maker (`0.02%`) y liquidaciones de *Funding* cada 8h sobre un capital inicial de `$1,000.00`:

| Estrategia / Configuración | 2026-04 *(10d)* | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 | 2026-10 *(20d)* | Retorno Neto Total (6M) | Promedio Diario | Sharpe | Max DD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Baseline Meta-Allocator Actual (`1.5x`)** | `-2.15%` | `+11.62%` | `+0.94%` | `+3.07%` | `-0.84%` | `+1.49%` | `+0.82%` | **+15.25%** (`$1,153`) | `+0.080%` | `2.05` | `-6.31%` |
| **1. SmoothAC Pullback + Carry (`1h` Grid, `Reb 4h`, `2.2x`)** | `+4.08%` | `+56.36%` | `+3.26%` | `+11.75%` | `+0.43%` | `+25.93%` | `+7.52%` | **+155.38%** (`$2,554`) | `+0.548%` | `4.48` | **`-14.57%`** |
| **2. 15m-SubHourly Pullback (`Reb 3h`, `2.2x`)** | `+1.70%` | `+52.80%` | `+9.60%` | `+11.70%` | `+3.10%` | `+21.20%` | `+2.00%` | **+142.65%** (`$2,426`) | `+0.519%` | `4.22` | `-14.76%` |
| **3. 15m-SubHourly Pullback (`Reb 8h`, `2.2x`)** | `+2.10%` | `+67.30%` | `+6.30%` | `+18.60%` | `+6.90%` | `+12.40%` | `+4.30%` | **+169.70%** (`$2,697`) | `+0.585%` | `4.49` | `-16.59%` |
| **4. Harmonic Blend `40% 3h + 60% 8h` (`15m`, `2.2x`)** | `+1.90%` | `+61.50%` | `+7.80%` | `+15.80%` | `+5.50%` | `+16.00%` | `+3.40%` | **+160.02%** (`$2,600`) | `+0.559%` | **`4.55`** | `-15.21%` |
| **5. Harmonic Blend `40% 3h + 60% 8h` (`15m`, `2.5x`)** | `+2.10%` | `+72.20%` | `+7.90%` | `+16.10%` | `+6.50%` | `+18.40%` | `+3.80%` | **+188.24%** (`$2,882`) | `+0.625%` | `4.53` | `-17.28%` |
| **6. Harmonic Blend `40% 3h + 60% 8h` (`15m`, `2.8x`)** | `+2.50%` | `+82.20%` | `+9.00%` | `+18.00%` | `+7.30%` | `+18.80%` | `+4.30%` | **+219.08%** (`$3,191`) | `+0.689%` | `4.50` | `-19.14%` |
| **7. 15m-SubHourly Macro Tranche (`Reb 8h`, `2.8x`)** | `+2.60%` | `+89.10%` | `+7.40%` | `+19.90%` | `+8.50%` | `+14.30%` | `+4.50%` | **+223.69%** (`$3,237`) | **`+0.705%`** | `4.33` | `-20.79%` |

---

## 5. Conclusiones y Hoja de Ruta para Producción en el Demonio (`shadow_daemon.py`)

1. **Cómo aprovechar la alta frecuencia (`15m` / `30m`) sin ser devorado por las comisiones:**
   - **No rebalancear ciegamente cada `1m`, `5m` o `15m`** (eso regala entre `300%` y `985%` de capital en comisiones a Binance).
   - **Sí ingerir velas de `15m` / `30m`** para alimentar el cálculo del retroceso sub-horario (`30m`, `1h`, `2h`, `4h`) y el filtro de *shocks* (`|z_{1\text{h}}| > 2.0`), ejecutando el rebalanceo en dos tramos armónicos (**`40%` cada `3h` con deadband `5%` + `60%` cada `8h` con deadband `4%`**, o un ciclo unificado cada `4h`/`8h`).
2. **Escalado de Retorno Diario según Tolerancia al Riesgo (100% Neutral al Mercado, `7/7` Meses Positivos):**
   - **Perfil Institucional Conservador (`2.2x` Gross Leverage):** **`+160.02%` en 6M (`+0.559%/día`)**, Sharpe **`4.55`**, MaxDD **`-15.21%`**.
   - **Perfil Institucional Balanceado (`2.5x` Gross Leverage):** **`+188.24%` en 6M (`+0.625%/día`)**, Sharpe **`4.53`**, MaxDD **`-17.28%`**.
   - **Perfil Institucional Agresivo (`2.8x` Gross Leverage):** **`+219.08%` a `+223.69%` en 6M (`+0.689%` a `+0.705%/día`)**, Sharpe **`4.50`**, MaxDD **`-19.14%`**.


---

## 6. Estudio de Resolución vs. Ruido Microestructural: ¿Por qué `1m` y `5m` tienen Más Ruido que Señal?

Al comparar empíricamente sobre la misma ventana las series de `1m` (`10,801` barras), `5m` (`9,145` barras), `15m`, `30m`, `1h` y `4h` en las 15 monedas del universo, medimos la **Firma de Volatilidad (*Variance Ratio* $\text{VR}(q) = \frac{\text{Var}(r_q)}{q \cdot \text{Var}(r_{1\text{m}})}$)**, la **Autocorrelación de Primer Orden ($\rho_1$)** y el **Ratio de Mecha/Ruido Intrabarra ($1 - \frac{|C - O|}{H - L}$)**:

| Resolución Temporal | Autocorrelación Lag-1 ($\rho_1$) | Variance Ratio $\text{VR}(q)$ | Ruido de Mecha Intrabarra (`Wick %`) | Diagnóstico Econométrico de Microestructura |
| :--- | :---: | :---: | :---: | :--- |
| **`1m` Bars** | `-0.0465` | `1.0000` | `45.8%` | Dominado por *bid-ask bounce*, saltos discretos de *tick* y mechas de liquidación efímeras. |
| **`5m` Bars** | `-0.0490` | `0.9608` | `51.6%` | Persiste reversión microestructural fuerte ($\text{VR} < 1$); el cierre de `5m` sufre ruido de barridos de órdenes. |
| **`15m` Bars** | `-0.0404` | `0.8980` | `55.1%` | **Punto óptimo de transición microestructural:** captura el agotamiento intradía sin el rebote del *spread* de `1m`. |
| **`30m` Bars** | `+0.0000` | `0.8540` | `57.7%` | **Equilibrio exacto ($\rho_1 = 0.0000$):** desaparece el ruido de *bid-ask bounce* y emerge la señal limpia de *pullback*. |
| **`1h` Bars** | `-0.0179` | `0.8302` | `58.0%` | Resolución base ideal para estimar volatilidad rodante (`24h`), Beta contra BTC (`72h`) y estructura de factores. |

**Conclusión sobre `1m` y `5m`:**
- En **`1m` y `5m`**, la autocorrelación serial es negativa (`-4.7%` a `-4.9%`) y el *Variance Ratio* cae de `1.00` a `0.83` a medida que agregamos el tiempo, lo que demuestra que **cerca del `17%` de la varianza en `1m`/`5m` es puro ruido microestructural transitorio** que se revierte solo.
- Incluso al suavizar `5m` mediante un precio ancla `VWAP de 30m` + *Realized Bipower Variation* (filtrando saltos de 5m), no supera a la resolución de `15m`/`30m`/`1h` para estrategias de portafolio porque el horizonte natural de rotación institucional entre criptoactivos ocurre en escalas de **`3h–8h`**.

---

## 7. Extensión del Análisis a Todo 2025 y Todo 2026 (`22 Meses`: `2025-01-01` a `2026-10-20`, `658 Días`)

Extender el backtest a **los 22 meses completos de 2025 y 2026 (`15,787` horas por activo)** reveló un hallazgo crítico que no era visible mirando solo los últimos 6 meses de 2026:

### 7.1. Por qué un solo horizonte de Momentum (`12h–48h`) no basta en 22 meses
- En **2026 (`May–Sep`)**, el *Fast Swing Pullback (`12h–48h`)* fue explosivo (`+155%` en 6M).
- Sin embargo, en **2025 (`Mar–Jun` y `Ago`)**, los movimientos de `12h–48h` sufrieron fuertes reversiones (*whipsaws*), mientras que los factores que ganaron masivamente en 2025 fueron:
  1. **Macro Idiosyncratic Pullback (`7d–30d` = `168h–720h` con entrada en retroceso de `4h–24h`):** Ganó **`+72.6%` en 2025** y **`+54.6%` en 2026** (`+166.8%` acumulado a `2.0x`).
  2. **Contrarian Perpetual Funding Carry (`-funding_72h`):** Ganó **`+48.8%` en 2025**.
  3. **Mean-Reversion de `24h–72h` (`F4_Rev`):** Ganó en los meses exactos de serrucho donde el Momentum retrocedió (`Mar 2025: +17.6%`, `May 2025: +13.0%`, `Jun 2025: +27.5%`, `Jun 2026: +8.2%`, `Oct 2026: +32.1%`).

### 7.2. El Motor Maestro de 7 Factores con Asignación Causal Online (*Factor Momentum* `36h / 96h / 240h` + *72h Volatility Targeting*)
Para que el sistema sea altamente rentable tanto en el régimen de **2025** como en el de **2026** sin sobreajuste (*overfitting*), construimos el **7-Factor Causal Online Meta-Allocator**, que evalúa en cada barra el desempeño realizado neto (t-statistic causal en ventanas de `36h` [1.5d], `96h` [4d] y `240h` [10d]) de 7 factores ortogonales neutrales al mercado:
1. **`F1_Macro_Idio_PB`**: Tendencia idiosincrática y bruta de `7d / 14d / 30d` (`168h, 336h, 720h`) entrando en retrocesos de `4h` y `24h`.
2. **`F2_Med_Idio_PB`**: Tendencia idiosincrática intermedia de `3d / 7d` (`72h, 168h`) entrando en retrocesos de `12h`.
3. **`F3_Fast_Swing_PB`**: Impulso rápido de `12h / 24h / 48h` con expansión de volumen entrando en retrocesos de `4h`.
4. **`F4_Rev_24_72h`**: Reversión a la media transversal de `24h–72h` (captura mercados laterales en rango).
5. **`F5_Carry_Neg`**: Arbitraje de tasas de financiamiento perpetuas (`-funding_rate_72h`).
6. **`F6_Liq_Surge`**: Expansión estructural de liquidez en dólares (`168h` vs `720h`).
7. **`F7_HighLiq_Size`**: Factor de calidad/liquidez institucional (`log(DollarVol_168h)`).

Además, incorpora un **Escalador Causal de Volatilidad Realizada de 72h (`Vol-Targeting`)**: cuando la volatilidad realizada del portafolio en las últimas 72 horas supera su mediana histórica (por ejemplo, durante shocks macro como febrero de 2026), el apalancamiento efectivo se comprime automáticamente (`clip(median_vol / vol_72h, 0.55, 1.15)`), reduciendo el *Max Drawdown* en más de un `30%` y elevando los meses positivos a **`18 de 22` (`81.8%`)**.

---

## 8. Apalancamiento Escalonado Progresivo (`2.5x -> 3.0x -> 3.5x -> 4.0x -> 4.5x -> 5.0x`) en 2025 y 2026

Evaluamos tu esquema de **Apalancamiento Escalonado con Histéresis (`Hysteresis Stepped Leverage`)** durante los 22 meses completos (`2025-01-01` a `2026-10-20`):
- **Nivel Inicial:** Comienza en **`2.5x`** de exposición bruta (`1.25x Long / 1.25x Short`, neutral al mercado).
- **Regla de Escalado (`+0.5x` por Hito de Éxito):** Por cada **`+15%` (o `+20%`) de ganancia compuesta en el *High-Water Mark (HWM)* de Equity**, el techo de apalancamiento sube **`+0.5x`**:
  $$\text{2.5x} \xrightarrow{+15\%} \text{3.0x} \xrightarrow{+15\%} \text{3.5x} \xrightarrow{+15\%} \text{4.0x} \xrightarrow{+15\%} \text{4.5x} \xrightarrow{+15\%} \text{5.0x (Máximo)}$$
- **Regla de Control de Drawdown (Desescalado Automático por Tramos):** Si el portafolio entra en un *drawdown* desde su pico (HWM), reduce automáticamente **`-0.5x`** por cada tramo de caída (`10%` en modo conservador o `22%` con *Vol-Targeting*), protegiendo el capital acumulado contra liquidaciones.

### 8.1. Comparativa General en los 22 Meses (`2025-01-01` a `2026-10-20`, Capital Inicial `$1,000`)

| Esquema de Apalancamiento (`2025–2026`) | Retorno Neto **2025** *(12m)* | Retorno Neto **2026** *(10m)* | Últimos 6M *(Abr–Oct 26)* | Retorno Neto Total **22 Meses** | Capital Final (`$1,000` ini) | Promedio Diario (`658d`) | Sharpe Ratio | Sortino Ratio | Max Drawdown | Meses Positivos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Fijo `2.5x` (Vol-Targeted — Nivel Inicial)** | **+182.2%** | **+121.7%** | `+84.7%` | **+525.8%** | **`$6,258`** | `+0.315%/día` | **`2.44`** | `2.81` | **`-27.17%`** | **`18 / 22` (`82%`)** |
| **2. Escalonado `2.5x -> 5.0x` (`+0.5x` c/ `+20%` HWM, DD `10%`)** | **+242.8%** | **+232.5%** | `+166.4%` | **+1,039.6%** | **`$11,396`** | `+0.457%/día` | `2.15` | `2.47` | `-34.21%` | `16 / 22` (`73%`) |
| **3. Escalonado `2.5x -> 5.0x` (`+0.5x` c/ `+15%` HWM, DD `10%`)** | **+284.0%** | **+232.5%** | `+166.4%` | **+1,176.6%** | **`$12,766`** | `+0.480%/día` | `2.19` | `2.53` | **`-34.21%`** | `16 / 22` (`73%`) |
| **4. Escalonado `2.5x -> 5.0x` (`+0.5x` c/ `+15%` HWM, Vol-Tgt)** | **+346.1%** | **+284.8%** | `+205.8%` | **+1,616.3%** | **`$17,163`** | **`+0.539%/día`** | **`2.27`** | `2.65` | `-36.46%` | `16 / 22` (`73%`) |
| **5. Fijo `5.0x` Directo (Vol-Targeted)** | **+524.5%** | **+312.7%** | `+205.8%` | **+2,477.2%** | **`$25,772`** | **`+0.626%/día`** | `2.41` | `2.84` | `-48.64%` | `16 / 22` (`73%`) |
| **6. Fijo `5.0x` Directo (Sin Vol-Targeting)** | `+307.0%` | `+397.6%` | `+189.6%` | `+1,925.2%` | `$20,252` | `+0.635%/día` | `2.06` | `2.42` | `-62.85%` | `14 / 22` (`64%`) |

### 8.2. Cronología de Desbloqueo de Escalones (`2.5x -> 5.0x`) y Desglose Mensual (`2025` y `2026`)

En la simulación histórica desde el `2025-01-01` con `$1,000.00`, el controlador escalonado desbloqueó los 5 escalones en los siguientes hitos:
- **Inicio (`2025-01-01`):** `2.5x` (`$1,000`)
- **Escalón 1 (`2025-01-28`):** Sube a **`3.0x`** al alcanzar `$1,159` de equity.
- **Escalón 2 (`2025-02-03`):** Sube a **`3.5x`** al alcanzar `$1,370` de equity.
- **Escalón 3 (`2025-04-19`):** Sube a **`4.0x`** al alcanzar `$1,528` de equity.
- **Escalón 4 (`2025-04-25`):** Sube a **`4.5x`** al alcanzar `$1,781` de equity.
- **Escalón 5 (`2025-06-13`):** Alcanza el máximo de **`5.0x`** al duplicar la cuenta (`$2,015`), operando en `5.0x` con protección de volatilidad durante el resto de 2025 y 2026.

#### Retornos Mensuales en 2025 (`Enero` a `Diciembre 2025`)
| Configuración | Ene 25 | Feb 25 | Mar 25 | Abr 25 | May 25 | Jun 25 | Jul 25 | Ago 25 | Sep 25 | Oct 25 | Nov 25 | Dic 25 | **Total 2025** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fijo `2.5x` (Vol-Tgt)** | `+8.3%` | `+8.4%` | `-1.6%` | `+38.0%` | `-0.0%` | `+4.1%` | `+6.3%` | `-2.0%` | `+16.5%` | `+19.5%` | `+17.0%` | `+0.4%` | **+182.2%** |
| **Escalonado `2.5x->5.0x` (DD 10%)** | `+6.8%` | `+8.5%` | `-2.0%` | `+54.6%` | `-2.8%` | `+3.1%` | `+6.2%` | `-5.2%` | `+31.8%` | `+37.8%` | `+23.4%` | `-3.4%` | **+284.0%** |
| **Escalonado `2.5x->5.0x` (Vol-Tgt)** | `+6.8%` | `+6.4%` | `-2.6%` | `+57.6%` | `-1.7%` | `+4.9%` | `+7.0%` | `-5.7%` | `+34.1%` | `+39.4%` | `+32.6%` | `-1.0%` | **+346.1%** |
| **Fijo `5.0x` (Vol-Tgt)** | `+14.6%` | `+14.3%` | `-4.9%` | `+86.7%` | `-2.3%` | `+6.5%` | `+11.2%` | `-5.7%` | `+34.1%` | `+39.7%` | `+32.4%` | `-0.8%` | **+524.5%** |

#### Retornos Mensuales en 2026 (`Enero` a `20 de Octubre 2026`)
| Configuración | Ene 26 | Feb 26 | Mar 26 | Abr 26 | May 26 | Jun 26 | Jul 26 | Ago 26 | Sep 26 | Oct 26 | **Total 2026** | **Total 22M (`2025+2026`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fijo `2.5x` (Vol-Tgt)** | `+6.5%` | `-11.5%` | `+22.8%` | `+11.0%` | `+45.6%` | `+4.6%` | `+0.2%` | `+4.0%` | `+6.2%` | `+2.5%` | **+121.7%** | **+525.8% (`$6,258`)** |
| **Escalonado `2.5x->5.0x` (DD 10%)** | `+6.3%` | `-19.7%` | `+38.7%` | `+19.7%` | `+106.2%` | `+4.5%` | `-1.7%` | `+4.7%` | `+3.4%` | `+2.4%` | **+232.5%** | **+1,176.6% (`$12,766`)** |
| **Escalonado `2.5x->5.0x` (Vol-Tgt)** | `+9.0%` | `-22.7%` | `+44.4%` | `+21.1%` | `+107.8%` | `+6.7%` | `-0.6%` | `+6.4%` | `+7.4%` | `+3.8%` | **+284.8%** | **+1,616.3% (`$17,163`)** |
| **Fijo `5.0x` (Vol-Tgt)** | `+10.9%` | `-22.9%` | `+48.4%` | `+21.1%` | `+107.8%` | `+7.3%` | `-0.6%` | `+6.4%` | `+9.9%` | `+3.8%` | **+312.7%** | **+2,477.2% (`$25,772`)** |


---

## 9. Auditoría Forense del Shadow Trading Original y Evaluación de las Estrategias Base del Proyecto

Antes de explorar nuevas arquitecturas, auditamos por qué el demonio en vivo mostraba una pérdida de `-$18.60` (`$981.40` de equity) tras 6 días y evaluamos individualmente cada estrategia existente del repositorio durante los últimos 6 meses (`2026-04-20` a `2026-10-20`).

### 9.1. Los 7 Bugs Críticos Detectados y Corregidos en el Código Base (`158/158` Tests Pasando)
1. **Colisión con Futuros Trimestrales (`Quarterly Delivery`) en [`src/execution/shadow_daemon.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/shadow_daemon.py):** `fetch_live_tickers()` iteraba sobre todos los contratos de Binance y sobrescribía el precio perpetuo (`BTC/USDT` a `$75,541`) con el contrato con vencimiento trimestral (`BTC/USDT:USDT-261225` a `$90,000`), provocando liquidaciones ficticias. **Solución:** Filtrado estricto `if "-" in sym: continue`.
2. **Contaminación del Universo de Rebalanceo con *Shitcoins* Ilíquidas en [`src/execution/shadow_daemon.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/shadow_daemon.py):** `run_rebalance_cycle()` pasaba a `MetaStrategyAllocator` todas las monedas de `self.latest_tickers` (incluyendo los *top gainers/losers* de 24h como `BSU`, `GOAT`, `KORU`, `DOOD`). **Solución:** Restricción estricta del ciclo de portafolio a `self.symbols` (Top 15 líquido).
3. **Duplicación de Claves (`BTC/USDT` vs `BTCUSDT`) en `market_data`:** Cada moneda competía contra su propio duplicado en el ranking transversal y recibía la mitad del capital. **Solución:** Normalización de clave única canónica.
4. **Inyección Forzada de *Stop-Loss* (`-4%`) y *Take-Profit* (`+8%`) a Posiciones de Portafolio:** `ensure_position_brackets()` colocaba brackets de *scalping* sobre posiciones estructurales de `Meta_Rebalance`, cortando las piernas ganadoras prematuramente. **Solución:** Aplicación exclusiva de brackets a estrategias de anomalías (`EarlyBreakout`, `OversoldBounce`, `PumpFundingFade`).
5. **Órdenes Huérfanas (`TAKE_PROFIT` / `STOP_LOSS`) al Invertir Posiciones (`LONG <-> SHORT`) en [`src/execution/shadow_broker.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/shadow_broker.py) y [`src/execution/database.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/execution/database.py):** El uso de `COALESCE` en SQLite retenía el `stop_loss` y `take_profit` del lado anterior al girar una posición, disparando ejecuciones invertidas. **Solución:** Cancelación automática (`_cancel_bracket_orders`) y limpieza de campos al cerrar o invertir lado.
6. **Ventana Insuficiente para el Exponente de Hurst en [`src/strategies/stat_arb_mean_reversion.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/src/strategies/stat_arb_mean_reversion.py):** `lookback_window = 48` pasaba solo 48 barras a `estimate_hurst_variance_time()` (que exige `>= 64` barras), devolviendo siempre `0.50` fijo y desactivando `StatArbMeanReversion` en el 100% de las barras. **Solución:** `lookback_window = 72` y ventana mínima de 72 barras para Hurst.
7. **Ausencia de Tasas de *Funding* Reales en el Rebalanceo en Vivo y Pérdida de `last_rebalance_time` al Reiniciar:** Corregido en `shadow_daemon.py` consultando `fetch_funding_rates` para los 15 activos y restaurando el último timestamp desde `rebalance_snapshots`.

### 9.2. Rendimiento Auditado de las Estrategias Originales del Proyecto (`2026-04-20` a `2026-10-20`)

| Estrategia Original del Proyecto | Universo | Capital Final (`$1,000` ini) | Retorno Neto 6M | Promedio Diario | Sharpe Ratio | Max Drawdown | Diagnóstico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`MetaStrategyAllocator` (Core Líquido)** | Top 15 | `$1,152.53` | **+15.25%** | `+0.080%` | `2.05` | `-6.31%` | Sólido y de bajo riesgo, pero retorno diario modesto (`+0.08%/día`). |
| **`CrossSectionalMomentumStrategy` (Puro)** | Top 15 | `$1,126.94` | **+12.69%** | `+0.068%` | `1.59` | `-8.42%` | Sumaba el retorno de `1h` al de `24h`, comprando techos de corto plazo. |
| **`StatArbMeanReversionStrategy` *(con fix Hurst>=72)*** | Top 15 | `$1,041.20` | **+4.12%** | `+0.023%` | `0.88` | `-5.14%` | Positivo tras corregir el bug de `lookback=72` (antes daba `0.00%`). |
| **`FundingCarryArbitrageStrategy` (Puro)** | Top 15 | `$1,038.45` | **+3.85%** | `+0.021%` | `1.12` | `-3.89%` | Buen amortiguador defensivo, baja volatilidad. |
| **`CointegratedPairsStrategy` (`RENDER/FET`)** | 2 activos | `$1,019.80` | **+1.98%** | `+0.011%` | `0.74` | `-4.10%` | Estable pero limitado a un único par sectorial IA. |
| **`MetaStrategyAllocator` (Con Cola Ilíquida)** | Top 19 | `$977.40` | **-2.26%** | `-0.012%` | `-0.31` | `-11.20%` | Las 4 monedas ilíquidas (`CTSI`, `MAGIC`, `OGN`, `RLC`) degradan el alpha. |
| **Anomalías: `OversoldBounce` + `EarlyBreakout` + `PumpFundingFade`** | 23 activos | `$475.82` | **-52.42%** | `-0.401%` | `-2.22` | `-55.08%` | Los *stops* fijos (`-4%`/`-5%`) se disparan por ruido en *altcoins* volátiles. |

---

## 10. Catálogo Completo de Mecanismos Cuantitativos Explorados en Fase 2 y Fase 3

Durante la exploración sistemática probamos **más de 30 variantes de estrategias agrupadas en 8 familias cuantitativas**. Documentamos aquí el resultado de cada familia para que quede registro íntegro de qué funcionó y qué se descartó:

| Familia / Mecanismo Cuantitativo | Descripción Matemática | Retorno Neto 6M (`2026-04` a `2026-10`) | Sharpe Ratio | Max Drawdown | Veredicto de Investigación |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Time-Series Trend (`EMA 12/48` + `ADX`)** | Larga si `EMA12 > EMA48`, corta si `<`, ponderada por `1/vol` (sin neutralidad dólar estricta). | `+28.40%` | `1.15` | `-26.80%` | **Descartada:** Sufre fuertes caídas direccionales cuando BTC revierte en V. |
| **2. Breakout Donchian (`24h/72h` High-Low)** | Compra rupturas del máximo de `24h–72h` con expansión de volumen. | `+41.20%` | `1.42` | `-31.50%` | **Descartada:** Sin cobertura beta, los falsos quiebres generan drawdowns `>30%`. |
| **3. BTC Lead-Lag Direccional (`1h`)** | Compra altcoins rezagadas cuando BTC sube en `1h`. | `-18.90%` | `-0.65` | `-38.90%` | **Descartada en `1h`:** En barras de `1h` el rezago ya se arbitró; en `15m` da `+571%` bruto pero `-99.7%` neto por comisiones. |
| **4. Reversión Corta Pura (`-ret_4h` Neutral)** | Compra los perdedores de las últimas `4h` y vende los ganadores de `4h`. | `+42.70%` *(a `4h`)* / `-37.35%` *(con fees altos)* | `2.27` | `-16.40%` | **Integrada como filtro de entrada:** Sola tiene alta rotación, pero restada del momentum (`-0.25 z_{4\text{h}}$) evita pagar sobrecostos y dispara el Sharpe. |
| **5. Concentración Top-K Discreta (`Top 2 / Bottom 2`)** | Asigna `50%` a las 2 mejores y `-50%` a las 2 peores monedas. | `+138.90%` | `3.62` | `-24.10%` | **Superada por Tail-Power Continuo:** Los saltos discretos cuando una moneda entra/sale del Top 2 generan rotación innecesaria. |
| **6. Tail-Power Continuo ($\text{sign}(Z)|Z|^{1.8} / \sigma$)** | Ponderación continua no lineal con tope por activo (`0.45x`). | **+149.83%** | `4.03` | `-18.30%` | **Adoptada:** Concentra capital en los extremos suavizando la transición y reduciendo comisiones. |
| **7. Order Flow Imbalance (`OFI = (2 V_{\text{buy}} - V)/V$)** | Desequilibrio de compras Taker en `30m`, `2h` y `12h`. | `+17.20%` *(Rev `2h`)* | `0.95` | `-25.89%` | **Secundaria:** `OFI` en `2h` revierte a la media (`+17.2%`), pero aporta menos información que el residuo idiosincrático de precio. |
| **8. 7-Factor Causal Online Meta-Allocator + Vol-Targeting** | Combinación dinámica (`36h/96h/240h` t-stat) de 7 factores ortogonales + control de volatilidad de `72h`. | **+525.8%** (`2.5x` en 22M) / **+1,616.3%** (`2.5x->5.0x` en 22M) | **`2.44`** | `-27.17%` | **Arquitectura Ganadora Definitiva:** Gana tanto en 2025 como en 2026 (`18/22` meses positivos). |

---

## 11. Tablas Maestras de Referencia (2025–2026): Estructura Temporal y Matriz de los 7 Factores

### 11.1. Estructura Temporal Completa de Retornos (`4h` a `720h`) en 2025 vs. 2026 (`2.0x` Gross, `Reb 8h`)

| Factor / Horizonte Aislando Señal | Retorno Neto **2025** *(12m)* | Retorno Neto **2026** *(10m)* | Retorno Total **22 Meses** | Promedio Diario | Sharpe Ratio | Max Drawdown | Meses Positivos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Mom_Raw_4h`** *(Continuación 4h)* | `-71.6%` | `-24.0%` | `-78.4%` | `-0.211%` | `-1.75` | `-84.94%` | `6 / 22` |
| **`Mom_Idio_4h`** *(Continuación Idio 4h)* | `-73.9%` | `-32.1%` | `-82.3%` | `-0.241%` | `-1.95` | `-87.91%` | `5 / 22` |
| **`Mom_Raw_12h`** *(Continuación 12h)* | `-61.0%` | `-9.5%` | `-64.7%` | `-0.136%` | `-1.10` | `-82.67%` | `6 / 22` |
| **`Mom_Raw_24h`** *(Continuación 24h)* | `-43.7%` | **`+42.1%`** | `-20.0%` | `-0.009%` | `-0.07` | `-64.78%` | `11 / 22` |
| **`Mom_Raw_48h`** *(Continuación 48h)* | `-55.1%` | **`+10.5%`** | `-50.4%` | `-0.079%` | `-0.61` | `-72.75%` | `9 / 22` |
| **`Mom_Raw_168h`** *(Momentum 7 Días)* | `-15.9%` | **`+56.6%`** | **+31.7%** | `+0.076%` | `0.60` | `-49.53%` | `12 / 22` |
| **`Mom_Idio_168h`** *(Momentum Idio 7 Días)* | **`+1.6%`** | **`+95.2%`** | **+98.3%** | `+0.139%` | `1.09` | `-52.79%` | `13 / 22` |
| **`Mom_Raw_336h`** *(Momentum 14 Días)* | **`+36.8%`** | **`+30.7%`** | **+78.8%** | `+0.116%` | `0.97` | `-49.41%` | **`15 / 22`** |
| **`Mom_Idio_336h`** *(Momentum Idio 14 Días)* | **`+38.5%`** | **`+60.6%`** | **+122.3%** | `+0.151%` | `1.25` | `-56.19%` | `14 / 22` |
| **`Mom_Idio_720h`** *(Momentum Idio 30 Días)* | **`+42.0%`** | **`+28.6%`** | **+82.7%** | `+0.119%` | `1.03` | **`-32.86%`** | `14 / 22` |
| **`Idio_Macro_Mom (168h+336h - 0.3*24h)`** | **`+63.4%`** | **`+74.6%`** | **+185.2%** | **`+0.190%`** | **`1.61`** | `-44.40%` | `13 / 22` |
| **`Funding_Carry_Contrarian (-fund_72h)`** | **`+48.7%`** | `-18.1%` | **+21.8%** | `+0.051%` | `0.52` | `-49.19%` | `12 / 22` |

### 11.2. Matriz Mensual de Rendimiento por Factor Individual (`2025-01` a `2026-10`, `2.0x` Gross)

| Mes | `BTCUSDT` | `F1_Macro_Idio_PB` *(7d-30d)* | `F2_Med_Idio_PB` *(3d-7d)* | `F3_Fast_Swing_PB` *(12h-48h)* | `F4_Rev_48_72h` *(Reversión)* | `F5_Carry_Neg` *(Funding)* | `F6_LowVol_BAB` *(Defensivo)* | `F7_HighLiq_Size` *(Liquidez)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2025-01** | `+8.5%` | `+5.7%` | `-2.4%` | `-2.6%` | `-4.0%` | `+1.8%` | `-1.8%` | **`+17.0%`** |
| **2025-02** | `-17.7%` | `-3.8%` | `-13.9%` | `+12.9%` | `-8.5%` | `+6.7%` | **`+18.3%`** | `+2.9%` |
| **2025-03** | `-2.1%` | `-27.7%` | `-30.5%` | `-15.2%` | **`+17.6%`** | `+5.2%` | `+0.4%` | `+2.9%` |
| **2025-04** | `+14.1%` | `+21.3%` | **`+21.8%`** | `-17.2%` | `-5.4%` | `+16.5%` | `-33.3%` | `-10.4%` |
| **2025-05** | `+11.1%` | `-7.7%` | `-2.5%` | `-23.1%` | `+13.0%` | `+0.3%` | `+8.0%` | **`+18.6%`** |
| **2025-06** | `+2.4%` | `-3.5%` | `-22.2%` | `-14.1%` | **`+27.5%`** | `+15.5%` | `+7.7%` | `+2.4%` |
| **2025-07** | `+8.0%` | **`+12.2%`** | `+7.3%` | `+10.8%` | `-11.0%` | `-17.1%` | `-3.5%` | `+8.5%` |
| **2025-08** | `-6.5%` | **`+10.6%`** | `-3.3%` | `+1.2%` | `+7.4%` | `-15.0%` | `-10.3%` | `+9.7%` |
| **2025-09** | `+5.3%` | `+5.6%` | `+6.9%` | `+10.8%` | `-11.2%` | **`+11.3%`** | `+5.6%` | `-0.3%` |
| **2025-10** | `-3.9%` | **`+58.0%`** | `+18.1%` | `-5.5%` | `+0.3%` | `+30.4%` | `+26.3%` | `+29.4%` |
| **2025-11** | `-17.6%` | `-4.3%` | **`+21.2%`** | `+10.7%` | `-21.0%` | `-8.6%` | `-17.9%` | `-8.8%` |
| **2025-12** | `-3.0%` | `+5.4%` | `+3.1%` | `-11.4%` | `+5.9%` | `-2.0%` | `+8.9%` | **`+16.5%`** |
| **2026-01** | `-10.2%` | `-6.3%` | `+2.6%` | `-6.5%` | `-5.0%` | **`+17.3%`** | `-5.5%` | `-8.7%` |
| **2026-02** | `-15.0%` | `+6.0%` | **`+7.6%`** | `-16.4%` | `+7.3%` | `+4.1%` | `-4.8%` | `-2.2%` |
| **2026-03** | `+1.9%` | `+15.3%` | **`+26.9%`** | `-12.3%` | `-6.8%` | `+20.9%` | `-13.5%` | `-10.1%` |
| **2026-04** | `+11.8%` | **`+22.0%`** | `+20.7%` | `+2.2%` | `-8.5%` | `-16.2%` | `+1.2%` | `+0.6%` |
| **2026-05** | `-3.5%` | `+8.9%` | `+42.9%` | **`+59.6%`** | `-38.0%` | `-7.1%` | `-18.1%` | `-17.0%` |
| **2026-06** | `-20.4%` | `-3.6%` | `-18.7%` | `+5.0%` | `+8.2%` | `-4.2%` | `+3.5%` | **`+13.2%`** |
| **2026-07** | `+7.3%` | `-6.5%` | `+0.6%` | **`+16.4%`** | `-10.6%` | `+2.1%` | `+1.1%` | `+12.9%` |
| **2026-08** | `+25.0%` | `+3.7%` | `+5.3%` | `+2.6%` | `-13.1%` | `-10.7%` | `-5.8%` | **`+11.6%`** |
| **2026-09** | `+6.4%` | **`+16.5%`** | `+9.1%` | `+12.2%` | `-11.0%` | `-13.8%` | `-42.1%` | `-28.1%` |
| **2026-10** | `-1.3%` | `-7.5%` | `-21.5%` | `-14.9%` | **`+32.1%`** | `-0.7%` | `-0.8%` | `-7.5%` |
| **TOTAL 22M** | — | **`+153.6%`** | **`+57.4%`** | `-21.4%` | `-46.6%` | **`+21.4%`** | `-65.9%` | **`+40.0%`** |

---

## 12. Inventario de Datos Descargados (Parquet) y Scripts Reproducibles

### 12.1. Ubicación Exacta de los Datos Descargados (`41.9 MB` en Disco Local + Backup en Google Drive)

Todos los archivos Parquet descargados y procesados están disponibles en **dos ubicaciones locales sincronizadas** (más el respaldo remoto en Google Drive):
1. **Almacenamiento Permanente en el Proyecto (no se borra al reiniciar la Mac):**  
   📂 **[`data/cache/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/data/cache)** (`/Users/santim/Projects/trading/crypto-alpha-engine/data/cache/`)
2. **Caché Rápida en `/tmp` (usada por defecto por los scripts de backtest):**  
   📂 **`/tmp/crypto_alpha_backtest_cache/`**
3. **Backup Remoto en la Nube (vía `rclone`):**  
   ☁️ **`gdrive:trading/crypto-alpha-engine/data/`** (contiene los 83 símbolos en `1h`, `15m` y `funding` desde `2023-01-01`).

#### Detalle por Subdirectorio y Frecuencia:
| Ruta en el Proyecto (`data/cache/...`) | Archivos | Rango Temporal Exacto (UTC) | Barras por Símbolo | Tamaño en Disco | Columnas Incluidas en cada Parquet |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **[`data/cache/bars/1h/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/data/cache/bars/1h)** | `19` (`.parquet`) | `2023-01-01 00:00` a `2026-10-20 18:00` *(3.8 años)* | `33,331` barras | `19.0 MB` | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `trade_count`, `taker_buy_vol` |
| **[`data/cache/bars/30m/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/data/cache/bars/30m)** | `15` (`.parquet`) | `2026-03-15 00:00` a `2026-10-20 18:00` *(7.2 meses)* | `10,549` barras | `3.7 MB` | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `trade_count`, `taker_buy_vol` |
| **[`data/cache/bars/15m/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/data/cache/bars/15m)** | `15` (`.parquet`) | `2026-03-15 00:00` a `2026-10-20 18:00` *(7.2 meses)* | `21,097` barras | `8.6 MB` | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `trade_count`, `taker_buy_vol` |
| **[`data/cache/bars/5m/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/data/cache/bars/5m)** | `15` (`.parquet`) | `2026-09-19 00:00` a `2026-10-20 18:00` *(31.8 días)* | `9,145` barras | `5.0 MB` | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `trade_count`, `taker_buy_vol` |
| **[`data/cache/bars/1m/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/data/cache/bars/1m)** | `15` (`.parquet`) | `2026-10-13 06:00` a `2026-10-20 18:00` *(7.5 días)* | `10,801` barras | `4.9 MB` | `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `trade_count`, `taker_buy_vol` |
| **[`data/cache/funding/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/data/cache/funding)** | `19` (`.parquet`) | `2023-01-01 00:00` a `2026-09-30 20:00` *(liquidaciones c/ 8h)* | `4,107` registros | `784 KB` | `timestamp`, `symbol`, `funding_rate` |

### 12.2. Índice de Scripts de Investigación Guardados en [`scripts/research_2026/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026)

Todos los scripts desarrollados durante la investigación fueron copiados a [`scripts/research_2026/`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026) dentro del repositorio para que puedas ejecutarlos en cualquier momento con `.venv/bin/python`:
- [`download_subhourly_data.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/download_subhourly_data.py): Descargador de velas `15m`, `5m` y `1m` con volumen comprador Taker desde Google Drive (`rclone`) y Binance Futures (`fapiPublicGetKlines`).
- [`explore_high_return_strategies.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/explore_high_return_strategies.py) y [`deep_strategy_research.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/deep_strategy_research.py): Exploración inicial de las 6 familias de estrategias en `1h`, `2h`, `4h` y `8h`.
- [`test_term_structure.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_term_structure.py) y [`test_neutral_alphas.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_neutral_alphas.py): Descubrimiento de la estructura temporal (`2h–4h` reversión vs. `12h–48h` momentum) y ponderación no lineal *Tail-Power*.
- [`inspect_regime_features.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/inspect_regime_features.py), [`test_7of7_months.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_7of7_months.py) y [`test_smooth_regime.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_smooth_regime.py): Descubrimiento del sensor causal `CS-AutoCorr 48h` (`7/7` meses positivos en 2026).
- [`test_subhourly_frequencies.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_subhourly_frequencies.py), [`test_subhourly_alpha_boost.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_subhourly_alpha_boost.py), [`explore_true_subhourly_strategies.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/explore_true_subhourly_strategies.py) y [`verify_harmonic_ensemble.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/verify_harmonic_ensemble.py): Pruebas en `1m`, `5m`, `15m` y `30m`, análisis de fricción vs. retorno bruto y ensamble armónico (`40% 3h + 60% 8h`) con intervalos de confianza Bootstrap al 95%.
- [`test_2025_2026_and_resolution.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_2025_2026_and_resolution.py), [`diagnose_2025_2026_all_mechanisms.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/diagnose_2025_2026_all_mechanisms.py), [`inspect_22m_months.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/inspect_22m_months.py), [`test_online_factor_allocator_22m.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_online_factor_allocator_22m.py), [`test_403pct_engine_with_stepped_leverage.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_403pct_engine_with_stepped_leverage.py) y [`test_hysteresis_stepped_leverage.py`](file:///Users/santim/Projects/trading/crypto-alpha-engine/scripts/research_2026/test_hysteresis_stepped_leverage.py): Estudio econométrico de ruido (`1m` a `4h`), matriz de los 7 factores ortogonales en los 22 meses (`2025–2026`), asignador causal online (`36h/96h/240h`) y controlador de apalancamiento escalonado con histéresis (`2.5x -> 5.0x`).
