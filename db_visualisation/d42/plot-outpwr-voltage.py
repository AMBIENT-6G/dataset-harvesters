import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

db_path = "032026_data.db"
conn = sqlite3.connect(db_path)

f_min = 910
f_max = 920
# f_min = 2450
# f_max = 2460

# v_min = 1000
# v_max = 1100

pwr = -10

# Stel marge in (bijvoorbeeld ±50 mV)
target_voltage_marge = 0.05

# Haal alle tabelnamen op
tables = pd.read_sql("""
    SELECT name FROM sqlite_master
    WHERE type='table'
""", conn)["name"].tolist()

dfs = []

volt = None
power = None

# plt.figure(figsize=(10,6))

for table in tables:

    # if table == "P2110B":
    #     continue

    if table == "sSUHFIPTIVA0":

        print(table)
        try:
            df = pd.read_sql(f"""
                SELECT frequency_mhz, level_dbm, efficiency, source, buffer_voltage_mv, tuning_frequency_mhz, target_voltage_mv, pwr_pw
                FROM "{table}"
                WHERE frequency_mhz BETWEEN ? AND ?
            """, conn, params=(f_min, f_max))

            df["harvester"] = table  # handig om te weten uit welke tabel het komt
            # dfs.append(df)

            unique_frequencies = sorted(df["frequency_mhz"].unique())


            print(unique_frequencies)

            for freq in unique_frequencies:

                print(freq)

                df_f = df.loc[
                    (df["frequency_mhz"] == freq) &
                    (df["level_dbm"] == pwr)
                ].copy()

                df_f["buffer_voltage"] = df_f["buffer_voltage_mv"] / 1000.0

                harvester = df_f["harvester"].iloc[0]
                tf_mhz = df_f["tuning_frequency_mhz"].iloc[0]

                # Sort
                df_plot = df_f.sort_values(by="buffer_voltage_mv")
                
                # global storage
                volt = df_plot["buffer_voltage_mv"]/1e3
                power = df_plot["pwr_pw"]/1e6

                # plt.plot(
                #     df_plot["buffer_voltage_mv"],
                #     df_plot["pwr_pw"]/1e6,
                #     marker="o",
                #     linestyle="-",
                #     label=f"{harvester}({round(tf_mhz)}), {round(freq,1)} MHz"
                # ) 

                
        except Exception as e:
            print(f"⚠️ Skipping table {table}: {e}")

conn.close()

import math

def eng_format(x, precision=1):
    if x == 0:
        return "0"
    
    exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
    mant = x / (10 ** exp)
    
    if exp == 0:
        return f"{mant:.{precision}f}"
    else:
        return f"{mant:.{precision}f}e{exp:+03d}"
    
# plt.figure()
# plt.plot(volt, power)
# plt.xlabel("Voltage (V)")
# plt.ylabel("Power (W)")
# plt.title("Power vs Voltage")
# plt.grid()

# plt.show()


# LS fit (graad 1 = lineair)
coeffs = np.polyfit(volt, power, 2)  # [a, b] voor y = a*x + b
# fit_line = np.polyval(coeffs, volt)
fit_curve = np.polyval(coeffs, volt)

volt_fit = np.linspace(0, max(volt), 100)
fit_curve = np.polyval(coeffs, volt_fit)

plt.figure()
plt.title(f"Output power level vs buffer voltage for input power {pwr} dBm")
plt.plot(volt, power, 'o', label="Data")
# plt.plot(volt, fit_line, label=f"Fit: y={coeffs[0]:.3f}x + {coeffs[1]:.3f}")
plt.plot(volt_fit, fit_curve, label=f"Fit: {eng_format(coeffs[0])}x² + {eng_format(coeffs[1])}x + {eng_format(coeffs[2])}")
plt.xlabel("Voltage (V)")
plt.ylabel("Power (uW)")
plt.grid()
plt.legend()

plt.show()


# print(np.polyval(coeffs, 0.2))

# exit()

# C = 10e-3
# dt = 0.1
# t_max = 60*60
# V_max = 2

# V = 0.05
# V_list = []
# t_list = []

# for t in np.arange(0, t_max, dt):
#     # P = 0.01  # voorbeeld: 10 mW (of maak functie van V)
#     power = np.polyval(coeffs, V)/1e6  # vermogen in W, afhankelijk van V

#     # power = 0.01
    
#     dVdt = power / (C * V)
#     V += dVdt * dt

#     if V >= V_max:
#         break

#     V_list.append(V)
#     t_list.append(t)

# print(len(V_list))


# indices = np.linspace(0, len(V_list)-1, 100, dtype=int)
# V_list_reduced = [V_list[i] for i in indices]
# t_list_reduced = [t_list[i] for i in indices]


# print(len(V_list))
# print(len(V_list_reduced))

# plt.title(f"Voltage is {max(V_list):.3f} V after {t_list[-1]:.1f} seconds")
# plt.plot(t_list_reduced, V_list_reduced, "o")
# plt.xlabel("Time (s)")
# plt.ylabel("Voltage (V)")
# plt.grid()
# plt.show()

# print(max(V_list))



