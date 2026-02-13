import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

db_path = "data.db"
conn = sqlite3.connect(db_path)

f_min = 910
f_max = 920

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


plt.figure(figsize=(10,6))

for table in tables:

    if table == "P2110B":
        continue

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

            plt.plot(
                df_plot["buffer_voltage_mv"],
                df_plot["pwr_pw"]/1e6,
                marker="o",
                linestyle="-",
                label=f"{harvester}({round(tf_mhz)}), {round(freq,1)} MHz"
            ) 

            
    except Exception as e:
        print(f"⚠️ Skipping table {table}: {e}")

conn.close()

plt.title(f"Input power level of {pwr} dBm")
plt.xlabel("Output voltage (V)")
plt.ylabel("Power (uW)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()