import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

db_path = "data.db"
conn = sqlite3.connect(db_path)

f_min = 900
f_max = 920

v_min = 1600
v_max = 1700

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
    print(table)
    try:
        df = pd.read_sql(f"""
            SELECT frequency_mhz, level_dbm, efficiency, source, buffer_voltage_mv, tuning_frequency_mhz, target_voltage_mv
            FROM "{table}"
            WHERE frequency_mhz BETWEEN ? AND ?
        """, conn, params=(f_min, f_max))

        df["harvester"] = table  # handig om te weten uit welke tabel het komt
        # dfs.append(df)

        unique_frequencies = sorted(df["frequency_mhz"].unique())

        print(unique_frequencies)

        unique_voltages = sorted(
            df.loc[
                (df["target_voltage_mv"] >= v_min) &
                (df["target_voltage_mv"] <= v_max),
                "target_voltage_mv"
            ].unique()
        )

        print(unique_voltages)

        for volt in unique_voltages:

            for freq in unique_frequencies:

                df_f = df[df["frequency_mhz"] == freq]

                df_f["buffer_voltage"] = df_f["buffer_voltage_mv"] / 1000.0

                df_f = df_f[df_f["target_voltage_mv"] == volt]

                if df_f.empty:
                    continue

                # Filter de DataFrame: alleen waarden binnen median ± marge
                df_f = df_f[
                    (df_f["buffer_voltage"] >= volt/1000 - target_voltage_marge) &
                    (df_f["buffer_voltage"] <= volt/1000 + target_voltage_marge)
                ]

                source = df_f["source"].iloc[0]
                harvester = df_f["harvester"].iloc[0]
                bv = round(np.mean(df_f["buffer_voltage"]),2)
                tf_mhz = df_f["tuning_frequency_mhz"].iloc[0]

                print(df_f["buffer_voltage"])

                # Bereken de mediaan van de kolom
                median_value = np.median(df_f["buffer_voltage"])

                # Sort
                df_plot = df_f.sort_values(by="level_dbm")
                
                plt.plot(
                    df_plot["level_dbm"],
                    df_plot["efficiency"],
                    marker="o",
                    linestyle="-",
                    label=f"{harvester}({round(tf_mhz)}), {round(freq)} MHz, {bv} V, t{volt}, {source}"
                ) 

    except Exception as e:
        print(f"⚠️ Skipping table {table}: {e}")

conn.close()


plt.xlabel("Input power (dBm)")
plt.ylabel("Efficiency (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()