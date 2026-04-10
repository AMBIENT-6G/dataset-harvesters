import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

db_path = "032026_data.db"
conn = sqlite3.connect(db_path)

band = 900

if band == 900:
    fbmin = 800
    fbmax = 1000

volt = 1000 
pwrs = [-15,-10,-5,0]

for pwr in pwrs:

    # Retrieve harvester list
    tables = pd.read_sql("""
        SELECT name FROM sqlite_master
        WHERE type='table'
    """, conn)["name"].tolist()

    harvesters = []

    for t in tables:
        result = pd.read_sql(f"""
            SELECT 1
            FROM "{t}"
            WHERE source = ?
            LIMIT 1
        """, conn, params=("measured",))
        
        if not result.empty:
            harvesters.append(t)

    print(harvesters)

    plt.figure(figsize=(10,6))

    for table in harvesters: 

        df = pd.read_sql(f"""
            SELECT *
            FROM {table}
            WHERE source = ?
        """, conn, params=("measured",))

        df = df[df["target_voltage_mv"] == volt]
        df = df[df["level_dbm"] == pwr]
        df = df[(df["frequency_mhz"] > fbmin) & (df["frequency_mhz"] < fbmax)]

        x = df["frequency_mhz"]
        y = df["efficiency"]

        tf_mhz = df["tuning_frequency_mhz"].iloc[0]

        plt.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            label=f"{table}({round(tf_mhz)})"#({round(tf_mhz)}), {round(freq,1)} MHz, {bv} V, t{round(volt)}, {source}"
        )

        idx_max = y.idxmax()

        x_max = x.loc[idx_max]
        y_max = y.loc[idx_max]

        plt.annotate(f"({x_max:.2f},{y_max:.2f})",
                    (x_max, y_max),
                    textcoords="offset points",
                    xytext=(10,10),
                    arrowprops=dict(arrowstyle="->"))


    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Efficiency [%]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()


    import os

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    parent_path = os.path.dirname(current_dir)
    filename = os.path.splitext(os.path.basename(current_file_path))[0]

    import matplot2tikz

    matplot2tikz.save(f"{current_dir}/plots/{filename}_{volt}_{pwr}.tex")



# exit()

# print(tables)

# SMS7630005LF

# dfs = dfs[dfs["target_voltage_mv"] == volt]


# # plt.figure(figsize=(10,6))

# for table in tables:
#     print(table)
#     try:
#         df = pd.read_sql(f"""
#             SELECT frequency_mhz, level_dbm, efficiency, source, buffer_voltage_mv, tuning_frequency_mhz, target_voltage_mv
#             FROM "{table}"
#             WHERE frequency_mhz BETWEEN ? AND ?
#         """, conn)

#         df["harvester"] = table  # handig om te weten uit welke tabel het komt
#         # dfs.append(df)

#         print(df["harvester"])


#         df_f = df_f[df_f["target_voltage_mv"] == volt]

#         df_f = df_f[df_f["level_dbm"] == pwr]

#         print(df_f)


#         plt.figure()
#         # plt.plot(df_f["frequency_mhz"], df_f["efficiency"])
#         plt.plot(
#                     df_f["frequency_mhz"],
#                     df_f["efficiency"],
#                     marker="o",
#                     linestyle="-",
#                     label=f"{harvester}"#({round(tf_mhz)}), {round(freq,1)} MHz, {bv} V, t{round(volt)}, {source}"
#                 ) 
        
#         plt.show()



#     #     unique_frequencies = sorted(df["frequency_mhz"].unique())

#     #     print(unique_frequencies)

#     #     unique_voltages = sorted(
#     #         df.loc[
#     #             (df["target_voltage_mv"] >= v_min) &
#     #             (df["target_voltage_mv"] <= v_max),
#     #             "target_voltage_mv"
#     #         ].unique()
#     #     )

#     #     print(unique_voltages)

#     #     unique_voltages = sorted(
#     #         df.loc[
#     #             (df["target_voltage_mv"] >= v_min) &
#     #             (df["target_voltage_mv"] <= v_max),
#     #             "target_voltage_mv"
#     #         ].unique()
#     #     )

#     #     for volt in unique_voltages:

#     #         for freq in unique_frequencies:

#     #             df_f = df[df["frequency_mhz"] == freq]

#     #             df_f["buffer_voltage"] = df_f["buffer_voltage_mv"] / 1000.0

#     #             df_f = df_f[df_f["target_voltage_mv"] == volt]

#     #             if df_f.empty:
#     #                 continue

#     #             # Filter de DataFrame: alleen waarden binnen median ± marge
#     #             df_f = df_f[
#     #                 (df_f["buffer_voltage"] >= volt/1000 - target_voltage_marge) &
#     #                 (df_f["buffer_voltage"] <= volt/1000 + target_voltage_marge)
#     #             ]

#     #             source = df_f["source"].iloc[0]
#     #             harvester = df_f["harvester"].iloc[0]
#     #             bv = round(np.mean(df_f["buffer_voltage"]),2)
#     #             tf_mhz = df_f["tuning_frequency_mhz"].iloc[0]

#     #             print(df_f["buffer_voltage"])

#     #             # Bereken de mediaan van de kolom
#     #             median_value = np.median(df_f["buffer_voltage"])

#     #             # Sort
#     #             df_plot = df_f.sort_values(by="level_dbm")
                
#     #             plt.plot(
#     #                 df_plot["level_dbm"],
#     #                 df_plot["efficiency"],
#     #                 marker="o",
#     #                 linestyle="-",
#     #                 label=f"{harvester}({round(tf_mhz)}), {round(freq,1)} MHz, {bv} V, t{round(volt)}, {source}"
#     #             ) 

#     except Exception as e:
#         print(f"⚠️ Skipping table {table}: {e}")

# conn.close()


# exit()

# plt.xlabel("Input power (dBm)")
# plt.ylabel("Efficiency (%)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# # plt.show()



# import os

# current_file_path = os.path.abspath(__file__)
# current_dir = os.path.dirname(current_file_path)
# parent_path = os.path.dirname(current_dir)
# filename = os.path.splitext(os.path.basename(current_file_path))[0]

# import matplot2tikz

# matplot2tikz.save(f"{current_dir}/plots/{filename}_{f_target}.tex")