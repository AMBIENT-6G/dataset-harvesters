import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

db_path = "032026_data.db"
conn = sqlite3.connect(db_path)

band = 900
# band = 2450

if band == 900:
    fbmin = 800
    fbmax = 1000

if band == 2450:
    fbmin = 2400
    fbmax = 2600

volt = 1000 
pwrs = np.arange(-20,0+2.5,2.5)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Retrieve harvester list (doe dit maar 1x!)
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

# Loop over harvesters
for table in harvesters:

    df = pd.read_sql(f"""
        SELECT *
        FROM "{table}"
        WHERE source = ?
    """, conn, params=("measured",))

    df = df[df["target_voltage_mv"] == volt]
    df = df[(df["frequency_mhz"] > fbmin) & (df["frequency_mhz"] < fbmax)]

    # assen
    x = df["frequency_mhz"]
    y = df["level_dbm"]
    z = df["efficiency"]

    if len(x) == 0:
        continue

    ax.scatter(x, y, z, label=table)

# labels
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Power [dBm]")
ax.set_zlabel("Efficiency [%]")

plt.legend()
plt.tight_layout()
plt.show()

# for pwr in pwrs:

#     # Retrieve harvester list
#     tables = pd.read_sql("""
#         SELECT name FROM sqlite_master
#         WHERE type='table'
#     """, conn)["name"].tolist()

#     harvesters = []

#     for t in tables:
#         result = pd.read_sql(f"""
#             SELECT 1
#             FROM "{t}"
#             WHERE source = ?
#             LIMIT 1
#         """, conn, params=("measured",))
        
#         if not result.empty:
#             harvesters.append(t)

#     print(harvesters)

#     plt.figure(figsize=(10,6))

#     for table in harvesters: 

#         df = pd.read_sql(f"""
#             SELECT *
#             FROM {table}
#             WHERE source = ?
#         """, conn, params=("measured",))

#         df = df[df["target_voltage_mv"] == volt]
#         df = df[df["level_dbm"] == pwr]
#         df = df[(df["frequency_mhz"] > fbmin) & (df["frequency_mhz"] < fbmax)]

#         x = df["frequency_mhz"]
#         y = df["efficiency"]

#         tf_mhz = df["tuning_frequency_mhz"].iloc[0]

#         plt.plot(
#             x,
#             y,
#             marker="o",
#             linestyle="-",
#             label=f"{table}({round(tf_mhz)})"#({round(tf_mhz)}), {round(freq,1)} MHz, {bv} V, t{round(volt)}, {source}"
#         )

#         idx_max = y.idxmax()

#         x_max = x.loc[idx_max]
#         y_max = y.loc[idx_max]

#         plt.annotate(f"({x_max:.2f},{y_max:.2f})",
#                     (x_max, y_max),
#                     textcoords="offset points",
#                     xytext=(10,10),
#                     arrowprops=dict(arrowstyle="->"))


#     plt.xlabel("Frequency [MHz]")
#     plt.ylabel("Efficiency [%]")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


    # import os

    # current_file_path = os.path.abspath(__file__)
    # current_dir = os.path.dirname(current_file_path)
    # parent_path = os.path.dirname(current_dir)
    # filename = os.path.splitext(os.path.basename(current_file_path))[0]

    # import matplot2tikz

    # matplot2tikz.save(f"{current_dir}/plots/{filename}_{volt}_{pwr}.tex")



