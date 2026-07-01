import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

db_path = "032026_data.db"
conn = sqlite3.connect(db_path)


freq = 912.5



volt = 1000 
pwrs = np.arange(-20,0+2.5,2.5)

h_all = []
x_all = []
y_all = []
z_all = []


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
    
    if table != "sSUHFIPTIVA0" and table != "SMS7630005LF" and table != "SMS7621005LF":
        continue

    df = pd.read_sql(f"""
        SELECT *
        FROM "{table}"
        WHERE source = ?
    """, conn, params=("measured",))

    df = df[df["frequency_mhz"] == freq]
    # df = df[(df["frequency_mhz"] > fbmin) & (df["frequency_mhz"] < fbmax)]

    # assen
    h = [table] * len(df)
    x = df["target_voltage_mv"]
    y = df["level_dbm"]
    z = df["efficiency"]

    if len(x) == 0:
        continue

    ax.scatter(x, y, z, label=table)


    h_all.extend(h)
    x_all.extend(x)
    y_all.extend(y)
    z_all.extend(z)

df_out = pd.DataFrame({
    "harvester": h_all,
    "target_voltage_mv": x_all,
    "level_dbm": y_all,
    "efficiency": z_all,
})

df_out.to_csv("output.csv", index=False)


# labels
ax.set_xlabel("Output voltage [mV]")
ax.set_ylabel("Power [dBm]")
ax.set_zlabel("Efficiency [%]")

plt.legend()
plt.tight_layout()
plt.show()



