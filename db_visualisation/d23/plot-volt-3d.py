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

freq = 2450

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

    df = df[df["frequency_mhz"] == freq]
    # df = df[(df["frequency_mhz"] > fbmin) & (df["frequency_mhz"] < fbmax)]

    # assen
    x = df["target_voltage_mv"]
    y = df["level_dbm"]
    z = df["efficiency"]

    if len(x) == 0:
        continue

    ax.scatter(x, y, z, label=table)

# labels
ax.set_xlabel("Output voltage [mV]")
ax.set_ylabel("Power [dBm]")
ax.set_zlabel("Efficiency [%]")

plt.legend()
plt.tight_layout()
plt.show()



