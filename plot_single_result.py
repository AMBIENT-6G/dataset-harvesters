import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


def plot_harvester_csv(csv_file):
    # ---- Extract info from filename ----
    filename = Path(csv_file).stem

    # Example: P2110B_datasheet_868mhz
    match = re.search(r"(?P<name>P\d+[A-Z]*)_.*_(?P<freq>\d+)\s*mhz", filename, re.IGNORECASE)
    if not match:
        raise ValueError("Filename does not match expected pattern")

    harvester_name = match.group("name")
    frequency_mhz = match.group("freq")
    print(f"Harvester: {harvester_name}, Frequency: {frequency_mhz} MHz")

    filename_path = Path(f"{harvester_name}/{filename}.csv")

    # ---- Load CSV ----
    df = pd.read_csv(filename_path)

    # ---- Plot ----
    plt.figure(figsize=(6, 4))
    plt.plot(
        df["input_power_dbm"],
        df["efficiency"],
        marker="o",
        linestyle="-"
    )

    plt.xlabel("Input Power (dBm)")
    plt.ylabel("Efficiency (-)")
    plt.title(f"{harvester_name} - {frequency_mhz} MHz")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ---- Example usage ----
plot_harvester_csv("P2110B_datasheet_868mhz.csv")
