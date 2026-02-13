import sqlite3
import csv
import os
import pandas as pd
from pathlib import Path
from helper import *
import glob

#----------------------------------------------------#
#                      Settings                      #
#----------------------------------------------------#
REPLACE = True
#-------------------sSUHFIPTIVA0---------------------#
# data_name = "sSUHFIPTIVA0_900_measured_t1600.csv"
#---------------------AEM40940-----------------------#
# data_name = "AEM40940_867_datasheet_t0.csv"
# data_name = "AEM40940_921_datasheet_t0.csv"
# data_name = "AEM40940_2400_datasheet_t0.csv"
#--------------------P1110B--------------------------#
# data_name = "P1110B_915_datasheet_t0.csv"
#----------------------------------------------------#
#--------------------P2110B--------------------------#
# data_name = "P2110B_915_datasheet_t0.csv"
#----------------------------------------------------#
#-------------------SMS7630-005LF--------------------# (LOW POWER DIODE)
#data_name = "SMS7630-005LF_915_measured_t*.csv"
#----------------------------------------------------#
#-------------------SMS7621-005LF--------------------# (HIGH POWER DIODE)
# data_name = "SMS7621005LF_915_measured_t*.csv"
data_name = "SMS7621005LF_2450_measured_t*.csv"
#----------------------------------------------------#


# Path to save the experiment data as a YAML file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_path = os.path.dirname(current_dir)

# Folder name
stem = Path(data_name).stem
folder_name = "_".join(stem.split("_")[:-2])

pattern = os.path.join(f"{parent_path}/{folder_name}", data_name)

files = sorted(glob.glob(pattern))

for f in files:

    data_name = Path(f).name

    # Extract harvester name and tuning frequency from folder name
    harvester_name, tuning_frequency_mhz = folder_name.rsplit("_", 1)

    stem_parts = Path(data_name).stem.split("_")
    target_voltage_str = next(p for p in stem_parts if p.startswith("t"))
    target_voltage_mv = float(target_voltage_str[1:])
    data_source = next(p for p in stem_parts if p.isalpha())

    # 1. Verbinding maken (bestand wordt automatisch aangemaakt)
    conn = sqlite3.connect(f"{parent_path}/data.db")
    cur = conn.cursor()

    # 2. Tabel aanmaken (eenmalig)
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {harvester_name} (
        tuning_frequency_mhz REAL,
        target_voltage_mv REAL,
        source TEXT,
        frequency_mhz REAL,
        level_dbm REAL,
        buffer_voltage_mv REAL,
        resistance REAL,
        pwr_pw REAL,
        efficiency REAL
    )
    """)

    csv_path = f"{parent_path}/{folder_name}/{data_name}"

    # Check and potentially add efficiency column to csv
    check_columns(csv_path)

    # Check all data is available
    if not check_header(csv_path):
        exit()

    # Round df
    df_round(csv_path)

    # 3. CSV inlezen en invoegen
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Check of dezelfde rij al bestaat
                cur.execute(f"""
                    SELECT COUNT(*) FROM {harvester_name}
                    WHERE tuning_frequency_mhz = ?
                    AND target_voltage_mv = ?
                    AND source = ?
                    AND frequency_mhz = ?
                    AND level_dbm = ?
                """, (
                    float(tuning_frequency_mhz),
                    float(target_voltage_mv),
                    str(data_source),
                    float(row["frequency_mhz"]),
                    float(row["level_dbm"])
                ))
                exists = cur.fetchone()[0]

                if exists:
                    if REPLACE:
                        # Update de bestaande rij
                        cur.execute(f"""
                            UPDATE {harvester_name}
                            SET buffer_voltage_mv = ?,
                                resistance = ?,
                                pwr_pw = ?,
                                efficiency = ?
                            WHERE tuning_frequency_mhz = ?
                            AND target_voltage_mv = ?
                            AND source = ?
                            AND frequency_mhz = ?
                            AND level_dbm = ?
                        """, (
                            float(row["buffer_voltage_mv"]),
                            float(row["resistance"]),
                            float(row["pwr_pw"]),
                            float(row["efficiency"]),
                            float(tuning_frequency_mhz),
                            float(target_voltage_mv),
                            str(data_source),
                            float(row["frequency_mhz"]),
                            float(row["level_dbm"])
                        ))
                        continue  # vervanging gedaan, ga naar volgende rij
                    else:
                        print(f"⚠️ Row already exists: freq={row['frequency_mhz']} Hz, level={row['level_dbm']} dbm, harvester={harvester_name}")
                        continue  # overslaan

                # Insert nieuwe rij
                cur.execute(f"""
                    INSERT INTO {harvester_name} (
                        tuning_frequency_mhz,
                        target_voltage_mv,
                        source,
                        frequency_mhz,
                        level_dbm,
                        buffer_voltage_mv,
                        resistance,
                        pwr_pw,
                        efficiency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    float(tuning_frequency_mhz),
                    float(target_voltage_mv),
                    str(data_source),
                    float(row["frequency_mhz"]),
                    float(row["level_dbm"]),
                    float(row["buffer_voltage_mv"]),
                    float(row["resistance"]),
                    float(row["pwr_pw"]),
                    float(row["efficiency"])
                ))

            except Exception as e:
                print(f"❌ Error inserting row: {e}")
    # 4. Opslaan en sluiten
    conn.commit()
    conn.close()
