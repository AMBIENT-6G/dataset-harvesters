import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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



def calculate_response_time(harvester, f_min, f_max, input_pwr_dbm, C, V_charge_target):

    db_path = "032026_data.db"
    conn = sqlite3.connect(db_path) 

    # Stel marge in (bijvoorbeeld ±50 mV)
    target_voltage_marge = 0.05

    # Haal alle tabelnamen op
    tables = pd.read_sql("""
        SELECT name FROM sqlite_master
        WHERE type='table'
    """, conn)["name"].tolist()

    volt = None
    power = None

    for table in tables:
        if table == harvester:
            try:
                # Retrieve data
                df = pd.read_sql(f"""
                    SELECT frequency_mhz, level_dbm, efficiency, source, buffer_voltage_mv, tuning_frequency_mhz, target_voltage_mv, pwr_pw
                    FROM "{table}"
                    WHERE frequency_mhz BETWEEN ? AND ?
                """, conn, params=(f_min, f_max))

                df["harvester"] = table

                unique_frequencies = sorted(df["frequency_mhz"].unique())

                for freq in unique_frequencies:

                    df_f = df.loc[
                        (df["frequency_mhz"] == freq) &
                        (df["level_dbm"] == input_pwr_dbm)
                    ].copy()

                    df_f["buffer_voltage"] = df_f["buffer_voltage_mv"] / 1000.0

                    harvester = df_f["harvester"].iloc[0]

                    # Sort
                    df_plot = df_f.sort_values(by="buffer_voltage_mv")
                    
                    # global storage
                    volt = df_plot["buffer_voltage_mv"]/1e3
                    power = df_plot["pwr_pw"]/1e6
                    
            except Exception as e:
                print(f"⚠️ Skipping table {table}: {e}")

    conn.close()

    # Add zero
    volt = [0] + (df_plot["buffer_voltage_mv"]/1e3).tolist()
    power = [0] + (df_plot["pwr_pw"]/1e6).tolist()

    # LS fit (graad 1 = lineair)
    coeffs = np.polyfit(volt, power, 2)  # [a, b] voor y = a*x + b

    # fit_line = np.polyval(coeffs, volt)
    fit_curve = np.polyval(coeffs, volt)

    # Residuals
    ss_res = np.sum((power - fit_curve)**2)
    # Total variance
    ss_tot = np.sum((power - np.mean(power))**2)
    # R²
    r2 = 1 - (ss_res / ss_tot)
    print(f"R² = {r2:.4f}")
    
    if r2 < 0.9:
        print("Waarschuwing: slechte fit (R² < 0.9)")
        return [0], [0]

    volt_fit = np.linspace(0, max(volt), 100)
    fit_curve = np.polyval(coeffs, volt_fit)

    # Plot raw data
    line, = plt.plot(volt, power, 'o', label=f"{harvester}")
    color = line.get_color()

    # Plot fitted data
    plt.plot(volt_fit, fit_curve, color=color, linestyle='--',
            # label=f"Fit: {eng_format(coeffs[0])}x² + {eng_format(coeffs[1])}x + {eng_format(coeffs[2])}")
            label=f"Fit")


import numpy as np
import matplotlib.pyplot as plt

harvesters = ['sSUHFIPTIVA0', 'SMS7630005LF', 'SMS7621005LF']
power_levels = [-15]
power_levels = np.arange(-15, 0 + 0.1, 5)
v_charge_target = 0.5
C = 10e-3

freq = 912.5
freq_min = freq-1
freq_max = freq+1
# freq_min = 2450
# freq_max = 2455

# Data verzamelen
results = {h: [] for h in harvesters}

for input_pwr_dbm in power_levels:
    plt.figure()
    plt.title(f"Output power level vs buffer voltage for input power {input_pwr_dbm} dBm")

    for harvester in harvesters:
        calculate_response_time(harvester, freq_min, freq_max, input_pwr_dbm, C, v_charge_target)

    plt.xlabel("Voltage (V)")
    plt.ylabel("Power (uW)")
    plt.grid()
    plt.legend()

    # plt.show()

    import os

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    parent_path = os.path.dirname(current_dir)
    filename = os.path.splitext(os.path.basename(current_file_path))[0]

    import matplot2tikz

    matplot2tikz.save(f"{current_dir}/plots/{filename}-{input_pwr_dbm}-{round(freq)}.tex")
