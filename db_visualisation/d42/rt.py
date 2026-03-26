import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

    # Add zero
    volt = [0] + (df_plot["buffer_voltage_mv"]/1e3).tolist()
    power = [0] + (df_plot["pwr_pw"]/1e6).tolist()


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

    # volt_fit = np.linspace(0, max(volt), 100)
    # fit_curve = np.polyval(coeffs, volt_fit)
    # plt.figure()
    # plt.title(f"Output power level vs buffer voltage for input power {input_pwr_dbm} dBm")
    # plt.plot(volt, power, 'o', label="Data")
    # # plt.plot(volt, fit_line, label=f"Fit: y={coeffs[0]:.3f}x + {coeffs[1]:.3f}")
    # plt.plot(volt_fit, fit_curve, label=f"Fit: {eng_format(coeffs[0])}x² + {eng_format(coeffs[1])}x + {eng_format(coeffs[2])}")
    # plt.xlabel("Voltage (V)")
    # plt.ylabel("Power (uW)")
    # plt.grid()
    # plt.legend()

    # plt.show()


    dt = 0.1
    t_max = 60*60*10


    V = 0.05
    V_list = []
    t_list = []

    for t in np.arange(0, t_max, dt):
        # P = 0.01  # voorbeeld: 10 mW (of maak functie van V)
        power = np.polyval(coeffs, V)/1e6  # vermogen in W, afhankelijk van V

        # power = 0.01
        
        dVdt = power / (C * V)
        V += dVdt * dt

        if V >= V_charge_target:
            break

        V_list.append(V)
        t_list.append(t)

    # print(len(V_list))


    indices = np.linspace(0, len(V_list)-1, 100, dtype=int)
    V_list_reduced = [V_list[i] for i in indices]
    t_list_reduced = [t_list[i] for i in indices]


    return V_list_reduced, t_list_reduced


# input_pwr_dbm = -10
# C = 10e-3
# # harvesters = ['sSUHFIPTIVA0', 'P2110B', 'SMS7630005LF', 'SMS7621005LF']
# harvesters = ['sSUHFIPTIVA0', 'SMS7630005LF', 'SMS7621005LF']

# power_levels = [-15, -10, -5, 0]

# for input_pwr_dbm in power_levels:
#     for harvester in harvesters:
#         x, y = calculate_response_time(harvester, 910, 913, input_pwr_dbm, 10e-3, 1.5)
#         print(f"{harvester}:")
#         # print(f"Final voltage: {x[-1]}")
#         print(f"Final time: {y[-1]}")


import numpy as np
import matplotlib.pyplot as plt

harvesters = ['sSUHFIPTIVA0', 'SMS7630005LF', 'SMS7621005LF']
power_levels = [-15, -10, -5, 0]
power_levels = np.arange(-15, 0 + 0.1, 2.5)
v_charge_target = 2
C = 10e-3

freq_min = 910
freq_max = 913
# freq_min = 2450
# freq_max = 2455

# Data verzamelen
results = {h: [] for h in harvesters}

for harvester in harvesters:
    for input_pwr_dbm in power_levels:
        x, y = calculate_response_time(harvester, freq_min, freq_max, input_pwr_dbm, C, v_charge_target)

        if y[-1] > 35000:
            results[harvester].append(0)
        else:
            results[harvester].append(y[-1])
        print(f"{harvester} at {input_pwr_dbm} dBm: Response time = {y[-1]:.2f} s")

# Bar plot
x = np.arange(len(power_levels))  # 0,1,2,3
width = 0.25  # breedte van de balkjes

plt.figure()

# for i, harvester in enumerate(harvesters):
#     plt.bar(x + i*width, results[harvester], width=width, label=harvester)


for i, harvester in enumerate(harvesters):
    bars = plt.bar(x + (i - 1)*width, results[harvester], width=width, label=harvester)

    # labels toevoegen
    for bar in bars:
        height = bar.get_height()
        
        if np.isnan(height):
            label = "Impossible"
            height = 1  # nodig voor log scale (kan niet op NaN/0)
        else:
            label = f"{height:.0f}"

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            label,
            ha='center',
            va='bottom',
            fontsize=8
        )


# X-as labels in het midden zetten
plt.xticks(x + width, power_levels)

plt.xlabel("Input Power (dBm)")
plt.ylabel("Charge Time (s)")
plt.yscale("log")
plt.title(f"Charge Time for {C}F and target voltage {v_charge_target}V")
plt.legend()
plt.grid(axis='y')

plt.show()