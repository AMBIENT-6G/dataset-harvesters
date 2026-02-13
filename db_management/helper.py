import pandas as pd

def check_header(csv_path):
    df = pd.read_csv(csv_path)

    # Gewenste kolommen
    required_columns = [
        "frequency_mhz",
        "level_dbm",
        "buffer_voltage_mv",
        "resistance",
        "pwr_pw",
        "efficiency"
    ]

    # Checken welke kolommen missen
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing columns in CSV: {missing_cols}")
        return False
    else:
        print("✅ All required columns are present.")
        return True

def df_round(csv_path):
    df = pd.read_csv(csv_path)

    # Rond alle numerieke kolommen af op 3 decimalen
    df = df.round(3)

    # Optioneel meteen overschrijven in CSV
    df.to_csv(csv_path, index=False)    
    print(f"✅ All numeric columns rounded to 3 decimals and saved to {csv_path}")

def check_columns(csv_path):
    df = pd.read_csv(csv_path)
    changed = False

    # Add 'efficiency' if missing
    if "efficiency" not in df.columns:
        df["efficiency"] = round(((df["pwr_pw"]/1e12) / (10**(df["level_dbm"]/10)/1e3)) * 100, 3)
        print(f"✅ Added 'efficiency' column")
        changed = True

    # Add 'resistance' if missing
    if "resistance" not in df.columns:
        df["resistance"] = 0
        print(f"✅ All values in 'resistance' set to 0")
        changed = True

    # Add 'pwr_pw' if missing
    if "pwr_pw" not in df.columns:
        df["pwr_pw"] = round((10**(df["level_dbm"]/10)/1e3) * df["efficiency"]/100 * 1e12)
        print(f"✅ All values in 'pwr_pw' set to 0")
        changed = True

    # Save if df was changed
    if changed:
        df.to_csv(csv_path, index=False)
        print(f"💾 CSV updated and saved to {csv_path}")
    else:
        print("ℹ️ CSV already contains all required columns")