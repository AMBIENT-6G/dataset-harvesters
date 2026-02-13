# Simple example on how to use the RsInstrument module for remote-controlling yor VISA instrument
# Preconditions:
# - Installed RsInstrument Python module (see the attached RsInstrument_PythonModule folder Readme.txt)
# - Installed VISA e.g. R&S Visa 5.12.x or newer

from RsInstrument.RsInstrument import RsInstrument
import time
import serial
import struct
import numpy as np
import csv
import os
from ep_handler import *

# Path to save the experiment data as a YAML file
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_path = os.path.dirname(current_dir)
filename = os.path.basename(current_dir)

print(filename)

resource_string_1 = 'TCPIP::10.128.48.6::INSTR'  # Standard LAN connection (also called VXI-11)
resource_string_2 = 'TCPIP::192.168.2.101::hislip0'  # Hi-Speed LAN connection - see 1MA208
resource_string_3 = 'GPIB::20::INSTR'  # GPIB Connection
resource_string_4 = 'USB::0x0AAD::0x0119::022019943::INSTR'  # USB-TMC (Test and Measurement Class)
resource_string_5 = 'RSNRP::0x0095::104015::INSTR'  # R&S Powersensor NRP-Z86
instr = RsInstrument(resource_string_1, True, False)

FREQ = 917e6
LVL = -20

#ser = serial.Serial('COM4', 115200, timeout=1)  # Change 'COM1' to your serial port

idn = instr.query_str('*IDN?')
print(f"\nHello, I am: '{idn}'")
print(f'RsInstrument driver version: {instr.driver_version}')
print(f'Visa manufacturer: {instr.visa_manufacturer}')
print(f'Instrument full name: {instr.full_instrument_model_name}')
print(f'Instrument installed options: {",".join(instr.instrument_options)}')

instr.write('*RST')
instr.write('*CLS')


def change_freq_lvl(frequency, level):
  instr.write('SOUR:FREQ:CW '+ str(frequency))
  instr.write('SOUR:POW:LEV:IMM:AMPL ' + str(level))

def output(status):
  instr.write('OUTP1:STAT ' + str(status))


csv_header = ['frequency_mhz', 'level_dbm', 'efficiency', 'buffer_voltage_mv', 'resistance', 'pwr_pw']

current_directory = os.getcwd()

# csv_file_path = f"{current_directory}/02-energy-profiler-v-1-2/meas_data/"
# csv_file_time = round(time.time())

# print(csv_file_path)

def append_to_csv(csv_file_path, data):
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, mode='a', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write header if the file is newly created
        if not file_exists:
            writer.writerow(csv_header)  # Modify header as needed

        # Write the data to the CSV file
        writer.writerow(data)


start_lvl = -20
stop_lvl = 0

lvl_buffer = np.linspace(start_lvl, stop_lvl, (stop_lvl-start_lvl)*2 + 1)
# lvl_buffer = np.linspace(start_lvl, stop_lvl, 5)

# frequency = FREQ

start_freq = 800e6
stop_freq = 950e6

freq_buffer = np.linspace(start_freq, stop_freq, int((stop_freq-start_freq)/12.5e6) + 1)

output(1)

change_freq_lvl(start_freq, start_lvl)

time.sleep(1)

output(1)

tvbuffer = np.linspace(250,2000,8)
# print(tvbuffer)

# target_voltage = 1000

for target_voltage in tvbuffer:

    # Convert to integer
    target_voltage = int(target_voltage)
    print(f"Target voltage: {target_voltage}")

    # Change target voltage
    set_ep_target_voltage(target_voltage)

    # Sweep
    for freq in freq_buffer:

        for lvl in lvl_buffer:

            #   Change freq and lvl SMC100A
            change_freq_lvl(freq, lvl)

            #   Print
            # print(f"Level: {lvl} - Frequency: {freq}")

            #   Wait 2 seconds for stabalizing profiler
            time.sleep(5)

            #   Read values
            vals = get_ep_data()

            while vals == None:
                print("Try again ... ")
                time.sleep(1)
                vals = get_ep_data()

            ep_results = np.asarray(vals)

            efficiency = round(vals['pwr_pw']/1e12/(10**(lvl/10)/1e3)*100,2)

            ep_results = [efficiency, vals['buffer_voltage_mv'], vals['resistance'], vals['pwr_pw']]

            settings = [freq/1e6, lvl]

            store_buffer = np.concatenate((settings, ep_results))

            #print(store_buffer)

            print(f"Level: {lvl} - Frequency: {freq} - Voltage: {vals['buffer_voltage_mv']} - Power [nW]: {round(vals['pwr_pw']/1e3)}")

            append_to_csv(f"{current_dir}/{filename}_measured_t{target_voltage}.csv", store_buffer)

output(0)

# Close the session
instr.close()