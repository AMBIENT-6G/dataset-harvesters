from ep_handler import *

while True:
    try:
        data = get_ep_data()
        if data is not None:
            print(f"Received data: {data}")
    except Exception as e:
        print(f"Error reading data: {e}")

    # time.sleep(1)