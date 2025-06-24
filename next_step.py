import time
import Legs
from Legs import lss_array
import lss_const as lssc, lss
import threading
from queue import Queue
import csv
from datetime import datetime

# Constants
LSS_BAUD = lssc.LSS_DefaultBaud 
CST_LSS_Port = "/dev/ttyUSB0"  # For Linux/Unix platforms
# CST_LSS_Port = "COM230"      # For Windows platforms

# Variables
userinput: str = ""
next_step = [-1] * 4  # array for next step
OMEGA = 20  # unit in RPM
ROTATION_UNITS = 3600  # Assuming 3600 units for a full rotation
is_manipulation = False

# Queue for user inputs
input_queue = Queue()

# CSV file setup
csv_file = 'expert_operation_log.csv'
csv_columns = ['run_timestamp', 'user_input_timestamp', 'user_input', 'annotation']

def write_to_csv(data):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writerow(data)

def verify_csv_columns():
    try:
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            if header != csv_columns:
                raise ValueError("CSV columns do not match expected columns")
    except (FileNotFoundError, StopIteration, ValueError):
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

def annotate_run(run_timestamp):
    annotation = input("Enter annotation for this run: ")
    # Read existing data and update the last row's annotation
    updated_rows = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['run_timestamp'] == run_timestamp and row['annotation'] == '':
                row['annotation'] = annotation
            updated_rows.append(row)

    # Write updated data back to the CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(updated_rows)

# PhaseBot PART I: The original document of PhasezBots
def LSS_init():
    def move_to_initial_position(index, leg):
        attempt = 0
        while attempt < 3:  # Retry up to 3 times
            leg.move(0)
            time.sleep(1.5)  # Reduce waiting time to 1.5 second
            current_position = leg.getPosition()
            #print(f"Leg {index} moved to position {current_position}")
            if current_position == 0:
                break
            attempt += 1
            #print(f"Retry {attempt} for leg {index}")

    lss.initBus(CST_LSS_Port, LSS_BAUD)  # Initialize connection
    time.sleep(1)  # Wait for 1 second

    threads = []
    for index, leg in enumerate(lss_array):
        thread = threading.Thread(target=move_to_initial_position, args=(index, leg))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    Legs.ALL.reset()  # RESET virtual angular position after moving to initial position
    time.sleep(1.25)  # Wait for 1.25 seconds to ensure the reset is complete

    # Set gyre directions
    Legs.R_LEAD_LEG.setGyre(+1)
    Legs.L_LEAD_LEG.setGyre(-1)
    Legs.L_HIND_LEG.setGyre(-1)
    Legs.R_HIND_LEG.setGyre(+1)
    time.sleep(0.5)

    stopMove()  # Stop all motors after initialization

def stopMove():
    for leg in lss_array:
        leg.wheelRPM(0)

# PhaseBot PART II: The original document of Phase_Space:
def get_user_input(run_timestamp):
    global next_step, userinput, is_manipulation
    manipulation = {
        "RF": [0, -1, -1, -1],
        "LF": [-1, 0, -1, -1],
        "LR": [-1, -1, 0, -1],
        "RR": [-1, -1, -1, 0],
    }

    operation = {
        "BOND": [0, 0, 400, 400],
        "TROT": [0, 400, 0, 400],
        "LEFT": [400, -400, -400, 400],
        "RIGHT": [-400, 400, 400, -400],
    }

    user_inputs = []

    while True:
        user_input = input("Enter next step (enter 'quit' to quit):").upper()
        user_input_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if user_input == 'QUIT':
            userinput = 'quit'
            input_queue.put('quit')
            user_inputs.append(user_input)
            write_to_csv({
                'run_timestamp': run_timestamp,
                'user_input_timestamp': user_input_timestamp,
                'user_input': " ".join(user_inputs),
                'annotation': ''
            })
            break
        elif user_input in manipulation:
            is_manipulation = True
            input_queue.put(manipulation[user_input])
            user_inputs.append(user_input)
        elif user_input in operation:
            is_manipulation = False
            input_queue.put(operation[user_input])
            user_inputs.append(user_input)
        else:
            try:
                next_step = list(map(int, user_input.split()))
                if len(next_step) != 4:
                    print("Please enter exactly 4 integers.")
                else:
                    for i in range(4):
                        if next_step[i] < -1:
                            raise ValueError("Invalid input, must be -1 or non-negative integer.")
                    input_queue.put(next_step)
                    user_inputs.append(user_input)
            except ValueError:
                print("Please enter a valid array of 4 integers, separated by spaces.")

def approach(next_step):
    global OMEGA, ROTATION_UNITS, is_manipulation
    def move_leg(leg_index, value, is_manipulation):
        if value != -1:
            rotation_units = 7200 if is_manipulation else ROTATION_UNITS
            if value < 0:
                rotation_units = -rotation_units
            time.sleep(abs(value) / 1000.0)  # Convert milliseconds to seconds
            lss_array[leg_index].moveRelative(rotation_units)  # Move leg relative to current position
            time.sleep(60.0 / OMEGA)  # Time to complete one rotation at OMEGA RPM
            lss_array[leg_index].wheelRPM(0)  # Stop after rotation

    threads = []
    for i, value in enumerate(next_step):
        thread = threading.Thread(target=move_leg, args=(i, value, is_manipulation))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def loop():
    global userinput, next_step
    if not input_queue.empty():
        next_step = input_queue.get()
        if next_step == 'quit':
            userinput = 'quit'
            stopMove()
            return

        approach(next_step)
        next_step = [-1] * 4  # Reset next_step to default after execution

# main
if __name__ == '__main__':
    # Verify CSV columns and create file with headers if it doesn't exist
    verify_csv_columns()

    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    LSS_init()

    input_thread = threading.Thread(target=get_user_input, args=(run_timestamp,))
    input_thread.start()

    try:
        while userinput != 'quit':
            loop()
            time.sleep(0.1)  # Add a small delay to reduce CPU usage
    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        input_thread.join()
        annotate_run(run_timestamp)
