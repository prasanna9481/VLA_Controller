#!/usr/bin/env python3

"""
Purpose:
Continuously reads and decodes the Robotiq 2F-85 gripper status over
Modbus RTU so the current activation, motion, object-detection, and
position state can be inspected.

Workflow:
1. Open the serial connection to the Robotiq gripper.
2. Send a Modbus RTU request for the gripper status registers.
3. Read the 11-byte response from the gripper.
4. Parse the returned status, fault, requested position, actual position,
   and motor-current fields.
5. Decode the Robotiq status bits:
   gACT, gGTO, gSTA, and gOBJ.

Technical details:
- Communication uses Modbus RTU over `/dev/ttyUSB0` at 115200 baud.
- The gripper slave ID is `0x09`.
- Status registers are read starting at address `0x07D0`.
- CRC16 is calculated and appended to every Modbus request.
- The actual gripper position is reported in the Robotiq raw range:
  approximately `0 = fully open` and `255 = fully closed`.
- The script is read-only and does not send motion commands to the gripper.
"""


import serial
import time


PORT = "/dev/ttyUSB0"
BAUD = 115200
SLAVE_ID = 0x09


def crc16(data):
    crc = 0xFFFF

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1

    return crc


def add_crc(frame):
    crc = crc16(frame)

    return frame + bytes([
        crc & 0xFF,
        (crc >> 8) & 0xFF,
    ])


def read_status(ser):
    request = bytes([
        SLAVE_ID,
        0x04,
        0x07, 0xD0,
        0x00, 0x03,
    ])

    request = add_crc(request)

    ser.reset_input_buffer()

    print("TX:", request.hex(" "))

    ser.write(request)
    ser.flush()

    time.sleep(0.1)

    response = ser.read(11)

    print("RX:", response.hex(" "))

    return response


def parse_status(response):
    if len(response) != 11:
        print("Unexpected response length:", len(response))
        return

    slave = response[0]
    function = response[1]
    byte_count = response[2]

    data = response[3:9]

    status = data[0]
    fault = data[1]
    requested_position = data[2]
    actual_position = data[3]
    current = data[4]

    # Decode status byte
    gACT = status & 0x01
    gGTO = (status >> 3) & 0x01
    gSTA = (status >> 4) & 0x03
    gOBJ = (status >> 6) & 0x03

    print()
    print("----- RAW FIELDS -----")
    print("Slave ID:          ", slave)
    print("Function:          ", hex(function))
    print("Byte count:        ", byte_count)
    print("Status byte:       ", hex(status))
    print("Fault byte:        ", hex(fault))
    print("Requested position:", requested_position)
    print("Actual position:   ", actual_position)
    print("Current raw:       ", current)

    print()
    print("----- STATUS BITS -----")
    print("gACT:", gACT)
    print("gGTO:", gGTO)
    print("gSTA:", gSTA)
    print("gOBJ:", gOBJ)

    print()
    print("----- INTERPRETATION -----")

    activation_states = {
        0: "Gripper reset / not activated",
        1: "Activation in progress",
        2: "Reserved",
        3: "Activation complete",
    }

    object_states = {
        0: "Fingers moving",
        1: "Stopped while opening: object detected",
        2: "Stopped while closing: object detected",
        3: "Reached requested position",
    }

    print("Activation:", activation_states.get(gSTA, "Unknown"))
    print("Object:    ", object_states.get(gOBJ, "Unknown"))

    print()

    if actual_position <= 10:
        print("Position: approximately FULLY OPEN")
    elif actual_position >= 245:
        print("Position: approximately FULLY CLOSED")
    else:
        print("Position: intermediate")

    print()


def main():
    ser = serial.Serial(
        port=PORT,
        baudrate=BAUD,
        bytesize=8,
        parity=serial.PARITY_NONE,
        stopbits=1,
        timeout=1.0,
    )

    print("Connected to", ser.name)
    print()

    try:
        while True:
            response = read_status(ser)

            parse_status(response)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        ser.close()


if __name__ == "__main__":
    main()