#!/usr/bin/python3
"""From https://github.com/csarron/emonpi/blob/master/firmware/energy_monitor.py"""

import argparse
import time
from datetime import datetime

import serial


IDLE_POWER = 180.5  # Watts


def main(args):
  output_file = args.output_file
  of = open(output_file, 'w')
  of.write('timestamp,value\n')
  of.flush()
  ser = serial.Serial('/dev/ttyUSB0', 115200)

  # ser.write(b'f10')
  # ser.write(b'f20')
  # ser.write(b'c1')
  try:
    while True:
      try:
        response = ser.readline()
        print(response)
        content = response.decode().strip()
        save_data = f'{time.clock_gettime(time.CLOCK_REALTIME):.6f},{content}\n'
        of.write(save_data)
        of.flush()
        print(f'{datetime.now()},{save_data}')
      except UnicodeDecodeError as e:
        print(e, 'decode error, skip')
        continue
      except serial.SerialException as e:
        print(e, 'SerialException, skip')
        continue
      # print("{}\n".format(datetime.now()))
      print()
  except KeyboardInterrupt:
    ser.close()
    of.close()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-o", "--output_file", type=str, required=True,
                      help="output file")
  main(parser.parse_args())
