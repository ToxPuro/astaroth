print("started", flush=True)
import os
from time import sleep

print("manager 1", flush=True)
os.system("python3 pipe_reader.py &")
sleep(3)
print("manager 2", flush=True)
os.system("python3 pipe_writer.py")
print("manager 3", flush=True)