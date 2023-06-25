from multiprocessing import Process
import time
import os

start_time = time.time()

def count(cnt):
  proc = os.getpid()
  for i in range(cnt):
    print(f"Process ID : {proc} -- {i}")

if __name__ == '__main__':
  num_arr = [10, 10, 10, 10]
  procs = []

  for index, number in enumerate(num_arr):
    proc = Process(target=count, args=(number, ))
    procs.append(proc)
    proc.start()
    
  for proc in procs:
    proc.join()