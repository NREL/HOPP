import concurrent.futures as cf
import time
import threading

def func(wait):
    time.sleep(wait)
    return f'Done... {wait}'

cond = threading.Condition

print(isinstance(cond, threading.Condition))