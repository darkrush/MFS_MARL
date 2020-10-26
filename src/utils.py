import time

class process_bar(object):
    def __init__(self, tik_number):
        self.total_tik_number = tik_number

    def start(self):
        self.start_time = time.time()
        self.tik_number = 0
    
    def tik(self):
        self.tik_number += 1
        process_past = self.tik_number / self.total_tik_number
        process_left = 1 - process_past
        now_time = time.time()
        time_past = now_time - self.start_time
        time_total = time_past / process_past
        time_left = time_total * process_left
        return process_past, time_total, time_left

def float2time(f_time):
    str_time = ''
    if f_time>3600*24:
        day = f_time // 3600*24
    if f_time>3600:
        str_time += ''
