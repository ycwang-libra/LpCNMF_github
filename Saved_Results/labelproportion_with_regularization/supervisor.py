# 监控内存用量
import psutil
import time
import datetime

def sleeptime(hour, min, sec):
    return hour * 3600 + min * 60 + sec

second = sleeptime(0,0,2)
while 1 == 1:
    time.sleep(second)

    Time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = psutil.virtual_memory()
    mem_percent = info.percent
    file = open('memory_recording.txt','a')
    file.write('The memory used percent is: ' + str(mem_percent) + '%' + ' Time is: ' + Time + '\n')