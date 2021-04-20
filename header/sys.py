from multiprocessing import process

def start_processes(processes_lst, start = None, end = None):
    if start is None:
        start = 0
    if end is None:
        end = len(processes_lst)
    
    for i in range(start, end):
        processes_lst[i].start()

def join_processes(processes_lst, start = None, end = None):
    if start is None:
        start = 0
    if end is None:
        end = len(processes_lst)
    
    for i in range(start, end):
        processes_lst[i].join()
        print(f'Decision tree no.{i} was trained.')