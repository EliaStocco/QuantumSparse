MEMORY_SAVING_EIGESTATES = False

def set_memory_saving(value:bool=True):
    global MEMORY_SAVING_EIGESTATES
    MEMORY_SAVING_EIGESTATES= value
    
def get_memory_saving()->bool:
    return MEMORY_SAVING_EIGESTATES
