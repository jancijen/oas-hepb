import multiprocessing


def run_parallelly(func, parameters):
    pool = multiprocessing.Pool()
    map_results = pool.starmap(func, parameters)

    pool.close()
    pool.join()
    
    return map_results