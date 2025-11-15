from find_optimum_tdoa_ch3 import find_optimum_tdoa_ch3
import multiprocessing

def process_file(file, peak_detection_method, recordings_path):
    # Get results
    local_good_pairs = []
    local_results = []
    local_nothing_found = []
    print(f"Processing {file} using peak detection method: {peak_detection_method}...")
    reasonable_options = find_optimum_tdoa_ch3(
        file=file, 
        recordings_path=recordings_path, 
        Lhat_bounds = (2000, 2000, 1), 
        start_threshold_bounds = (0.15, 0.45, 40), 
        epsi_bounds=(0.0001,0.08, 100), 
        peak_detection_method=peak_detection_method
    )
    if len(reasonable_options) == 0:
        local_nothing_found.append((file, peak_detection_method))
    print(f"{len(reasonable_options)} reasonable options for {file}")
    local_results.append((file, reasonable_options))
    for option in reasonable_options:
        option[-1].params['pdm'] = peak_detection_method
        local_good_pairs.append((file, option[-1].params, option[-1].errorcm))
    return local_nothing_found, local_results, local_good_pairs

def run_parallel_processing(files, peak_detection_methods, recordings_path):
    nothing_found = []
    results = []
    good_pairs = []

    # Set up the pool of workers (one worker per CPU core)
    with multiprocessing.Pool() as pool:
        # Use pool.map to apply the function to each file
        # pool.map takes a function and an iterable (files in this case) and applies the function to each item in parallel
        tasks = [(file, pdm, recordings_path) for pdm in peak_detection_methods for file in files]
        results_list = pool.starmap(process_file, tasks)
        
        # After parallel execution, merge the results from all workers
        for local_nothing_found, local_results, local_good_pairs in results_list:
            nothing_found.extend(local_nothing_found)
            results.extend(local_results)
            good_pairs.extend(local_good_pairs)

    # Return the final aggregated results
    return nothing_found, results, good_pairs

if __name__ == "__main__":
    pass