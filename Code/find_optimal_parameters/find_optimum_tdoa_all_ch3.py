import numpy as np
from tdoa import classify
from find_optimum_tdoa_ch3_functions import run_parallel_processing

if __name__ == "__main__":
    recordings_path = "./Finished recordings/"
    files = ["recording-beacon-50cm.wav", "recording-beacon-50cm2.wav", "recording-beacon-50cm3.wav", "recording-beacon-50cm4.wav",
           "recording-beacon-100cm.wav", "recording-beacon-100cm2.wav", "recording-beacon-100cm3.wav", "recording-beacon-100cm4.wav"]
    peak_detection_methods = ["abs", "absreal", "abssign","real"]
    # peak_detection_methods = ["abssign","real"]

    nothing_found, results, good_pairs = run_parallel_processing(files, peak_detection_methods, recordings_path)   
    correct_pairs = {}

    print("Collecting best params...")
    # Transform to correct_pairs
    for file, params, errorcm in good_pairs:
        if str(params) in correct_pairs:
            correct_pairs[str(params)]["count"] += 1
            correct_pairs[str(params)]["errors_square"].append(errorcm**2)
            correct_pairs[str(params)]["class"].append(classify(errorcm))
            correct_pairs[str(params)]["file"].append(file)
        else:
            correct_pairs[str(params)] = {"count": 1, "errors_square": [errorcm**2], "class": [classify(errorcm)], "file": [file]}
        
    # Calculate MSE    
    for params in correct_pairs.keys():
        correct_pairs[params]["MSE"] = np.sqrt(np.mean(correct_pairs[params]['errors_square']))
        
    # Get max hits of best params
    max_count = max([data['count'] for data in correct_pairs.values()])
    if max_count < len(files) - len(nothing_found):
        print("WARNING: No params found that work for all files with reasonable results.")
    else:
        print(f"INFO: Found params that work for {max_count} files.")

    print("Files with no good result:")
    for file in nothing_found:
        print(f"  - {file}")
        

    print("Most common best params for ch3:")
    for params, data in correct_pairs.items():
        if data['count'] < max_count:
            continue
        classes = ""
        values, counts = np.unique(data['class'], return_counts=True)
        for value, count in zip(values, counts):
            classes += f"{value}: {count}, "
        print(f"  - Params: {params} - Count: {data['count']} - Avg Error cm: {data["MSE"]:.4f} cm")
        print(f"    ({classes}), <{",".join(data['file'])}>")
        
    with open("best_ch3_params.txt", "w") as f:
        if max_count < len(files) - len(nothing_found):
            f.write("WARNING: No params found that work for all files with reasonable results.\n")
        else:
            f.write(f"INFO: Found params that work for {max_count} files.\n")
        f.write("\n")
        f.write("\n\nFiles with no good result:\n")
        for file in nothing_found:
            f.write(f"  - {file}\n")
        f.write("\n")
        
        f.write("Most common best params for ch3:\n")
        for params, data in correct_pairs.items():
            if data['count'] < max_count:
                continue
            classes = ""
            values, counts = np.unique(data['class'], return_counts=True)
            for value, count in zip(values, counts):
                classes += f"{value}: {count}, "
            f.write(f"  - Params: {params} - Count: {data['count']} - Avg Error cm: {data["MSE"]:.4f} cm\n")
            f.write(f"    ({classes}), <{",".join(data['file'])}>\n")
        f.write("\n")
        
        f.write("Best options per file:\n\n")
        for file, best_options in results:
            f.write(f"File: {file}\n")
            for option in best_options:
                f.write(f"{option[-1]}\n")
            f.write("\n")
            