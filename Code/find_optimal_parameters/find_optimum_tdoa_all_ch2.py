from find_optimum_tdoa_ch2 import find_optimum_tdoa_ch2
import numpy as np
from tdoa import classify

recordings_path = "./Finished recordings/"
files = ["recording-beacon-50cm.wav", "recording-beacon-50cm2.wav", "recording-beacon-50cm3.wav", "recording-beacon-50cm4.wav",
         "recording-beacon-100cm.wav", "recording-beacon-100cm2.wav", "recording-beacon-100cm3.wav", "recording-beacon-100cm4.wav"]
start_detection_methods = ["abs", "absreal", "abssign","real"]
peak_detection_methods = ["abs", "absreal", "real"]

# Get results
good_pairs = []
results = []
nothing_found = []
for file in files:
    print(f"Processing {file}...")
    for peak_detection_method in peak_detection_methods:
        print(f"  Using peak detection method: {peak_detection_method}")
        for start_detection_method in start_detection_methods:
            print(f"  Using start detection method: {start_detection_method}")
            reasonable_options = find_optimum_tdoa_ch2(file=file, recordings_path=recordings_path, Lhat_bounds = (2000, 2000, 1), start_threshold_bounds = (0.001, 0.45, 400), peak_detection_method=peak_detection_method, start_detection_method=start_detection_method)
            if len(reasonable_options) == 0:
                nothing_found.append((file, peak_detection_method))
            print(f"{len(reasonable_options)} reasonable options for {file}")
            results.append((file, reasonable_options))
            for option in reasonable_options:
                option[-1].params['pdm'] = peak_detection_method
                good_pairs.append((file, option[-1].params, option[-1].errorcm, option[-1].file_name))
    
correct_pairs = {}

print("Collecting best params...")
# Transform to correct_pairs
for file, params, errorcm, file in good_pairs:
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
    
no_solutions = list(set(files) - set(file for data in correct_pairs.values() for file in data['file'] if data['count'] == max_count))

print("Files with no good result:")
for file in no_solutions:
    print(f"  - {file}")
    

print("Most common best params for ch2:")
for params, data in correct_pairs.items():
    if data['count'] < max_count:
        continue
    classes = ""
    values, counts = np.unique(data['class'], return_counts=True)
    for value, count in zip(values, counts):
        classes += f"{value}: {count}, "
    print(f"  - Params: {params} - Count: {data['count']} - Avg Error cm: {data["MSE"]:.4f} cm")
    print(f"    ({classes}), <{",".join(data['file'])}>")
    
with open("best_ch2_params.txt", "w") as f:
    if max_count < len(files) - len(nothing_found):
        f.write("WARNING: No params found that work for all files with reasonable results.\n")
    else:
        f.write(f"INFO: Found params that work for {max_count} files.\n")
        
    f.write("Files with no good result:\n")
    for file in no_solutions:
        print(f"  - {file}\n")
        
    
    f.write("\n")
    f.write("\n\nFiles with no good result:\n")
    for file in nothing_found:
        f.write(f"  - {file}\n")
    f.write("\n")
    
    f.write("Most common best params for ch2:\n")
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
        