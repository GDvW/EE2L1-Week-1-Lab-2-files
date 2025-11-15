from tdoa import tdoa, tdoa_prepare_x
# remove, just for convenience 
# Packages
import numpy as np
from scipy.io import wavfile

from wavaudioread import wavaudioread

def find_optimum_tdoa_ch2(file = "recording-beacon-50cm3.wav", Lhat_bounds = (2000, 4000, 50), start_threshold_bounds = (0.001, 0.45, 100), recordings_path = "./Finished recordings/", original_file_path='audio-beacon.wav', Fs_RX = 48000, peak_detection_method="abs"): 
    Fs, x_original = wavfile.read(original_file_path)
    y = wavaudioread(recordings_path+file, Fs_RX)
    # Optimize ch2 params (start_threshold, Lhat)
    results = []
    errors = []

    for start_threshold in np.linspace(*start_threshold_bounds):
        x = tdoa_prepare_x(x_original, start_threshold)
        for Lhat in np.linspace(*Lhat_bounds).astype(int):
            result = tdoa(x, y, Lhat, method="ch2", start_threshold=start_threshold, Fs_RX=Fs_RX, file=file, peak_detection_method=peak_detection_method)
            result.params = {'Lhat': Lhat, 'start_threshold': start_threshold}
            results.append([result.errorcm, Lhat, start_threshold, result])
            errors.append(result.errorcm)
        print(start_threshold, errors[-1])
    min_error = np.min(errors)
    
    if min_error <= 1.5:
        print(f"INFO: Good results found for {file}")
    elif min_error <= 3:
        print(f"WARNING: No realistic result found for {file}")
    elif min_error <= 15:
        print(f"ERROR: Too big error, still in the direction found for {file}")
    else:
        print(f"CRITICAL: Completely outside the ballpark for {file}")
        
    reasonable_options = []
    for result in results:
        if result[0] <= 15:
            reasonable_options.append(result)
            
    return reasonable_options

if __name__ == "__main__":
    # Example usage
    recordings_path = "./Finished recordings/"
    file = "recording-beacon-50cm2.wav"
    peak_detection_method = "real"
    reasonable_options = find_optimum_tdoa_ch2(file=file, recordings_path=recordings_path, Lhat_bounds = (2000, 2000, 1), start_threshold_bounds = (0.001, 0.45, 400), peak_detection_method=peak_detection_method)
    print(f"{len(reasonable_options)} reasonable options for {file}")
    for option in reasonable_options:
        print(option[-1])