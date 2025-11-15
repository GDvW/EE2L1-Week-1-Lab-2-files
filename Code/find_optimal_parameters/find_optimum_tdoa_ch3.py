from tdoa import tdoa, tdoa_prepare_x
# remove, just for convenience 
# Packages
import numpy as np
from scipy.io import wavfile

from wavaudioread import wavaudioread

def find_optimum_tdoa_ch3(file = "recording-beacon-50cm3.wav", Lhat_bounds = (2000, 4000, 50), start_threshold_bounds = (0.001, 0.45, 100), epsi_bounds = (0.00001,0.0001, 10), recordings_path = "./Finished recordings/", original_file_path='audio-beacon.wav', Fs_RX = 48000, peak_detection_method="abs"):
    Fs, x_original = wavfile.read(original_file_path)
    y = wavaudioread(recordings_path+file, Fs_RX)
    # Optimize ch2 params (start_threshold, Lhat)
    results = []
    errors = []

    for start_threshold in np.linspace(*start_threshold_bounds):
        x = tdoa_prepare_x(x_original, start_threshold)
        # for Lhat in np.linspace(*Lhat_bounds).astype(int):
        for Lhat in np.linspace(*Lhat_bounds).astype(int):
            for epsi in np.linspace(*epsi_bounds):
                result = tdoa(x, y, Lhat, epsi=epsi, method="ch3", start_threshold=start_threshold, Fs_RX=Fs_RX, file=file)
                result.params = {'Lhat': Lhat, 'start_threshold': start_threshold, 'epsi': epsi}
                results.append([result.errorcm, Lhat, start_threshold, epsi, result])
                errors.append(result.errorcm)
        # print(start_threshold, errors[-1])
    min_error = np.min(errors)
    
    # if min_error <= 1.5:
    #     print(f"INFO: Good results found for {file}")
    # elif min_error <= 3:
    #     print(f"WARNING: No realistic result found for {file}")
    # elif min_error <= 15:
    #     print(f"ERROR: Too big error, still in the direction found for {file}")
    # else:
    #     print(f"CRITICAL: Completely outside the ballpark for {file}")
        
    reasonable_options = []
    for result in results:
        if result[0] <= 15:
            reasonable_options.append(result)
            
    return reasonable_options

if __name__ == "__main__":
    pass