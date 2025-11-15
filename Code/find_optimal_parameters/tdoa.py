import numpy as np
from chFunctions import ch2, ch3

class Distance:
    def __init__(self, dT, Fs_RX, file_name, method, speed_of_sound=343.2, params = {}):
        self.dT = dT
        self.dS = dT * Fs_RX
        self.estimated_distance = dT * speed_of_sound
        self.Fs_RX = Fs_RX
        self.expected_distance = 1 if "100" in file_name else 0.5
        self.error = abs(self.estimated_distance - self.expected_distance)
        self.errorcm = abs(self.estimated_distance - self.expected_distance)*100
        self.file_name = file_name
        self.method = method
        self.speed_of_sound = speed_of_sound
        self.params = params
    def __repr__(self):
        return (f"File: {self.file_name}\n  - Method: {self.method}\n  - Estimated Distance: {self.estimated_distance:.4f} m\n  - Expected Distance: {self.expected_distance:.4f} m\n  - Error: {self.errorcm:.4f} cm\n  - dT: {self.dT:.6f} s\n  - dS: {self.dS:.2f} samples\n  - params: {self.params}")
    def __str__(self):
        return self.__repr__()
        
def tdoa_prepare_x(x, start_threshold=0.15):
    start_signal = np.where(np.abs(x) > start_threshold)[0][0]
    end_signal = np.where(np.abs(x) > start_threshold)[0][-1]
    x = np.array(x[start_signal:end_signal+1])
    return x

def tdoa(x, y, Lhat, method='ch2', epsi=0.001, Fs_RX = 48000, start_threshold = 0.15, file="", peak_detection_method="abs"):
    data = np.array(y)
    y_near = data[:,0]
    y_far = data[:,1]
    match peak_detection_method:
        case "abs":
            cond = np.abs(y_near) > start_threshold
        case "absreal":
            cond = np.abs(np.real(y_near)) > start_threshold
        case "real":
            cond = np.real(y_near) > start_threshold
        case "abssign":
            cond = (np.abs(y_near) > start_threshold) & (np.real(y_near) >= 0)
        case _:
            raise ValueError("peak_detection_method must be either 'abs', 'absreal', 'abssign' or 'real'.")
    start_response = np.where(cond)[0][0]
    y_near = y_near[start_response:]
    y_far = y_far[start_response:]
    if method == 'ch2':
        h_near: np.ndarray = ch2(x, y_near, Lhat)
        h_far: np.ndarray = ch2(x, y_far, Lhat)
    elif method == 'ch3':
        h_near: np.ndarray = ch3(x, y_near, Lhat, epsi)
        h_far: np.ndarray = ch3(x, y_far, Lhat, epsi)
    else:
        raise ValueError("Method must be either 'ch2' or 'ch3'.")
    
    peak_near_t = np.argmax(np.real(h_near))/Fs_RX
    peak_far_t = np.argmax(np.real(h_far))/Fs_RX
    
    dist = Distance(peak_far_t - peak_near_t, Fs_RX, file, method)
    
    return dist

def classify(errorcm):
    if errorcm <= 1.5:
        return "Good"
    elif errorcm <= 3:
        return "Okay"
    elif errorcm <= 15:
        return "Reasonable"
    else:
        return "Fail"