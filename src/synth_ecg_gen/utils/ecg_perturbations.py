# def add_noise(ecg, f_min, f_max, amplitude, fs=512):
#     noise = band_limited_noise(ecg.shape, f_min, f_max, fs=fs)

#     norm_noise = (noise - np.mean(noise, axis=0)) / np.std(noise, axis=0)

#     scaled_noise = amplitude * norm_noise

#     return ecg + scaled_noise


# # generate random noise
# # filter
# # return filtered noise
# def band_limited_noise(shape, f_min, f_max, fs=512):
#     raw = np.random.normal(size=shape)

#     filt = signal.butter(10, [f_min, f_max], "bandpass", fs=fs, output="sos")

#     filtered_noise = signal.sosfilt(filt, raw)

#     return filtered_noise
