import copy

import numpy as np
from scipy import signal

"""
Methods to uniformly perturb an ECG in a controlled fashion.
"""


def change_HR(vcg_ode_original, target_hr):
    vcg_ode = copy.deepcopy(vcg_ode_original)
    vcg_ode.set_HR(target_hr)

    return vcg_ode


# input the ode object for the ECG you want to perturb
def qt_elongation(vcg_ode_original, ms_forward=0):
    vcg_ode = copy.deepcopy(vcg_ode_original)

    # TODO warning for pushing qt interval too far forward
    # TODO find out -- should QT elongation result in a longer HR here?
    # TODO this approach is a bit messy because of how the rest of the ECG is affected...

    th_x = vcg_ode.theta_x
    th_y = vcg_ode.theta_y
    th_z = vcg_ode.theta_z

    beat_duration = 1 / vcg_ode.f

    s_forward = ms_forward / 1000

    degrees_forward = 2 * np.pi * s_forward / beat_duration

    # t-wave is made out two gaussians
    th_x[-3] += degrees_forward
    th_x[-2] += degrees_forward
    # th_x[-1] += degrees_forward

    th_y[-3] += degrees_forward
    th_y[-2] += degrees_forward
    # th_y[-1] += degrees_forward

    th_z[-4] += degrees_forward
    th_z[-3] += degrees_forward
    th_z[-2] += degrees_forward
    # th_z[-1] += degrees_forward

    vcg_ode.theta_x = th_x
    vcg_ode.theta_y = th_y
    vcg_ode.theta_z = th_z

    return vcg_ode


# scaledown is to compensate for increase in height from widening.
def wide_qrs(vcg_ode_original, percent_widened=0, scaledown=1):
    vcg_ode = copy.deepcopy(vcg_ode_original)

    b_x = vcg_ode.b_x
    b_y = vcg_ode.b_y
    b_z = vcg_ode.b_z

    alpha_x = vcg_ode.alpha_x
    alpha_y = vcg_ode.alpha_y
    alpha_z = vcg_ode.alpha_z

    # widen
    b_x[3] *= 1 + percent_widened / 100
    b_x[4] *= 1 + percent_widened / 100
    b_x[5] *= 1 + percent_widened / 100

    b_y[3] *= 1 + percent_widened / 100
    b_y[4] *= 1 + percent_widened / 100

    b_z[5] *= 1 + percent_widened / 100

    # shorten a little
    alpha_x[3] *= scaledown
    alpha_x[4] *= scaledown
    alpha_x[5] *= scaledown

    alpha_y[3] *= scaledown
    alpha_y[4] *= scaledown

    alpha_z[5] *= scaledown

    vcg_ode.b_x = b_x
    vcg_ode.b_y = b_y
    vcg_ode.b_z = b_z

    vcg_ode.alpha_x = alpha_x
    vcg_ode.alpha_y = alpha_y
    vcg_ode.alpha_z = alpha_z

    return vcg_ode


def invert_T_waves(vcg_ode_original):
    raise NotImplementedError


# options for type are concave, convex, and straight. TODO
def ST_elevation(vcg_ode_original, type="convex"):
    raise NotImplementedError


# options for type should be downsloping, upsloping TODO
def ST_depression(vcg_ode_original, type="downsloping"):
    raise NotImplementedError


def add_noise(ecg, f_min, f_max, amplitude, fs=512):
    noise = band_limited_noise(ecg.shape, f_min, f_max, fs=fs)

    norm_noise = (noise - np.mean(noise, axis=0)) / np.std(noise, axis=0)

    scaled_noise = amplitude * norm_noise

    return ecg + scaled_noise


# generate random noise
# filter
# return filtered noise
def band_limited_noise(shape, f_min, f_max, fs=512):
    raw = np.random.normal(size=shape)

    filt = signal.butter(10, [f_min, f_max], "bandpass", fs=fs, output="sos")

    filtered_noise = signal.sosfilt(filt, raw)

    return filtered_noise


# make a new, slightly different ECG
def modify_parameters(vcg_ode_original, perturbation_scale=0.01):
    vcg_ode = copy.deepcopy(vcg_ode_original)

    vcg_ode.alpha_x += np.random.normal(scale=perturbation_scale, size=vcg_ode.alpha_x.shape)
    vcg_ode.alpha_y += np.random.normal(scale=perturbation_scale, size=vcg_ode.alpha_y.shape)
    vcg_ode.alpha_z += np.random.normal(scale=perturbation_scale, size=vcg_ode.alpha_z.shape)

    vcg_ode.b_x += np.random.normal(scale=perturbation_scale, size=vcg_ode.b_x.shape)
    vcg_ode.b_y += np.random.normal(scale=perturbation_scale, size=vcg_ode.b_y.shape)
    vcg_ode.b_z += np.random.normal(scale=perturbation_scale, size=vcg_ode.b_z.shape)

    vcg_ode.theta_x += np.random.normal(scale=perturbation_scale, size=vcg_ode.theta_x.shape)
    vcg_ode.theta_y += np.random.normal(scale=perturbation_scale, size=vcg_ode.theta_y.shape)
    vcg_ode.theta_z += np.random.normal(scale=perturbation_scale, size=vcg_ode.theta_z.shape)

    return vcg_ode
