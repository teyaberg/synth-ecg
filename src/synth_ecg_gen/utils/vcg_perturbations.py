import copy
from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from omegaconf import DictConfig

T = TypeVar("T", bound="Perturbation")


def change_HR(vcg_ode_original, target_hr):
    vcg_ode = copy.deepcopy(vcg_ode_original)
    vcg_ode.set_HR(target_hr)

    return vcg_ode


class Perturbation(ABC):
    def __init__(self, probability=0):
        super().__init__()
        self.probability = probability

    @classmethod
    def initialize(cls: type[T], **kwargs) -> T:
        return cls(DictConfig(kwargs, flags={"allow_objects": True}))

    @abstractmethod
    def apply_perturbation(self, vcg_ode_original):
        pass

    def __call__(self, vcg_ode):
        if np.random.rand() < self.probability:
            return self.apply_perturbation(vcg_ode)
        return vcg_ode


class QTElongation(Perturbation):
    def __init__(self, cfg):
        super().__init__()
        self.name = "QT Elongation"
        self.min = cfg.ms_forward.min
        self.max = cfg.ms_forward.max

    def apply_perturbation(self, vcg_ode_original):
        vcg_ode = copy.deepcopy(vcg_ode_original)
        # randomly select a value between min and max
        ms_forward = np.random.randint(self.min, self.max)

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


class WideQRS(Perturbation):
    def __init__(self, cfg):
        super().__init__()
        self.name = "Wide QRS"
        self.wide_min = cfg.percent_widened.min
        self.wide_max = cfg.percent_widened.max
        self.scale_min = cfg.scale.min
        self.scale_max = cfg.scale.max

    def apply_perturbation(self, vcg_ode_original):
        vcg_ode = copy.deepcopy(vcg_ode_original)
        # randomly select a value between min and max
        percent_widened = np.random.randint(self.min, self.max)
        scaledown = np.random.randint(self.scale_min, self.scale_max)

        # th_x = vcg_ode.theta_x
        # th_y = vcg_ode.theta_y
        # th_z = vcg_ode.theta_z

        # beat_duration = 1 / vcg_ode.f

        # s_widened = percent_widened / 100

        # degrees_widened = 2 * np.pi * s_widened / beat_duration

        # # qrs complex is made out of two gaussians
        # th_x[-4] -= degrees_widened
        # th_x[-3] -= degrees_widened
        # th_x[-2] += degrees_widened
        # th_x[-1] += degrees_widened

        # th_y[-4] -= degrees_widened
        # th_y[-3] -= degrees_widened
        # th_y[-2] += degrees_widened
        # th_y[-1] += degrees_widened

        # th_z[-4] -= degrees_widened
        # th_z[-3] -= degrees_widened
        # th_z[-2] += degrees_widened
        # th_z[-1] += degrees_widened

        # vcg_ode.theta_x = th_x
        # vcg_ode.theta_y = th_y
        # vcg_ode.theta_z = th_z

        # return vcg_ode
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

        b_y[4] *= 1 + percent_widened / 100
        b_y[5] *= 1 + percent_widened / 100

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


class QRSAmplitude(Perturbation):
    def __init__(self, scale):
        super().__init__()
        self.name = "QRS Amplitude"
        self.min = scale.min
        self.max = scale.max

    def apply_perturbation(self, vcg_ode_original):
        scale = np.random.randint(self.min, self.max)

        vcg_ode = copy.deepcopy(vcg_ode_original)

        b_x = vcg_ode.b_x
        b_y = vcg_ode.b_y
        b_z = vcg_ode.b_z

        alpha_x = vcg_ode.alpha_x
        alpha_y = vcg_ode.alpha_y
        alpha_z = vcg_ode.alpha_z

        # make bigger
        alpha_x[3] *= scale
        alpha_x[4] *= scale
        alpha_x[5] *= scale

        alpha_y[3] *= scale
        alpha_y[4] *= scale
        alpha_y[5] *= scale

        alpha_z[3] *= scale
        alpha_z[4] *= scale
        alpha_z[5] *= scale

        vcg_ode.b_x = b_x
        vcg_ode.b_y = b_y
        vcg_ode.b_z = b_z

        vcg_ode.alpha_x = alpha_x
        vcg_ode.alpha_y = alpha_y
        vcg_ode.alpha_z = alpha_z

        return vcg_ode


class PWaveAmplitude(Perturbation):
    def __init__(self, scale):
        super().__init__()
        self.name = "P Wave Amplitude"
        self.min = scale.min
        self.max = scale.max

    def apply_perturbation(self, vcg_ode_original):
        scale = np.random.randint(self.min, self.max)

        vcg_ode = copy.deepcopy(vcg_ode_original)

        b_x = vcg_ode.b_x
        b_y = vcg_ode.b_y
        b_z = vcg_ode.b_z

        alpha_x = vcg_ode.alpha_x
        alpha_y = vcg_ode.alpha_y
        alpha_z = vcg_ode.alpha_z

        # make bigger
        alpha_x[0] *= scale
        alpha_x[1] *= scale
        alpha_x[2] *= scale

        alpha_y[0] *= scale
        alpha_y[1] *= scale
        alpha_y[2] *= scale

        alpha_z[0] *= scale
        alpha_z[1] *= scale
        alpha_z[2] *= scale

        vcg_ode.b_x = b_x
        vcg_ode.b_y = b_y
        vcg_ode.b_z = b_z

        vcg_ode.alpha_x = alpha_x
        vcg_ode.alpha_y = alpha_y
        vcg_ode.alpha_z = alpha_z

        return vcg_ode


class TWaveAmplitude(Perturbation):
    def __init__(self, scale):
        super().__init__()
        self.name = "T Wave Amplitude"
        self.min = scale.min
        self.max = scale.max

    def apply_perturbation(self, vcg_ode_original):
        scale = np.random.randint(self.min, self.max)

        vcg_ode = copy.deepcopy(vcg_ode_original)

        b_x = vcg_ode.b_x
        b_y = vcg_ode.b_y
        b_z = vcg_ode.b_z

        alpha_x = vcg_ode.alpha_x
        alpha_y = vcg_ode.alpha_y
        alpha_z = vcg_ode.alpha_z

        # make bigger
        alpha_x[8] *= scale
        alpha_x[4] *= scale
        alpha_x[5] *= scale

        alpha_y[8] *= scale
        alpha_y[4] *= scale
        alpha_y[5] *= scale

        alpha_z[8] *= scale
        alpha_z[4] *= scale
        alpha_z[5] *= scale

        vcg_ode.b_x = b_x
        vcg_ode.b_y = b_y
        vcg_ode.b_z = b_z

        vcg_ode.alpha_x = alpha_x
        vcg_ode.alpha_y = alpha_y
        vcg_ode.alpha_z = alpha_z

        return vcg_ode


class STChange(Perturbation):
    def __init__(self, scale):
        super().__init__()
        self.name = "ST Change"
        self.min = scale.min
        self.max = scale.max

    def apply_perturbation(self, vcg_ode_original):
        scale = np.random.randint(self.min, self.max)

        vcg_ode = copy.deepcopy(vcg_ode_original)

        b_x = vcg_ode.b_x
        b_y = vcg_ode.b_y
        b_z = vcg_ode.b_z

        alpha_x = vcg_ode.alpha_x
        alpha_y = vcg_ode.alpha_y
        alpha_z = vcg_ode.alpha_z

        # make bigger
        alpha_x[8] *= scale
        alpha_x[6] *= scale
        alpha_x[4] *= scale

        alpha_y[8] *= scale
        alpha_y[6] *= scale
        alpha_y[4] *= scale

        alpha_z[8] *= scale
        alpha_z[6] *= scale
        alpha_z[4] *= scale

        vcg_ode.b_x = b_x
        vcg_ode.b_y = b_y
        vcg_ode.b_z = b_z

        vcg_ode.alpha_x = alpha_x
        vcg_ode.alpha_y = alpha_y
        vcg_ode.alpha_z = alpha_z

        return vcg_ode


class InvertTWaves(Perturbation):
    def __init__(self, invert_prob=0):
        super().__init__()
        self.name = "Invert T Waves"
        self.invert_prob = invert_prob

    def apply_perturbation(self, vcg_ode_original):
        # vcg_ode = copy.deepcopy(vcg_ode_original)
        # # randomly select a value between min and max
        # if np.random.rand() < self.invert_prob:
        #     th_x = vcg_ode.theta_x
        #     th_y = vcg_ode.theta_y
        #     th_z = vcg_ode.theta_z

        #     th_x[-3] = -th_x[-3]
        #     th_x[-2] = -th_x[-2]

        #     th_y[-3] = -th_y[-3]
        #     th_y[-2] = -th_y[-2]

        #     th_z[-4] = -th_z[-4]
        #     th_z[-3] = -th_z[-3]
        #     th_z[-2] = -th_z[-2]

        #     vcg_ode.theta_x = th_x
        #     vcg_ode.theta_y = th_y
        #     vcg_ode.theta_z = th_z

        # return vcg_ode
        vcg_ode = copy.deepcopy(vcg_ode_original)
        return vcg_ode


class STElevation(Perturbation):
    def __init__(self, percent_elevated):
        self.name = "ST Elevation"
        self.min = percent_elevated.min
        self.max = percent_elevated.max

    def apply_perturbation(self, vcg_ode_original):
        vcg_ode = copy.deepcopy(vcg_ode_original)
        return vcg_ode
        # randomly select a value between min and max
        percent_elevated = np.random.randint(self.min, self.max)

        th_x = vcg_ode.theta_x
        th_y = vcg_ode.theta_y
        th_z = vcg_ode.theta_z

        beat_duration = 1 / vcg_ode.f

        s_elevated = percent_elevated / 100

        degrees_elevated = 2 * np.pi * s_elevated / beat_duration

        # st-segment is made out of two gaussians
        th_x[-2] += degrees_elevated
        th_x[-1] += degrees_elevated

        th_y[-2] += degrees_elevated
        th_y[-1] += degrees_elevated

        th_z[-3] += degrees_elevated
        th_z[-2] += degrees_elevated

        vcg_ode.theta_x = th_x
        vcg_ode.theta_y = th_y
        vcg_ode.theta_z = th_z

        return vcg_ode


class STDepression(Perturbation):
    def __init__(self, percent_depressed):
        super().__init__()
        self.name = "ST Depression"
        self.min = percent_depressed.min
        self.max = percent_depressed.max

    def apply_perturbation(self, vcg_ode_original):
        vcg_ode = copy.deepcopy(vcg_ode_original)
        return vcg_ode
        # randomly select a value between min and max
        percent_depressed = np.random.randint(self.min, self.max)

        th_x = vcg_ode.theta_x
        th_y = vcg_ode.theta_y
        th_z = vcg_ode.theta_z

        beat_duration = 1 / vcg_ode.f

        s_depressed = percent_depressed / 100

        degrees_depressed = 2 * np.pi * s_depressed / beat_duration

        # st-segment is made out of two gaussians
        th_x[-2] -= degrees_depressed
        th_x[-1] -= degrees_depressed

        th_y[-2] -= degrees_depressed
        th_y[-1] -= degrees_depressed

        th_z[-3] -= degrees_depressed
        th_z[-2] -= degrees_depressed

        vcg_ode.theta_x = th_x
        vcg_ode.theta_y = th_y
        vcg_ode.theta_z = th_z

        return vcg_ode


class ModifyParameters(Perturbation):
    def __init__(self, scale):
        super().__init__()
        self.name = "Modify Parameters"
        self.scale_min = scale.min
        self.scale_max = scale.max

    def apply_perturbation(self, vcg_ode_original):
        vcg_ode = copy.deepcopy(vcg_ode_original)
        perturbation_scale = np.random.uniform(self.scale_min, self.scale_max)

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
