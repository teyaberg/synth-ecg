import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from loguru import logger

from synth_ecg.utils.tools import convert_vcg_to_12lead, solve_vcg_object
from synth_ecg.utils.vcg import VCG


class ECGGenerator:
    def __init__(self, params):
        self.cfg = params
        self.duration = self.cfg.sample_params.duration
        self.frequency = self.cfg.sample_params.frequency

        self.save_duration = self.cfg.sample_params.save_duration
        self.perturbations = (
            self.cfg.generation_params.perturbations
            if hasattr(self.cfg.generation_params, "perturbations")
            else []
        )
        logger.debug(
            f"Generator initialized with perturbations {[perturb.name for perturb in self.perturbations]}"
        )

    # Generate ECG
    def generate_vcg(self, hr):
        vcg_ode = VCG(hr)
        for perturbation in self.perturbations:
            vcg_ode = perturbation(vcg_ode)
        return vcg_ode

    def generate_ecg(self, hr):
        vcg_ode = self.generate_vcg(hr)
        t, vcg = solve_vcg_object(vcg_ode, fs=self.frequency, duration=self.duration)
        ecg = convert_vcg_to_12lead(vcg)
        # return only the save duration

        start_point = np.random.randint(0, int((self.duration - self.save_duration) * self.frequency) + 1)
        t = t[start_point : start_point + int(self.save_duration * self.frequency)]
        ecg = ecg[start_point : start_point + int(self.save_duration * self.frequency)]

        # TODO: fix this so you can return specific leads
        if self.cfg.sample_params.leads is not None:
            ecg = ecg[:, range(self.cfg.sample_params.leads)]
        return ecg

    def generate_ecgs(self):
        logger.info("Generating ECGs...")
        heart_rates = np.random.randint(
            self.cfg.generation_params.heart_rate.min,
            self.cfg.generation_params.heart_rate.max,
            size=self.cfg.n_samples,
        )
        ecgs = []
        with ProcessPoolExecutor(max_workers=self.cfg.n_jobs if self.cfg.n_jobs > 0 else None) as executor:
            future_to_index = {executor.submit(self.generate_ecg, hr): i for i, hr in enumerate(heart_rates)}

            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    ecg = future.result()
                    ecgs.append(ecg)
                except Exception as e:
                    logger.error(f"Error generating ECG {i+1}: {e}")

        logger.debug(f"Generated {len(ecgs)} ECGs, with shape {np.array(ecgs).shape}")
        return ecgs

    def save_ecgs(self, ecgs):
        logger.info("Saving ECGs...")
        # make a directory
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        # save as npy
        np.save(f"{self.cfg.output_dir}/ecgs.npy", ecgs)

        return f"{self.cfg.output_dir}/ecgs.npy"
