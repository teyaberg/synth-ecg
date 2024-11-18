import numpy as np

# default parameters
theta_x = np.array([-1.09, -0.83, -0.19, -0.07, 0.00, 0.06, 0.22, 1.20, 1.42, 1.68, 2.90])
theta_y = np.array([-1.10, -0.90, -0.76, -0.11, -0.01, 0.07, 0.80, 1.58, 2.90])
theta_z = np.array([-1.10, -0.93, -0.70, -0.40, -0.15, 0.10, 1.05, 1.25, 1.55, 2.80])

alpha_x = np.array([0.03, 0.08, -0.13, 0.85, 1.11, 0.75, 0.06, 0.10, 0.17, 0.39, 0.03])
alpha_y = np.array([0.04, 0.02, -0.0, 0.32, 0.51, -0.32, 0.04, 0.08, 0.01])
alpha_z = np.array([-0.03, -0.14, -0.04, 0.05, -0.40, 0.46, -0.12, -0.20, -0.35, -0.04])

b_x = np.array([0.09, 0.11, 0.05, 0.04, 0.03, 0.03, 0.24, 0.60, 0.30, 0.18, 0.50])
b_y = np.array([0.07, 0.07, 0.04, 0.06, 0.04, 0.06, 0.45, 0.30, 0.50])
b_z = np.array([0.03, 0.12, 0.04, 0.40, 0.05, 0.05, 0.80, 0.40, 0.20, 0.40])


class VCG:
    def __init__(
        self,
        HR,
        theta_x=theta_x,
        theta_y=theta_y,
        theta_z=theta_z,
        alpha_x=alpha_x,
        alpha_y=alpha_y,
        alpha_z=alpha_z,
        b_x=b_x,
        b_y=b_y,
        b_z=b_z,
    ):
        super().__init__()

        self.HR = HR

        # rotational frequency
        self.f = self.HR / 60.0
        self.w = 2 * np.pi * self.f

        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.alpha_z = alpha_z
        self.b_x = b_x
        self.b_y = b_y
        self.b_z = b_z

    def set_HR(self, hr):
        self.HR = hr
        # rotational frequency
        self.f = self.HR / 60.0
        self.w = 2 * np.pi * self.f

    def call(self, t, v):
        theta = v[0]
        # x = v[1]
        # y = v[2]
        # z = v[3]

        dtheta_x = np.remainder((theta - self.theta_x), 2 * np.pi) - np.pi
        dtheta_y = np.remainder((theta - self.theta_y), 2 * np.pi) - np.pi
        dtheta_z = np.remainder((theta - self.theta_z), 2 * np.pi) - np.pi

        dtheta_dt = self.w

        dx_dt = -np.sum(
            (self.w * self.alpha_x / (self.b_x**2))
            * dtheta_x
            * np.exp(-(dtheta_x**2) / (2 * (self.b_x**2)))
        )
        dy_dt = -np.sum(
            (self.w * self.alpha_y / (self.b_y**2))
            * dtheta_y
            * np.exp(-(dtheta_y**2) / (2 * (self.b_y**2)))
        )
        dz_dt = -np.sum(
            (self.w * self.alpha_z / (self.b_z**2))
            * dtheta_z
            * np.exp(-(dtheta_z**2) / (2 * (self.b_z**2)))
        )

        return np.array([dtheta_dt, dx_dt, dy_dt, dz_dt])
