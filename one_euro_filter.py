import math

class OneEuroFilter:
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta            # ✅ FIX
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return x if x_prev is None else a * x + (1 - a) * x_prev

    def __call__(self, x, dx=None):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0 * x
            return x

        t_e = max(1e-6, 1.0 / self.freq)
        dx = (x - self.x_prev) / t_e if dx is None else dx

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
