import numpy as np

class Kalman3D:
    """
    Simple Kalman Filter pour un point 3D.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-2):
        self.P = np.eye(3)          # covariance
        self.X = np.zeros(3)        # état initial
        self.Q = process_noise * np.eye(3)
        self.R = measurement_noise * np.eye(3)

    def apply(self, measurement):
        """
        Applique le filtre Kalman sur la mesure 3D.
        """
        # Prediction
        X_pred = self.X
        P_pred = self.P + self.Q

        # Update
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        self.X = X_pred + K @ (measurement - X_pred)
        self.P = (np.eye(3) - K) @ P_pred
        return self.X.copy()
