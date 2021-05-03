class Controller:
    # Velocity controller parameters
    target_velocity = 0
    kp_vel = 0.1
    ki_vel = 0.01
    kd_vel = 0.001

    vel_error_0 = 0
    I_vel_error = 0
    D_vel_error = 0

    # Lateral controller parameters
    kp_lat = 0.1
    ki_lat = 0.01
    kd_lat = 0.001

    deviation_0 = 0
    I_deviation = 0
    D_deviation = 0

    def set_target_velocity(self, velocity):
        self.target_velocity = velocity
        self.D_vel_error = 0
        self.vel_error_0 = 0
        self.I_vel_error = 0

    def velocity_controller(self, dt, current_velocity):
        throttle = 0
        brake = 0
        if dt == 0:
            dt = 1e-3

        vel_error = self.target_velocity - current_velocity  # velocity error
        self.D_vel_error = (vel_error - self.vel_error_0) / dt  # velocity error difference
        self.I_vel_error = self.I_vel_error + vel_error * dt  # velocity error integral over time
        self.vel_error_0 = vel_error

        throttle_signal = self.kp_vel * vel_error + self.ki_vel * self.I_vel_error + self.kd_vel * self.D_vel_error

        if throttle_signal > 1:
            throttle = 1
        elif throttle_signal > 0:
            throttle = throttle_signal
        elif throttle_signal > -1:
            brake = abs(throttle)
        else:
            brake = 1

        return throttle, brake

    def lateral_controller(self, dt, deviation):
        if dt == 0:
            dt = 1e-3

        self.D_deviation = (self.deviation_0 - deviation) / dt
        self.I_deviation = self.I_deviation + deviation * dt
        self.deviation_0 = deviation

        steering = self.kp_lat * deviation + self.ki_lat * self.I_deviation + self.kd_lat * self.D_deviation
        steering = max(min(steering, 1), -1)
        return steering
