"""This module supports angle converter functions.
"""
def azimuth_angle_to_motor_angle(degree):
  # In ISO8855, 0 degree is axis of longitude(12 o'clock of vehicle) and clockwise is minus, anti anti-clockwise is plus
  # In lidar motor angle, 0 degress is 9 o'clock of vehicle.
  if degree < 0:
    # right side ( 12:00 ~ 06:00 clockwise)
    return 90 - degree
  else:
    # left side ( 06:00 ~ 12:00 anti-clockwise)
    return 360 - (degree - 90) if degree > 90 else (90 - degree)
