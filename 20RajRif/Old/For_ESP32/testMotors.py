def map_range(value):
    old_min, old_max = 1, 100
    new_min, new_max = 300, 1000
    speed = new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)
    motor_speed = str(int(speed))
    return motor_speed

print(map_range(100))    # 300
