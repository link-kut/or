from random import randint, expovariate

TIME_STEP_SCALE = 1 / 100
print(expovariate(0.002 * (1.0 / TIME_STEP_SCALE)))