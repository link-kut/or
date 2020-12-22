from random import randint, expovariate

TIME_STEP_SCALE = 1 / 10
for _ in range(100):
    print(int(expovariate(0.002 * (1.0 / TIME_STEP_SCALE))))