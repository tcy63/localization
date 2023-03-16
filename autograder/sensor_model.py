import numpy as np
import matplotlib.pyplot as plt

def sensor_model(a_hit, a_short, a_max, a_rand, sigma, eps, z_max, d, z_k):
    p_hit = p_short = p_max = p_rand = 0

    # Case 1: the laser scan hit the obstacle; Gaussian distribution around the ground truth obstacle d
    if 0 <= z_k and z_k <= z_max:
        p_hit = 1 / np.sqrt(2*np.pi*sigma*sigma) * np.exp(-(z_k - d)**2 / (2*sigma*sigma))
    # Case 2: the scan is too short due to unknown obstacles
    if 0 <= z_k and z_k <= d and d != 0:
        p_short = 2 / d * (1 - z_k / d)
    # Case 3: the scan is very large due to beams not bouncing back
    if z_max - eps <= z_k and z_k <= z_max:
        p_max = 1 / eps
    # Case 4: assume a random scan
    if 0 <= z_k and z_k <= z_max:
        p_rand = 1 / z_max
    
    p_zk = a_hit * p_hit + a_short * p_short + a_max * p_max + a_rand * p_rand
    return p_zk, p_hit, p_short, p_max, p_rand

answer = []
for z_k in [0, 3, 5, 8, 10]:
    p_zk, p_hit, p_short, p_max, p_rand = sensor_model(0.74, 0.07, 0.07, 0.12, 0.5, 0.1, 10, 7, z_k)
    print(f"z_k: {z_k}; P(z_k|x_k, m): {p_zk}; P_hit: {p_hit}; P_short: {p_short}; P_max: {p_max}; P_rand: {p_rand}")
    answer.append(p_zk)
print(answer)

p = []
x = []
for z in np.linspace(0, 10, 200):
    p_z, _, _, _, _ = sensor_model(0.74, 0.07, 0.07, 0.12, 0.5, 0.1, 10, 7, z)
    p.append(p_z)
    x.append(z)

plt.plot(x, p)
plt.show()