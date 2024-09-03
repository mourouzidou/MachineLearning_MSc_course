import numpy as np


## exercise 1.b

def mle_function(theta, x):
    return -np.sum(2 * np.log(theta) + np.log(x + 1) + x * np.log(1 - theta))


def calculate_mle(samples, initial_theta=0.15, threshold=0.001):
    # Initialize theta with an initial value
    theta = initial_theta

    # Initialize variable to check previous theta
    prev_theta = None

    # Iteratively update theta until convergence with precision = threshold
    while prev_theta is None or abs(theta - prev_theta) >= threshold:
        prev_theta = theta
        dltheta = -2 * len(samples) / theta + np.sum(samples / (1 - theta))
        dl2_theta = 2 * len(samples) / (theta ** 2) + np.sum(samples / ((1 - theta) ** 2))

        # Update theta using the Newton-Raphson method
        theta -= dltheta / dl2_theta

    return theta


samples = np.array([3.2, 1.4, 2.2, 7, 0.5, 3.3, 9, 0.15, 2, 3.21, 6.13, 5.5, 1.8, 1.2, 11])
estimated_theta = calculate_mle(samples)

print(estimated_theta)



