import math
import random

import numpy as np
from scipy.special import expit, gamma
from tqdm import tqdm


def levy_flight(lambda_=1.5, min_step_size=1, max_step_size=10):
    step = random.uniform(min_step_size, max_step_size)
    v = ((lambda_ * gamma(lambda_) * math.sin(lambda_)) / math.pi) * (1 / step ** (1 + lambda_))
    return v


class BinaryFlowerPollinationAlgorithm:

    def __init__(self, num_flowers, flower_dim, estimator):
        self.num_flowers = num_flowers
        self.flower_dim = flower_dim
        self.estimator = estimator

    def fit(self, x_train, y_train, x_val, y_val, alpha=0.01, p=0.5, iterations=100):
        # Initialize flowers
        x_hat = np.random.rand(iterations + 1, self.num_flowers, self.flower_dim)
        x_hat[1:, :, :] = 0
        x = x_hat > 0.5

        # Initialize fitness values vector
        fitness_vector = [-np.inf] * self.num_flowers

        # Initialize global fitness value and global best flower
        global_fitness_value = -np.inf
        global_best_flower = None

        for i in tqdm(range(iterations)):
            for flower_index in range(self.num_flowers):

                # If none of the feature are selected, we select the first one as default
                if not x[i, flower_index].any():
                    x[0, flower_index] = True

                # Build training and validation set selecting features marked as 1 in current flower vector
                x_train_selected = x_train[:, x[i, flower_index]]
                x_val_selected = x_val[:, x[i, flower_index]]

                # Fit and evaluate model
                self.estimator.fit(x_train_selected, y_train)
                model_score = self.estimator.score(x_val_selected, y_val)

                # Update current flower score
                if model_score > fitness_vector[flower_index]:
                    fitness_vector[flower_index] = model_score

            # Update global fitness value and global best flower
            max_fitness_index = np.argmax(fitness_vector)
            max_fitness_value = fitness_vector[max_fitness_index]

            if max_fitness_value > global_fitness_value:
                global_fitness_value = max_fitness_value
                global_best_flower = x[i, max_fitness_index]

            # Update flowers' pollination
            for flower_index in range(self.num_flowers):
                for d in range(self.flower_dim):
                    rand = random.random()

                    if rand < p:
                        x_hat[i + 1][flower_index][d] = x_hat[i][flower_index][d] + alpha * levy_flight() * \
                                                        (global_fitness_value - x_hat[i][flower_index][d])
                    else:
                        e = random.random()
                        x_j = random.choice(range(self.num_flowers))
                        x_k = random.choice(range(self.num_flowers))
                        x_hat[i + 1][flower_index][d] = x_hat[i][flower_index][d] + e * \
                                                        (x_hat[i][x_j][d] - x_hat[i][x_k][d])

                    if 0.5 < expit(x_hat[i][flower_index][d]):
                        x[i + 1][flower_index][d] = 1
                    else:
                        x[i + 1][flower_index][d] = 0

        return global_fitness_value, global_best_flower
