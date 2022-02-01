from ec_augmented import Augmented

# Example
shape = (9, 3, 2)

Ckj = [
    [35, 40, 38],
    [-20, -22, -25]
]
Aij = [
    [2, 1.75, 2.1],
    [0.5, 0.6, 0.5],
    [35, 40, 38],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1]
]
Bi = [20_000, 15_000, 450_000, 5_000, -3_000, 7_000, -4_000, 4_000, -2_000]

augmented_obj = Augmented(shape, 50, 0)

# Enter Data
augmented_obj.cons_coefficients = Aij
augmented_obj.obj_coefficients = Ckj
augmented_obj.cons_values = Bi

# Run augmented v1
augmented_obj.run(version='v1')

# Plot
augmented_obj.plot_objectives((0, 1))