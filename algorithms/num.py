import numpy as np

# Load the similarity matrix
sim_scores = np.loadtxt('similarity_result.txt')

# Print its dimensions
rows, cols = sim_scores.shape
print(f"The matrix in 'similarity_result.txt' has {rows} rows and {cols} columns.")
