import numpy as np

# For a page with four handwritten letters and three possible styles
# We get a list of four probability distributions over styles, example:
ProbabilityMatrix = [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]

ProbabilityMatrix_2 = [
    [0.1, 0.1, 0.8],
    [0.1, 0.8, 0.1],
    [0.8, 0.1, 0.1],
    [0.1, 0.1, 0.8],
]


concat_prob_matrices = ProbabilityMatrix + ProbabilityMatrix_2

print(concat_prob_matrices)
# Naive Bayes Classification of the handwritten page in a single line:
print(np.argmax(sum(list(map(lambda x: np.log(x), ProbabilityMatrix)))))
