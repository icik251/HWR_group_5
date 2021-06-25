import numpy as np

# For a page with four handwritten letters and three possible styles
# We get a list of four probability distributions over styles, example:
ProbabilityMatrix = [
    [0.1, 0.1, 0.8], 
    [0.1, 0.8, 0.1], 
    [0.8, 0.1, 0.1], 
    [0.1, 0.1, 0.8]]

ProbabilityMatrix_2 = [
    [0.0, 0.0, 1.0],
    [0.1, 0.8, 0.1],
    [0.8, 0.1, 0.1],
    [0.1, 0.1, 0.8],
]

concat_prob_matrices = ProbabilityMatrix + ProbabilityMatrix_2

linearTransform = ( lambda probability: (probability -1/3) *(1-3*0.05) +1/3 )

transformed_prob_matrices = list(map( lambda x: list(map(linearTransform,x)) , concat_prob_matrices))

print(transformed_prob_matrices)

# Naive Bayes Classification of the handwritten page in a single line:
print(np.argmax(sum(list(map(lambda x: np.log(x), transformed_prob_matrices)))))


