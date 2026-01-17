import random
from typing import List
import Levenshtein
import matplotlib.pyplot as plt


# computes levenstein distances for lists consisting of pairs (str, str)
def compute_levenstein(num : int, model_outputs : List) -> List:
    tests = random.sample(model_outputs, num)
    return [Levenshtein.distance(t[1], t[0]) for t in tests]

# computes levenstein distances for lists consisting of pairs (str, [str, str...])
# in this case, retuns the minimal levestein distance from one of the list to the first parameter
def compute_levenstein_several(num : int, model_outputs : List) -> List:
    tests = random.sample(model_outputs, num)
    return [Levenshtein.distance(t[1], t[0]) for t in tests]


def print_picture_distances(numbers : List) -> None:
    plt.figure()
    plt.hist(
        numbers,
        bins=range(min(numbers), max(numbers) + 2),
        density=True
    )

    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Histogram of Integer Distribution")
    plt.show()

