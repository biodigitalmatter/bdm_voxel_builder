import numpy as np

print("hello")
array = np.asarray([0, 0, 2, 5, 9, 5, 5, 0, 5])
n = 3
a = np.sort(array)
best_of = a[(9 - n) :]
print(best_of)

random_choice_from_the_best_nth = np.random.choice(best_of)
print(random_choice_from_the_best_nth)
matching_i = np.argwhere(array == random_choice_from_the_best_nth).transpose()
print("matching i:", matching_i[0])
best_index = np.random.choice(matching_i[0])
print(
    "random value of the best [{}] value = {}, index = {}".format(
        n, array[best_index], best_index
    )
)


def random_choice_index_from_best_n(list_array, n):
    array = list_array
    a = np.sort(array)
    best_of = a[(len(array) - n) :]
    random_choice_from_the_best_nth = np.random.choice(best_of)
    matching_i = np.argwhere(array == random_choice_from_the_best_nth).transpose()
    random_choice_index_from_best_n = np.random.choice(matching_i[0])
    # print('random value of the best [{}] value = {}, index = {}'.format(n, array[best_index], index_of_random_choice_of_best_n_value))
    return random_choice_index_from_best_n


b = random_choice_index_from_best_n(array, n)
print("random value of the best [{}] value = {}, index = {}".format(n, array[b], b))
