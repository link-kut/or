def combine(terms, accumulation, combinations):
    last = (len(terms) == 1)
    n = len(terms[0])
    for i in range(n):
        item = accumulation + [terms[0][i]]
        if last:
            combinations.append(item)
        else:
            combine(terms[1:], item, combinations)


a = [
    [1, 2, 3],
    [10, 11, 12],
    [100, 101, 102]
]

combinations = []
combine(a, [], combinations)

for combination in combinations:
    print(combination)
