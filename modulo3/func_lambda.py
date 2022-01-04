
def somaquadrado(a, b):
    x = a**2 + b**2
    return x


# lambda param_1, param_n: operacao_a_realizar
sq = lambda a, b: a**2 + b**2

print(somaquadrado(3, 8))
print(sq(3, 8))
