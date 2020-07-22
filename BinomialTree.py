import numpy as np


def Binomial(n, S, K, r, v, t, PutCall):
    At = t / n
    u = np.exp(v * np.sqrt(At))
    d = 1. / u
    p = (np.exp(r * At) - d) / (u - d)
    # Binomial price tree
    stockvalue = np.zeros((n + 1, n + 1))
    stockvalue[0, 0] = S
    for i in range(1, n + 1):
        stockvalue[i, 0] = stockvalue[i - 1, 0] * u
        for j in range(1, i + 1):
            stockvalue[i, j] = stockvalue[i - 1, j - 1] * d
    # option value at final node
    optionvalue = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        if PutCall == "C":  # Call
            optionvalue[n, j] = max(0, stockvalue[n, j] - K)
        elif PutCall == "P":  # Put
            optionvalue[n, j] = max(0, K - stockvalue[n, j])

    # backward calculation for option price
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            if PutCall == "P":
                optionvalue[i, j] = max(0, K - stockvalue[i, j], np.exp(-r * At) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1]))
            elif PutCall == "C":
                optionvalue[i, j] = max(0, stockvalue[i, j] - K, np.exp(-r * At) * (
                            p * optionvalue[i + 1, j] + (1 - p) * optionvalue[i + 1, j + 1]))
    return optionvalue[0, 0]

    # Inputs
n = 1000  # input("Enter number of binomial steps: ")           #number of steps
S = 40  # input("Enter the initial underlying asset price: ") #initial underlying asset price
r = 0.06  # input("Enter the risk-free interest rate: ")        #risk-free interest rate
K = 40  # input("Enter the option strike price: ")            #strike price
v = 0.2  # input("Enter the volatility factor: ")              #volatility
t = 1.

print("American Put Price: %s" % (Binomial(n, S, K, r, v, t, PutCall="P")))
