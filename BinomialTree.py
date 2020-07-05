# Author: Victor Lopez Lopez,
# Group Members: Shedrack Lutembeka, Bo Yuan, Victor Lopez Lopez
# Binomial tree (Cox-Rox-Rubenstein) for American Option Valuation 

# Inspired on: 
# Binomial Tree for America and European options by Mehdi Bounouar
# Binomial Tree Option Valuation Cox, Ross, Rubinstein method by "www.quantandfinancial.com"
import matplotlib.pyplot as plt
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





# Graphs and results for the Option prices
"""y = [-Binomial(n, S, K, r, v, t, "C")] * (K)
y += [x - Binomial(n, S, K, r, v, t, "C") for x in range(K)]

plt.plot(range(2 * K), y)
plt.axis([0, 2 * K, min(y) - 10, max(y) + 10])
plt.xlabel('Underlying asset price')
plt.ylabel('Profits')
plt.axvline(x=K, linestyle='--', color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.title('American Call Option')
plt.text(105, 0, 'K')
plt.show()

print(
"American Call Price: %s" % (Binomial(n, S, K, r, v, t, PutCall="C")))

z = [-x + K - Binomial(n, S, K, r, v, t, "P") for x in range(K)]
z += [-Binomial(n, S, K, r, v, t, "P")] * (K)

plt.plot(range(2 * K), z, color='red')
plt.axis([0, 2 * K, min(y) - 10, max(y) + 10])
plt.xlabel('Underlying asset price')
plt.ylabel('Profits')
plt.axvline(x=K, linestyle='--', color='black')
plt.axhline(y=0, linestyle=':', color='black')
plt.title('American Put Option')
plt.text(105, 0, 'K')
plt.show()
"""

print(
"American Put Price: %s" % (Binomial(n, S, K, r, v, t, PutCall="P")))
