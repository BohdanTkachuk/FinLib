from math import exp, log, pi

from scipy.stats import norm


def d_1(S, K, r, delta, v, t):
    """
    d_j = \frac{log(\frac{S}{K})+(r+(-1)^{j-1} \frac{1}{2}v^2)T}{v sqrt(T)}
    """
    return (log(S/K) + (r - delta + v**2 / 2)*t) /(v*(t**0.5))


def d_2(S, K, r, delta, v, t):
    return d_1(S,K,r,delta,v,t) - v * t**0.5




def vanilla_put_price(S, K, r, delta, v, t):
    """
    Price of a European put option struck at K, with
    spot S, constant rate r, constant vol v (over the
    life of the option) and time to maturity T
    """
    return K*exp(-r*t)*norm.cdf(-d_2(S,K,r,delta,v,t)) - S*exp(-delta*t)*norm.cdf(-d_1(S,K,r,delta,v,t))





