n = 2
gamma = 0.5
memory = [
    [0], [0], [1]
]
T = len(memory)
t = 0
tau = t - n + 1

while True:
    # import pudb; pudb.set_trace()
    tau = t - n + 1

    G = 0
    if tau >= 0:

        i = tau
        while True:    
            G += gamma**(i - tau) * memory[i][0]
            i += 1
            if i >= min(tau + n, T): break

        print(f"final G {G}, tau {tau}, t{t}", "next gamma", n)
    t += 1

    if tau == T - 1: break