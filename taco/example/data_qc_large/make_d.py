import random
AUX = 900
with open("D.tns", "w") as dfile:
    for k in range(1, AUX+1):
        for l in range(1, AUX+1):
            dfile.write(f"{k} {l} {random.random()}\n")

