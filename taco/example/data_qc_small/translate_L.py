import csv

with open("IX.csv", "r") as f:
    reader = csv.reader(f)
    L = list(reader)

L_mod = []
for (i, x, _) in L:
    i = int(i) - 1
    x = int(x) - 1
    for k in range(int(x/3)*7, int(x/3)*7+7):
        L_mod.append((k+1, int(i)+1, 1))

with open("L.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(L_mod)
