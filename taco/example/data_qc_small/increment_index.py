import csv

def increment_rows(rows):
    for row in rows:
        for i in range(0, len(row)-1):
            row[i] = str(int(row[i]) + 1)
    return rows

files = ["Ci.shape.csv", "E.shape.csv", "P.shape.csv", "IX.csv"]

for fname in files:
    with open(fname, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        data = increment_rows(data)

    with open(fname, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)

