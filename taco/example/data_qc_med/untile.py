import csv
TILE_AO = 3
TILE_K = 7

def transpose2d(rows):
    rows_mod = []
    for row in rows:
        rows_mod.append([row[1], row[0], row[2]])
    return rows_mod
def increment_rows(rows):
    for row in rows:
        for i in range(0, len(row)-1):
            row[i] = str(int(row[i]) + 1)
    return rows

def untileC(tensor):
    tensor_mod = []
    for row in tensor:
        nnz_val = float(row[-1])
        pao, mo = int(row[0]) - 1, int(row[1]) - 1
        linearized_paos = [pao * TILE_AO + i for i in range(0, TILE_AO)]
        for new_pao in linearized_paos:
            tensor_mod.append([new_pao, mo, nnz_val])
    return tensor_mod

def untileE(tensor):
    tensor_mod = []
    for row in tensor:
        nnz_val = float(row[-1])
        pao1, pao2, k = int(row[0]) - 1, int(row[1]) - 1, int(row[2]) - 1
        linearized_paos1 = [pao1 * TILE_AO + i for i in range(0, TILE_AO)]
        linearized_paos2 = [pao2 * TILE_AO + i for i in range(0, TILE_AO)]
        linearized_k = [k * TILE_K + i for i in range(0, TILE_K)]
        for new_pao1 in linearized_paos1:
            for new_pao2 in linearized_paos2:
                for new_k in linearized_k:
                    tensor_mod.append([new_pao1, new_pao2, new_k, nnz_val])
    return tensor_mod

def untileIX(tensor):
    tensor_mod = []
    for row in tensor:
        nnz_val = float(row[-1])
        mo, k = int(row[0]) - 1, int(row[1]) - 1
        linearized_k = [k * TILE_K + i for i in range(0, TILE_K)]
        for new_k in linearized_k:
            tensor_mod.append([mo, new_k, nnz_val])
    return tensor_mod

def untileP(tensor):
    tensor_mod = []
    for row in tensor:
        nnz_val = float(row[-1])
        pao1, pao2 = int(row[0]) - 1, int(row[1]) - 1
        linearized_paos1 = [pao1 * TILE_AO + i for i in range(0, TILE_AO)]
        linearized_paos2 = [pao2 * TILE_AO + i for i in range(0, TILE_AO)]
        for new_pao1 in linearized_paos1:
            for new_pao2 in linearized_paos2:
                tensor_mod.append([new_pao1, new_pao2, nnz_val])
    return tensor_mod




files = ["Ci.shape.csv", "E.shape.csv", "P.shape.csv", "IX.csv"]
functions = [untileC, untileE, untileP, (untileIX, transpose2d)]
new_names = ["C.tns", "Int.tns", "Phat.tns", "L.tns"]

for ind, fname in enumerate(files):
    with open(fname, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        if type(functions[ind]) == tuple:
            data = functions[ind][0](data)
            data = functions[ind][1](data)
        else:
            data = functions[ind](data)
        data = increment_rows(data)

    with open(new_names[ind], "w") as f:
        for row in data:
            f.write(" ".join(map(lambda s: str(s), row)) + "\n")

