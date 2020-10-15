with open('rough_score.txt') as file:
    lines = file.readlines()

with open('rough_score.csv', 'w') as file:
    file.write('filename,precision,recall,f1measure\n')
    for line in lines:
        if line.split('.')[0].isdecimal():
            filename = line.strip()
        tokens = line.split()
        if tokens[0] == '"rouge1:':
            p = tokens[1].split('=')[1]
            r = tokens[2].split('=')[1]
            f1 = tokens[2].split('=')[1][:-1]
            file.write(filename + ',' + p + r + f1 + '\n')
