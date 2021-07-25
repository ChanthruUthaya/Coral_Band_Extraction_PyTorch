import os
max_dim = 22
excluded = []
if __name__ == '__main__':
    file_read = open('exclude.txt', 'r')
    lines = file_read.readlines()
    for line in lines:
        line = line.replace('\n', "").strip()
        first, second = line.split("-")
        r_1 = tuple(map(int, first.split(",")))
        r_2 = tuple(map(int, second.split(",")))
        #print(r_1, r_2)
        mult_r_1 = r_1[0]
        mult_r_2 = r_2[0]
        x = list(range(mult_r_1*max_dim+r_1[1], mult_r_2*max_dim+r_2[1]+1))
        out = [(i//max_dim, i%max_dim) for i in x]
        excluded = excluded + out

    print(excluded)
