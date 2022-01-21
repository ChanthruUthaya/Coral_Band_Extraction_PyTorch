from random import randint
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--dims_x", type=int, default=100, help="default x dimension")
parser.add_argument("--dims_y", type=int, default=100, help="default y dimension")
args =parser.parse_args()

DIMS = (args.dims_x, args.dims_y)

def largest_theoretical_distance():
    return math.sqrt(pow(args.dims_x,2) + pow(args.dims_y,2))/2

def create_ds():
    return [np.array([randint(0,DIMS[0]), randint(0,DIMS[1])]) for i in range(0,100)]


def euclidean(point1, point2):
    return min(list(map(lambda p: np.linalg.norm(point1-p), point2)))


def hausdorff(ds1, ds2):
    return np.array(list(map(lambda p: euclidean(p, ds2), ds1))).mean()


def accuracy(h1, h2):
    dist = largest_theoretical_distance()
    avg_h = (h1+h2)/2
    acc = (dist-avg_h)/dist
    return acc


if __name__ == "__main__":

    ds1 = create_ds()
    ds2 = create_ds()

    hausdorff_set1 = hausdorff(ds1, ds2)
    hausdorff_set2 = hausdorff(ds2, ds1)

    accuracy = accuracy(hausdorff_set1, hausdorff_set2)
    print(accuracy)

