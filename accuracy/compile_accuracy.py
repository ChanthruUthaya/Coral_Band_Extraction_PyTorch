from ctypes import CDLL
import ctypes
import os

ret = os.system("cc -fPIC -shared -std=c99 -o accuracy.so accuracy.c -lm")

if ret == 0:
    print("compiled library")
    C = CDLL(os.path.abspath("accuracy.so"))
else:
    print("couldnt compile C lib")
    exit()