import os 

if __name__ == '__main__':
    dir = os.readlink('scratch')
    print(os.listdir(dir))