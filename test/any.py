from src.util.tools import *
from multiprocessing import Pool, Process, Value, Queue, Lock, Manager

class Test:
    m = Manager()
    a = m.dict()

    def t(self):
        with Pool(4) as pool:
            pool.map(Test.aa, [1,2,3,4])


    @staticmethod
    def aa(id):
        Test.a[id] = 10

        for i in Test.a.keys():
            print(Test.a[i])


if __name__ == "__main__":
    test = Test()
    test.t()
