from divideData import dividePerCountry
from analysisData import start
from dataViz import main as makePlots
from priorityChangesScript import makePlots2
from makePPTX import main as makePPTX
from centralizedData import mutableFolders


def deleteFolders():
    import shutil
    for folder in mutableFolders:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass


ALL_ALL = 0
if __name__ == '__main__':
    if ALL_ALL:
        deleteFolders()
        dividePerCountry()
        start()
    makePlots()
    makePlots2()
    makePPTX()
