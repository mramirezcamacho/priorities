import sys
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


if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = sys.argv[1]
        if text.lower() == 'delete':
            ALL_ALL = 1
        else:
            ALL_ALL = 0

    else:
        print("No input provided, using default settings (don't delete folders).")
        file = 'NewRankDataset_cristobalnavarro_20241129002848.csv'
        ALL_ALL = 0

    if ALL_ALL:
        deleteFolders()
        dividePerCountry()
        start()
    makePlots()
    makePlots2()
    makePPTX()
