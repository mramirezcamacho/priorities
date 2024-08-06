from divideData import dividePerCountry
from analysisData import start
from dataViz import main as makePlots
from priorityChangesScript import makePlots2
from makePPTX import main as makePPTX

if __name__ == '__main__':
    dividePerCountry()
    start()
    makePlots()
    makePlots2()
    makePPTX()
