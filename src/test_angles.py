import sys
sys.setrecursionlimit(1500)

import quadtree4 as quadtree4
import pylab, numpy
import time
from math import sin, cos
'''
dif_BAK02.py  :  good enough for t = 0
dif_testGradient.py : introducing the spatial gradient to refine cells.
dif_testSplit_BAK02  : only good before refinement

Since after refinement, cell config. changes so we must find a flexible
way to deal with cell differential scheme, rather than relying on their
positions.

dif_testSplit_BAK03  : write all the scheme formulas, although have not
yet test the correctness.
dif_testSplit_BAK04  :  quite symmetric answer but still not tested all cases.

dif_testSplit_BAK06  : make a dictionary (wrong way).
dif_testSplit_BAK07  : attempt to differentiate velocity
dif_testSplit_BAK08  : differentiate velocity lead to overflow !?
dif_testSplit_BAK09  : advection seems okay.
dif_testSplit_BAK10  : including viscosity.
'''
MAX_DEPTH = 8
# THETA_C = input('Theta C = ')
# THETA_U = input('Theta U = ')   # numpy.inf
THETA_C = 500
THETA_U = 1E-4
nu = 1E-7 # s^-1 since the scale of model is 100.
pylab.close('all')

N = 8

tmpList = [ ]
def search_DFS(mesh, cell, maxDepth=None):
    for key in cell['neighbors']:
        if key in ['N', 'S', 'E', 'W']:
            nb = cell['neighbors'][key]
            if ( type(nb['id']) is str ):     # implying a real, non-boundary cell.
                if ( (cell['level'] == nb['level'] + 1) and (nb['level'] < maxDepth) ):
                    search_DFS(mesh, nb, maxDepth)
    
    tmpList.append( cell )
    return tmpList
###

def configurator(mesh, cell):
    nbkeys = cell['neighbors'].keys()
    conf = 'Unknown'
    ratio = 1.0
    coef = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    if set(nbkeys) == set(['N','S','E','W']):   # type a
        conf = 'a'
        ratio = 1.0     # just changed
        coef = (1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,-4)
        
    if set(nbkeys) == set(['WNW','WSW','N','E','S']):   # type b
        conf = 'bW'
        ratio = 4.0 / 21
        coef = (5,0,0,0,6,0,0,0,5,0,0,4,0,4,0,0,-24)
    if set(nbkeys) == set(['NNW','NNE','E','S','W']):   # type b
        conf = 'bN'
        ratio = 4.0 / 21
        coef = (0,4,0,0,5,0,0,0,6,0,0,0,5,0,0,4,-24)
    if set(nbkeys) == set(['ENE','ESE','S','W','N']):   # type b
        conf = 'bE'
        ratio = 4.0 / 21
        coef = (5,0,0,4,0,4,0,0,5,0,0,0,6,0,0,0,-24)
    if set(nbkeys) == set(['SSE','SSW','W','N','E']):   # type b
        conf = 'bS'
        ratio = 4.0 / 21
        coef = (6,0,0,0,5,0,0,4,0,4,0,0,5,0,0,0,-24)
        
    if set(nbkeys) == set(['NNW','NNE','ENE','ESE','S','W']):   # type c
        conf = 'cNE'
        ratio = 4.0 / 11
        coef = (0,2,0,2,0,2,0,0,3,0,0,0,3,0,0,2,-14)
        
    if set(nbkeys) == set(['SSW','SSE','ENE','ESE','N','W']):   # type c
        conf = 'cSE'
        ratio = 4.0 / 11
        coef = (3,0,0,2,0,2,0,2,0,2,0,0,3,0,0,0,-14)
        
    if set(nbkeys) == set(['SSW','SSE','WNW','WSW','N','E']):   # type c
        conf = 'cSW'
        ratio = 4.0 / 11
        coef = (3,0,0,0,3,0,0,2,0,2,0,2,0,2,0,0,-14)
        
    if set(nbkeys) == set(['NNW','NNE','ENE','ESE','S','W']):   # type c
        conf = 'cNW'
        ratio = 4.0 / 11
        coef = (0,2,0,2,0,0,3,0,0,0,3,0,0,2,0,2,-14)
        
    if set(nbkeys) == set(['WNW','WSW','ENE','ESE','S','N']):   # type d
        conf = 'dEW'
        ratio = 8.0 / 9
        coef = (1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,0,-6)
        
    if set(nbkeys) == set(['NNW','NNE','SSE','SSW','E','W']):   # type d
        conf = 'dNS'
        ratio = 8.0 / 9
        coef = (0,1,0,0,1,0,1,0,0,1,0,0,1,0,1,0,-6)
        
    if set(nbkeys) == set(['WNW','WSW','ENE','ESE','SSW','SSE','N']):   # type e
        conf = 'eN'
        ratio = 8.0 / 47
        coef = (6,0,0,5,0,5,0,4,0,4,0,5,0,5,0,0,-34)
        
    if set(nbkeys) == set(['WNW','WSW','NNE','NNW','SSW','SSE','E']):   # type e
        conf = 'eE'
        ratio = 8.0 / 47
        coef = (0,5,0,0,6,0,0,5,0,5,0,4,0,4,0,5,-34)
        
    if set(nbkeys) == set(['WNW','WSW','ENE','ESE','NNW','NNE','S']):   # type e
        conf = 'eS'
        ratio = 8.0 / 47
        coef = (0,4,0,4,0,5,0,5,0,0,6,0,0,5,0,5,-34)
        
    if set(nbkeys) == set(['NNW','NNE','ENE','ESE','SSW','SSE','W']):   # type e
        conf = 'eW'
        ratio = 8.0 / 47
        coef = (0,5,0,4,0,4,0,5,0,5,0,0,6,0,0,5,-34)
        
    if set(nbkeys) == set(['WNW','WSW','ENE','ESE','SSW','SSE','NNW','NNE']):   # type f
        conf = 'f'
        ratio = 4.0 / 5
        coef = (0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,-8)
        
    if ( set(nbkeys) == set(['WNW','WSW','N','NE','E','S'])
        and cell['level'] == cell['neighbors']['NE']['level'] ):   # type g
        conf = 'gESE'
        ratio = 4.0 / 75
        coef = (13,0,6,0,8,0,0,0,15,0,0,12,0,12,0,0,-66)
        
    if ( set(nbkeys) == set(['NNW','NNE','E','SE','S','W'])
        and cell['level'] == cell['neighbors']['SE']['level'] ):   # type g
        conf = 'gSSW'
        ratio = 4.0 / 75
        coef = (0,12,0,0,13,0,6,0,8,0,0,0,15,0,0,12,-66)
        
    if ( set(nbkeys) == set(['ENE','ESE','S','SW','W','N'])
        and cell['level'] == cell['neighbors']['SW']['level'] ):   # type g
        conf = 'gWNW'
        ratio = 4.0 / 75
        coef = (15,0,0,12,0,12,0,0,13,0,6,0,8,0,0,0,-66)
        
    if ( set(nbkeys) ==  set(['SSW','SSE','W','NW','N','E'])
        and cell['level'] == cell['neighbors']['NW']['level'] ):   # type g
        conf = 'gNNE'
        ratio = 4.0 / 75
        coef = (8,0,0,0,15,0,0,12,0,12,0,0,13,0,6,0,-66)
        
    if ( set(nbkeys) == set(['WNW','WSW','N','SE','E','S'])
        and cell['level'] == cell['neighbors']['SE']['level'] ):   # type g
        conf = 'gENE'
        ratio = 4.0 / 75
        coef = (15,0,0,0,8,0,6,0,13,0,0,12,0,12,0,0,-66)
        
    if ( set(nbkeys) == set(['NNW','NNE','E','SW','S','W'])
        and cell['level'] == cell['neighbors']['SW']['level'] ):   # type g
        conf = 'gSSE'
        ratio = 4.0 / 75
        coef = (0,12,0,0,15,0,0,0,8,0,6,0,13,0,0,12,-66)
        
    if ( set(nbkeys) ==  set(['ENE','ESE','S','W','NW','N'])
        and cell['level'] == cell['neighbors']['NW']['level'] ):   # type g
        conf = 'gWSW'
        ratio = 4.0 / 75
        coef = (13,0,0,12,0,12,0,0,15,0,0,0,8,0,6,0,-66)
        
    if ( set(nbkeys) ==  set(['SSW','SSE','W','NE','N','E'])
        and cell['level'] == cell['neighbors']['NE']['level'] ):   # type g
        conf = 'gNNW'
        ratio = 4.0 / 75
        coef = (8,0,6,0,13,0,0,12,0,12,0,0,15,0,0,0,-66)
        
    if ( set(nbkeys) == set(['WNW','WSW','N','NE','E','S'])
        and cell['level'] == cell['neighbors']['NE']['level'] + 1) :   # type h
        conf = 'hESE'
        ratio = 4.0 / 27
        coef = (5,0,1,0,3,0,0,0,5,0,0,4,0,4,0,0,-22)
        
    if ( set(nbkeys) == set(['NNW','NNE','E','SE','S','W'])
        and cell['level'] == cell['neighbors']['SE']['level'] + 1):   # type h
        conf = 'hSSW'
        ratio = 4.0 / 27
        coef = (0,4,0,0,5,0,1,0,3,0,0,0,5,0,0,4,-22)
        
    if ( set(nbkeys) == set(['ENE','ESE','S','SW','W','N'])
        and cell['level'] == cell['neighbors']['SW']['level'] + 1):   # type h
        conf = 'hWNW'
        ratio = 4.0 / 27
        coef = (5,0,0,4,0,4,0,0,5,0,1,0,3,0,0,0,-22)
        
    if ( set(nbkeys) ==  set(['SSW','SSE','W','NW','N','E'])
        and cell['level'] == cell['neighbors']['NW']['level'] + 1):   # type h
        conf = 'hNNE'
        ratio = 4.0 / 27
        coef = (3,0,0,0,5,0,0,4,0,4,0,0,5,0,1,0,-22)
        
    if ( set(nbkeys) == set(['WNW','WSW','N','SE','E','S'])
        and cell['level'] == cell['neighbors']['SE']['level'] + 1):   # type h
        conf = 'hENE'
        ratio = 4.0 / 27
        coef = (5,0,0,0,3,0,1,0,5,0,0,4,0,4,0,0,-22)
        
    if ( set(nbkeys) == set(['NNW','NNE','E','SW','S','W'])
        and cell['level'] == cell['neighbors']['SW']['level'] + 1):   # type h
        conf = 'hSSE'
        ratio = 4.0 / 27
        coef = (0,4,0,0,5,0,0,0,3,0,1,0,5,0,0,4,-22)
        
    if ( set(nbkeys) ==  set(['ENE','ESE','S','W','NW','N'])
        and cell['level'] == cell['neighbors']['NW']['level'] + 1):   # type h
        conf = 'hWSW'
        ratio = 4.0 / 27
        coef = (5,0,0,4,0,4,0,0,5,0,0,0,3,0,1,0,-22)
        
    if ( set(nbkeys) ==  set(['SSW','SSE','W','NE','N','E'])
        and cell['level'] == cell['neighbors']['NE']['level'] + 1):   # type h
        conf = 'hNNW'
        ratio = 4.0 / 27
        coef = (3,0,1,0,5,0,0,4,0,4,0,0,5,0,0,0,-22)
#    
    if ( set(nbkeys) == set(['W','N','SW','E','S'])
        and cell['level'] == cell['neighbors']['SW']['level']) :   # type i
        conf = 'iSSE'
        ratio = 2.0 / 21
        coef = (9,0,0,0,9,0,0,0,4,0,3,0,8,0,0,0,-33)
        
    if ( set(nbkeys) == set(['N','E','NW','S','W'])
        and cell['level'] == cell['neighbors']['NW']['level'] ):   # type i
        conf = 'iWSW'
        ratio = 2.0 / 21
        coef = (8,0,0,0,9,0,0,0,9,0,0,0,4,0,3,0,-33)
        
    if ( set(nbkeys) == set(['E','S','NE','W','N'])
        and cell['level'] == cell['neighbors']['NE']['level'] ):   # type i
        conf = 'iNNW'
        ratio = 2.0 / 21
        coef = (4,0,3,0,8,0,0,0,9,0,0,0,9,0,0,0,-33)
        
    if ( set(nbkeys) ==  set(['S','W','SE','N','E'])
        and cell['level'] == cell['neighbors']['SE']['level'] ):   # type i
        conf = 'iENE'
        ratio = 2.0 / 21
        coef = (9,0,0,0,4,0,3,0,8,0,0,0,9,0,0,0,-33)
        
    if ( set(nbkeys) == set(['W','N','SE','E','S'])
        and cell['level'] == cell['neighbors']['SE']['level']) :   # type i
        conf = 'iSSW'
        ratio = 2.0 / 21
        coef = (9,0,0,0,9,0,3,0,4,0,0,0,8,0,0,0,-33)
        
    if ( set(nbkeys) == set(['N','E','SW','S','W'])
        and cell['level'] == cell['neighbors']['SW']['level'] ):   # type i
        conf = 'iWNW'
        ratio = 2.0 / 21
        coef = (8,0,0,0,9,0,0,0,9,0,3,0,4,0,0,0,-33)
        
    if ( set(nbkeys) == set(['E','S','NW','W','N'])
        and cell['level'] == cell['neighbors']['NW']['level'] ):   # type i
        conf = 'iNNE'
        ratio = 2.0 / 21
        coef = (4,0,0,0,8,0,0,0,9,0,0,0,9,0,3,0,-33)
        
    if ( set(nbkeys) ==  set(['S','W','NE','N','E'])
        and cell['level'] == cell['neighbors']['NE']['level'] ):   # type i
        conf = 'iESE'
        ratio = 2.0 / 21
        coef = (9,0,3,0,4,0,0,0,8,0,0,0,9,0,0,0,-33)
#
    if ( set(nbkeys) == set(['E','S','SE','W','N'])
        and ( cell['level'] - 1 == cell['neighbors']['SE']['level'] - 1
        == cell['neighbors']['S']['level'] == cell['neighbors']['E']['level'] ) ):   # type j
        conf = 'jSE'
        ratio = 2.0 / 13
        coef = (5,0,0,0,2,0,3,0,2,0,0,0,5,0,0,0,-17)
        
    if ( set(nbkeys) == set(['E','S','SW','W','N'])
        and ( cell['level'] - 1 == cell['neighbors']['SW']['level'] - 1
        == cell['neighbors']['S']['level'] == cell['neighbors']['W']['level'] ) ):   # type j
        conf = 'jSW'
        ratio = 2.0 / 13
        coef = (5,0,0,0,5,0,0,0,2,0,3,0,2,0,0,0,-17)
        
    if ( set(nbkeys) == set(['E','S','NW','W','N'])
        and ( cell['level'] -1 == cell['neighbors']['NW']['level'] - 1
        == cell['neighbors']['N']['level'] == cell['neighbors']['W']['level'] ) ):   # type j
        conf = 'jNW'
        ratio = 2.0 / 13
        coef = (2,0,0,0,5,0,0,0,5,0,0,0,2,0,3,0,-17)
        
    if ( set(nbkeys) == set(['E','S','NE','W','N'])
        and ( cell['level'] - 1 == cell['neighbors']['NE']['level'] - 1
        == cell['neighbors']['N']['level'] == cell['neighbors']['E']['level'] ) ):   # type j
        conf = 'jNE'
        ratio = 2.0 / 13
        coef = (2,0,3,0,2,0,0,0,5,0,0,0,5,0,0,0,-17)
#
    if ( set(nbkeys) == set(['E','S','SE','W','N'])
        and cell['level'] == cell['neighbors']['E']['level'] 
        and cell['neighbors']['S']['level'] == cell['neighbors']['SE']['level'] ):   # type k
        conf = 'kSSW'
        ratio = 2.0 / 15
        coef = (6,0,0,0,6,0,1,0,3,0,0,0,6,0,0,0,-22)

    if ( set(nbkeys) == set(['E','S','SW','W','N'])
        and cell['level'] == cell['neighbors']['S']['level'] 
        and cell['neighbors']['W']['level'] == cell['neighbors']['SW']['level'] ):   # type k
        conf = 'kWNW'
        ratio = 2.0 / 15
        coef = (6,0,0,0,6,0,0,0,6,0,1,0,3,0,0,0,-22)

    if ( set(nbkeys) == set(['E','S','NW','W','N'])
        and cell['level'] == cell['neighbors']['W']['level'] 
        and cell['neighbors']['N']['level'] == cell['neighbors']['NW']['level'] ):   # type k
        conf = 'kNNE'
        ratio = 2.0 / 15
        coef = (3,0,0,0,6,0,0,0,6,0,0,0,6,0,1,0,-22)

    if ( set(nbkeys) == set(['E','S','NE','W','N'])
        and cell['level'] == cell['neighbors']['N']['level'] 
        and cell['neighbors']['E']['level'] == cell['neighbors']['NE']['level'] ):   # type k
        conf = 'kESE'
        ratio = 2.0 / 15
        coef = (6,0,1,0,3,0,0,0,6,0,0,0,6,0,0,0,-22)

    if ( set(nbkeys) == set(['E','S','SW','W','N'])
        and cell['level'] == cell['neighbors']['W']['level'] 
        and cell['neighbors']['S']['level'] == cell['neighbors']['SW']['level'] ):   # type k
        conf = 'kSSE'
        ratio = 2.0 / 15
        coef = (6,0,0,0,6,0,0,0,3,0,1,0,6,0,0,0,-22)

    if ( set(nbkeys) == set(['E','S','NW','W','N'])
        and cell['level'] == cell['neighbors']['N']['level'] 
        and cell['neighbors']['W']['level'] == cell['neighbors']['NW']['level'] ):   # type k
        conf = 'kWSW'
        ratio = 2.0 / 15
        coef = (6,0,0,0,6,0,0,0,6,0,0,0,3,0,1,0,-22)

    if ( set(nbkeys) == set(['E','S','NE','W','N'])
        and cell['level'] == cell['neighbors']['E']['level'] 
        and cell['neighbors']['N']['level'] == cell['neighbors']['NE']['level'] ):   # type k
        conf = 'kNNW'
        ratio = 2.0 / 15
        coef = (3,0,1,0,6,0,0,0,6,0,0,0,6,0,0,0,-22)

    if ( set(nbkeys) == set(['E','S','SE','W','N'])
        and cell['level'] == cell['neighbors']['S']['level'] 
        and cell['neighbors']['E']['level'] == cell['neighbors']['SE']['level'] ):   # type k
        conf = 'kENE'
        ratio = 2.0 / 15
        coef = (6,0,0,0,3,0,1,0,6,0,0,0,6,0,0,0,-22)
#
    if ( set(nbkeys) == set(['E','S','SE','W','N'])
        and cell['neighbors']['E']['level'] ==
        cell['neighbors']['S']['level'] >= cell['neighbors']['SE']['level'] ):   # type l
        conf = 'lSE'
        ratio = 1.0 / 9
        coef = (6,0,0,0,3,0,2,0,3,0,0,0,6,0,0,0,-20)

    if ( set(nbkeys) == set(['E','S','SW','W','N'])
        and cell['neighbors']['S']['level'] ==
        cell['neighbors']['W']['level'] >= cell['neighbors']['SW']['level'] ):   # type l
        conf = 'lSW'
        ratio = 1.0 / 9
        coef = (6,0,0,0,6,0,0,0,3,0,2,0,3,0,0,0,-20)

    if ( set(nbkeys) == set(['E','S','NW','W','N'])
        and cell['neighbors']['N']['level'] ==
        cell['neighbors']['W']['level'] >= cell['neighbors']['NW']['level'] ):   # type l
        conf = 'lNW'
        ratio = 1.0 / 9
        coef = (3,0,0,0,6,0,0,0,6,0,0,0,3,0,2,0,-20)

    if ( set(nbkeys) == set(['E','S','NE','W','N'])
        and cell['neighbors']['N']['level'] ==
        cell['neighbors']['E']['level'] >= cell['neighbors']['NE']['level'] ):   # type l
        conf = 'lNE'
        ratio = 1.0 / 9
        coef = (3,0,2,0,3,0,0,0,6,0,0,0,6,0,0,0,-20)
    
    return conf, ratio, coef
    
def extractUW(cell, point):
    xP, yP = point
    if cell['isLeaf']:
        # print "Found %.2f, %.2f at Cell %s; U = %.2f, W = %.2f" %(xP, yP, cell['id'], cell['U'], cell['W'])
        return (cell['U'], cell['W'])
    else:
        if cell['xL'] <= xP <= cell['xC'] and cell['yB'] <= yP <= cell['yC']:
            return extractUW(cell['SW'], point)
        elif cell['xL'] <= xP <= cell['xC'] and cell['yC'] <= yP <= cell['yT']:
            return extractUW(cell['NW'], point)
        elif cell['xC'] <= xP <= cell['xR'] and cell['yB'] <= yP <= cell['yC']:
            return extractUW(cell['SE'], point)
        elif cell['xC'] <= xP <= cell['xR'] and cell['yC'] <= yP <= cell['yT']:
            return extractUW(cell['NE'], point)



COLOR_BW = (raw_input('B/W output? [y/n]') == 'y')

# Initialization
# C matrix
side = 2**N
matC = numpy.zeros((side, side), dtype=float)

xLport = 0.5 - 1.0 / side
xRport = 0.5 + 1.0 / side
matC[side-1][side/2-1] = matC[side-1][side/2] = Co = 100

meshC = quadtree4.QTree(matrix=matC, propnames=['C','Cnew'])
meshC.split_BFS(meshC.rootCell, threshold=1, maxDepth=MAX_DEPTH)
for cel in meshC.leafList:
    meshC.assignDiagNeighbors(cel)

# UW matrix
try:
    Ua = input('Ambient velocity (/s) [0.002] = ')
except:     # SyntaxError / NameError : Use default value
    Ua = 0.002

matUW = numpy.zeros((side, side), dtype=float) + Ua

try:
    Wo = input('Jet velocity (/s) [0.005] = ')   # to refine the mesh
except:     # SyntaxError / NameError : Use default value
    Wo = 0.005

matUW[side-1][side/2-1] = matUW[side-1][side/2] = Wo

try:
    alpha = input('Incline angle jet [90] = ') * 3.1416 / 180
except:     # SyntaxError / NameError : Use default value
    alpha = 90

meshUW = quadtree4.QTree(matrix=matUW,  propnames=['U','Unew','W','Wnew'])
meshUW.split_BFS(meshUW.rootCell, threshold=0, maxDepth=MAX_DEPTH)

# Calculation

try:
    delta_t = input('Time step (s) [0.05] = ')
except:     # SyntaxError / NameError : Use default value
    delta_t = 0.05

try:
    eps = input('Eddy diffusivity (unit^2/s) [1E-7] = ')      # 0.001 m^2/s
except:     # SyntaxError / NameError : Use default value
    eps = 1E-7

try:
    nt = input('# time steps [1000] = ')
except:     # SyntaxError / NameError : Use default value
    nt = 1000

starttime = time.clock()

for timeCounter in range(nt):
    t =  timeCounter * delta_t
    if (timeCounter % 500 == 0):
        print
        print 't (s) = ', t
    
    meshC.patchesList = []      # reset the patches for newer time step
    
    refineList = []
    refineListUW = []
    
    for cell in meshUW.leafList:
        h = cell['side']
        nbdict = cell['neighbors']
        Uc = cell['U']
        Wc = cell['W']
        xL = cell['xL']
        xC = cell['xC']
        yC = cell['yC']
        xR = cell['xR']
        yB = cell['yB']
        yT = cell['yT']
        
        ududx = udwdx = wdudy = wdwdy = 0
        if (yB == 0) and (xLport <= xC <= xRport):     # at the mouth of outfall
            _dir = "*"
            conf = ' (Outfall) '
            ratio = 0
            coef = (0,)
            cell['U'] = cell['Unew'] = Wo * cos(alpha)
            cell['W'] = cell['Wnew'] = Wo * sin(alpha)
        
        elif (yB >= 0.5):     # in the air. Zero concentration
            _dir = "o"
            conf = 'air'
            ratio = 0
            coef = (0,)
            cell['U'] = cell['W'] = cell['Unew'] = cell['Wnew'] = 0
        
        else:
            # to ensure that every neighbor C has a value.
            Un = Unne = Une = Uene = Ue = Uese = Use = Usse = 0
            Wn = Wnne = Wne = Wene = We = Wese = Wse = Wsse = 0
            Us = Ussw = Usw = Uwsw = Uw = Uwnw = Unw = Unnw = 0
            Ws = Wssw = Wsw = Wwsw = Ww = Wwnw = Wnw = Wnnw = 0
            
            try: 
                Un = nbdict['N']['U']
                Wn = nbdict['N']['W']
            except: pass
            try: 
                Unne = nbdict['NNE']['U']
                Wnne = nbdict['NNE']['W']
            except: pass
            try: 
                Uene = nbdict['ENE']['U']
                Wene = nbdict['ENE']['W']
            except: pass
            try: 
                Ue = nbdict['E']['U']
                We = nbdict['E']['W']
            except: pass
            try: 
                Uese = nbdict['ESE']['U']
                Wese = nbdict['ESE']['W']
            except: pass
            try: 
                Usse = nbdict['SSE']['U']
                Wsse = nbdict['SSE']['W']
            except: pass
            try: 
                Us = nbdict['S']['U']
                Ws = nbdict['S']['W']
            except: pass
            try: 
                Ussw = nbdict['SSW']['U']
                Wssw = nbdict['SSW']['W']
            except: pass
            try: 
                Uwsw = nbdict['WSW']['U']
                Wwsw = nbdict['WSW']['W']
            except: pass
            try: 
                Uw = nbdict['W']['U']
                Ww = nbdict['W']['W']
            except: pass
            try: 
                Uwnw = nbdict['WNW']['U']
                Wwnw = nbdict['WNW']['W']
            except: pass
            try: 
                Unnw = nbdict['NNW']['U']
                Wnnw = nbdict['NNW']['W']
            except: pass
            
            # at boundaries we have to link to virtual cells before configuration & solve
            if yB == 0:
                Ws = -Wc
            if (xL == 0) and (yT <= 0.5):
                Ww = 0
                Uw = Uc
                Wnw = 0
                Unw = Un
            if (xR == 1) and (yT <= 0.5):
                We = 0
                Ue = Uc

            if xL == 0.5: 
                _dir = '|^'
            elif xR == 0.5: 
                _dir = '^|'
            elif yB == 0 : 
                _dir = '--'
            else:
                _dir = '  '
                
            K = nbdict.keys()
            if 'W' in K:
                dxw = xC - nbdict['W']['xC']
                duwdx = (Uc - Uw) / dxw
                dwwdx = (Wc - Ww) / dxw
                UMw = Uw if Uw > 0 else Uc
                WMw = (Ww + Wc) / 2.0
                
            if 'E' in K:
                dxe = nbdict['E']['xC'] - xC
                duedx = (Ue - Uc) / dxe
                dwedx = (We - Wc) / dxe
                UMe = Ue if Ue < 0 else Uc
                WMe = (Wc + We) / 2.0
            
            if 'WNW' in K and 'WSW' in K:
                dxw = xC - nbdict['WNW']['xC']
                duwdx = 0.5 * ( (Uc - Uwnw)  + (Uc - Uwsw) ) / dxw
                dwwdx = 0.5 * ( (Wc - Wwnw)  + (Wc - Wwsw) ) / dxw
                UMw = (Uwnw + Uwsw) / 2.0 if (Uwnw + Uwsw) > 0 else Uc
                WMw = (Wwnw + Wwsw + 2 * Wc) / 4.0
                
            if 'ENE' in K and 'ESE' in K:
                dxe = nbdict['ENE']['xC'] - xC
                duedx = (0.5 * (Uene + Uese) - Uc) / dxe
                dwedx = (0.5 * (Wene + Wese) - Wc) / dxe
                UMe = (Uene + Uese) / 2.0 if (Uene + Uese) < 0 else Uc
                WMe = (Wene + Wese + 2 * Wc) / 4.0
                
            if 'S' in K:
                dys = yC - nbdict['S']['yC']
                dusdy = (Uc - Us) / dys
                dwsdy = (Wc - Ws) / dys
                UMs = (Us + Uc) / 2.0
                WMs = Ws if Ws > 0 else Wc
                
            if 'N' in K:
                dyn = nbdict['N']['yC'] - yC
                dundy = (Un - Uc) / dyn
                dwndy = (Wn - Wc) / dyn
                UMn = (Uc + Un) / 2.0
                WMn = Wn if Wn < 0 else Wc
                
            if 'SSW' in K and 'SSE' in K:
                dys = yC - nbdict['SSW']['yC']
                dusdy = 0.5 * ( (Uc - Ussw)  + (Uc - Usse) ) / dys
                dwsdy = 0.5 * ( (Wc - Wssw)  + (Wc - Wsse) ) / dys
                UMs = (Ussw + Usse + 2 * Uc) / 4.0
                WMs = (Wssw + Wsse) / 2.0 if (Wssw + Wsse) > 0 else Wc
                
            if 'NNE' in K and 'NNW' in K:
                dyn = nbdict['NNE']['yC'] - yC
                dundy = 0.5 * ( (Unne - Uc)  + (Unnw - Uc) ) / dyn
                dwndy = 0.5 * ( (Wnne - Wc)  + (Wnnw - Wc) ) / dyn
                UMn = (Unnw + Unne + 2 * Uc) / 4.0
                WMn = (Wnnw + Wnne) / 2.0 if (Wnnw + Wnne) < 0 else Wc
            
            if Uc > 0:
                ududx = UMw * duwdx
                udwdx = UMw * dwwdx
            elif Uc < 0:
                ududx = UMe * duedx
                udwdx = UMe * dwedx
            else:
                ududx = (UMw * duwdx + UMe * duedx) / 2.0
                udwdx = (UMw * dwwdx + UMe * dwedx) / 2.0
            
            if Wc > 0:
                wdudy = WMs * dusdy
                wdwdy = WMs * dwsdy
            elif Wc < 0:
                wdudy = WMn * dundy
                wdwdy = WMn * dwndy
            else:
                wdudy = (WMs * dusdy + WMn * dundy) / 2.0
                wdwdy = (WMs * dwsdy + WMn * dwndy) / 2.0
            
            # viscosity
            visu = nu * 2 * ((duedx - duwdx) / (dxw + dxe) + (dundy - dusdy) / (dys + dyn))
            visw = nu * 2 * ((dwedx - dwwdx) / (dxw + dxe) + (dwndy - dwsdy) / (dys + dyn))
            Unew = Uc - delta_t * (ududx + wdudy - visu)
            Wnew = Wc - delta_t * (udwdx + wdwdy - visw)
            
            cell['Unew'] = Unew
            cell['Wnew'] = Wnew
            
        graU = numpy.sqrt(ududx**2 + wdwdy**2)
        
        tmpList = [ ]
        if (graU > THETA_U) and (cell['yT'] <= 0.5) and (cell['level'] < MAX_DEPTH): 
            # sort ascending based on the level
            search_DFS( meshUW, cell, MAX_DEPTH )
            for anycell in tmpList:
                if not (anycell in refineListUW):
                    refineListUW.append( anycell )

    refineListUW = sorted(refineListUW, key=lambda k: k['level'])

    while refineListUW != []:
        c = refineListUW.pop(0)
        meshUW.refine(c)
        
    
    for cell in meshC.leafList:
        h = cell['side']
        nbdict = cell['neighbors']
        Cc = cell['C']
        xL = cell['xL']
        xC = cell['xC']
        yC = cell['yC']
        xR = cell['xR']
        yB = cell['yB']
        yT = cell['yT']
        graC = 0
        dCdx = dCdy = 0
        fluxw = fluxe = fluxn = fluxs = 0
        
        if (yB == 0) and (xLport <= xC <= xRport):     # at the mouth of outfall
            _dir = ' *'
            conf = ' (Outfall) '
            ratio = 0
            coef = (0,)
            Cadv = 0
            Cdif = 0
            
        elif (yB >= 0.5):     # in the air. Zero concentration
            _dir = ' o'
            conf = 'air'
            ratio = 0
            coef = (0,)
            Cadv = 0
            Cdif = 0
        
        else:
            # to ensure that every neighbor C has a value.
            Cn = Cnne = Cne = Cene = Ce = Cese = Cse = Csse = 0
            Cs = Cssw = Csw = Cwsw = Cw = Cwnw = Cnw = Cnnw = 0
            try: Cn = nbdict['N']['C']
            except: pass
            try: Cnne = nbdict['NNE']['C']
            except: pass
            try: Cne = nbdict['NE']['C']
            except: pass
            try: Cene = nbdict['ENE']['C']
            except: pass
            try: Ce = nbdict['E']['C']
            except: pass
            try: Cese = nbdict['ESE']['C']
            except: pass
            try: Cse = nbdict['SE']['C']
            except: pass
            try: Csse = nbdict['SSE']['C']
            except: pass
            try: Cs = nbdict['S']['C']
            except: pass
            try: Cssw = nbdict['SSW']['C']
            except: pass
            try: Csw = nbdict['SW']['C']
            except: pass
            try: Cwsw = nbdict['WSW']['C']
            except: pass
            try: Cw = nbdict['W']['C']
            except: pass
            try: Cwnw = nbdict['WNW']['C']
            except: pass
            try: Cnw = nbdict['NW']['C']
            except: pass
            try: Cnnw = nbdict['NNW']['C']
            except: pass
            # at boundaries we have to link to virtual cells before configuration & solve
            if yB == 0:
                Cs = Cc
                Cse = Ce
                Csw = Cw
            if (xL == 0) and (yT <= 0.5):
                Cw = 0
                Cnw = 0
            if (xR == 1) and (yT <= 0.5):
                Ce = 0
                Cne = 0
                
            # Advection ->
            
            # get the velocity at the center points of N/S/E/W faces
            Ucenn, Wcenn = extractUW(meshUW.rootCell, (xC, yT))
            Ucens, Wcens = extractUW(meshUW.rootCell, (xC, yB))
            Ucene, Wcene = extractUW(meshUW.rootCell, (xR, yC))
            Ucenw, Wcenw = extractUW(meshUW.rootCell, (xL, yC))
                
            if Ucenw == 0:
                fluxw = 0
            elif Ucenw < 0:
                fluxw = Cc * Ucenw
            elif Ucenw > 0:
                if nbdict.has_key('WNW'):
                    fluxw = 0.5 * (Cwnw + Cwsw) * Ucenw
                else:
                    fluxw = Cw * Ucenw
                
            if Ucene == 0:
                fluxe = 0
            elif Ucene > 0:
                fluxe = Cc * Ucene
            elif Ucene < 0:
                if nbdict.has_key('ENE'):
                    fluxe = 0.5 * (Cene + Cese) * Ucene
                else:
                    fluxe = Ce * Ucenw
            
            if Wcens == 0:
                fluxs = 0
            elif Wcens < 0:
                fluxs = Cc * Wcens
            elif Wcens > 0:
                if nbdict.has_key('SSW'):
                    fluxs = 0.5 * (Cssw + Csse) * Wcens
                else:
                    fluxs = Cs * Wcens
            
            if Wcenn == 0:
                fluxn = 0
            elif Wcenn > 0:
                fluxn = Cc * Wcenn
            elif Wcenn < 0:
                if nbdict.has_key('NNW'):
                    fluxn = 0.5 * (Cnnw + Cnne) * Wcenn
                else:
                    fluxn = Cn * Wcenn
            
            Cadv = -(fluxe - fluxw + fluxn - fluxs) * delta_t / h
            # dealing with diffusion ...
            CList = ( Cn, Cnne, Cne, Cene, Ce, Cese, Cse, Csse, 
                    Cs, Cssw, Csw, Cwsw, Cw, Cwnw, Cnw, Cnnw, Cc )
            
            # Solving for diffusion eq
            conf, ratio, coef = configurator(meshC, cell)
            
            assert( len(coef) == 17 )
            Cdif = 0
            for i in range(17):
                Cdif += coef[i] * CList[i]
            
            Cdif *= ratio * eps * delta_t / (h * h)
        
        Ccnew = Cc + Cadv + Cdif
        
        cell['Cnew'] = Ccnew
        cell['conf'] = conf
        cell['ratio'] = ratio
        cell['coef'] = coef
        
        if xL == 0.5: 
            _dir = '|^'
        elif xR == 0.5: 
            _dir = '^|'
        elif yB == 0 : 
            _dir = '--'
        else:
            _dir = '  '
            
        K = nbdict.keys()
        try:
            if 'W' in K and 'E' in K:
                dCdx = (Ce - Cw) / (nbdict['E']['xC'] - nbdict['W']['xC'])
            elif 'WNW' in K and 'WSW' in K and 'E' in K:
                dCdx = 0.5 * ( (Ce - Cwnw)  + (Ce - Cwsw) ) / (nbdict['E']['xC'] - nbdict['WNW']['xC'])
            elif 'WNW' in K and 'WSW' in K and 'ENE' in K and 'ESE' in K:
                dCdx = 0.5 * ( (Cene - Cwnw)  + (Cese - Cwsw) ) / (nbdict['ENE']['xC'] - nbdict['WNW']['xC'])
            elif 'W' in K and 'ENE' in K and 'ESE' in K:
                dCdx = 0.5 * ( (Cene - Cw)  + (Cese - Cw) ) / (nbdict['ENE']['xC'] - nbdict['W']['xC'])
        except:
            print cell['id'], cell['neighbors'].keys()
        
        try:
            if 'N' in K and 'S' in K:
                dCdy = (Cn - Cs) / (nbdict['N']['yC'] - nbdict['S']['yC'])
            elif 'SSW' in K and 'SSE' in K and 'N' in K:
                dCdy = 0.5 * ( (Cn - Cssw)  + (Cn - Csse) ) / (nbdict['N']['yC'] - nbdict['SSW']['yC'])
            elif 'SSW' in K and 'SSE' in K and 'NNW' in K and 'NNE' in K:
                dCdy = 0.5 * ( (Cnne - Csse)  + (Cnnw - Cssw) ) / (nbdict['NNE']['yC'] - nbdict['SSE']['yC'])
            elif 'S' in K and 'NNE' in K and 'NNW' in K:
                dCdy = 0.5 * ( (Cnne - Cs)  + (Cnnw - Cs) ) / (nbdict['NNE']['yC'] - nbdict['S']['yC'])
        except:
            print cell['id']
            
        graC = numpy.sqrt(dCdx**2 + dCdy**2)
        
        if (timeCounter % 200 == 0) and cell['level'] == MAX_DEPTH:
            print _dir, cell['id'], cell['C'], fluxn, fluxs, fluxe, fluxw
        
        tmpList = [ ]
        if (graC > THETA_C) and (cell['yT'] <= 0.5) and (cell['level'] < MAX_DEPTH): 
            # sort ascending based on the level
            search_DFS( meshC, cell, MAX_DEPTH )
            for anycell in tmpList:
                if not (anycell in refineList):
                    refineList.append( anycell )

    refineList = sorted(refineList, key=lambda k: k['level'])

    while refineList != []:
        c = refineList.pop(0)
        meshC.refine(c)
        print "Refine cell ", c['id']
    
    # Updating new values
    for cell in meshC.leafList:
        cell['C'] = cell['Cnew']
        meshC.assignDiagNeighbors(cell)         # update neighborhood of newer cells
    
    for cell in meshUW.leafList:    # update
        cell['U'] = cell['Unew']
        cell['W'] = cell['Wnew']
        if (timeCounter % 200 == 0) and (cell['xL'] == 0.5 or cell['xR'] == 0.5):
            # print t, _dir, cell['id'], cell['Unew'], cell['Wnew'] 
            pass
            
print 'Ellapsed time: ', time.clock() - starttime, ' seconds'

# Post-processing: draw the meshes

def getColor(v, vmin, vmax):
    if v < vmin:
        v = vmin
        return [0, 0, 1]
    if v > vmax:
        v = vmax
        return [1, 0, 0]
    dv = 1.0 * (vmax - vmin)
    if v < vmin + 0.25 * dv:
        return [0, 4 * (v - vmin) / dv, 1]
    elif v < vmin + 0.5 * dv:
        return [0, 1, 1 + 4 * (vmin + 0.25 * dv - v) / dv]
    elif v < vmin + 0.75 * dv:
        return [4 * (v - vmin - 0.5 * dv) / dv, 1, 0]
    else:
        return [1, 1 + 4 * (vmin + 0.75 * dv - v) / dv, 0]

    
for cell in meshC.leafList:
    h = cell['side']
    Ccnew = cell['Cnew']
    meshC.textList.append((cell['xC'], cell['yC'], Ccnew))
    color = 1.0 - numpy.log(abs(Ccnew) + 1.0) / numpy.log(Co + 1)      # log scale
    # color = 1.0 - float(Ccnew / Co)
    if COLOR_BW:
        color = 0 if color < 0 else color
        color = 1 if color > 1 else color
        meshC.patchesList.append( [(cell['xL'], cell['yB']), h, h, str(color), '0'] )
    else:
        color = getColor(color, 0, 1)
        meshC.patchesList.append( [(cell['xL'], cell['yB']), h, h, color, '0'] )
    

for cell in meshUW.leafList:
    Unew = cell['Unew']
    Wnew = cell['Wnew']
    vel = (Unew * Unew + Wnew * Wnew) ** 0.5
    
    color = 1.0 - float(vel / Wo)
    color = 0 if color < 0 else color
    color = 1 if color > 1 else color
    h = cell['side']
    meshUW.patchesList.append( [(cell['xL'], cell['yB']), h, h, str(color), '0'] )
    meshUW.textList.append([cell['xC'], cell['yC'], str(vel)])
    
meshC.draw(t, grid=True, num=True, arrow=False, patches = False, quiver=False)

meshUW.draw(t, grid=True, num=True, arrow=False, patches=False, quiver=False)

pylab.show()
