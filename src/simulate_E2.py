# file simulate_E2.py
'''
Simulate the flow and concentration using quadtree grids
Only for Case E2 because the boundary condition and matrix seeding is different
Outlet on the Left Boundary (x0 = -0.5, y0 = 0.125)
'''

import quadtree4 as quadtree4
import numpy
from math import sin, cos
import time


UNIT_LEN = 100.    # Length of unit square
MAX_DEPTH = 9
THETA_C = 1000
THETA_U = 1E-5

SURFACE = 0.5
size_matr_seed = 2**(MAX_DEPTH - 1)

tmpList = [ ]  # initialised here, to be used in `search_DFS`

def search_DFS(mesh, cell, maxDepth=None):
    for key in cell['neighbors']:
        if key in ['N', 'S', 'E', 'W']:
            nb = cell['neighbors'][key]
            if ( type(nb['id']) is str ):     # implying a real, non-boundary cell.
                if ( (cell['level'] == nb['level'] + 1) and (nb['level'] < maxDepth) ):
                    search_DFS(mesh, nb, maxDepth)
    
    tmpList.append( cell )
    return tmpList


def configurator(mesh, cell):
    # TODO: use a dictionary instead of multibranch if
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


def extractC(cell, point):
    xP, yP = point
    if cell['isLeaf']:
        return cell['C']
    else:
        if cell['xL'] <= xP <= cell['xC'] and cell['yB'] <= yP <= cell['yC']:
            return extractC(cell['SW'], point)
        elif cell['xL'] <= xP <= cell['xC'] and cell['yC'] <= yP <= cell['yT']:
            return extractC(cell['NW'], point)
        elif cell['xC'] <= xP <= cell['xR'] and cell['yB'] <= yP <= cell['yC']:
            return extractC(cell['SE'], point)
        elif cell['xC'] <= xP <= cell['xR'] and cell['yC'] <= yP <= cell['yT']:
            return extractC(cell['NE'], point)


## Data input
## Data input
scenario = 'E2'
print('Running scenario E2')

SURFACE = 1 # 0.25 # 1
Ua = 0.5 / UNIT_LEN
Wo = 2 / UNIT_LEN
alpha = 0 * numpy.pi / 180
Sc = 0.67  # (turbulent) Schmidt number
dens_deficit = 0.025
eps = 0.01 / (UNIT_LEN**2)
nuh = 0.01 / (UNIT_LEN**2)
nuv = 0.01 / (UNIT_LEN**2)
THETA_U = 1E-3
THETA_C = 1200 # 1500
THETA_U2 = 5E-4
THETA_C2 = 1000
delta_t = 0.001
nt = 30_000


## Initialization

matC = numpy.zeros((size_matr_seed, size_matr_seed), dtype=float)
matU = numpy.zeros((size_matr_seed, size_matr_seed), dtype=float) + Ua
if scenario not in ['E2']:
    xo_port = 0.5
    yo_port = 0
    xLport = xo_port - 1.0 / size_matr_seed
    xRport = xo_port + 1.0 / size_matr_seed
    width_outlet = xRport - xLport
    matC[size_matr_seed-1][size_matr_seed//2-1] = matC[size_matr_seed-1][size_matr_seed//2] = Co = 100
    matU[size_matr_seed-1][size_matr_seed//2-1] = matU[size_matr_seed-1][size_matr_seed//2] = Wo
else:
    xo_port = 0
    yo_port = yo = 0.9375 # 15/16
    yBport = yo - 1.0 / size_matr_seed
    yTport = yo + 1.0 / size_matr_seed
    width_outlet = yTport - yBport
    matC[size_matr_seed*1//16-1][0] = matC[size_matr_seed*1//16][0] = Co = 100
    matU[size_matr_seed*1//16-1][0] = matU[size_matr_seed*1//16][0] = Wo

meshC = quadtree4.QTree(matrix=matC, propnames=['C','Cnew'])
meshC.split_BFS(meshC.rootCell, threshold=1, maxDepth=MAX_DEPTH)
for cel in meshC.leafList:
    meshC.assignDiagNeighbors(cel)

meshU = quadtree4.QTree(matrix=matU, propnames=['U','Unew','W','Wnew','P','Pnew'])
meshU.split_BFS(meshU.rootCell, threshold=0, maxDepth=MAX_DEPTH)



## Calculation

starttime = time.time()

for timeCounter in range(nt):
    t = timeCounter * delta_t
    if (timeCounter % 300 == 0):
        print(f'({timeCounter})\tt = {t} \tElapsed: {time.time() - starttime:.0f} s', end='')
        print('\tMesh C:', len(meshC.leafList), '\tMeshU:', len(meshU.leafList))
    
    meshC.patchesList = []      # reset the patches for newer time step
    refineList = []
    refineListU = []
    
    for cell in meshU.leafList:
        h = cell['side']
        nbdict = cell['neighbors']
        Up = cell['U']
        Wp = cell['W']
        Pp = cell['P']
        xL = cell['xL']
        xC = cell['xC']
        yC = cell['yC']
        xR = cell['xR']
        yB = cell['yB']
        yT = cell['yT']
        
        ududx = udwdx = wdudy = wdwdy = 0
        if (xL == 0) and (yBport <= yC <= yTport):     # (xL == 0.5) old
            conf = ' (Outfall) '
            ratio = 0
            coef = (0,)
            cell['U'] = cell['Unew'] = Wo * cos(alpha)
            cell['W'] = cell['Wnew'] = Wo * sin(alpha)
        
        elif (yB >= SURFACE):     # in the air. Zero concentration
            conf = 'air'
            ratio = 0
            coef = (0,)
            cell['U'] = cell['W'] = cell['P'] = 0
            cell['Unew'] = cell['Wnew'] = cell['Pnew'] = 0
        
        else:
            # obtain pressure and velocity components if available
            Un = nbdict.get('N', {}).get('U', 0)
            Wn = nbdict.get('N', {}).get('W', 0)
            Pn = nbdict.get('N', {}).get('P', 0)
            Une = nbdict.get('NE', {}).get('U', 0)
            Wne = nbdict.get('NE', {}).get('W', 0)
            Pne = nbdict.get('NE', {}).get('P', 0)
            Unne = nbdict.get('NNE', {}).get('U', 0)
            Wnne = nbdict.get('NNE', {}).get('W', 0)
            Pnne = nbdict.get('NNE', {}).get('P', 0)
            Uene = nbdict.get('ENE', {}).get('U', 0)
            Wene = nbdict.get('ENE', {}).get('W', 0)
            Pene = nbdict.get('ENE', {}).get('P', 0)
            Ue = nbdict.get('E', {}).get('U', 0)
            We = nbdict.get('E', {}).get('W', 0)
            Pe = nbdict.get('E', {}).get('P', 0)
            Use = nbdict.get('SE', {}).get('U', 0)
            Wse = nbdict.get('SE', {}).get('W', 0)
            Pse = nbdict.get('SE', {}).get('P', 0)
            Uese = nbdict.get('ESE', {}).get('U', 0)
            Wese = nbdict.get('ESE', {}).get('W', 0)
            Pese = nbdict.get('ESE', {}).get('P', 0)
            Usse = nbdict.get('SSE', {}).get('U', 0)
            Wsse = nbdict.get('SSE', {}).get('W', 0)
            Psse = nbdict.get('SSE', {}).get('P', 0)
            Us = nbdict.get('S', {}).get('U', 0)
            Ws = nbdict.get('S', {}).get('W', 0)
            Ps = nbdict.get('S', {}).get('P', 0)
            Usw = nbdict.get('SW', {}).get('U', 0)
            Wsw = nbdict.get('SW', {}).get('W', 0)
            Psw = nbdict.get('SW', {}).get('P', 0)
            Ussw = nbdict.get('SSW', {}).get('U', 0)
            Wssw = nbdict.get('SSW', {}).get('W', 0)
            Pssw = nbdict.get('SSW', {}).get('P', 0)
            Uwsw = nbdict.get('WSW', {}).get('U', 0)
            Wwsw = nbdict.get('WSW', {}).get('W', 0)
            Pwsw = nbdict.get('WSW', {}).get('P', 0)
            Uw = nbdict.get('W', {}).get('U', 0)
            Ww = nbdict.get('W', {}).get('W', 0)
            Pw = nbdict.get('W', {}).get('P', 0)
            Unw = nbdict.get('NW', {}).get('U', 0)
            Wnw = nbdict.get('NW', {}).get('W', 0)
            Pnw = nbdict.get('NW', {}).get('P', 0)
            Uwnw = nbdict.get('WNW', {}).get('U', 0)
            Wwnw = nbdict.get('WNW', {}).get('W', 0)
            Pwnw = nbdict.get('WNW', {}).get('P', 0)
            Unnw = nbdict.get('NNW', {}).get('U', 0)
            Wnnw = nbdict.get('NNW', {}).get('W', 0)
            Pnnw = nbdict.get('NNW', {}).get('P', 0)
            
            # at boundaries we have to link to virtual cells before configuration & solve
            if yB == 0:
                Ws = 0  # Ws = -Wp
            elif yT == SURFACE:
                Wn = 0
            if (xL == 0) and (yT <= SURFACE):
                Ww = 0
                Uw = Up
                Wnw = 0
                Unw = Un
            if (xR == 1) and (yT <= SURFACE):
                We = 0
                Ue = Up

            K = nbdict.keys()
            if 'W' in K:
                dxw = xC - nbdict['W']['xC']
                duwdx = (Up - Uw) / dxw
                dwwdx = (Wp - Ww) / dxw
                UMw = Uw if Uw > 0 else Up
                WMw = (Ww + Wp) / 2.0  # not sure, how about bordering a bigger cell?
            
            if 'E' in K:
                dxe = nbdict['E']['xC'] - xC
                duedx = (Ue - Up) / dxe
                dwedx = (We - Wp) / dxe
                UMe = Ue if Ue < 0 else Up
                WMe = (Wp + We) / 2.0
            
            if 'WNW' in K and 'WSW' in K:
                dxw = xC - nbdict['WNW']['xC']
                duwdx = 0.5 * ( (Up - Uwnw)  + (Up - Uwsw) ) / dxw
                dwwdx = 0.5 * ( (Wp - Wwnw)  + (Wp - Wwsw) ) / dxw
                UMw = (Uwnw + Uwsw) / 2.0 if (Uwnw + Uwsw) > 0 else Up
                WMw = (Wwnw + Wwsw + 2 * Wp) / 4.0
            
            if 'ENE' in K and 'ESE' in K:
                dxe = nbdict['ENE']['xC'] - xC
                duedx = (0.5 * (Uene + Uese) - Up) / dxe
                dwedx = (0.5 * (Wene + Wese) - Wp) / dxe
                UMe = (Uene + Uese) / 2.0 if (Uene + Uese) < 0 else Up
                WMe = (Wene + Wese + 2 * Wp) / 4.0
            
            if 'S' in K:
                dys = yC - nbdict['S']['yC']
                dusdy = (Up - Us) / dys
                dwsdy = (Wp - Ws) / dys
                UMs = (Us + Up) / 2.0
                WMs = Ws if Ws > 0 else Wp
            
            if 'N' in K:
                dyn = nbdict['N']['yC'] - yC
                dundy = (Un - Up) / dyn
                dwndy = (Wn - Wp) / dyn
                UMn = (Up + Un) / 2.0
                WMn = Wn if Wn < 0 else Wp
            
            if 'SSW' in K and 'SSE' in K:
                dys = yC - nbdict['SSW']['yC']
                dusdy = 0.5 * ( (Up - Ussw)  + (Up - Usse) ) / dys
                dwsdy = 0.5 * ( (Wp - Wssw)  + (Wp - Wsse) ) / dys
                UMs = (Ussw + Usse + 2 * Up) / 4.0
                WMs = (Wssw + Wsse) / 2.0 if (Wssw + Wsse) > 0 else Wp
            
            if 'NNE' in K and 'NNW' in K:
                dyn = nbdict['NNE']['yC'] - yC
                dundy = 0.5 * ( (Unne - Up)  + (Unnw - Up) ) / dyn
                dwndy = 0.5 * ( (Wnne - Wp)  + (Wnnw - Wp) ) / dyn
                UMn = (Unnw + Unne + 2 * Up) / 4.0
                WMn = (Wnnw + Wnne) / 2.0 if (Wnnw + Wnne) < 0 else Wp
            
            if Up > 0:
                ududx = UMw * duwdx
                udwdx = UMw * dwwdx
            elif Up < 0:
                ududx = UMe * duedx
                udwdx = UMe * dwedx
            else:
                ududx = (UMw * duwdx + UMe * duedx) / 2.0
                udwdx = (UMw * dwwdx + UMe * dwedx) / 2.0
            
            if Wp > 0:
                wdudy = WMs * dusdy
                wdwdy = WMs * dwsdy
            elif Wp < 0:
                wdudy = WMn * dundy
                wdwdy = WMn * dwndy
            else:
                wdudy = (WMs * dusdy + WMn * dundy) / 2.0
                wdwdy = (WMs * dwsdy + WMn * dwndy) / 2.0
            
            # viscosity, finite difference approach
            # nu_z = 0.41 * yC * (Wo*0.1) * (1 - yC / SURFACE)
            # nuh = nu_z
            # nuv = nu_z
            # visu = nuh * 2 * (duedx - duwdx) / (dxw + dxe) + nuv * 2 * (dundy - dusdy) / (dys + dyn)
            # visw = nuh * 2 * (dwedx - dwwdx) / (dxw + dxe) + nuv * 2 * (dwndy - dwsdy) / (dys + dyn)
            
            # Unew = Up - delta_t * (ududx + wdudy - visu)
            # Wnew = Wp - delta_t * (udwdx + wdwdy - visw)

            # if visu * visw != 0 and cell['level'] == MAX_DEPTH:
            #     print(' ', (delta_t*visu, delta_t*visw), end='')

            # alternatively - viscosity from diffusion approach -- now use this with proper Une, Use, Unw, Usw
            UList = ( Un, Unne, Une, Uene, Ue, Uese, Use, Usse, 
                    Us, Ussw, Usw, Uwsw, Uw, Uwnw, Unw, Unnw, Up )
            WList = ( Wn, Wnne, Wne, Wene, We, Wese, Wse, Wsse, 
                    Ws, Wssw, Wsw, Wwsw, Ww, Wwnw, Wnw, Wnnw, Wp )
            
            conf, ratio, coef = configurator(meshU, cell)
            
            assert len(coef) == 17, "Error configurator with NR coeff != 17"
            Udif = 0
            Wdif = 0
            for i in range(17):
                Udif += coef[i] * UList[i]
                Wdif += coef[i] * WList[i]
            
            nu_z = 0.41 * yC * (Wo*0.1) * (1 - yC / SURFACE)
            Udif *= ratio * nu_z * delta_t / (h * h)
            Wdif *= ratio * nu_z * delta_t / (h * h)

            Unew = Up - delta_t * (ududx + wdudy) + Udif

            conc = extractC(meshC.rootCell, (xC, yC))
            gprime = (9.81/UNIT_LEN) * dens_deficit * (conc/100)  # buoyancy
            Wnew = Wp - delta_t * (udwdx + wdwdy - gprime) + Wdif
            
            # if Udif * Wdif != 0 and cell['level'] == MAX_DEPTH:
            #     print(' ', (Udif, Wdif), end='')
            
            # MISSING THE PRESSURE CORRECTION!
            Upred = Unew
            Wpred = Wnew

            cell['P'] = 1
            Ucorr = Upred
            Wcorr = Wpred            
            
            cell['Unew'] = Ucorr
            cell['Wnew'] = Wcorr
        
        graU = numpy.sqrt(ududx*ududx + wdwdy*wdwdy)
        
        tmpList = [ ]
        if (cell['level'] < MAX_DEPTH) and (
            (graU > THETA_U) and (cell['yT'] <= SURFACE - 0.04) or 
            (graU > THETA_U2) and (SURFACE - 0.04 < cell['yT'] <= SURFACE) 
        ):
            # sort ascending based on the level
            search_DFS( meshU, cell, MAX_DEPTH )
            for anycell in tmpList:
                if not (anycell in refineListU):
                    refineListU.append( anycell )

    # print()
    # Update the velocity here (when all cells are calculated, 
    # but before refinement), so that the velocities are
    # retained and passed to the subcells.
    for cell in meshU.leafList:
        cell['U'] = cell['Unew']    # update
        cell['W'] = cell['Wnew']    # update

    refineListU = sorted(refineListU, key=lambda k: k['level'])

    while refineListU != []:
        c = refineListU.pop(0)
        meshU.refine(c)
    
    # Concentration
    for cell in meshC.leafList:
        h = cell['side']
        nbdict = cell['neighbors']
        Cp = cell['C']
        xL = cell['xL']
        xC = cell['xC']
        yC = cell['yC']
        xR = cell['xR']
        yB = cell['yB']
        yT = cell['yT']
        graC = 0
        dCdx = dCdy = 0
        fluxw = fluxe = fluxn = fluxs = 0
        
        if (xL == 0) and (yBport <= yC <= yTport):     # old (xL == 0.5) at the mouth of outfall
            _dir = ' *'
            conf = ' (Outfall) '
            ratio = 0
            coef = (0,)
            Cadv = 0
            Cdif = 0
            Cp = Co
            
        elif (yB >= SURFACE):     # in the air. Zero concentration
            _dir = ' o'
            conf = 'air'
            ratio = 0
            coef = (0,)
            Cadv = 0
            Cdif = 0
            Cp = 0
        
        else:
            Cn = nbdict.get('N', {}).get('C', 0)
            Cne = nbdict.get('NE', {}).get('C', 0)
            Cnne = nbdict.get('NNE', {}).get('C', 0)
            Cene = nbdict.get('ENE', {}).get('C', 0)
            Ce = nbdict.get('E', {}).get('C', 0)
            Cse = nbdict.get('SE', {}).get('C', 0)
            Cese = nbdict.get('ESE', {}).get('C', 0)
            Csse = nbdict.get('SSE', {}).get('C', 0)
            Cs = nbdict.get('S', {}).get('C', 0)
            Csw = nbdict.get('SW', {}).get('C', 0)
            Cssw = nbdict.get('SSW', {}).get('C', 0)
            Cwsw = nbdict.get('WSW', {}).get('C', 0)
            Cw = nbdict.get('W', {}).get('C', 0)
            Cnw = nbdict.get('NW', {}).get('C', 0)
            Cwnw = nbdict.get('WNW', {}).get('C', 0)
            Cnnw = nbdict.get('NNW', {}).get('C', 0)
            
            # at boundaries we have to link to virtual cells before configuration & solve
            if yB == 0:
                Cs = Cp
                Cse = Ce
                Csw = Cw
            if (xL == 0) and (yT <= SURFACE):
                Cw = 0
                Cnw = 0
            if (xR == 1) and (yT <= SURFACE):
                Ce = 0
                Cne = 0
                
            # Advection ->
            
            # get the velocity at the center points of N/S/E/W faces
            Ucenn, Wcenn = extractUW(meshU.rootCell, (xC, yT))
            Ucens, Wcens = extractUW(meshU.rootCell, (xC, yB))
            Ucene, Wcene = extractUW(meshU.rootCell, (xR, yC))
            Ucenw, Wcenw = extractUW(meshU.rootCell, (xL, yC))
                
            if Ucenw == 0:
                fluxw = 0
            elif Ucenw < 0:
                fluxw = Cp * Ucenw
            elif Ucenw > 0:
                if 'WNW' in nbdict:
                    fluxw = 0.25 * (Cwnw + Cwsw) * Ucenw
                else:
                    fluxw = Cw * Ucenw
                
            if Ucene == 0:
                fluxe = 0
            elif Ucene > 0:
                fluxe = Cp * Ucene
            elif Ucene < 0:
                if 'ENE' in nbdict:
                    fluxe = 0.25 * (Cene + Cese) * Ucene
                else:
                    fluxe = Ce * Ucenw
            
            if Wcens == 0:
                fluxs = 0
            elif Wcens < 0:
                fluxs = Cp * Wcens
            elif Wcens > 0:
                if 'SSW' in nbdict:
                    fluxs = 0.25 * (Cssw + Csse) * Wcens
                else:
                    fluxs = Cs * Wcens
            
            if Wcenn == 0:
                fluxn = 0
            elif Wcenn > 0:
                fluxn = Cp * Wcenn
            elif Wcenn < 0:
                if 'NNW' in nbdict:
                    fluxn = 0.25 * (Cnnw + Cnne) * Wcenn
                else:
                    fluxn = Cn * Wcenn
            
            Cadv = -(fluxe - fluxw + fluxn - fluxs) * delta_t / h
            # dealing with diffusion ...
            CList = ( Cn, Cnne, Cne, Cene, Ce, Cese, Cse, Csse, 
                    Cs, Cssw, Csw, Cwsw, Cw, Cwnw, Cnw, Cnnw, Cp )
            
            # Solving for diffusion eq
            conf, ratio, coef = configurator(meshC, cell)
            
            assert len(coef) == 17, "Error configurator with NR coeff != 17"
            Cdif = 0
            for i in range(17):
                Cdif += coef[i] * CList[i]
            
            eps = Sc * nu_z  # wrong
            Cdif *= ratio * eps * delta_t / (h * h)
        
        Cpnew = Cp + Cadv + Cdif
        if Cpnew > Co:
            Cpnew = Co
        
        cell['Cnew'] = Cpnew
        cell['conf'] = conf
        cell['ratio'] = ratio
        cell['coef'] = coef
        
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
            print(cell['id'], cell['neighbors'].keys())
        
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
            print(cell['id'])
            
        graC = numpy.sqrt(dCdx*dCdx + dCdy*dCdy)
        
        tmpList = [ ]
        if (cell['level'] < MAX_DEPTH) and (
            (graC > THETA_C) and (cell['yT'] <= SURFACE - 0.04) or 
            (graC > THETA_C2) and (SURFACE - 0.04 < cell['yT'] <= SURFACE) 
        ):
            # sort ascending based on the level
            search_DFS( meshC, cell, MAX_DEPTH )
            for anycell in tmpList:
                if not (anycell in refineList):
                    refineList.append( anycell )

    # Updating new values - move before refining
    Cmax = 0
    for cell in meshC.leafList:
        cell['C'] = cell['Cnew']
        if cell['C'] > Cmax:
            Cmax = cell['C']
        meshC.assignDiagNeighbors(cell)         # update neighborhood of newer cells
    
    refineList = sorted(refineList, key=lambda k: k['level'])

    while refineList != []:
        c = refineList.pop(0)
        meshC.refine(c)
    
    
print(f'({timeCounter+1})\tt = {t+delta_t:.1f} \tElapsed: {time.time() - starttime:.0f} s', end='')
print('\tMesh C:', len(meshC.leafList), '\tMeshU:', len(meshU.leafList))


# Post-processing: draw the meshes
for cell in meshC.leafList:
    Cpnew = cell['Cnew']
    color = float(Cpnew / Co)  # log scale implemented in color map
    color = 0 if color < 0 else color
    color = 1 if color > 1 else color
    color = numpy.NaN if cell['yC'] > SURFACE else color
    h = cell['side']
    meshC.patchesList.append( [(cell['xL'], cell['yB']), h, h, color, '0'] )
    val = str(int(round(100 * Cpnew / Co)))
    if val != "0":
        meshC.textList.append([cell['xC'], cell['yC'], val])
    
for cell in meshU.leafList:
    Unew = cell['Unew']
    Wnew = cell['Wnew']
    vel = numpy.sqrt(Unew * Unew + Wnew * Wnew)
    val = str(int(round(100 * vel / Wo)))
    if val != "0":
        meshU.textList.append([cell['xC'], cell['yC'], val])
    
    color = 1.0 - float(vel / Wo)
    color = 0 if color < 0 else color
    color = 1 if color > 1 else color
    h = cell['side']
    meshU.patchesList.append( [(cell['xL'], cell['yB']), h, h, color, '0'] )
    
# Visualization
w = width_outlet
plot_extent=(0, 31*w, yo-8*w, yo+8*w)
plot_xticks=[0, 6*w, 12*w, 18*w, 24*w]
plot_yticks=[yo-6*w, yo-3*w, yo, yo+3*w, yo+6*w]

st1 = f'Concentration at t = {t:.2f}'
st2 = f'Velocity at t = {t:.2f}'
meshC.draw(st1, grid=True, num=False, arrow=False, patches=True, quiver=False, 
            xo_port=xo_port, yo_port=yo_port,
            extent=plot_extent, xticks=plot_xticks, yticks=plot_yticks
        )
# meshC.draw(st1, grid=True, num=True, arrow=False, patches=False, quiver=False, extent=(0.48, 0.63, 0, 0.15))
# meshU.draw(st2, grid=True, num=True, arrow=False, patches=False, quiver=False, extent=(0.48, 0.63, 0, 0.15))
if scenario in ['C1', 'D1', 'E2', 'G2']:
    qvscale = 5E-4
else:
    qvscale = 1E-4

meshU.draw(st2, grid=True, num=False, arrow=False, patches=False, quiver=True, 
            qvscale=qvscale, color='blue', alpha=0.4,
            xo_port=xo_port, yo_port=yo_port,
            extent=plot_extent, xticks=plot_xticks, yticks=plot_yticks
            )
