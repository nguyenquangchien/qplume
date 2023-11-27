# 
# Try shallow water by filling 9999 on the top of initial matrix.
#
'''Consider only 2-DV hydrodynamic modelling'''
import quadtree4 as quadtree4
import pylab, numpy
from math import sin, cos
import time

import sys
sys.setrecursionlimit(1500)

import psyco
psyco.full()

UNIT_LEN = 100.    # Length of unit square = 100.0 m
MAX_DEPTH = 8
THETA_C = 1000
THETA_U = 1E-5

grav = 9.81 / UNIT_LEN
N = 8
SURFACE = 0.5

tmpList = [ ]

pylab.close('all')

def search_DFS(mesh, cell, maxDepth=None):
    for key in cell['neighbors']:
        if key in ['N', 'S', 'E', 'W']:
            nb = cell['neighbors'][key]
            if ( type(nb['id']) is str ):     # implying a real, non-boundary cell.
                if ( (cell['level'] == nb['level'] + 1) and (nb['level'] < maxDepth) ):
                    search_DFS(mesh, nb, maxDepth)
    
    tmpList.append( cell )
    return tmpList

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


# Initialization
side = 2**N

# UW matrix
try:
    Ua = input('Ambient velocity (m/s) [0] = ') / UNIT_LEN
except:     # SyntaxError / NameError : Use default value
    Ua = 0 / UNIT_LEN

matUW = numpy.zeros((side, side), dtype=float) + Ua

try:
    Wo = input('Jet velocity (m/s) [0.5] = ') / UNIT_LEN
except:     # SyntaxError / NameError : Use default value
    Wo = 0.5 / UNIT_LEN

matUW[side-1][side/2-1] = matUW[side-1][side/2] = Wo

try:
    alpha = input('Incline angle jet (degrees) [90] = ') * numpy.pi / 180
except:     # SyntaxError / NameError : Use default value
    alpha = 90 * numpy.pi / 180

try:
    sdo = input('Specific relative density of effluent [0.00255] = ')
except:     # SyntaxError / NameError : Use default value
    sdo = 0.00255

meshUW = quadtree4.QTree(matrix=matUW,  propnames=['U','Unew','W','Wnew'])
meshUW.split_BFS(meshUW.rootCell, threshold=0, maxDepth=MAX_DEPTH)

try:
    eps = input('Eddy diffusivity (m^2/s) [0.01] = ')  / (UNIT_LEN**2)
except:     # SyntaxError / NameError : Use default value
    eps = 0.01 / (UNIT_LEN*UNIT_LEN)

try:
    nuh = input('Horizontal eddy viscosity (m^2/s) [0.01] = ')  / (UNIT_LEN**2)
except:     # SyntaxError / NameError : Use default value
    nuh = 0.01 / (UNIT_LEN*UNIT_LEN)

try:
    nuv = input('Vertical eddy viscosity (m^2/s) [0.01] = ')  / (UNIT_LEN**2)
except:     # SyntaxError / NameError : Use default value
    nuv = 0.01 / (UNIT_LEN*UNIT_LEN)

# Calculation
try:
    delta_t = input('Time step (s) [0.005] = ')
except:     # SyntaxError / NameError : Use default value
    delta_t = 0.005

try:
    nt = input('# time steps [1000] = ')
except:     # SyntaxError / NameError : Use default value
    nt = 1000

starttime = time.clock()

for timeCounter in range(nt):
    t =  timeCounter * delta_t
    if (timeCounter % 500 == 0):
        print
        print 't = ', t, '\tEllapsed time: ', time.clock() - starttime, ' seconds'
    
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
        
        elif (yB >= SURFACE):     # in the air. Zero concentration
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
                Uc = 0
                Ws = -Wc
            if yT == SURFACE:
                Un = Uc
                Wn = -Wc
            if (xL == 0) and (yT <= SURFACE):
                Ww = 0
                Uw = Uc = Ua
                Wnw = 0
                Unw = Un
            if (xR == 1) and (yT <= SURFACE):
                We = 0
                Ue = Uc = Ua

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
            visu = nuh * 2 * (duedx - duwdx) / (dxw + dxe) + nuv * 2 * (dundy - dusdy) / (dys + dyn)
            visw = nuh * 2 * (dwedx - dwwdx) / (dxw + dxe) + nuv * 2* (dwndy - dwsdy) / (dys + dyn)
            
            # buoyancy (reduced gravity)
            gprime = 0
            
            # momentum equations
            Unew = Uc - delta_t * (ududx + wdudy - visu)
            Wnew = Wc - delta_t * (udwdx + wdwdy - visw + gprime)
            
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
    
    for cell in meshUW.leafList:    # update
        cell['U'] = cell['Unew']
        cell['W'] = cell['Wnew']
    
print 'Ellapsed time: ', time.clock() - starttime, ' seconds'

# Post-processing: draw the meshes
for cell in meshUW.leafList:
    Unew = cell['Unew']
    Wnew = cell['Wnew']
    vel = (Unew * Unew + Wnew * Wnew) ** 0.5
    val = str(int(round(100 * Wnew / Ua)))  # or
    # val = str(int(round(10000 * Unew)))   # or
    # val = str(int(round(100 * Unew / Wo)))
    if val != "0":
        meshUW.textList.append([cell['xC'], cell['yC'], val])
    
    # color = 1.0 - float(vel / Wo)
    # color = 0 if color < 0 else color
    # color = 1 if color > 1 else color
    # h = cell['side']
    # meshUW.patchesList.append( [(cell['xL'], cell['yB']), h, h, str(color), '0'] )
    # meshUW.textList.append([cell['xC'], cell['yC'], str(vel)])

st2 = 'Velocity at t = ' + str(t)
meshUW.draw(st2, grid=True, num=True, arrow=False, patches=False, quiver=False, qvscale=0.01)

pylab.show()
