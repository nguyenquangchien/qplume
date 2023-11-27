import diff1 as diff
# import sys
# sys.setrecursionlimit(2000)

import quadtree4 as quadtree4
reload(quadtree4)
import pylab, numpy, matplotlib
import time
import os
from math import sin, cos
'''
test_side: The jet is located in the mid water upstream.

Square cell for the entire domain. No cell for air.

Since after refinement, cell config. changes so we must find a flexible
way to deal with cell differential scheme, rather than relying on their
positions.

A minor ammendment: U, W, for NE, SE, NW, SW
'''
MAX_DEPTH = 8
# THETA_C = input('Theta C = ')
# THETA_U = input('Theta U = ')   # numpy.inf
THETA_C = 500
THETA_U = 1E-4
SCALE = 0.01
# nu = 5E-5 # s^-1 since the scale of model is 100. Eddy viscosity ~ O(50 m^2/s)
nu = 0.001 * SCALE * SCALE
eps = 0.01 * SCALE * SCALE

pylab.close('all')

N = 9

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


# Here we define separate extraction methods since they operate on different meshes.
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

def getColor(v, vmin, vmax):
    if v < vmin:
        v = vmin
        return (0, 0, 1)
    if v > vmax:
        v = vmax
        return (1, 0, 0)
    # otherwise, v is within the range (vmin, vmax)
    dv = 1.0 * (vmax - vmin)
    if v < vmin + 0.25 * dv:
        return (0, 4 * (v - vmin) / dv, 1)
    if v < vmin + 0.5 * dv:
        return (0, 1, 1 + 4 * (vmin + 0.25 * dv - v) / dv)
    if v < vmin + 0.75 * dv:
        return (4 * (v - vmin - 0.5 * dv) / dv, 1, 0)
    if v < vmax:
        return (1, 1 + 4 * (vmin + 0.75 * dv - v) / dv, 0)


# MAIN PROGRAM #

scen = raw_input("Scenario: ")
# Make directory to save the output.
if os.access(scen, os.F_OK) == 0:
    os.mkdir(scen)

COLOR_BW = (raw_input('B/W output? [y/n] ') == 'y')

# Initialization
side = 2**N
Ua = 0.000  # ambient current
CENTRE_OUTFALL = 0.25   # z_centre of outfall

# Upstream position of the jet
yUport = CENTRE_OUTFALL + 1.0 / side
yBport = CENTRE_OUTFALL - 1.0 / side

matC = numpy.zeros((side, side), dtype=float)
# Note: vertical ordering is different between the matrix and the domain
matC[int(round(1 - CENTRE_OUTFALL * side)) - 1][0] = \
    matC[int(round(1 - CENTRE_OUTFALL * side)) - 2][0] = Co = 100

matUW = numpy.zeros((side, side), dtype=float) + Ua
matUW[int(round(1 - CENTRE_OUTFALL * side)) - 1][0] = \
    matUW[int(round(1 - CENTRE_OUTFALL * side)) - 2][0] = Wo = input('Wo = ') 

meshC = quadtree4.QTree(matrix=matC, propnames=['C','Cnew'])
meshC.split_BFS(meshC.rootCell, threshold=1, maxDepth=MAX_DEPTH)

for cel in meshC.leafList:
    meshC.assignDiagNeighbors(cel)

meshUW = quadtree4.QTree(matrix=matUW,  propnames=['U','Unew','W','Wnew'])
meshUW.split_BFS(meshUW.rootCell, threshold=0, maxDepth=MAX_DEPTH)

for cel in meshUW.leafList:
    meshUW.assignDiagNeighbors(cel)

alpha = input('Incline angle jet = ') * numpy.pi / 180

# Calculation
delta_t = input('delta_t = ')

nt = input('nt = ')

starttime = time.clock()

for timeCounter in range(nt):
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
        if (xL == 0) and (yBport <= yC <= yUport):     # at the mouth of outfall
            _dir = "*"
            conf = ' (Outfall) '
            ratio = 0
            coef = (0,)
            cell['U'] = cell['Unew'] = Wo * cos(alpha)
            cell['W'] = cell['Wnew'] = Wo * sin(alpha)
        
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
                Une = nbdict['NE']['U']
                Wne = nbdict['NE']['W']
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
                Use = nbdict['SE']['U']
                Wse = nbdict['SE']['W']
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
                Usw = nbdict['SW']['U']
                Wsw = nbdict['SW']['W']
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
                Unw = nbdict['NW']['U']
                Wnw = nbdict['NW']['W']
            except: pass
            try: 
                Unnw = nbdict['NNW']['U']
                Wnnw = nbdict['NNW']['W']
            except: pass
            
            # at boundaries we have to link to virtual cells before configuration & solve
            
            # bottom & surface: reflective boundaries
            if yB == 0:
                Us = Uc
                Ws = -Wc
            if yT == 1:
                Un = Uc
                Wn = -Wc
            
            # lateral boundaries: Neumann: zero gradient velocity
            if (xL == 0):
                Ww = 0
                Uw = Uc
                Wnw = 0
                Unw = Un
            if (xR == 1):
                We = 0
                Ue = Uc
                Wne = 0
                Une = Uc
            
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
            
            # method 1, discretization
            # visu = nu * 2 * ((duedx - duwdx) / (dxw + dxe) + (dundy - dusdy) / (dys + dyn))
            # visw = nu * 2 * ((dwedx - dwwdx) / (dxw + dxe) + (dwndy - dwsdy) / (dys + dyn))
            # Unew = Uc - delta_t * (ududx + wdudy - visu)
            # Wnew = Wc - delta_t * (udwdx + wdwdy - visw)
            
            # method 2, use result Borthwick et al. (2000)
            UList = ( Un, Unne, Une, Uene, Ue, Uese, Use, Usse, 
                    Us, Ussw, Usw, Uwsw, Uw, Uwnw, Unw, Unnw, Uc )
            WList = ( Wn, Wnne, Wne, Wene, We, Wese, Wse, Wsse, 
                    Ws, Wssw, Wsw, Wwsw, Ww, Wwnw, Wnw, Wnnw, Wc )
            
            # Solving for diffusion eq
            conf, ratio, coef = diff.configurator(meshUW, cell)
            
            assert( len(coef) == 17 )
            Udif = 0
            Wdif = 0
            for i in range(17):
                Udif += coef[i] * UList[i]
                Wdif += coef[i] * WList[i]
            
            Udif *= ratio * nu * delta_t / (h * h)
            Wdif *= ratio * nu * delta_t / (h * h)
            
            Unew = Uc - delta_t * (ududx + wdudy) + Udif
            Wnew = Wc - delta_t * (udwdx + wdwdy) + Wdif
            cell['Unew'] = Unew
            cell['Wnew'] = Wnew
            
        graU = numpy.sqrt(ududx**2 + wdwdy**2)
        
        tmpList = [ ]
        if (graU > THETA_U) and (cell['yT'] <= 0.5) and (cell['level'] < MAX_DEPTH): 
            # sort ascending based on the level
            search_DFS( meshUW, cell, MAX_DEPTH )
            for anycell in tmpList:
                try:
                    if not (anycell in refineListUW):
                        refineListUW.append( anycell )
                except:
                    print "Error max recursion (meshUW) at #t, #C, #UW ", timeCounter, len(meshC.leafList), len(meshUW.leafList)
        

    refineListUW = sorted(refineListUW, key=lambda k: k['level'])

    while refineListUW:
        c = refineListUW.pop(0)
        meshUW.refine(c)
    
    ####################
    # For concentration#
    ####################
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
        
        if (xL == 0) and (yBport <= yC <= yUport):     # at the mouth of outfall
            _dir = ' *'
            conf = ' (Outfall) '
            ratio = 0
            coef = (0,)
            Cadv = 0
            Cdif = 0
            cell['C'] = cell['Cnew'] = Co
        
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
            if xL == 0:
                Cw = 0
                Cnw = 0
            if xR == 1:
                Ce = 0
                Cne = 0
            if yT == 1:
                Cn = Cc
                Cne = Ce
                Cnw = Cw
            
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
            # conf, ratio, coef = diff.configurator(meshC, cell)
            
            # assert( len(coef) == 17 )
            # Cdif = 0
            # for i in range(17):
                # Cdif += coef[i] * CList[i]
            
            # Cdif *= ratio * eps * delta_t / (h * h)
        
        # Ccnew = Cc + Cadv + Cdif
        Ccnew = Cc + Cadv
        
        cell['Cnew'] = Ccnew
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
        
        tmpList = [ ]
        if (graC > THETA_C) and (cell['yT'] <= 0.5) and (cell['level'] < MAX_DEPTH): 
            # sort ascending based on the level
            search_DFS( meshC, cell, MAX_DEPTH )
            for anycell in tmpList:
                try:
                    if not (anycell in refineList):
                        refineList.append( anycell )
                except:
                    print "Error max recursion (meshC) at #t, #C, #UW ", timeCounter, len(meshC.leafList), len(meshUW.leafList)

    refineList = sorted(refineList, key=lambda k: k['level'])

    while refineList:
        c = refineList.pop(0)
        meshC.refine(c)
    
    # Updating new values
    Cmax = 0
    for cell in meshC.leafList:
        cell['C'] = cell['Cnew']
        if cell['Cnew'] > Cmax:
            Cmax = cell['Cnew']
        meshC.assignDiagNeighbors(cell)         # update neighborhood of newer cells
    
    Umax = 0
    for cell in meshUW.leafList:    # update
        cell['U'] = cell['Unew']
        cell['W'] = cell['Wnew']
        if  cell['Unew'] > Umax:
            Umax = cell['Unew']
        meshUW.assignDiagNeighbors(cell)
    
    if timeCounter % 30 == 0:
        print "Cmax = ", int(Cmax), 
        print "\tUmax = ", Umax
    
    # Text output ...
    t =  timeCounter * delta_t
    if (timeCounter % 300 == 0):
        print
        print 't = ', t
        # Print the velocity near the mouth
        dx = 1.0 / (2**N)
        dz = 1.0 / (2**N)
        for i in range(20):
            print
            for j in range(6):
                x = (i + 0.5) * dx
                z = CENTRE_OUTFALL + (j - 2.5) * dz
                c = extractC(meshC.rootCell, (x, z))
                u, w = extractUW(meshUW.rootCell, (x, z))
                print "(%d, %.4f) " %(int(c),u) ,
    
    # Graphical plot saved to file
    if (timeCounter % 600 == 0):
        meshC.patchesList = []      # reset the patches for newer time step
        for cell in meshC.leafList:
            h = cell['side']
            Ccnew = cell['Cnew']
            # color = 1.0 - numpy.log(abs(Ccnew) + 1.0) / numpy.log(Co + 1)      # log scale
            color = 1.0 - float(Ccnew / Cmax)
            if COLOR_BW:
                # color = 0 if color < 0 else color
                # color = 1 if color > 1 else color
                meshC.patchesList.append( [(cell['xL'], cell['yB']), h, h, str(color), '0'] )
            else:
                color = getColor(color, 0, 1)
                meshC.patchesList.append( [(cell['xL'], cell['yB']), h, h, color, '0'] )

        for cell in meshUW.leafList:
            Unew = cell['Unew']
            Wnew = cell['Wnew']
            vel = (Unew * Unew + Wnew * Wnew) ** 0.5
            
            color = 1.0 - float(Unew / Umax)
            # color = 0 if color < 0 else color
            # color = 1 if color > 1 else color
            h = cell['side']
            meshUW.patchesList.append( [(cell['xL'], cell['yB']), h, h, str(color), '0'] )
            # meshUW.textList.append([cell['xC'], cell['yC'], str(vel)])
        
        fnc = scen + '/C' + str(t) + '.png'
        fnuw = scen + '/V' + str(t) + '.png'
        meshC.draw(t, grid=False, num=False, arrow=False, patches = True, quiver=False, 
                outfile = fnc, lim = (0, 0.5, 0, 0.5))

        meshUW.draw(t, grid=False, num=False, arrow=False, patches=True, quiver=False,
                outfile = fnuw, lim = (0, 0.5, 0, 0.5))
    
    # End print out result

# Finally ...
print 'Ellapsed time: ', time.clock() - starttime, ' seconds'
