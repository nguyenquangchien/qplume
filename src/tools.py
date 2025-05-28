""" tools module for the mesh generation and manipulation."""

# tmpList = [ ]  # initialised here, to be used in `search_DFS`


def search_DFS(mesh, cell, tmpList, maxDepth=None):
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
