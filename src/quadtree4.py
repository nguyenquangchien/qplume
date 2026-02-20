# file:  quadtree.py

"""
Implement a list-type tree with BFS algorithm.
"""

import numpy
import pickle
import cmocean
import matplotlib.patches
import matplotlib.pyplot as plt

_NO_BND = 0
_NORTH = 1
_SOUTH = 2
_EAST = 4
_WEST = 8

class QTree(object):
    def __init__(self, matrix=None, propnames=None):
        self.nLeaves = 0
        self.patchesList = []
        self.lineCrossList = []
        self.markXList = []
        self.markYList = []
        self.arrowList = []
        self.textList = []   # plotting numbers
        self.valList = []    # storing values of variable
        self.leafList = []      # main data
        self.cellQueue = []     # main data
        self.flux = {}      # fluxes between cells
        self.propnames = propnames
        self.matrix = matrix
        shape = matrix.shape
        N = numpy.log2(shape[0])
        assert (shape[0] == shape[1]) or (N != numpy.floor(N)), \
            "Matrix should be square with size 2^p "
            
        self.rootCell = {"pos" : "root", 
                "id" : "root",
                "parent":None, 
                "neighbors": {"N": _NORTH, "S": _SOUTH, "E": _EAST, "W": _WEST},
                "bnd" : _NORTH | _SOUTH | _EAST | _WEST,
                "level" : 0,
                "side" : 1,
                "matrix" : self.matrix,
                "xL" : 0 ,
                "xC" : 0.5 ,
                "xR" : 1 ,
                "yB" : 0 ,
                "yC" : 0.5,
                "yT" : 1 ,
                }
        
    def split_BFS(self, aCell, threshold, maxDepth):
        
        self.cellQueue.append(aCell)
        
        # start traversing
        while self.cellQueue != []:
            cell = self.cellQueue.pop(0)
            
            xL = cell["xL"]
            yB = cell["yB"]
            xR = cell["xR"]
            yT = cell["yT"]
            xC = cell["xC"]
            yC = cell["yC"]
            
            # check if further division should be made
            matrix = cell["matrix"]
            maxval = numpy.max(matrix)
            minval = numpy.min(matrix)
            avgval = numpy.average(matrix)
            
            for propname in self.propnames:
                cell[propname] = avgval if (propname[0] == "U" or propname[0] == "C") else 0
            
            leafCriteria = (numpy.size(matrix) == 1) or \
                    (maxval - minval <= threshold) or (cell["level"] == maxDepth)
            if leafCriteria:
                cell["isLeaf"] = True
                self.nLeaves += 1
                self.leafList.append(cell)
                
            else:
                SWCell, NWCell, SECell, NECell = self.divide(cell)
                self.cellQueue += [SWCell, NWCell, SECell, NECell]
    
    def divide(self, cell):
        # just divide into 4
        cell["isLeaf"] = False
        
        xL = cell["xL"]
        yB = cell["yB"]
        xR = cell["xR"]
        yT = cell["yT"]
        xC = cell["xC"]
        yC = cell["yC"]
        lvl = cell["level"]
        side = cell["side"]
        mat = cell["matrix"]
        size = mat.shape[0]
        bnd = cell["bnd"]
        
        SWCell = {"pos" : "SW", 
                "id" : cell["id"] + " SW",
                "bnd" : (bnd & _SOUTH) | (bnd & _WEST),     # or other predefined boundaries
                "matrix" : mat[size//2 : size, 0: size//2],
                "xL" : xL ,
                "xC" : xL + side / 4.0 ,
                "xR" : xC ,
                "yB" : yB ,
                "yC" : yB + side / 4.0,
                "yT" : yC ,
                }
        NWCell = {"pos" : "NW",
                "id" : cell["id"] + " NW",
                "bnd" : (bnd & _NORTH) | (bnd & _WEST),
                "matrix" : mat[0: size//2, 0: size//2],
                "cornerBL" : (xL, yC) ,
                "cornerTR" : (xC, yT) ,
                "center" : (xL + side / 4.0, yC + side / 4.0) ,
                "xL" : xL ,
                "xC" : xL + side / 4.0 ,
                "xR" : xC ,
                "yB" : yC ,
                "yC" : yC + side / 4.0,
                "yT" : yT , 
                }
        SECell = {"pos" : "SE",
                "id" : cell["id"] + " SE",
                "bnd" : (bnd & _SOUTH) | (bnd & _EAST),
                "matrix" : mat[size//2 : size, size//2 : size],
                "cornerBL" : (xC, yB) ,
                "cornerTR" : (xR, yC) ,
                "center" : (xC + side / 4.0, yB + side / 4.0) ,
                "xL" : xC ,
                "xC" : xC + side / 4.0 ,
                "xR" : xR ,
                "yB" : yB ,
                "yC" : yB + side / 4.0,
                "yT" : yC ,
                }
                
        NECell = {"pos" : "NE",
                "id" : cell["id"] + " NE",
                "bnd" : (bnd & _NORTH) | (bnd & _EAST),
                "matrix" : mat[0: size//2, size//2 : size],
                "cornerBL" : (xC, yC) ,
                "cornerTR" : (xR, yT) ,
                "center" : (xC + side / 4.0, yC + side / 4.0) ,
                "xL" : xC ,
                "xC" : xC + side / 4.0 ,
                "xR" : xR ,
                "yB" : yC ,
                "yC" : yC + side / 4.0 ,
                "yT" : yT , 
                }
        
        for everychild in [SWCell, NWCell, SECell, NECell]:
            everychild['parent'] = cell
            everychild['level'] = lvl + 1
            everychild['side'] = side / 2.0
            everychild['neighbors'] = {}
            for p in self.propnames:
                assert(not p in everychild), f"Property {p} already exists!"
                everychild[p] = cell[p]
        
        # remember to link the cell with its children
        cell["SW"] = SWCell
        cell["NW"] = NWCell
        cell["SE"] = SECell
        cell["NE"] = NECell
        
        # put a cross whenever the cell is split into 4 quadrants
        self.lineCrossList.append( [(xC, xC), (yB, yT)] )
        self.lineCrossList.append( [(xL, xR), (yC, yC)] )
        
        # intra-neighborhood
        SWCell["neighbors"]["N"] = NWCell
        SWCell["neighbors"]["E"] = SECell
        NWCell["neighbors"]["S"] = SWCell
        NWCell["neighbors"]["E"] = NECell
        SECell["neighbors"]["N"] = NECell
        SECell["neighbors"]["W"] = SWCell
        NECell["neighbors"]["S"] = SECell
        NECell["neighbors"]["W"] = NWCell
        
        #inter-neighborhood
        nbdict = cell['neighbors']
        # boundaries.
        # note that two boundaries may coexist
        if cell["bnd"] & _NORTH:
            NECell["neighbors"]["N"] = {'id':_NORTH, 
                    'level':lvl + 1,
                    'xC' : NECell['xC'],
                    'yC' : yT + side / 4.0}
            NWCell["neighbors"]["N"] = {'id':_NORTH, 
                    'level':lvl + 1,
                    'xC' : NWCell['xC'],
                    'yC' : yT + side / 4.0}
        if cell["bnd"] & _SOUTH:
            SWCell["neighbors"]["S"] = {'id':_SOUTH, 
                    'level':lvl + 1,
                    'xC' : SWCell['xC'],
                    'yC' : yB - side / 4.0}
            SECell["neighbors"]["S"] = {'id':_SOUTH, 
                    'level':lvl + 1,
                    'xC' : SECell['xC'],
                    'yC' : yB - side / 4.0}
        if cell["bnd"] & _EAST:
            NECell["neighbors"]["E"] = {'id':_EAST, 
                    'level':lvl + 1,
                    'xC' : xR + side / 4.0,
                    'yC': NECell['yC']}
            SECell["neighbors"]["E"] = {'id':_EAST, 
                    'level':lvl + 1,
                    'xC' : xR + side / 4.0,
                    'yC': SECell['yC']}
        if cell["bnd"] & _WEST:
            NWCell["neighbors"]["W"] = {'id':_WEST, 
                    'level':lvl + 1,
                    'xC' : xL - side / 4.0,
                    'yC': NWCell['yC']}
            SWCell["neighbors"]["W"] = {'id':_WEST, 
                    'level':lvl + 1,
                    'xC' : xL - side / 4.0,
                    'yC': SWCell['yC']}
        
        # The following part is performed regardless a cell adjacent to bnd or not

        # Neighbors having the same parent
        # north neighbors
        try:    # must use "try" as the keys N S E W may not exist
            # we can iterate through the keys --> terser code
            nbN = cell["neighbors"]["N"]
            if type(nbN) is dict:
                if cell["level"] == nbN["level"]:
                    NWCell["neighbors"]["N"] = nbN
                    NECell["neighbors"]["N"] = nbN
                    nbN["neighbors"]["SSW"] = NWCell 
                    nbN["neighbors"]["SSE"] = NECell  
                    del cell["neighbors"]["N"]      # here del nbN does not work !!
                    del nbN["neighbors"]["S"]
                # don't worry. This deletes the entry but not the other cell itself
        except:
            pass
        # south neighbors
        try:
            nbS = cell["neighbors"]["S"]
            if type(nbS) is dict:
                if cell["level"] == nbS["level"]:
                    SWCell["neighbors"]["S"] = nbS
                    SECell["neighbors"]["S"] = nbS
                    nbS["neighbors"]["NNW"] = SWCell 
                    nbS["neighbors"]["NNE"] = SECell
                    del cell["neighbors"]["S"] 
                    del nbS["neighbors"]["N"]
        except:
            pass
        # east neighbors
        try:
            nbE = cell["neighbors"]["E"]
            if type(nbE) is dict:
                if cell["level"] == nbE["level"]:
                    NECell["neighbors"]["E"] = nbE
                    SECell["neighbors"]["E"] = nbE
                    nbE["neighbors"]["WNW"] = NECell 
                    nbE["neighbors"]["WSW"] = SECell  
                    del cell["neighbors"]["E"] 
                    del nbE["neighbors"]["W"]
        except:
            pass
            
        # west neighbors
        try:
            nbW = cell["neighbors"]["W"]
            if type(nbW) is dict:
                if cell["level"] == nbW["level"]:
                    NWCell["neighbors"]["W"] = nbW
                    SWCell["neighbors"]["W"] = nbW
                    nbW["neighbors"]["ENE"] = NWCell 
                    nbW["neighbors"]["ESE"] = SWCell  
                    del cell["neighbors"]["W"] 
                    del nbW["neighbors"]["E"]
        except:
            pass
            
        # other adjacent directions
        try:
            nbNNW = cell["neighbors"]["NNW"]
            if cell["level"] == nbNNW["level"] - 1:
                NWCell["neighbors"]["N"] = nbNNW
                nbNNW["neighbors"]["S"] = NWCell
                del cell["neighbors"]["NNW"]
        except:
            pass
            
        try:
            nbNNE = cell["neighbors"]["NNE"]
            if cell["level"] == nbNNE["level"] - 1:
                NECell["neighbors"]["N"] = nbNNE
                nbNNE["neighbors"]["S"] = NECell
                del cell["neighbors"]["NNE"]
        except:
            pass
            
        try:
            nbENE = cell["neighbors"]["ENE"]
            if cell["level"] == nbENE["level"] - 1:
                NECell["neighbors"]["E"] = nbENE
                nbENE["neighbors"]["W"] = NECell
                del cell["neighbors"]["ENE"]
        except:
            pass
            
        try:
            nbESE = cell["neighbors"]["ESE"]
            if cell["level"] == nbESE["level"] - 1:
                SECell["neighbors"]["E"] = nbESE
                nbESE["neighbors"]["W"] = SECell
                del cell["neighbors"]["ESE"]
        except:
            pass
            
        try:
            nbSSE = cell["neighbors"]["SSE"]
            if cell["level"] == nbSSE["level"] - 1:
                SECell["neighbors"]["S"] = nbSSE
                nbSSE["neighbors"]["N"] = SECell
                del cell["neighbors"]["SSE"]
        except:
            pass
            
        try:
            nbSSW = cell["neighbors"]["SSW"]
            if cell["level"] == nbSSW["level"] - 1:
                SWCell["neighbors"]["S"] = nbSSW
                nbSSW["neighbors"]["N"] = SWCell
                del cell["neighbors"]["SSW"]
        except:
            pass
            
        try:
            nbWSW = cell["neighbors"]["WSW"]
            if cell["level"] == nbWSW["level"] - 1:
                SWCell["neighbors"]["W"] = nbWSW
                nbWSW["neighbors"]["E"] = SWCell
                del cell["neighbors"]["WSW"]
        except:
            pass
        
        try:
            nbWNW = cell["neighbors"]["WNW"]
            if cell["level"] == nbWNW["level"] - 1:
                NWCell["neighbors"]["W"] = nbWNW
                nbWNW["neighbors"]["E"] = NWCell
                del cell["neighbors"]["WNW"]
        except:
            pass
        
        return SWCell, NWCell, SECell, NECell
    
    
    def assignDiagNeighbors(self, cell):
        nbdict = cell['neighbors']
        if ( 'E' in nbdict and 'N' in nbdict 
            # and type( nbdict['E'] ) is dict
            and (not cell['bnd'] & _EAST)
            and (not cell['bnd'] & _NORTH)
            and (cell['xL'] == nbdict['N']['xC'] or cell['yB'] == nbdict['E']['yC'])
            # and nbdict['N']['level'] + nbdict['E']['level'] < 2 * cell['level'] 
            ):
            # either N or E neighbors has lower level
            try:
                nbdict['NE'] = nbdict['E']['neighbors']['N']
            except:
                pass
            # the diagonal neighborhood relationship is not two-way
            # so no need to update the neighbor cell.
        else:
            try:
                del nbdict['NE']
            except:
                pass
            
        if ( 'E' in nbdict and 'S' in nbdict 
            and (not cell['bnd'] & _EAST)
            and (not cell['bnd'] & _SOUTH)
            and (cell['xL'] == nbdict['S']['xC'] or cell['yT'] == nbdict['E']['yC'])
            # and nbdict['S']['level'] + nbdict['E']['level'] < 2 * cell['level'] 
            ):
            try:
                nbdict['SE'] = nbdict['E']['neighbors']['S']
            except:
                pass
        else:
            try:
                del nbdict['SE']
            except:
                pass
        
        if ( 'W' in nbdict and 'N' in nbdict 
            and (not cell['bnd'] & _WEST)
            and (not cell['bnd'] & _NORTH)
            and (cell['xR'] == nbdict['N']['xC'] or cell['yB'] == nbdict['W']['yC'])
            # and nbdict['N']['level'] + nbdict['W']['level'] < 2 * cell['level'] 
            ):
            try:
                nbdict['NW'] = nbdict['W']['neighbors']['N']
            except:
                pass
        else:
            try:
                del nbdict['NW']
            except:
                pass
        
        if ( 'W' in nbdict and 'S' in nbdict 
            and (not cell['bnd'] & _WEST)
            and (not cell['bnd'] & _SOUTH)
            and (cell['xR'] == nbdict['S']['xC'] or cell['yT'] == nbdict['W']['yC'])
            # and nbdict['S']['level'] + nbdict['W']['level'] < 2 * cell['level'] 
            ):
            try:
                nbdict['SW'] = nbdict['W']['neighbors']['S']
            except:
                pass
        else:
            try:
                del nbdict['SW']
            except:
                pass
        
        # and for boundary cells:
        if cell['bnd'] & _NORTH and cell['xL'] > 0 and cell['xR'] < 1 :
            cell['neighbors']['N']['level'] = cell['level']
            if 'E' in nbdict and nbdict['E']['level'] == cell['level'] - 1:
                cell['neighbors']['NE'] = {'id':_NORTH, 'level':cell['level'] - 1}
            elif 'W' in nbdict and nbdict['W']['level'] == cell['level'] - 1:
                cell['neighbors']['NW'] = {'id':_NORTH, 'level':cell['level'] - 1}
                
        if cell['bnd'] & _SOUTH and cell['xL'] > 0 and cell['xR'] < 1:
            cell['neighbors']['S']['level'] = cell['level']
            if 'E' in nbdict and nbdict['E']['level'] == cell['level'] - 1:
                cell['neighbors']['SE'] = {'id':_SOUTH, 'level':cell['level'] - 1}
            elif 'W' in nbdict and nbdict['W']['level'] == cell['level'] - 1:
                cell['neighbors']['SW'] = {'id':_SOUTH, 'level':cell['level'] - 1}
                
        if cell['bnd'] & _WEST and cell['yB'] > 0 and cell['yT'] < 1:
            cell['neighbors']['W']['level'] = cell['level']
            if 'N' in nbdict and nbdict['N']['level'] == cell['level'] - 1:
                cell['neighbors']['NW'] = {'id':_WEST, 'level':cell['level'] - 1}
            elif 'S' in nbdict and nbdict['S']['level'] == cell['level'] - 1:
                cell['neighbors']['SW'] = {'id':_WEST, 'level':cell['level'] - 1}
                
        if cell['bnd'] & _EAST and cell['yB'] > 0 and cell['yT'] < 1:
            cell['neighbors']['E']['level'] = cell['level']
            if 'N' in nbdict and nbdict['N']['level'] == cell['level'] - 1:
                cell['neighbors']['NE'] = {'id':_EAST, 'level':cell['level'] - 1}
            elif 'S' in nbdict and nbdict['S']['level'] == cell['level'] - 1:
                cell['neighbors']['SE'] = {'id':_EAST, 'level':cell['level'] - 1}


    def refine(self, cell):
        SWCell, NWCell, SECell, NECell = self.divide(cell)
        self.nLeaves += 3
        self.leafList.remove( cell )
        NWCell['isLeaf'] = SWCell['isLeaf'] = NECell['isLeaf'] = SECell['isLeaf'] = True
        self.leafList += [NWCell, SWCell, NECell, SECell]
    

    def draw(self, t, grid=True, xo_port=0.5, yo_port=0, width_port=0.0078125, 
            num=False, arrow=True, dot=False, color_scale='norm',
            color='blue', alpha=0.5,
            patches=False, quiver=True, qvscale=None, 
            extent=(0.48, 0.52, 0, 0.1), thin=None,
            xticks=None, yticks=None,
            file_save=None):
        """ Draw the quadtree with matplotlib. 
            t: time, for the title of the plot.
            grid: the quadtree grid or mesh object.
            xo_port: x offset for the port.
            yo_port: y offset for the port.
            width_port: width of the port.
            num: draw numbers in the cells.
            arrow: draw arrows between cells.
            dot: draw dots in the cells.
            color_scale: 'norm' for linear scaling, 'log' for logarithmic scaling.
            patches: draw patches for the cells.
            quiver: draw quiver plot for the leaf cells.
            qvscale: scale for the quiver plot.
            thin: 'x' or 'y': velocity vector thining (distance 4*d_0) in either direction
            extent: the extent of the plot in the form (xmin, xmax, ymin, ymax).
            xticks, yticks: list of tick mark locations on both axes.
            file_save: if not None, save the plot to a file using pickle.
            This method generates a plot of the quadtree structure with various options for visualization.
        """
        print("Generating plot ...")
        plt.figure()
        plt.plot( [0, 0, 1, 1, 0], [0, 1, 1, 0, 0], 'k-')
        
        if patches:
            cmap = cmocean.cm.haline  # plt.cm.rainbow
            if color_scale == 'norm':
                val2color = matplotlib.colors.Normalize(vmin=0, vmax=1)
            elif color_scale == 'log':
                val2color = matplotlib.colors.LogNorm(vmin=0.0001, vmax=1)

            # Is this patchesList an old version matplotlib object?
            # We have to unzip it and convert to a Rectangle object, 
            # then add it to a collection.
            new_patch_list = []
            for square in self.patchesList:
                xy, wid, hgt, fc, ec, *_ = square
                p = matplotlib.patches.Rectangle(xy, wid, hgt, 
                        facecolor=cmap(val2color(float(fc))), edgecolor=ec, linewidth=0.1)
                plt.gca().add_patch(p)
                new_patch_list.append(p)

            pc = matplotlib.collections.PatchCollection(new_patch_list, cmap=cmap, alpha=1.)
            # plt.colorbar(pc, shrink=0.5)  # stackoverflow.com/a/18665162  # error in newer matplotlib versions
            
        if dot:
            plt.scatter(self.markXList, self.markYList)
        
        if num:
            for txt in self.textList:
                plt.text(txt[0], txt[1], txt[2], fontsize=8, color='black', 
                        ha='center', va='center')
        
        if grid:
            for cross in self.lineCrossList:
                plt.plot(cross[0], cross[1], 'k:', linewidth=0.3)
        
        if quiver:
            if thin is None:
                data = [(ci['xC'], ci['yC'], ci['Unew'], ci['Wnew']) 
                        for ci in self.leafList]
            elif thin == 'x':
                data = [(ci['xC'], ci['yC'], ci['Unew'], ci['Wnew']) 
                        for ci in self.leafList
                        if abs(ci['xL'] % width_port) < 1E-4 or abs(ci['xL'] % width_port - width_port) < 1E-4]
            elif thin == 'y':
                data = [(ci['xC'], ci['yC'], ci['Unew'], ci['Wnew']) 
                        for ci in self.leafList
                        if abs(ci['yB'] % width_port) < 1E-4 or abs(ci['yB'] % width_port - width_port) < 1E-4]
            list_zip = list(zip(*data))
            xs, ys, us, ws = list(list_zip[0]), list(list_zip[1]), \
                             list(list_zip[2]), list(list_zip[3])
            
            plt.quiver(xs, ys, us, ws, units='dots', width=1, scale=qvscale, 
                        headwidth=7, headaxislength=3, headlength=5,
                        color=color, alpha=alpha)

        if arrow:
            for cell in self.leafList:
                for dir in cell["neighbors"]:
                    nb = cell["neighbors"][dir] 
                    if (type(nb['id']) is str) and nb["isLeaf"]:
                        self.arrowList += [ (cell["xC"], cell["yC"], nb['xC'], nb['yC']) ]
            
            for arrow in self.arrowList:
                plt.axes()
                plt.arrow(arrow[0], arrow[1], arrow[2] - arrow[0], arrow[3] - arrow[1])
        
        if file_save is not None:
            with open(file_save, 'wb') as fh:
                pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

        offset_x = lambda x, _: f'{(x - xo_port) / width_port:g}'
        offset_y = lambda y, _: f'{(y - yo_port) / width_port:g}'
        axes = plt.gca()
        axes.xaxis.set_major_formatter(offset_x)
        axes.yaxis.set_major_formatter(offset_y)
        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)

        xmin, xmax, ymin, ymax, *_ = extent
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal', 'box')
        plt.title(t)
        
        plt.show()
        print("Completed one plot")
