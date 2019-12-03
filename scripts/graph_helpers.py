import os,sys
from pathlib import Path

################################################################################
### Path setup
################################################################################
PROJ_ROOT = Path(os.getcwd()).parent;print(PROJ_ROOT)
DATA_DIR = PROJ_ROOT/'data/raw'
SRC_DIR = PROJ_ROOT/'scripts'
paths2add = [PROJ_ROOT, SRC_DIR]

# Add project root and src directory to the path
for p in paths2add:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
        print("Prepened to path: ", p)

import numpy as np
import pdb
from pprint import pprint

def nprint(*args, header=True):
    if header:
        print("="*80)
    for arg in args:
        print(arg)
                
################################################################################
### Graph related classes
################################################################################
class Vertex:
    def __init__(self, vid):
        self.vid = vid # integer
        self.neighbors = set() # set of adjacent Vertex objects
        
    def add_neighbors(self, new_vs):
        for new_v in new_vs:
            self.neighbors.add(new_v)
    def print_neighbors(self):
        for v in self.neighbors:
            print('\t<->', v)
            
    def __eq__(self, other):
        return self.vid == other.vid and self.neighbors == other.neighbors
    
    def __gt__(self, other):
        return self.vid > other.vid
    
    def __ge__(self, other):
        return self == other or self > other
    def __hash__(self):
        return hash(self.vid)
    
    def __str__(self):
        return "Vertex {}".format(self.vid) #": connected to {}".format(self.vid, [n.vid for n in self.neighbors])

    
class UndirectedEdge:
    def __init__(self, v1, v2):
        """
        Args:
        - v1, v2 (Vertex object)
        """
        self.endpoints = set([v1,v2])
        
    def get_endpoints(self, toSort=True, **kwargs):
        epts = list(self.endpoints)
        if toSort:
            epts.sort(**kwargs)
        return epts
    
    def __hash__(self):
        return hash( tuple(self.get_endpoints(toSort=True)) )
        
    def __eq__(self, other):
        if not isinstance(other, UndirectedEdge):
            return NotImplemented
        return self.endpoints == other.endpoints
    
    def __str__(self):
        v1, v2 = self.get_endpoints()
        return "Edge({} <-> {})".format(v1.vid,v2.vid)
        
        
class DirectedEdge:
    def __init__(self, v1, v2):
        """
        Args:
        - v1, v2 (Vertex object)
        """
        self.endpoints = (v1,v2)
        
    def get_endpoints(self):
        return self.endpoints
    
    def __hash__(self):
        return hash( self.endpoints )
        
    def __eq__(self, other):
        if not isinstance(other, UndirectedEdge):
            return NotImplemented
        return self.endpoints == other.endpoints
    
    def __str__(self):
        v1, v2 = self.endpoints
        return "Edge({} -> {})".format(v1.vid,v2.vid)
        
class UndirectedGraph:
    def __init__(self, vertices):
        self.V = {v.vid: v for v in vertices} # set of Vertex objs 
        
        # create self.E based on the adjacency of Vertices in self.V
        self.__init_edges_from_vertices()
        
    def __init_edges_from_vertices(self):
        self.E = set()
        for v in self.V.values():
            for n in v.neighbors:
                assert n.vid in self.V, "Vertex {} not in Graph".format(n.vid)
                self.E.add(UndirectedEdge(v,n))

    def __str__(self):
        pass #todo: tree like printout
    
    def has_vertex(self, v):
        return self.V[v.vid] == v
    
    def has_edge(self, e):
        return e in self.E
    
    def add_vertex(self, v):
        if v.vid in self.V:
            print("No update: Vertex {} already in graph".format(v.vid))
        else: 
            self.V[v.vid] = v
        
    def add_edge(self, e):
        v1, v2 = e.get_endpoints
        if not (v1.vid in self.V and v2.vid in self.V):
            raise ValueError("Both endpoints of the edge must in already in graph. Try add_vertex first")
        
        # Add to graph
        self.E.add(e)
        
        # Add to vertices' `neighbors`
        v1.add_neighbors([v2])
        v2.add_neighbors([v1])
        
    def get_neighbors(self, vid):
        return self.V[vid].neighbors
    
    def print_neighbors(self, vid):
        return self.V[vid].print_neighbors()
    
    def print_edges(self):
        for e in self.E:
            print(e)
    def print_structure(self):
        for v in self.V.values():
            print("="*80)
            print(v)
            v.print_neighbors()
            
class DirectedGraph:
    """
    Directed graph abstraction that contains a dictionary of Vertex objects
    and a set of DirectedEdge objects
    
    Args:
    - vertices (list): list of Vertex objects
    """
    def __init__(self, vertices):
        self.V = {v.vid: v for v in vertices} # set of Vertex objs 
        
        # create self.E based on the adjacency of Vertices in self.V
        self.__init_edges_from_vertices()
        
    def __init_edges_from_vertices(self):
        self.E = set()
        for v in self.V.values():
            for n in v.neighbors:
                assert n.vid in self.V, "Vertex {} not in Graph".format(n.vid)
                self.E.add(DirectedEdge(v,n))

    def __str__(self):
        pass #todo: tree like printout
    
    def has_vertex(self, v):
        return self.V[v.vid] == v
    
    def has_edge(self, e):
        return e in self.E
    
    def add_vertex(self, v):
        if v.vid in self.V:
            print("No update: Vertex {} already in graph".format(v.vid))
        else: 
            self.V[v.vid] = v
        
    def add_edge(self, e):
        """Adds a directed edge
        """
        if not isinstance(e, DirectedEdge):
            raise TypeError(f"{e} must be of type DirectedEdge: {type(e)}")
        v1, v2 = e.get_endpoints
        if not (v1.vid in self.V and v2.vid in self.V):
            raise ValueError("Both endpoints of the edge must in already in graph. Try add_vertex first")
        
        # Add to graph
        self.E.add(e)
        
        # Add to vertices' `neighbors`
        v1.add_neighbors([v2])        
        
    def get_neighbors(self, vid):
        return self.V[vid].neighbors
    
    def print_neighbors(self, vid):
        return self.V[vid].print_neighbors()
    
    def print_edges(self):
        for e in self.E:
            print(e)
    def print_structure(self):
        for v in self.V.values():
            print("="*80)
            print(v)
            v.print_neighbors()

class DAG(DirectedGraph):
    """ 
    Fully connected DAG with vertices Vertex(0), Vertex(1), ..., Vertex(n-1)
    
    Args:
    - n (int): number of vertices
    """
    def __init__(self, n):
        verts = [Vertex(i) for i in range(n)]
        # add DAG edges
        for i,v in enumerate(verts):
            v.add_neighbors( [verts[j] for j in range(i+1, n)] )
            
        super().__init__(verts)

class Node:
    """
    Node representing a search path on a graph
    
    Args:
    - vid (int): vertex id in a graph 
    - parent (Node): its parent Node object
    
    """
    def __init__(self, vid, parent):
        self.vid = vid
        self.parent = parent 
        
        # depth(ie. level) in the search tree
        # Note a root is at level 1, not zero
        self.path_len = 1 if parent is None else parent.path_len + 1 
        self.depth = self.path_len #alias
        
    def __str__(self):
        clsname =  self.__class__.__name__
        return "{}({}, p={}, depth={})".format(clsname,
                                             self.vid, 
                                            self.parent.vid if self.parent is not None else "None",
                                            self.depth)
# Alternate naming
class Path(Node):
    """
    Args:
    - vid (int): the id of the last vertex of this path
    - parent (Path): Path object that represents the parent path (ie. path leading upto 
        the `end_vid`
    - path_len (int): length of this path (ie. number of vertices on this path(
    """
    def __init__(self, vid, parent):
        super().__init__(vid, parent) #same as above
        
#     def __str__(self):
#         NotImplemented
#         pass #same as above

class CombNode(Node):
    """
    Combinatorial node
    Each node instance contains self.vid = a set of elements(not the indices)
    from the original list of elements
    
    Args:
    - vid (frozenset): a set of elements from the orig_list
        - Note this is not a set of indices (integers) indicating the element's index i
    - parent( Node)
    
    Below two are problem-specific 
    - orig_list (list): the original list containing members (not its indices)
    - binsizes (list or tuple): a list of binsizes to divide the elements in orig_list into.
        It must satisfy that sum(binsize) == len(orig_list)
    
    For example, 
    - orig_list = [0,1,2,3,4] 
    - binsize could be (1,1,3) or (1,2,2) if we want to group the elements into three groups
    or (1,4), (2,3) if into 2 groups.
    
    Another example,
    - orig_list = ['highway' , 'primary', 'tertiary', 'residential', 'path', 'cycleway']
    - binsize = (1,1,4), (1,2,3) if we want to group the elements into three groups
    or (1,5), (2,4), (3,3) if into two groups
    """
    def __init__(self, vid, parent, orig_list, binsizes):
        if not isinstance(vid, set):
            raise ValueError("vid must be a set: {}".format(type(vid)))
        if not isinstance(vid, frozenset):
            vid = frozenset(vid)
#         assert np.sum(binsizes) == len(orig_list), "Binsize must sum upto {}".format(len(orig_list))
        

        super().__init__(vid, parent)
        
        if not isinstance(orig_list, np.ndarray):
            orig_list = np.array(orig_list)
        self.orig_list = orig_list
        self.binsizes = binsizes
        
        # element-based (not index-based) list history
        _prev_remaining = self.parent.remaining if self.parent is not None else self.orig_list
        self.remaining = np.array([ele for ele in _prev_remaining if ele not in self.vid])
    
    def get_children(self, verbose=False):
        """
        Returns a list of its children nodes (CombNode objects)
        """
        # If no more remaining elements, return right away
        if len(self.remaining) == 0:
            return []
        
        # Find all possible combinations of size `k` from `self.remaining` list
        n = len(self.remaining)
        k = self.binsizes[self.depth] # children level's binsize since self.depth starts at 1
        idxset_list = nCk(n, k)        
        vidset_list = [ set(self.remaining[ list(idxset) ]) for idxset in idxset_list ]
        
        if verbose:
            print('remaining elements: ', self.remaining)
            print('child binsize: ', k)
            print('idxset_list: ',idxset_list)
            print('vidset_list: ',vidset_list)

        
        # Create CombNodes for children 
        children = [CombNode(vid=vidset, parent=self, orig_list=self.orig_list, binsizes=self.binsizes) 
                    for vidset in vidset_list]
        return children
    
    def __str__(self):
        clsname =  self.__class__.__name__
        p = set(self.parent.vid) if self.parent is not None else "None"
        descr = ( f"{clsname}(id={set(self.vid)}, p={p}, level={self.path_len})"
#                   f"\n\tOriginal: {self.orig_list}"
                  f", Remaining: {self.remaining}" )
        return descr
        
################################################################################
### Traceback Helper
################################################################################

def get_trace(n, tlist=None):
    """
    Print full path from Node $n$ till its root
    Args:
    - n (Node):
    - tlist (list or None): list of vertex ids (integers)
    """
    if tlist is None: #n is the last node in the trace tree
        tlist = []
    tlist.append(n.vid)
    
    if n.parent is None: # n is the first node in the trace tree. End tracing.
        return tlist
    
    # recurse
    return get_trace(n.parent, tlist)
            
################################################################################
### Graph search helpers
################################################################################
def DFS(G, start_vid, goal_vid,
       verbose=False):
    """
    Args:
    - G (Graph)
        - Has a method `G.neighbors(v)` which returns a list of vertex ids neighboring vertex `v`
    - start_vid (int: Vertex id): vertex id of the start vertex in G
    - goal_vid (int: Vertex id): vertex id of the goal vertex in G
    """
    assert start_vid in G.V, "start node not found in Graph"
    assert goal_vid in G.V, "goal node not found in Graph"
    
    S = Node(vid=start_vid, parent=None)
    Q = [S] # Queue that is actually a stack.  List of Node objs
    Expanded = set() # a set of integers for vertex ids
    while len(Q) > 0:
        N = Q.pop(0)
        if N.vid == goal_vid:
            return get_trace(N)
        
        # Expand N and add its children nodes that haven't been explored yet
        Expanded.add(N.vid)
        cnodes = [Node(vid=c.vid, parent=N) for c in G.get_neighbors(N.vid) if c.vid not in Expanded]
        cnodes.sort(key=lambda n: n.vid)
        Q = cnodes + Q # prepend the children nodes
        
        if verbose:
            print("Expanding: ", N, "...")
            print("Updated stack: ", [node.vid for node in Q])
            print("So far, Expanded: ", Expanded)


def kBFS(G, start_vid, k, verbose=False):
    """
    Given a graph G=(V,E), find all paths of length `k` starting from vertex `start_vid`
    
    Args:
    - G (Graph)
    - start_vid (int): start vertex to compute the path and path length
    - k (int): length of the paths we are looking for
    
    Returns:
    - collection (list): a list of sets where each set contains `k` integers indicating 
                         vertices on the path of length `k`          
    """
    assert start_vid in G.V, "start node not in the graph"
    
    S = Node(vid = start_vid, parent=None)
    Q = [S] # Queue containing Node objects
#     Expanded = set() # a set of integers for vertex ids visited/expanded 
## 3/26/2019 (w) Keeping track of Expanded node is erronous for BFS
    collection = []
    while len(Q) > 0: 
        N = Q.pop(0)
        
        # check if the search goal is met
        if N.path_len == k:
            collection.append(frozenset(get_trace(N)))

            if verbose:
                print("Woohoo. Found a path of length {}: {}".format(k, N))
                print("\t {}".format(get_trace(N)))
        else:
            # This node needs to be expanded, so that its children paths can be 
            # explored further
            # 3/26/2019 (W): Expanded list is erronous for BFS
#             Expanded.add(N.vid)
            
            # 3/26/2019 (W): Found a bug for `nCn`
            # Below is wrong! nCn will return empty.
#             cnodes = [Node(vid=v.vid, parent=N) for v in G.get_neighbors(N.vid) if v.vid not in Expanded]
            # Instead exclude any node in this node's path trace from the children list
            # Better to implement `get_children` method for Node class
            cnodes = [Node(vid=v.vid, parent=N) for v in G.get_neighbors(N.vid) if v.vid not in get_trace(N)]
        
            # Give order by vertex id
            cnodes.sort(key=lambda n:n.vid)
            Q.extend(cnodes) # add the children nodes to the end of queue (BFS)
            
            if verbose:
                nprint("Expanding: ", N, "...")
                print("Updated queue: ", [node.vid for node in Q])
    
    return collection


def nCk(n, k, **kwargs):
    """
    Args:
    - n (int): number of elements in the full list 
    - k (int): number of items to choose
    
    Returns:
    - collections (list): a list of sets where each set contains `k` integers 
        indicating the ids of selected vertices
        
    """
    dag = DAG(n)
#     pdb.set_trace()
    collections = []
    for start_vid in range(n):
        collections.extend(kBFS(dag, start_vid, k, **kwargs))
    return collections
                           
    
def get_all_combs(orig_list, binsizes, verbose=False):
    
    # First, get the level0 nodes and initiate the queue with them
    n = len(orig_list)
    search_depth = len(binsizes)
    level0_idxset_list = nCk(n, binsizes[0])
    level0_vidset_list = [ set(orig_list[list(idxset)])  for idxset in level0_idxset_list ]
    level0_nodes = [CombNode(vid=vidset, parent=None, orig_list=orig_list, binsizes=binsizes)
                   for vidset in level0_vidset_list]
    Q = level0_nodes
    
    if verbose: 
        print('orig_list: ', orig_list)
        print('binsizes: ', binsizes)
        nprint(f"Initial Queue of nodes [{len(level0_nodes)}]: \n", *Q)
        nprint()
    
    # Collect the possible combinations using DFS
    collection = set([])
    while len(Q) > 0:
        if verbose:
            nprint("Current Q: ")
            for n in Q: print(n)
            
        N = Q.pop(0)
        
        if N.depth == search_depth: 
            collection.add(frozenset(get_trace(N)))
        else:
            children = N.get_children()
            
            ## debugging
            if verbose:
                nprint(f"--> Expanding: {N}", header=False)
                
            # add to the top of stack
            Q = children + Q
    if verbose:
        print("Num of all possible combs: ", len(collection)) 
        nprint("\nCollection: ", *collection, header=False)
    
    return collection 

def solve_problem():
    orig_list = np.array(range(7))
#     orig_list = np.array( [ 'a','b','c','d', 'e','f','g'] )
    binsizes_list = [ [1,1,5], [1,2,4], [1,3,3], [2,2,3] ]
    all_combs = []
    for binsizes in binsizes_list:
        combs = get_all_combs(orig_list, binsizes)
        
        all_combs.extend(combs)
        nprint(f"binsizes: {binsizes}", len(combs))
    print("Total: ", len(all_combs))
    return all_combs
