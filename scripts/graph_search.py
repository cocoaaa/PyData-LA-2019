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
        

from graph import *
import numpy as np
from pprint import pprint

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
