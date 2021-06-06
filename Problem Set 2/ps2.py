# Y4CEP Problem Set 2
# Graph optimization
# Name:


#
# Finding shortest paths through a map
#
import unittest
from graph import Digraph, Node, WeightedEdge
from typing import Dict, List, Optional, Tuple

#
# Problem 2: Building up the Map
#
# Problem 2a: Designing your graph
#
# What do the graph's nodes represent in this problem? What
# do the graph's edges represent? Where are the distances
# represented?
#
# Answer:
# 
# Each node represents one building in the map. Each directed edge represents
# a path going from a source node (building) to a destination node with 2
# costs in between them, one for the total distance between the 2 buildings
# and another for the length of the unsheltered portion between them. These
# distances are stored as part of an edge defined as a `graph.WeightedEdge`.
#


# Problem 2b: Implementing load_map
def load_map(map_filename: str):
    """
    Parses the map file and constructs a directed graph

    Parameters:
        map_filename : name of the map file

    Assumes:
        Each entry in the map file consists of the following four positive
        integers, separated by a blank space:
            From To TotalDistance DistanceOutdoors
        e.g.
            32 76 54 23
        This entry would become an edge from 32 to 76.

    Returns:
        a Digraph representing the map
    """
    print("Loading map from file...")
    with open(map_filename, "r") as buffer:
        digraph = Digraph()
        for line in buffer:
            tokens = line.strip().split(" ")
            source, destination, distance, unsheltered = tokens
            if not digraph.has_node(source):
                digraph.add_node(source)
            if not digraph.has_node(destination):
                digraph.add_node(destination)
            digraph.add_edge(
                WeightedEdge(
                    source,
                    destination,
                    int(distance),
                    int(unsheltered)
                )
            )
        return digraph


# Problem 2c: Testing load_map
# Include the lines used to test load_map below, but comment them out

def test_load_map(clear: bool = True):
    """
    Convert the graph in `map.txt` into a digraph and write its string
    representation to a file called `test_load_map.txt`.

    Example
    =======

        test_load_map(clear=True) # Clear `test_load_map.txt` before writing
    
    Parameters
    ==========
    clear: bool = True
        Whether to clear the output file before writing the graph to it.
    
    Raises
    ======
    FileNotFoundError
        If this function cannot find `map.txt`.
    
    Returns
    =======
    None
    """
    OUTPUT_NAME: str = "test_load_map.txt"
    digraph: Digraph = load_map("map.txt")
    if clear:
        f = open(OUTPUT_NAME, "w")
        f.close()
    with open(OUTPUT_NAME, "a") as f:
        for node in digraph.nodes:
            for edge in digraph.get_edges_for_node(node):
                f.write(str(edge) + "\n")

#
# Problem 3: Finding the Shorest Path using Optimized Search Method
#
# Problem 3a: Objective function
#
# What is the objective function for this problem? What are the constraints?
#
# Answer:
# Visit all possible paths using DFS. If at any point the distance
# outdoors is reached, the current path is abandoned and the next cheapest
# node is considered. If the distance travelled on that path exceeds that of
# the best path, the current path is also abandoned. If a better path is found,
# that path will become the new best path. The shortest path that does not
# exceed the maximum distance outdoors is the path to be returned.
#

class DfsResult(object):
    """
    A structure containing the current path being traversed, the total
    distance across the path and the total unsheltered distance. Any object
    of this class should be used immutably, through self.append and
    self.remove instead of directly editing the data inside.

    Paramters and Fields
    ====================
    path: List[Node]
        The current path.
    
    distance: int
        The total distance.
    
    unsheltered: int
        The unsheltered distance.
    """
    def __init__(
        self,
        path: List[Node] = [],
        distance: int = 0,
        unsheltered: int = 0
    ):
        self.path = path
        self.distance = distance
        self.unsheltered = unsheltered
    
    def __str__(self):
        return "%s, %s, %s" % (self.path, self.distance, self.unsheltered)
    
    def __repr__(self):
        return "DfsResult(%s)" % str(self)
    
    def __contains__(self, node):
        return self.has(node)

    def has(self, node):
        """
        Check if a node is in the path. `node` must be type `graph.Node`.
        """
        return node in self.path

    @classmethod
    def apply(cls, path, distance, unsheltered):
        return cls(path, distance, unsheltered)
    
    def unapply(self):
        return tuple(self.path), self.distance, self.unsheltered
    
    def append(self, nodes: List[Node], cost: int, unsheltered: int):
        """
        Create a new DfsResult with new nodes plus the cost to get to the
        last node, including the unsheltered distance.

        Parameters
        ==========
        nodes: graph.Node
            The nodes to be pushed onto the path.
        
        cost: int
            The cost to get to the last node in `nodes` from the last node in
            the current path.
        
        unsheltered: int
            Unsheltered distance to get to the last node in `nodes` from the
            last node in the current path.
        
        Returns
        =======
        DfsResult
        """
        return DfsResult(
            self.path + nodes,
            self.distance + cost,
            self.unsheltered + unsheltered)
    
    def remove(self, cost: int, unsheltered: int):
        """
        Create a new DfsResult without the last node, minus the cost from the
        second-last node to the last node with the unsheltered distance.

        Parameters
        ==========
        cost: int
            The cost to go from path[-2] to path[-1].
        
        unsheltered: int
            The unsheltered distance between path[-2] and path[-1].
        
        Returns
        =======
        DfsResult
        """
        if len(self.path) < 1:
            return self
        else:
            return DfsResult(
                self.path[:-1],
                self.distance - cost,
                self.unsheltered - unsheltered
            )

# Problem 3b: Implement get_best_path
def get_best_path(
    digraph: Digraph,
    start: Node,
    end: Node,
    path: DfsResult,
    max_dist_outdoors: int,
    best_dist: int,
    best_path: DfsResult):
    """
    Finds the shortest path between buildings subject to constraints.

    Parameters
    ==========
        digraph: Digraph
            The graph on which to carry out the search
        start: Node
            Building number at which to start
        end: Node
            Building number at which to end
        path: DfsResult
            Represents the current path of nodes being traversed. Contains
            a list of node names, total distance traveled, and total
            distance outdoors.
        max_dist_outdoors: int
            Maximum distance spent outdoors on a path
        best_dist: int (UNUSED)
            The smallest distance between the original start and end node
            for the initial problem that you are trying to solve
        best_path: DfsResult
            The shortest path found so far between the original start
            and end node.

    Returns
    =======
    DfsResult
        A result consisting of a list of nodes, the total distance and the
        total unsheltered distance. The list of nodes contains the nodes in
        the order to be travelled.

        If no available path is found, None is returned.
    """
    #print()
    #print("get_best_path: Starting node is %s" % repr(start))
    #print("get_best_path: End node is %s" % repr(end))
    if not digraph.has_node(start):
        raise ValueError("Unrecognised start node: %s" % start)
    if not digraph.has_node(end):
        raise ValueError("Unrecognised end node: %s" % end)
    if start in path:
        return None
    current_path = path.append([start], 0, 0)
    #print(current_path)
    if current_path.distance > best_path.distance:
        return None
    if current_path.unsheltered > max_dist_outdoors:
        return None
    if start == end:
        if current_path.distance <= best_path.distance:
            return current_path
        else:
            return None
    adjacents = digraph.get_edges_for_node(start)
    if len(adjacents) < 1:
        return None
    new_best_path = None
    for adjacent in adjacents:
        result = get_best_path(
            digraph,
            Node(adjacent.dest),
            end,
            current_path.append(
                [],
                adjacent.distance,
                adjacent.unsheltered
            ),
            max_dist_outdoors,
            best_dist,
            best_path if new_best_path is None else new_best_path
        )
        if result is None:
            continue
        elif new_best_path is None or new_best_path.distance > result.distance:
            new_best_path = result
    return new_best_path


# Problem 3c: Implement directed_dfs
def directed_dfs(digraph, start, end, max_total_dist, max_dist_unsheltered):
    """
    Finds the shortest path from start to end using a directed depth-first
    search. The total distance traveled on the path must not
    exceed max_total_dist, and the distance spent unsheltered on this path must
    not exceed max_dist_unsheltered.

    Parameters:
        digraph: Digraph instance
            The graph on which to carry out the search
        start: string
            Building number at which to start
        end: string
            Building number at which to end
        max_total_dist: int
            Maximum total distance on a path
        max_dist_sheltered: int
            Maximum distance spent unsheltered on a path

    Returns:
        The shortest-path from start to end, represented by
        a list of building numbers (in strings), [n_1, n_2, ..., n_k],
        where there exists an edge from n_i to n_(i+1) in digraph,
        for all 1 <= i < k

        If there exists no path that satisfies max_total_dist and
        max_dist_unsheltered constraints, then raises a ValueError.
    """
    best_path = DfsResult([], max_total_dist, 0)
    if not digraph.are_connected(
        Node(start) if type(start) == str else start,
        Node(end) if type(end) == str else end
    ):
        raise ValueError("%s and %s are not connected." % (start, end))
    result = get_best_path(
        digraph,
        Node(start) if type(start) == str else start,
        Node(end) if type(end) == str else end,
        DfsResult([], 0, 0),
        max_dist_unsheltered,
        max_total_dist,
        best_path
    )
    print(result)
    if result is None or result.path == []:
        raise ValueError(
            ("Could not find a path between %s and %s that is"
            + "less than length %s with unsheltered distance less than %s")
            % (start, end, max_total_dist, max_dist_unsheltered)
        )
    else:
        return [node.name for node in result.path]


# ================================================================
# Begin tests -- you do not need to modify anything below this line
# ================================================================

class Ps2Test(unittest.TestCase):
    LARGE_DIST = 99999

    def setUp(self):
        self.graph = load_map("map.txt")

    def test_load_map_basic(self):
        self.assertTrue(isinstance(self.graph, Digraph))
        self.assertEqual(len(self.graph.nodes), 37)
        all_edges = []
        for _, edges in self.graph.edges.items():
            all_edges += edges  # edges must be dict of node -> list of edges
        all_edges = set(all_edges)
        self.assertEqual(len(all_edges), 129)

    def _print_path_description(self, start, end, total_dist, outdoor_dist):
        constraint = ""
        if outdoor_dist != Ps2Test.LARGE_DIST:
            constraint = "without walking more than {}m outdoors".format(
                outdoor_dist)
        if total_dist != Ps2Test.LARGE_DIST:
            if constraint:
                constraint += ' or {}m total'.format(total_dist)
            else:
                constraint = "without walking more than {}m total".format(
                    total_dist)

        print("------------------------")
        print("Shortest path from Building {} to {} {}".format(
            start, end, constraint))

    def _test_path(self,
                   expectedPath,
                   total_dist=LARGE_DIST,
                   outdoor_dist=LARGE_DIST):
        start, end = expectedPath[0], expectedPath[-1]
        self._print_path_description(start, end, total_dist, outdoor_dist)
        dfsPath = directed_dfs(self.graph, start, end, total_dist, outdoor_dist)
        print("Expected: ", expectedPath)
        print("DFS: ", dfsPath)
        self.assertEqual(expectedPath, dfsPath)

    def _test_impossible_path(self,
                              start,
                              end,
                              total_dist=LARGE_DIST,
                              outdoor_dist=LARGE_DIST):
        self._print_path_description(start, end, total_dist, outdoor_dist)
        with self.assertRaises(ValueError):
            directed_dfs(self.graph, start, end, total_dist, outdoor_dist)

    def test_path_one_step(self):
        self._test_path(expectedPath=['32', '56'])

    def test_path_no_outdoors(self):
        self._test_path(
            expectedPath=['32', '36', '26', '16', '56'], outdoor_dist=0)

    def test_path_multi_step(self):
        self._test_path(expectedPath=['2', '3', '7', '9'])

    def test_path_multi_step_no_outdoors(self):
        self._test_path(
            expectedPath=['2', '4', '10', '13', '9'], outdoor_dist=0)

    def test_path_multi_step2(self):
        self._test_path(expectedPath=['1', '4', '12', '32'])

    def test_path_multi_step_no_outdoors2(self):
        self._test_path(
            expectedPath=['1', '3', '10', '4', '12', '24', '34', '36', '32'],
            outdoor_dist=0)

    def test_impossible_path1(self):
        self._test_impossible_path('8', '50', outdoor_dist=0)

    def test_impossible_path2(self):
        self._test_impossible_path('10', '32', total_dist=100)


if __name__ == "__main__":
    test_load_map()
    unittest.main()
