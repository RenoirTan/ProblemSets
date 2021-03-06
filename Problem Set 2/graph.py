# Y4 CEP Problem Set 2
# Graph optimization
# Name:


import unittest

#
# A set of data structures to represent graphs
#

class Node(object):
    """Represents a node in the graph"""
    def __init__(self, name):
        self.name = str(name)

    def get_name(self):
        return self.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Node(%s)" % repr(self.name)

    def __eq__(self, other):
        if type(other) == str:
            return self.name == other
        return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # This function is necessary so that Nodes can be used as
        # keys in a dictionary, even though Nodes are mutable
        return self.name.__hash__()


class Edge(object):
    """Represents an edge in the dictionary. Includes a source and
    a destination."""
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest

    def get_source(self):
        return self.src

    def get_destination(self):
        return self.dest

    def __str__(self):
        return '{}->{}'.format(self.src, self.dest)
    
    def __repr__(self):
        return "Edge(src=%s, dest=%s)" % (self.src, self.dest)


class WeightedEdge(Edge):
    def __init__(self, src, dest, distance, unsheltered):
        self.src = src
        self.dest = dest
        self.distance = distance
        self.unsheltered = unsheltered

    def get_total_distance(self):
        return self.distance

    def get_outdoor_distance(self):
        return self.unsheltered

    def __str__(self):
        return "%s->%s (%s, %s)" % (
            self.src,
            self.dest,
            self.distance,
            self.unsheltered
        )
    
    def __repr__(self):
        return "WeightedEdge(%s, %s, %s, %s)" % (
            self.src,
            self.dest,
            self.distance,
            self.unsheltered
        )

class EdgeToDest(object):
    """
    Edge with a destination but no source.
    """
    def __init__(self, to, cost):
        self.to = to
        self.cost = cost
    
    @classmethod
    def apply(cls, to, cost):
        return cls(to, cost)
    
    def unapply(self):
        return (self.to, self.cost)
    
    def spaceship(self, other):
        if self.cost > other.cost:
            return 1
        elif self.cost < other.cost:
            return -1
        else:
            return 0
    
    def __gt__(self, other):
        return self.spaceship(other) > 0
    
    def __ge__(self, other):
        return self.spaceship(other) >= 0
    
    def __eq__(self, other):
        return self.spaceship(other) == 0
    
    def __ne__(self, other):
        return not self == other
    
    def __le__(self, other):
        return self.spaceship(other) <= 0
    
    def __lt__(self, other):
        return self.spaceship(other) < 0


class Digraph(object):
    """Represents a directed graph of Node and Edge objects"""
    def __init__(self):
        self.nodes = set([])
        self.edges = {}  # must be a dict of Node -> list of edges

    def __str__(self):
        edge_strs = []
        for edges in self.edges.values():
            for edge in edges:
                edge_strs.append(str(edge))
        edge_strs = sorted(edge_strs)  # sort alphabetically
        return '\n'.join(edge_strs)  # concat edge_strs with "\n"s between them

    def get_edges_for_node(self, node):
        return self.edges[node]

    def has_node(self, node):
        return node in self.nodes

    def add_node(self, node):
        """Adds a Node object to the Digraph. Raises a ValueError if it is
        already in the graph."""
        if self.has_node(node):
            raise ValueError("%s is already in the graph" % node)
        self.nodes.add(node)
        self.edges[node] = []

    def add_edge(self, edge):
        """Adds an Edge or WeightedEdge instance to the Digraph. Raises a
        ValueError if either of the nodes associated with the edge is not
        in the  graph."""
        if not self.has_node(edge.src):
            raise ValueError(
                "%s has not been added to the graph." % edge.src
            )
        elif not self.has_node(edge.dest):
            raise ValueError(
                "%s has not been added to the graph." % edge.dest
            )
        else:
            self.edges[edge.src].append(edge)
    
    def are_connected(self, start, end):
        """
        DFS search to check if 2 nodes are connected to each other.
        """
        if not self.has_node(start):
            raise ValueError("Unrecognized start node: %s" % start)
        if not self.has_node(end):
            raise ValueError("Unrecognized end node: %s" % end)
        stack = [start]
        visited = {node: False for node in self.nodes}
        while len(stack) > 0:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                for adjacent in self.get_edges_for_node(current):
                    if Node(adjacent.dest) == end:
                        return True
                    stack.append(Node(adjacent.dest))
        return False


# ================================================================
# Begin tests -- you do not need to modify anything below this line
# ================================================================

class TestGraph(unittest.TestCase):

    def setUp(self):
        self.g = Digraph()
        self.na = Node('a')
        self.nb = Node('b')
        self.nc = Node('c')
        self.g.add_node(self.na)
        self.g.add_node(self.nb)
        self.g.add_node(self.nc)
        self.e1 = WeightedEdge(self.na, self.nb, 15, 10)
        self.e2 = WeightedEdge(self.na, self.nc, 14, 6)
        self.e3 = WeightedEdge(self.nb, self.nc, 3, 1)
        self.g.add_edge(self.e1)
        self.g.add_edge(self.e2)
        self.g.add_edge(self.e3)

    def test_weighted_edge_str(self):
        self.assertEqual(str(self.e1), "a->b (15, 10)")
        self.assertEqual(str(self.e2), "a->c (14, 6)")
        self.assertEqual(str(self.e3), "b->c (3, 1)")

    def test_weighted_edge_total_distance(self):
        self.assertEqual(self.e1.get_total_distance(), 15)
        self.assertEqual(self.e2.get_total_distance(), 14)
        self.assertEqual(self.e3.get_total_distance(), 3)

    def test_weighted_edge_outdoor_distance(self):
        self.assertEqual(self.e1.get_outdoor_distance(), 10)
        self.assertEqual(self.e2.get_outdoor_distance(), 6)
        self.assertEqual(self.e3.get_outdoor_distance(), 1)

    def test_add_edge_to_nonexistent_node_raises(self):
        node_not_in_graph = Node('q')
        no_src = WeightedEdge(self.nb, node_not_in_graph, 5, 5)
        no_dest = WeightedEdge(node_not_in_graph, self.na, 5, 5)

        with self.assertRaises(ValueError):
            self.g.add_edge(no_src)
        with self.assertRaises(ValueError):
            self.g.add_edge(no_dest)

    def test_add_existing_node_raises(self):
        with self.assertRaises(ValueError):
            self.g.add_node(self.na)

    def test_graph_str(self):
        expected = "a->b (15, 10)\na->c (14, 6)\nb->c (3, 1)"
        self.assertEqual(str(self.g), expected)


if __name__ == "__main__":
    unittest.main()
