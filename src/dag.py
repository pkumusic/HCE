from __future__ import print_function
import sys
from timeit import default_timer as timer

class Node(object):
    """ DAG nodes with parents and children. """

    def __init__(self, idx=None):
        self.idx = idx  # index in the DAG nlist
        self.parents = []
        self.children = []
        self.level = 0

class LinkNode(object):
    """ Typical linked list node to save the path from on category to another. """

    def __init__(self, data=None):
        self.data = data
        self.link = None

    def __eq__(self, o):
        return self.data == o.data

    def __cmp__(self, o):
        if self.data < o.data:
            return -1
        elif self.data == o.data:
            return 0
        else:
            return 1

    def __hash__(self):
        return self.data

    def point(self, node):
        self.link = node

class Dag(object):

    @staticmethod
    def loads_from(path):
        print("Loading...", file=sys.stderr)
        fin = open(path)
        hlist = fin.readlines()
        fin.close()
        print("Constructing...", file=sys.stderr)
        dag = Dag(len(hlist))
        dag.loads(hlist)
        print("Setting level...", file=sys.stderr)
        dag.set_level()
        return dag

    def __init__(self, length):
        self._root = None
        self.nlist = []
        for i in range(length):
            self.nlist.append(Node(i))

    @property
    def root(self):
        if self._root is None:
            for node in self.nlist:
                if len(node.parents) == 0:
                    self._root = node
                    break
        return self._root

    def set_level(self):
        root = self.root
        root.level = 0
        queue = [root]
        while len(queue) > 0:
            head = queue.pop()
            children = [self.nlist[c] for c in head.children]
            for child in children:
                child.level = max(child.level, head.level + 1)
                queue.append(child)

    def loads(self, hlist):
        for h in hlist:
            clist = [int(c) for c in h.strip().split('\t')]
            father = clist[0]
            for i in range(1, len(clist)):
                child = clist[i]
                self.nlist[father].children.append(child)
                self.nlist[child].parents.append(father)

    def shortest_path(self, idx1, idx2):
        node1 = self.nlist[idx1]
        node2 = self.nlist[idx2]
        if node1.level > node2.level:
            node1, node2 = node2, node1
        ancestors = {}
        lca = None
        queue = [LinkNode(node1.idx)]
        while len(queue) > 0:
            head = queue.pop()
            if not head.data in ancestors:
                ancestors[head.data] = head
            parents = self.nlist[head.data].parents
            for parent in parents:
                node = LinkNode(parent)
                node.point(head)
                queue.append(node)
        queue = [LinkNode(node2.idx)]
        while len(queue) > 0:
            head = queue.pop()
            if head.data in ancestors:
                lca = head
                break
            parents = self.nlist[head.data].parents
            for parent in parents:
                node = LinkNode(parent)
                node.point(head)
                queue.append(node)
        path = None
        if lca is None:
            raise NameError("No lowest common ancestor")
        else:
            p = lca
            path = [p.data]
            while p.link is not None:
                p = p.link
                path.append(p.data)
            path.reverse()
            p = ancestors[lca.data]
            while p.link is not None:
                p = p.link
                path.append(p.data)
        return path

    def ancestor_path(self, clist):
        count_stat = {} # store the number of paths
        sum_stat = {} # store the sum of length of paths
        queue = []
        for c in clist:
            node = LinkNode(c)
            node.link = 1
            queue.append(node)
            count_stat[node.data] = 1
            sum_stat[node.data] = 1
        while len(queue) > 0:
            head = queue.pop(0)
            parents = self.nlist[head.data].parents
            for p in parents:
                node = LinkNode(p)
                node.link = head.link + 1
                if p in count_stat:
                    count_stat[p] += 1
                    sum_stat[p] += node.link
                else:
                    count_stat[p] = 1
                    sum_stat[p] = node.link
                queue.append(node)
        stat = {}
        for key in count_stat:
            stat[key] = float(sum_stat[key])/count_stat[key]
        return stat.items()

    def level_list(self):
        return [n.level for n in self.nlist]

if __name__ == "__main__":
    dag = Dag.loads_from('./hierarchy_id.txt')
    print("Calculating...", file=sys.stderr)
    start = timer()
    for i in range(1000):
        dag.shortest_path(504593, 421917)
    end = timer()
    print("Used: {0}secs".format(end - start), file=sys.stderr)
