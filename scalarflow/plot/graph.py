from graphviz import Digraph


def create_graph(root):
    nodes, edges = set(), set()

    def build(node):
        if node not in nodes:
            nodes.add(node)
            for child in node._prev:
                edges.add((child, node))
                build(child)

    build(root)
    return nodes, edges


def draw_graph(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = create_graph(root)

    for node in nodes:
        uid = str(id(node))

        dot.node(
            name=uid,
            label=f"{node.label}|{node.data:.4f}|{node.grad:.4f}",
            shape="record",
        )

        if node.op:
            dot.node(
                name=f"{uid}{node.op}",
                label=node.op,
            )
            dot.edge(f"{uid}{node.op}", uid)

    for node1, node2 in edges:
        uid1 = str(id(node1))
        uid2 = str(f"{id(node2)}{node2.op}")
        dot.edge(uid1, uid2)

    return dot
