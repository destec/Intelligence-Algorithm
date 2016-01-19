from igraph import *
network_data = '/Users/catalystx/Code/Python/Intelligence-Algorithm/community_detection_problem/128nodes/d16z5/network.dat';
g = Graph.Read_Edgelist(network_data);
g.delete_vertices(0)
print(g)
plot(g)
# g = Graph.Erdos_Renyi(50, 0.1)
# h = Graph.Full(5)
# g2 = g + h
# g2.add_edges([(0, 50), (1, 51)])
# mcliques = g2.maximal_cliques(4, 7)
# print mcliques
# # for(clique in m)
# print len(mcliques)
# group_markers = [(clique, "green") for clique in mcliques]
# # group_markers = [('black', 'gray')]
# print group_markers
# # group_markers = [("green", "yellow")]
# plot(g2, mark_groups=group_markers)
# import igraph
# from random import randint
#
# def _plot(g, membership=None):
#     layout = g.layout("kk")
#     visual_style = {}
#     visual_style["edge_color"] = "gray"
#     visual_style["vertex_size"] = 30
#     visual_style["layout"] = layout
#     visual_style["bbox"] = (1024, 768)
#     visual_style["margin"] = 40
#     visual_style["mark_groups"] = True
#     for vertex in g.vs():
#         vertex["label"] = vertex.index
#     if membership is not None:
#         colors = []
#         for i in range(0, max(membership)+1):
#             colors.append('%06X' % randint(0, 0xFFFFFF))
#         for vertex in g.vs():
#             vertex["color"] = str('#') + colors[membership[vertex.index]]
#         visual_style["vertex_color"] = g.vs["color"]
#     igraph.plot(g, **visual_style)
#
# if __name__ == "__main__":
#     karate = igraph.Nexus.get("karate")
#     cl = karate.community_fastgreedy()
#     membership = cl.as_clustering().membership
#     _plot(karate, membership)
