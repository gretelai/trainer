import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from gretel_trainer.relational.core import RelationalData


def draw(rel_data: RelationalData) -> None:
    pos = graphviz_layout(rel_data.graph, prog="dot")
    nx.draw(rel_data.graph, pos, with_labels=True, arrows=True)
    plt.show()
