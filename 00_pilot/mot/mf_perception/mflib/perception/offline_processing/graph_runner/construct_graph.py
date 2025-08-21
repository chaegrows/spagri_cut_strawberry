from mflib.perception.offline_processing.grain.grain_base import (
  GrainRegistry, 
  import_all_grains,
  find_grain_by_class_names)

from mflib.perception.offline_processing.procedure.procedure_base import (
  ProcedureRegistry, 
  import_all_procedures,
  find_procedure_by_class_names)
import networkx as nx
from matplotlib import pyplot as plt

# grain_instances = {
#     name: grain_cls.get_instance()
#     for name, grain_cls in GrainRegistry.all().items()
#     if name != 'GrainBase'  # Exclude the base class
#   }
#   procedure_instances = {
#     name: procedure_cls()
#     for name, procedure_cls in ProcedureRegistry.all().items()
#     if name != 'ProcedureBase'  # Exclude the base class
#   }

def save_graph_as_png(G: nx.DiGraph, path: str = "graph.png"):
  pos = nx.spring_layout(G, seed=42)  # seed for consistent layout
  labels = {node: str(node) for node in G.nodes}
  node_colors = ['skyblue' if G.nodes[n]['type'] == 'grain' else 'lightgreen' for n in G.nodes]

  plt.figure(figsize=(12, 8))
  nx.draw(G, pos, with_labels=True, labels=labels,
          node_color=node_colors, node_size=1800, font_size=8, edge_color="gray")
  plt.title("Grain-Procedure Graph")
  plt.tight_layout()
  plt.savefig(path, dpi=300)
  plt.close()

def main():
  import_all_grains()  # ensure all grains are loaded and registered
  import_all_procedures()  # ensure all procedures are loaded and registered

  G = nx.DiGraph()
  # 1. Add all grain nodes
  for grain_cls in GrainRegistry.all().values():
    grain_id = tuple(grain_cls.grain_class_names)
    G.add_node(grain_id, type='grain', cls=grain_cls)

  # 2. Add all procedure nodes + edges
  for procedure_cls in ProcedureRegistry.all().values():
    proc_id = tuple(procedure_cls.procedure_class_names)
    G.add_node(proc_id, type='procedure', cls=procedure_cls)

    # (1) Add input edges: grain → procedure
    for grain_names in procedure_cls.from_grain_names_tuple:
        G.add_edge(tuple(grain_names), proc_id)

    # (2) Add output edge: procedure → grain
    G.add_edge(proc_id, tuple(procedure_cls.output_grain_names))

  save_graph_as_png(G, "grain_procedure_graph.png")

  # get grain instance
  # grain_type = find_grain_by_class_names(('crop_type', 'strawberry', 'fruit'))
  # print(f"Grain instance: {grain_type}")

  # procedure_type = find_procedure_by_class_names(('deep_learning', 'data_preprocessing', 'example_procedure'))
  # print(f"Procedure instance: {procedure_type}")
  

if __name__ == "__main__":
  main()