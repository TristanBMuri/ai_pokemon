from typing import List, Dict, Optional, Set, Any
import uuid

class GauntletNode:
    """
    Represents a single node in the Gauntlet Graph.
    type: "GYM" (Mandatory), "EVENT" (Optional), "ITEM" (Optional)
    """
    TYPE_GYM = "GYM"     # Mandatory Trainer Fight
    TYPE_EVENT = "EVENT" # Optional Fight (e.g. Route Trainer / Mini Boss) or Encounter
    TYPE_ITEM = "ITEM"   # Optional Item Pickup
    
    def __init__(self, node_id: str, node_type: str, name: str, data: Dict[str, Any] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.name = name
        self.data = data or {} # Stores TrainerSpec, ItemSpec, etc.
        self.next_nodes: List[str] = [] # IDs of possible next nodes
        
    def add_edge(self, target_node_id: str):
        if target_node_id not in self.next_nodes:
            self.next_nodes.append(target_node_id)

class GauntletMap:
    """
    The Graph representing the Gauntlet.
    """
    def __init__(self):
        self.nodes: Dict[str, GauntletNode] = {}
        self.start_node_id: Optional[str] = None
        self.end_node_id: Optional[str] = None
        
    def add_node(self, node: GauntletNode):
        self.nodes[node.node_id] = node
        
    def set_start(self, node_id: str):
        self.start_node_id = node_id
        
    def get_node(self, node_id: str) -> Optional[GauntletNode]:
        return self.nodes.get(node_id)
        
    def get_successors(self, node_id: str) -> List[GauntletNode]:
        node = self.get_node(node_id)
        if not node: return []
        return [self.nodes[nid] for nid in node.next_nodes if nid in self.nodes]
