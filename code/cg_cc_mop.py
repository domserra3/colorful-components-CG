from collections import defaultdict

import networkx as nx

from cg_cc import (
    ColorfulComponentsHeuristics,
    AlreadyContractedException,
    _clock,
)


class MOPReductionHeuristics(ColorfulComponentsHeuristics):

    def __init__(self, G, node_color, num_colors, verbose=False, debug=False):
        super().__init__(G=G, node_color=node_color, num_colors=num_colors,
                         verbose=verbose, debug=debug)
        self.mop_reduction = True
        self.locked_ids = set()
        self.locked_edges = set()
        self.reduction_frac = 0.75
        self.reduction_time = 0.0
        self._reduction_deadline = None
        self._reduction_use_time = False

    def _reduction_deadline_reached(self):
        return (self._reduction_deadline is not None
                and _clock(self._reduction_use_time) >= self._reduction_deadline)

    def _pre_greedy_reduction(self, CG, component_index, index_component, components_color,
                              node_component, edge_component, component_edge, edge_weights,
                              contractions, contractions_components,
                              start_time=None, time_limit=None, use_time=False):
        self.reduction_time = 0.0
        if not self.mop_reduction:
            return
        self._reduction_use_time = use_time
        if (start_time is not None and time_limit is not None
                and self.reduction_frac is not None and self.reduction_frac < 1.0):
            self._reduction_deadline = start_time + self.reduction_frac * time_limit
        else:
            self._reduction_deadline = None
        red_start = _clock(use_time)
        self._run_rule3_fixpoint(CG, component_index, index_component, components_color,
                                 node_component, edge_component, component_edge, edge_weights)
        self.reduction_time = round(_clock(use_time) - red_start, 4)

    def _run_rule3_fixpoint(self, CG, component_index, index_component, components_color,
                            node_component, edge_component, component_edge, edge_weights):
        changed = True
        while changed:
            if self._reduction_deadline_reached():
                break
            changed = False
            comp_adj = self._build_comp_adjacency(node_component)
            used = set()
            for seed in list(index_component.keys()):
                if self._reduction_deadline_reached():
                    break
                if seed in used or seed not in index_component:
                    continue
                Vp = self._grow_rule3_blob(seed, comp_adj, components_color, used)
                if not Vp or len(Vp) < 2:
                    continue
                used |= Vp
                self._contract_blob(Vp, CG, component_index, index_component, components_color,
                                    node_component, edge_component, component_edge, edge_weights)
                changed = True

    def _grow_rule3_blob(self, seed, comp_adj, components_color, used):
        Vp = {seed}
        color = components_color[seed]
        while True:
            if self._reduction_deadline_reached():
                return None
            if len(Vp) >= 2:
                cut = self._blob_cut(Vp, comp_adj)
                conn = self._blob_conn(Vp, comp_adj)
                if cut <= conn:
                    return Vp
            neighbors = set()
            for a in Vp:
                for b in comp_adj[a]:
                    if b in Vp or b in used:
                        continue
                    if components_color[b] & color:
                        continue
                    neighbors.add(b)
            if not neighbors:
                return None
            best, best_cut = None, float('inf')
            for c in neighbors:
                nc = self._blob_cut(Vp | {c}, comp_adj)
                if nc < best_cut:
                    best_cut, best = nc, c
            Vp.add(best)
            color |= components_color[best]
            if len(Vp) >= self.num_colors:
                cut = self._blob_cut(Vp, comp_adj)
                conn = self._blob_conn(Vp, comp_adj)
                return Vp if (len(Vp) >= 2 and cut <= conn) else None

    def _blob_cut(self, Vp, comp_adj):
        total = 0
        for a in Vp:
            for b, w in comp_adj[a].items():
                if b not in Vp:
                    total += w
        return total

    def _blob_conn(self, Vp, comp_adj):
        if len(Vp) < 2:
            return float('inf')
        H = nx.Graph()
        H.add_nodes_from(Vp)
        for a in Vp:
            for b, w in comp_adj[a].items():
                if b in Vp and a < b:
                    H.add_edge(a, b, weight=w)
        if H.number_of_nodes() < 2 or not nx.is_connected(H):
            return 0
        cut_value, _ = nx.stoer_wagner(H)
        return cut_value

    def _build_comp_adjacency(self, node_component):
        comp_adj = defaultdict(lambda: defaultdict(int))
        for u, v in self.G.edges():
            a = node_component[u]
            b = node_component[v]
            if a != b:
                comp_adj[a][b] += 1
                comp_adj[b][a] += 1
        return comp_adj

    def _contract_blob(self, Vp, CG, component_index, index_component, components_color,
                       node_component, edge_component, component_edge, edge_weights):
        blob_vertices = set()
        for cid in Vp:
            blob_vertices |= set(index_component[cid])

        while True:
            found = None
            for u in blob_vertices:
                if u not in CG:
                    continue
                cu = node_component[u]
                for v in CG._adj[u]:
                    if v in blob_vertices and node_component[v] != cu:
                        found = (min(u, v), max(u, v))
                        break
                if found:
                    break
            if found is None:
                break

            self.contract_edge(
                CG, found, component_index, index_component, components_color,
                node_component, edge_component, component_edge, edge_weights)

            self.locked_edges.add(found)

        self.locked_ids.add(node_component[next(iter(blob_vertices))])

    def singleton_decontraction(self, e, C_e, edge_component, index_component,
                                node_component, component_index, components_color,
                                component_edge, CG, edge_weights):
        if e in self.locked_edges or tuple(e) in self.locked_edges:
            raise AlreadyContractedException("locked by the MOP preprocessing")
        return super().singleton_decontraction(
            e, C_e, edge_component, index_component, node_component, component_index,
            components_color, component_edge, CG, edge_weights)

    def enhanced_singleton_decontraction(self, e, C_e, edge_component, index_component,
                                         node_component, component_index, components_color,
                                         component_edge, CG, edge_weights, use_wcc):
        if e in self.locked_edges or tuple(e) in self.locked_edges:
            raise AlreadyContractedException("locked by the MOP preprocessing")
        return super().enhanced_singleton_decontraction(
            e, C_e, edge_component, index_component, node_component, component_index,
            components_color, component_edge, CG, edge_weights, use_wcc)

    def block_decontraction(self, decontractions, decontractions_components, edge_component,
                            index_component, node_component, component_index, components_color,
                            component_edge, CG, edge_weights):
        if self.locked_edges:
            kept = [(d, dc) for d, dc in zip(decontractions, decontractions_components)
                    if d not in self.locked_edges]
            decontractions = [p[0] for p in kept]
            decontractions_components = [p[1] for p in kept]
        return super().block_decontraction(
            decontractions, decontractions_components, edge_component, index_component,
            node_component, component_index, components_color, component_edge, CG, edge_weights)
