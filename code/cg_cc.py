import copy
import sys
import time
import heapq
import datetime
import networkx as nx
from itertools import combinations

WEIGHT_FUNCTIONS = ("wCC", "wCS", "wEC")


def _clock(use_time=False):
    if (sys.version_info.major == 3 and sys.version_info.minor == 6) or use_time or not hasattr(time, "thread_time"):
        return time.time()
    return time.thread_time()


def bit_count(n):
    return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')


class PriorityQueue:

    def __init__(self, elements):
        self.elements = elements
        heapq.heapify(elements)

    def pop(self):
        return heapq.heappop(self.elements)

    def push(self, item):
        heapq.heappush(self.elements, item)

    def __len__(self):
        return len(self.elements)


def compute_mec_value(index_component):
    tot = 0
    for i, C in index_component.items():
        tot += (len(C) * (len(C) - 1)) / 2
    return tot


class EmptyHeapException(Exception):
    def __init__(self):
        self.message = "EmptyHeapException"
        super().__init__(self.message)


class AlreadyContractedException(Exception):
    def __init__(self, mex=""):
        self.message = f"Components Already Contracted {mex}"
        super().__init__(self.message)


class ColorfulComponentsHeuristics:

    def __init__(self, G: nx.Graph, node_color: dict, num_colors, verbose=False, debug=False):
        self.G = G
        self.node_color = node_color
        self.verbose = verbose
        self.debug = debug
        self.num_colors = num_colors
        self.weight = None
        self.ec_lambda = 1.0

        for node in self.G.nodes:
            self.G.nodes[node]["Colors"] = (1 << self.node_color[node])

    def feasibility_check(self, components):
        connected = True
        color_feasible = True
        for C in components:
            if connected:
                G_C = nx.induced_subgraph(self.G, C)
                if not nx.is_connected(G_C):
                    connected = False
            if color_feasible:
                for v, u in combinations(C, 2):
                    if self.node_color[u] == self.node_color[v]:
                        color_feasible = False
                        break

            if connected == False and color_feasible == False:
                break
        return connected, color_feasible

    def init_priority_queue_from_edges(self, edge_list, CG, components_color, node_component, edge_weights, use_wcc):
        elements = []
        for e in edge_list:
            e0, e1 = e
            e = (min(e0, e1), max(e0, e1))
            elements.append((
                self._edge_key(e, edge_weights, use_wcc),
                frozenset(e)
            ))

        return PriorityQueue(elements=elements)

    def compute_shrinkable_colors(self, CG, e, components_color, node_component, index_component):
        id0 = node_component[e[0]]
        id1 = node_component[e[1]]
        C0 = index_component[id0]
        C1 = index_component[id1]
        color_union = components_color[id0] | components_color[id1]

        shrinkable_colors = 0

        visited_components = {id0, id1}

        adj = CG._adj
        for C in (C0, C1):
            for v0 in C:
                for v in adj[v0]:
                    idv = node_component[v]
                    if idv in visited_components:
                        continue
                    visited_components.add(idv)
                    cc = components_color[idv]
                    if not (cc & color_union):
                        shrinkable_colors |= cc

        return shrinkable_colors, color_union

    def init_priority_queue(self, CG: nx.Graph, components_color, node_component, edge_weights, use_wcc):
        if self.verbose:
            print("\nPriorityQueue Initialization :")
        elements = []
        for e0, e1 in CG.edges:
            e = (min(e0, e1), max(e0, e1))
            elements.append(
                (
                    self._edge_key(e, edge_weights, use_wcc),
                    frozenset(e)
                )
            )

        return PriorityQueue(elements=elements)

    def _edge_key(self, e, edge_weights, use_wcc):
        w = edge_weights[e]
        if self.weight == 'wEC':
            return (self.ec_lambda * w[2] - w[3], -bit_count(w[0]))
        return -bit_count(w[0]) if use_wcc else -(bit_count(w[0]) + bit_count(w[1]))

    def _compute_edge_weight(self, CG, e, components_color, node_component, index_component):
        shrinkable, color_union = self.compute_shrinkable_colors(
            CG, e, components_color, node_component, index_component)
        if self.weight == 'wEC':
            d = self._count_new_non_contractible(
                CG, e, components_color, node_component, index_component)
            m = CG[e[0]][e[1]].get('m', 1) if CG.has_edge(*e) else 1
            return (shrinkable, color_union, d, m)
        return (shrinkable, color_union)

    def _count_new_non_contractible(self, CG, e, components_color, node_component, index_component):
        id0 = node_component[e[0]]
        id1 = node_component[e[1]]
        C0 = index_component[id0]
        C1 = index_component[id1]
        cc0 = components_color[id0]
        cc1 = components_color[id1]
        adj = CG._adj
        deaths = 0
        seen = {id0, id1}
        for u in C0:
            for v in adj[u]:
                idv = node_component[v]
                if idv in seen:
                    continue
                seen.add(idv)
                ccv = components_color[idv]
                if not (ccv & cc0) and (ccv & cc1):
                    deaths += 1
        seen = {id0, id1}
        for u in C1:
            for v in adj[u]:
                idv = node_component[v]
                if idv in seen:
                    continue
                seen.add(idv)
                ccv = components_color[idv]
                if not (ccv & cc1) and (ccv & cc0):
                    deaths += 1
        return deaths

    def _pre_greedy_reduction(self, CG, component_index, index_component, components_color,
                              node_component, edge_component, component_edge, edge_weights,
                              contractions, contractions_components,
                              start_time=None, time_limit=None, use_time=False):
        return

    def edge_contraction_heuristic(self, start_time, time_limit, use_wcc, use_time=False):

        contractions = []
        contractions_components = []
        CG = self.G.copy()

        component_index = {}
        index_component = {}
        components_color = {}
        node_component = {}
        component_edge = {}
        edge_component = {}

        history = []

        for i, u in enumerate(CG.nodes()):
            CG.nodes[u]["Colors"] = (1 << self.node_color[u])
            C = frozenset([u])
            component_index[C] = i
            index_component[i] = C
            node_component[u] = i
            components_color[i] = (1 << self.node_color[u])
            component_edge[i] = []

        edge_weights = {}
        self._pre_greedy_reduction(CG, component_index, index_component, components_color,
                                   node_component, edge_component, component_edge, edge_weights,
                                   contractions, contractions_components,
                                   start_time=start_time, time_limit=time_limit, use_time=use_time)
        for e0, e1 in CG.edges:
            e = (min(e0, e1), max(e0, e1))
            if components_color[node_component[e0]] & components_color[node_component[e1]]:
                CG.remove_edge(*e)
            w = self._compute_edge_weight(CG, e, components_color, node_component, index_component)
            edge_weights[e] = w
        Q = self.init_priority_queue(CG, components_color, node_component, edge_weights, use_wcc=use_wcc)

        if self.verbose:
            print("Number of initial components {}".format(len(component_index)))
            print("Number of starting edges {}".format(self.G.number_of_edges()))
            print("Number of contractible edges {}".format(len(Q)))

        round = 0
        while len(Q) != 0:

            if self.verbose:
                print("\nRound {}".format(round))
                print(edge_weights)
                print()
            key_stored, efs = Q.pop()
            edge = tuple(efs)
            if not CG.has_edge(*edge):
                continue
            if node_component[edge[0]] == node_component[edge[1]]:
                CG.remove_edge(*edge)
                continue

            if components_color[node_component[edge[0]]] & components_color[node_component[edge[1]]]:
                CG.remove_edge(*edge)
                continue

            e = (min(*edge), max(*edge))
            if e not in edge_weights or key_stored != self._edge_key(e, edge_weights, use_wcc):
                continue
            C_i = index_component[node_component[e[0]]]
            C_j = index_component[node_component[e[1]]]
            CC_i = components_color[node_component[e[0]]]
            CC_j = components_color[node_component[e[1]]]

            edges_to_updates, to_remove = \
                self.contract_edge(CG, e, component_index, index_component,
                                                     components_color, node_component,
                                                    edge_component, component_edge,
                                                    edge_weights)

            for ue in edges_to_updates:
                if CG.has_edge(*ue):
                    Q.push((self._edge_key(ue, edge_weights, use_wcc), frozenset(ue)))

            contractions.append(e)
            contractions_components.append(((C_i, C_j), (CC_i, CC_j)))
            t_c = _clock(use_time)
            if t_c - start_time >= time_limit:
                break

            round += 1
            current_time = _clock(use_time) - start_time

            date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
            greedy_val_MEC = compute_mec_value(index_component)
            sol_MOP = self.compute_mop_value(index_component, node_component)
            history.append(f"{len(index_component)}:{greedy_val_MEC}:{sol_MOP}:{round}:{current_time}:{date_time};")
        return component_index, index_component, components_color, contractions, contractions_components, CG, \
               node_component, component_edge, edge_component, edge_weights, history

    def is_optimum(self, index_component, components_color, node_component, node_color, component_index):
        optimum = True
        components_left = []
        nodes = []
        for i, C in index_component.items():
            if bit_count(components_color[i]) != self.num_colors:
                components_left.append(C)
                nodes.extend(list(C))
        if len(components_left) != 0:
            G_ind = nx.induced_subgraph(self.G, nodes)
            components = list(nx.connected_components(G_ind))
            if len(components) != components_left:
                return False
            else:
                for C in components_left:
                    for u in C:
                        for v in self.G._adj[u]:
                            if v not in C:
                                if not (components_color[component_index[C]] >> node_color[v]) & 1:
                                    return False
        return optimum

    def compute_mop_value(self, index_component, node_component=None):
        if node_component is None:
            node_comp = {}
            for i, C in index_component.items():
                for u in C:
                    node_comp[u] = i
        else:
            node_comp = node_component
        sol_MOP = 0
        for u, v in self.G.edges():
            if node_comp[u] != node_comp[v]:
                sol_MOP += 1
        return sol_MOP

    def carousel_greedy(self, component_index, index_component, components_color, contractions,
                        contractions_components, component_edge, edge_component, CG: nx.Graph,
                        node_component, edge_weights, greedy_time, time_limit, node_color, use_wcc,
                        alpha=20, beta=40, esd=True, use_time=False):
        greedy_val = len(index_component)
        greedy_val_mec = compute_mec_value(index_component)
        greedy_val_mop = self.compute_mop_value(index_component, node_component)

        start_time = _clock(use_time)

        best_sol = (dict(index_component), dict(component_index), greedy_val, CG.copy())
        best_sol_mec = (dict(index_component), dict(component_index), greedy_val_mec, CG.copy())
        best_sol_mop = (dict(index_component), dict(component_index), greedy_val_mop, CG.copy())

        date_time_greedy = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')

        if self.verbose:
            print(f"ECH number of components: {best_sol[2]}")

        len_initial_sol = len(contractions_components)
        len_carousel_start = len_initial_sol - int(len_initial_sol * beta / 100)
        cs_contractions = contractions[:len_carousel_start]
        cs_contractions_components = contractions_components[:len_carousel_start]
        cs_decontractions = contractions[len_carousel_start:]
        cs_decontractions_components = contractions_components[len_carousel_start:]

        last_update = -1
        history = [f"{greedy_val}:{last_update}:{0.0}:{date_time_greedy};"]
        history_mec = [f"{greedy_val_mec}:{last_update}:{0.0}:{date_time_greedy};"]
        history_mop = [f"{greedy_val_mop}:{last_update}:{0.0}:{date_time_greedy};"]

        if self.verbose:
            print(f"Initial number of contractions |S|: {len_initial_sol}")
            print(f"Carousel start size: {len_carousel_start}")

        self.block_decontraction(decontractions=cs_decontractions, decontractions_components=cs_decontractions_components,
                                 edge_component=edge_component, index_component=index_component,
                                 node_component=node_component, component_index=component_index,
                                 components_color=components_color, component_edge=component_edge, CG=CG,
                                 edge_weights=edge_weights)

        index_component_complete = None
        tot_iteration = alpha * len_initial_sol
        for iteration in range(0, tot_iteration):
            t_c = _clock(use_time)
            if (t_c - start_time >= time_limit - greedy_time) or ((time_limit - greedy_time) < 0):
                break
            if self.debug:
                for idx_C in index_component:
                    CC = index_component[idx_C]
                    if bit_count(components_color[idx_C]) > len(CC):
                        raise RuntimeError(f"Component {idx_C} has more colors than nodes")
                    for u in CC:
                        if not (components_color[idx_C] >> self.node_color[u]) & 1:
                            raise RuntimeError(f"Color of {u} ({self.node_color[u]}) not in component {idx_C} ({components_color[idx_C]})")

            if len(cs_decontractions) > 1 and len(cs_contractions) > 0:

                e = cs_contractions[0]
                C_e = cs_contractions_components[0]
                edges_to_checks = set(component_edge[edge_component[e]])

                indexes_to_rem = []
                for idx, edge in enumerate(cs_contractions):
                    if edge == e:
                        continue
                    if edge in edges_to_checks:
                        indexes_to_rem.append((cs_contractions[idx], cs_contractions_components[idx]))
                for cs_e, cs_e_c in indexes_to_rem:
                    cs_contractions.remove(cs_e)
                    cs_contractions_components.remove(cs_e_c)

                del cs_contractions[0]
                del cs_contractions_components[0]
                try:
                    if not esd:
                        t_c = _clock(use_time)
                        if t_c - start_time >= time_limit - greedy_time:
                            break
                        self.singleton_decontraction(
                            e=e,
                            C_e = C_e,
                            edge_component=edge_component,
                            index_component=index_component,
                            node_component=node_component,
                            component_index=component_index,
                            components_color=components_color,
                            component_edge=component_edge,
                            CG=CG,
                            edge_weights=edge_weights)
                        t_c = _clock(use_time)
                        if t_c - start_time >= time_limit - greedy_time:
                            break
                    else:
                        t_c = _clock(use_time)
                        if t_c - start_time >= time_limit - greedy_time:
                            break
                        C, contractions, contractions_components, edge_weights = self.enhanced_singleton_decontraction(
                            e=e,
                            C_e = C_e,
                            edge_component=edge_component,
                            index_component=index_component,
                            node_component=node_component,
                            component_index=component_index,
                            components_color=components_color,
                            component_edge=component_edge,
                            CG=CG,
                            edge_weights=edge_weights,
                            use_wcc=use_wcc
                        )
                        cs_contractions += contractions
                        cs_contractions_components += contractions_components
                        t_c = _clock(use_time)
                        if t_c - start_time >= time_limit - greedy_time:
                            break
                except AlreadyContractedException:
                    continue

                if self.verbose:
                    print(f"Current number of components {len(index_component)}")
                    print(index_component)
                try:
                    t_c = _clock(use_time)
                    if t_c - start_time >= time_limit - greedy_time:
                        break
                    next = None
                    while (next == None) and (CG.number_of_edges() > 0):
                        next = self.next_contraction(CG=CG, component_index=component_index, index_component=index_component, components_color=components_color,
                                        node_component=node_component, edge_component=edge_component,
                                            component_edge=component_edge, contractions=cs_contractions, use_wcc=use_wcc,
                                            contractions_components= cs_contractions_components, edge_weights=edge_weights)

                    t_c = _clock(use_time)
                    if t_c - start_time >= time_limit - greedy_time:
                        break
                except EmptyHeapException:
                    pass
                current_mec_val = compute_mec_value(index_component)
                current_mop_val = self.compute_mop_value(index_component, node_component)

                if len(index_component) < best_sol[2]:
                    if self.verbose:
                        print(f"CG - UPGRADE {len(index_component)}")

                    best_sol = (dict(index_component), dict(component_index),len(index_component), CG.copy())

                    current_time = _clock(use_time) - start_time

                    date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                    last_update = iteration
                    history.append(f"{best_sol[2]}:{iteration}:{current_time}:{date_time};")
                if current_mec_val > best_sol_mec[2]:
                    best_sol_mec = (
                    dict(index_component), dict(component_index), current_mec_val, CG.copy())

                    current_time = _clock(use_time) - start_time

                    date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                    last_update = iteration
                    history_mec.append(f"{best_sol_mec[2]}:{iteration}:{current_time}:{date_time};")
                if current_mop_val < best_sol_mop[2]:
                    if self.verbose:
                        print(f"CG - UPGRADE {len(index_component)}")

                    best_sol_mop = (dict(index_component), dict(component_index),current_mop_val, CG.copy())

                    current_time = _clock(use_time) - start_time

                    date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                    last_update = iteration
                    history_mop.append(f"{best_sol_mop[2]}:{iteration}:{current_time}:{date_time};")
                if iteration % len_initial_sol == 0:
                    try:
                        t_c = _clock(use_time)
                        if t_c - start_time >= time_limit - greedy_time:
                            break

                        index_component_complete, component_index_complete, \
                        CG_complete, components_color_complete,\
                        node_component_complete = self.complete_solution(CG=CG.copy(),
                                                                             component_index=component_index.copy(),
                                                                             index_component=index_component.copy(),
                                                                             components_color=components_color.copy(),
                                                                             node_component=node_component.copy(),
                                                                             edge_component=edge_component.copy(),
                                                                             component_edge=copy.deepcopy(component_edge),
                                                                             contractions=cs_contractions.copy(),
                                                                             contractions_components=cs_contractions_components.copy(),
                                                                             edge_weights=edge_weights.copy(),
                                                                             use_wcc=use_wcc)

                        t_c = _clock(use_time)
                        if t_c - start_time >= time_limit - greedy_time:
                            break

                    except EmptyHeapException:
                        if self.verbose:
                            print("Empty Heap Exception ... Continue")
                            break

                    if index_component_complete != None:
                        current_mec_val = compute_mec_value(index_component_complete)
                        current_mop_val = self.compute_mop_value(index_component_complete, node_component_complete)
                        if len(index_component_complete) < best_sol[2]:
                            if self.verbose:
                                print(f"CG UPGRADE {len(index_component_complete)}")
                            t_c = _clock(use_time)
                            if t_c - start_time >= time_limit - greedy_time:
                                break
                            best_sol = (index_component_complete, component_index_complete,len(index_component_complete), CG_complete)

                            current_time = _clock(use_time) - start_time
                            date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                            last_update = iteration
                            history.append(f"{best_sol[2]}:{iteration}:{current_time}:{date_time};")
                            t_c = _clock(use_time)
                            if t_c - start_time >= time_limit - greedy_time:
                                break
                        if current_mop_val < best_sol_mop[2]:
                            if self.verbose:
                                print(f"CG - UPGRADE {len(index_component)}")

                            best_sol_mop = (index_component_complete, component_index_complete, current_mop_val, CG_complete)

                            current_time = _clock(use_time) - start_time

                            date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                            last_update = iteration
                            history_mop.append(f"{best_sol_mop[2]}:{iteration}:{current_time}:{date_time};")
                        if current_mec_val > best_sol_mec[2]:
                            if self.verbose:
                                print(f"CG UPGRADE {len(index_component_complete)}")
                            t_c = _clock(use_time)
                            if t_c - start_time >= time_limit - greedy_time:
                                break
                            best_sol_mec = (
                            index_component_complete, component_index_complete, current_mec_val,
                            CG_complete)

                            current_time = _clock(use_time) - start_time
                            date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                            last_update = iteration
                            history_mec.append(f"{best_sol_mec[2]}:{iteration}:{current_time}:{date_time};")
                            t_c = _clock(use_time)
                            if t_c - start_time >= time_limit - greedy_time:
                                break
                else:
                    if self.verbose:
                        print("Complete Solution Still None")
            t_c = _clock(use_time)
            if t_c - start_time >= time_limit - greedy_time:
                break

        return best_sol[0], best_sol[1], best_sol[2], history, last_update, tot_iteration, best_sol[3], len_initial_sol, best_sol_mec[0], best_sol_mec[1], best_sol_mec[2], history_mec, best_sol_mop[0], best_sol_mop[1], best_sol_mop[2], history_mop

    def complete_solution(self, CG, component_index, index_component, components_color, node_component, edge_component,
                                      component_edge, contractions, contractions_components, edge_weights, use_wcc):

        Q = self.init_priority_queue(CG, components_color, node_component, edge_weights, use_wcc=use_wcc)

        round = 0
        while len(Q) != 0:
            if self.verbose:
                print(f"Elements in the queue : {len(Q)}")
                print("\nRound {}\n".format(round))
            key_stored, efs = Q.pop()
            edge = tuple(efs)
            if not CG.has_edge(*edge):
                continue
            if node_component[edge[0]] == node_component[edge[1]]:
                CG.remove_edge(*edge)
                continue
            if components_color[node_component[edge[0]]] & components_color[node_component[edge[1]]]:
                CG.remove_edge(*edge)
                continue
            e = (min(*edge), max(*edge))
            if e not in edge_weights or key_stored != self._edge_key(e, edge_weights, use_wcc):
                continue

            C_i = index_component[node_component[e[0]]]
            C_j = index_component[node_component[e[1]]]
            CC_i = components_color[node_component[e[0]]]
            CC_j = components_color[node_component[e[1]]]
            edges_to_updates, to_remove = self.contract_edge(CG, e, component_index, index_component,
                                                                      components_color, node_component,
                                                                      edge_component=edge_component, component_edge=component_edge, edge_weights=edge_weights)

            for ue in edges_to_updates:
                if CG.has_edge(*ue):
                    Q.push((self._edge_key(ue, edge_weights, use_wcc), frozenset(ue)))

            contractions.append(e)
            contractions_components.append(((C_i, C_j), (CC_i, CC_j)))

            round += 1

        return index_component, component_index, CG, components_color ,node_component

    def next_contraction(self, CG, component_index, index_component, components_color, node_component, edge_component,
                       component_edge, contractions, contractions_components, edge_weights, use_wcc):

        e = None
        best_key = None
        for e0, e1 in CG.edges:
            edge = (min(e0, e1), max(e0, e1))
            k = self._edge_key(edge, edge_weights, use_wcc)
            if e is None or k <= best_key:
                e = edge
                best_key = k

        if e:
            if node_component[e[0]] == node_component[e[1]]:
                CG.remove_edge(*e)
                return None

            if components_color[node_component[e[0]]] & components_color[node_component[e[1]]]:
                CG.remove_edge(*e)
                return None
            CG.remove_edge(*e)
            C_i = index_component[node_component[e[0]]]
            C_j = index_component[node_component[e[1]]]
            CC_i = components_color[node_component[e[0]]]
            CC_j = components_color[node_component[e[1]]]
            edges_to_updates, to_remove = self.contract_edge(CG, e, component_index, index_component,
                                                                      components_color, node_component, edge_component=edge_component,
                                                                      component_edge=component_edge, edge_weights=edge_weights)

            contractions.append(e)
            contractions_components.append(((C_i, C_j), (CC_i, CC_j)))

        return e

    def block_decontraction(self, decontractions, decontractions_components, edge_component, index_component,
                            node_component, component_index,components_color, component_edge, CG, edge_weights):
        if self.verbose:
            print("\nDecontractions", decontractions)
        indices = list(range(len(decontractions_components) - 1, -1, -1))
        for enum in indices:
            Cis, CCis = decontractions_components[enum]
            C_1, C_2, = Cis
            CC_1, CC_2 = CCis
            e = decontractions[enum]

            if self.verbose:
                print(f"\nIteration {enum}: decontracting {e}\n")
                print("index_component", index_component)
                print("edge_component", edge_component)
                print("component_edge", component_edge)

            try:
                C_id = edge_component[e]
            except Exception as ex:
                raise ex

            if self.verbose:
                print(f"Component {C_id}")

            try:
                C = index_component[C_id]
            except Exception as ex:
                raise ex

            edges_to_updates = component_edge[C_id]
            try:
                edges_to_updates.remove(e)
            except ValueError as ex:
                raise ex

            del edge_component[e]
            del index_component[C_id]
            del component_index[C]
            del components_color[C_id]
            del component_edge[C_id]

            self.max_key += 1
            C_1_id = self.max_key
            index_component[C_1_id] = C_1
            components_color[C_1_id] = CC_1
            component_index[C_1] = C_1_id
            for v in C_1:
                node_component[v] = C_1_id

            self.max_key += 1
            C_2_id = self.max_key
            index_component[C_2_id] = C_2
            components_color[C_2_id] = CC_2
            component_index[C_2] = C_2_id
            for v in C_2:
                node_component[v] = C_2_id

            component_edge[C_1_id] = []
            component_edge[C_2_id] = []
            for e_to_update in edges_to_updates:
                if e_to_update[0] in C_1 and e_to_update[1] in C_1:
                    edge_component[e_to_update] = C_1_id
                    component_edge[C_1_id].append(e_to_update)
                elif e_to_update[0] in C_2 and e_to_update[1] in C_2:
                    edge_component[e_to_update] = C_2_id
                    component_edge[C_2_id].append(e_to_update)
                else:
                    raise RuntimeError(f"Edge {e_to_update} belongs to neither C_1 nor C_2")

            edges_new = []
            edges_beetwen = {}
            for v in C_1:
                for u in self.G._adj[v]:
                    id_u = node_component[u]
                    if u in C_1 or id_u == C_1_id:
                        continue
                    chiave = (min(C_1_id, id_u))
                    if chiave in edges_beetwen:
                        continue
                    e_new = (min(u, v), max(u, v))
                    edges_beetwen[chiave] = e_new
                    if components_color[C_1_id] & components_color[id_u]:
                        continue

                    if e_new not in edges_new:
                        edges_new.append(e_new)
                        if not CG.has_edge(*e_new):
                            CG.add_edge(*e_new)
                    for uu in index_component[id_u]:
                        if uu != u:
                            for vv in CG._adj[uu]:
                                id_vv = node_component[vv]
                                e_new = (min(uu, vv), max(uu, vv))
                                chiave = (min(id_vv, id_u))
                                if chiave in edges_beetwen:
                                    continue
                                edges_beetwen[chiave] = e_new
                                if id_vv != id_u and id_vv != C_1_id:
                                    if e_new not in edges_new:
                                        edges_new.append(e_new)
            for v in C_2:
                for u in self.G._adj[v]:
                    if u in C_2:
                        continue
                    id_u = node_component[u]
                    if u in C_1 or id_u == C_2_id:
                        continue
                    chiave = (min(C_1_id, id_u))
                    if chiave in edges_beetwen:
                        continue
                    e_new = (min(u, v), max(u, v))
                    edges_beetwen[chiave] = e_new
                    if components_color[node_component[v]] & components_color[node_component[u]]:
                        continue
                    if e_new not in edges_new:
                        edges_new.append(e_new)
                        if not CG.has_edge(*e_new):
                            CG.add_edge(*e_new)
                    for uu in index_component[id_u]:
                        if uu != u:
                            for vv in CG._adj[uu]:
                                id_vv = node_component[vv]
                                e_new = (min(uu, vv), max(uu, vv))
                                chiave = (min(id_vv, id_u))
                                if chiave in edges_beetwen:
                                    continue
                                edges_beetwen[chiave] = e_new
                                if id_vv != id_u and id_vv != C_2_id:
                                    if e_new not in edges_new:
                                        edges_new.append(e_new)
            CG.add_edge(*e)

            for e_new in edges_new:
                w = self._compute_edge_weight(CG, e_new, components_color, node_component, index_component)
                edge_weights[e_new] = w
            w = self._compute_edge_weight(CG, e, components_color, node_component, index_component)
            edge_weights[e] = w

    def enhanced_singleton_decontraction(self, e, C_e, edge_component, index_component, node_component,
                                             component_index, components_color, component_edge, CG, edge_weights,
                                             use_wcc):

        Cis, CCis = C_e
        C_1, C_2 = Cis
        CC_1, CC_2 = CCis
        try:
            C_id = edge_component[e]
            if self.verbose:
                print(f"edge {e} in component {C_id}")
        except KeyError as ex:
            raise ex

        try:
            C = index_component[C_id]
        except KeyError as ex:
            print(C_id)
            print(index_component)
            raise ex

        edges_to_updates = component_edge[C_id]
        try:
            edges_to_updates.remove(e)
        except ValueError as ex:
            raise ex

        for edge in edges_to_updates:
            del edge_component[edge]
        del component_edge[C_id]

        del edge_component[e]

        del index_component[C_id]
        del component_index[C]
        del components_color[C_id]

        for u in C:
            self.max_key += 1
            maxx = self.max_key
            C_u = frozenset([u])
            component_index[C_u] = maxx
            index_component[maxx] = C_u
            node_component[u] = maxx
            components_color[maxx] = (1 << self.node_color[u])
            component_edge[maxx] = []
        to_update = []
        edges_betweens = {}
        n_nodes = CG.number_of_nodes()
        d = (CG.number_of_edges() * 2) / (n_nodes * (n_nodes - 1))
        for u_ in C:
            id_u_ = node_component[u_]
            for v_ in self.G._adj[u_]:
                to_dd = (min(u_, v_), max(u_, v_))
                if to_dd in to_update:
                    continue
                id_v_ = node_component[v_]
                chiave = (min(id_u_, id_v_), max(id_u_, id_v_))
                if chiave in edges_betweens:
                    continue
                edges_betweens[chiave]= (u_, v_)
                if components_color[id_v_] & components_color[id_u_]:
                    continue
                if not CG.has_edge(*to_dd):
                    CG.add_edge(*to_dd)
                if to_dd not in to_update:
                    to_update.append(to_dd)
                for k_ in self.G._adj[v_]:
                    if k_ in C or k_ == u_:
                        continue
                    id_k_ = node_component[k_]
                    chiave = (min(id_k_, id_v_), max(id_k_, id_v_))
                    if chiave in edges_betweens:
                        continue
                    to_dd = (min(v_, k_), max(v_, k_))
                    edges_betweens[chiave] = to_dd
                    if to_dd in to_update:
                        continue

                    if components_color[id_v_] & components_color[id_k_]:
                        continue

                    if not CG.has_edge(*to_dd):
                        CG.add_edge(*to_dd)
                    if to_dd not in to_update:
                        if d < 0.2:
                            to_update.append(to_dd)

        edges_added_out_component = []
        for edge_to_keep in to_update:
            edge_weights[edge_to_keep] = self._compute_edge_weight(CG, edge_to_keep, components_color,
                                                                        node_component, index_component)
            if not (edge_to_keep[0] in C and edge_to_keep[1] in C) and (edge_to_keep[0] in C or edge_to_keep[1] in C):
                edges_added_out_component.append(edge_to_keep)

        component_index, index_component, components_color, contractions, contractions_components, component_edge_beans, \
        edge_bean_component, CG, node_component, edge_weights = self.esd_contraction_step(CG=CG, edges_added_in_component = [],
            edges_added_out_component= edges_added_out_component, component_index=component_index,
                          index_component=index_component, components_color=components_color,
            node_component=node_component, edge_component=edge_component,
                          component_edge=component_edge, edge_weights=edge_weights, use_wcc=use_wcc)

        return C, contractions, contractions_components, edge_weights
    def singleton_decontraction(self, e, C_e, edge_component, index_component, node_component,
                                             component_index, components_color, component_edge, CG, edge_weights):

        Cis, CCis = C_e
        C_1, C_2 = Cis
        CC_1, CC_2 = CCis
        try:
            C_id = edge_component[e]
            if self.verbose:
                print(f"edge {e} in component {C_id}")
        except KeyError as ex:
            raise ex

        try:
            C = index_component[C_id]
        except KeyError as ex:
            print(C_id)
            print(index_component)
            raise ex

        edges_to_updates = component_edge[C_id]
        try:
            edges_to_updates.remove(e)
        except ValueError as ex:
            raise ex

        for edge in edges_to_updates:
            del edge_component[edge]
        del component_edge[C_id]

        del edge_component[e]

        del index_component[C_id]
        del component_index[C]
        del components_color[C_id]

        for u in C:
            self.max_key += 1
            maxx = self.max_key
            C_u = frozenset([u])
            component_index[C_u] = maxx
            index_component[maxx] = C_u
            node_component[u] = maxx
            components_color[maxx] = (1 << self.node_color[u])
            component_edge[maxx] = []

        to_update = []
        edges_betweens = {}
        for u_ in C:
            id_u_ = node_component[u_]
            for v_ in self.G._adj[u_]:
                to_dd = (min(u_, v_), max(u_, v_))
                if to_dd in to_update:
                    continue
                id_v_ = node_component[v_]
                chiave = (min(id_u_, id_v_), max(id_u_, id_v_))
                if chiave in edges_betweens:
                    continue
                edges_betweens[chiave] = (u_, v_)
                if components_color[id_v_] & components_color[id_u_]:
                    continue
                if not CG.has_edge(*to_dd):
                    CG.add_edge(*to_dd)
                if to_dd not in to_update:
                    to_update.append(to_dd)
                for k_ in self.G._adj[v_]:
                    if k_ in C or k_ == u_:
                        continue
                    id_k_ = node_component[k_]
                    chiave = (min(id_k_, id_v_), max(id_k_, id_v_))
                    if chiave in edges_betweens:
                        continue
                    to_dd = (min(v_, k_), max(v_, k_))
                    edges_betweens[chiave] = to_dd
                    if to_dd in to_update:
                        continue
                    if components_color[id_v_] & components_color[id_k_]:
                        continue

                    if not CG.has_edge(*to_dd):
                        CG.add_edge(*to_dd)
                    if to_dd not in to_update:
                        to_update.append(to_dd)

        for edge_to_keep in to_update:
            edge_weights[edge_to_keep] = self._compute_edge_weight(CG, edge_to_keep, components_color,
                                                                        node_component, index_component)

    def esd_contraction_step(self, CG, edges_added_in_component, edges_added_out_component, component_index, index_component,
        components_color, node_component, edge_component, component_edge, edge_weights, use_wcc):

        sol_val = len(component_index)

        contractions = []
        contractions_components = []
        Q = self.init_priority_queue_from_edges(edge_list= edges_added_out_component, CG=CG,
                                        components_color= components_color, node_component=node_component, edge_weights=edge_weights, use_wcc=use_wcc)

        if self.verbose:
            print("Number of initial components {}".format(sol_val))
            print("Number of starting edges {}".format(self.G.number_of_edges()))
            print("Number of contractible edges {}".format(len(Q)))
        round = 0
        while len(Q) != 0:

            if self.verbose:
                print(f"Elements in the queue : {len(Q)}")
                print("\nRound {}\n".format(round))
            edge = tuple(Q.pop()[1])
            e = (min(*edge), max(*edge))

            if node_component[edge[0]] == node_component[edge[1]]:
                if CG.has_edge(*edge):
                    CG.remove_edge(*edge)
                if e in edges_added_out_component:
                    edges_added_out_component.remove(e)
                continue

            if components_color[node_component[edge[0]]] & components_color[node_component[edge[1]]]:
                if CG.has_edge(*edge):
                    CG.remove_edge(*edge)
                if e in edges_added_out_component:
                    edges_added_out_component.remove(e)
                continue

            C_i = index_component[node_component[e[0]]]
            C_j = index_component[node_component[e[1]]]
            CC_i = components_color[node_component[e[0]]]
            CC_j = components_color[node_component[e[1]]]

            edges_to_updates, to_remove = self.contract_edge(CG, e, component_index, index_component,
                                            components_color, node_component, edge_component, component_edge, edge_weights)

            edges_added_out_component.remove(e)
            for ee in to_remove:
                if ee in edges_added_out_component:
                    edges_added_out_component.remove(ee)
            contractions.append(e)
            contractions_components.append(((C_i, C_j), (CC_i, CC_j)))

            Q = self.init_priority_queue_from_edges(edge_list=edges_added_out_component, CG=CG,
                                                                    components_color=components_color,
                                                                    node_component=node_component,edge_weights=edge_weights, use_wcc=use_wcc)

            round += 1

        return component_index, index_component, components_color, contractions, contractions_components, component_edge, edge_component, CG, node_component, edge_weights

    def contract_edge(self, CG: nx.Graph, e, component_index, index_component, components_color, node_component,
                               edge_component, component_edge, edge_weights):
        if self.verbose:
            print("\nLet's contract {}".format(e))
            print(CG.nodes[e[0]])
            print(CG.nodes[e[1]])
        i_ = node_component[e[0]]
        j_ = node_component[e[1]]

        i = min(i_, j_)
        j = max(i_, j_)

        Ci = index_component[i]
        Cj = index_component[j]

        Ci_new = Ci.union(Cj)

        if self.verbose:
            print(f"Indices of the endpoint components i:{i} j:{j}")
            print(f"Component {Ci} of {e[0]}")
            print(f"Component {Cj} of {e[1]}")
            print(f"New component {Ci_new}")
        del component_index[Cj]
        del index_component[j]
        del component_index[Ci]
        component_index[Ci_new] = i
        index_component[i] = Ci_new

        edges = component_edge[j]
        del component_edge[j]
        for edge in edges:
            edge_component[edge] = i
        component_edge[i].extend(edges)

        edge_component[e] = i
        component_edge[i].append(e)

        components_color[i] = components_color[i] | components_color[j]

        del components_color[j]

        for u in Cj:
            node_component[u] = i

        edges_to_remove = []
        edges_betweens = {}
        for u_ in Ci_new:
            for v_ in CG._adj[u_]:
                if v_ not in Ci_new:

                    id_v_ = node_component[v_]
                    if (i, id_v_) not in edges_betweens:
                        edges_betweens[(i, id_v_)] = [(min(u_, v_), max(u_, v_))]
                    else:
                        edges_betweens[(i, id_v_)].append((min(u_, v_), max(u_, v_)))
        for k, edges_between in edges_betweens.items():
            if len(edges_between) > 1:
                edge_to_keep = edges_between[0] if edges_between else None
                if edge_to_keep and len(edges_between) > 1:
                    if self.weight == 'wEC':
                        total_m = 0
                        for a_, b_ in edges_between:
                            if CG.has_edge(a_, b_):
                                total_m += CG[a_][b_].get('m', 1)
                        CG[edge_to_keep[0]][edge_to_keep[1]]['m'] = total_m
                    edges_to_remove = set(edges_between)
                    edges_to_remove.difference_update({edge_to_keep, e})
                    CG.remove_edges_from(edges_to_remove)

        n_nodes = CG.number_of_nodes()
        d = (CG.number_of_edges() * 2)/(n_nodes*(n_nodes-1))
        edges_to_updates = []
        seen_updates = set()
        to_remove = []
        for ui in Ci_new:
            for u in CG._adj[ui]:
                id_u = node_component[u]
                if id_u == i:
                    continue
                if components_color[id_u] & components_color[i]:
                    to_remove.append((min((ui, u)), max((ui, u))))
                else:
                    to_up = (min(ui, u), max((ui, u)))
                    if to_up not in seen_updates:
                        seen_updates.add(to_up)
                        edge_weights[to_up] = self._compute_edge_weight(CG, to_up, components_color,
                                                                             node_component, index_component)
                        edges_to_updates.append(to_up)
                if d < 0.2:
                    for uu in CG._adj[u]:
                        id_uu = node_component[uu]
                        if uu != e[1] and uu != e[0] and id_uu != i:
                            to_up = (min(uu, u), max((uu, u)))
                            if to_up not in seen_updates:
                                seen_updates.add(to_up)
                                edge_weights[to_up] = self._compute_edge_weight(CG, to_up, components_color,
                                                                                     node_component, index_component)
                                edges_to_updates.append(to_up)
        CG.remove_edges_from(to_remove)
        if CG.has_edge(*e):
            CG.remove_edge(*e)

        return edges_to_updates, to_remove

