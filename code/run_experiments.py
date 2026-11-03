import argparse
import csv
import datetime
import os
import platform
import subprocess
import sys

import networkx as nx

from instance_reader import read_instance
from cg_cc import ColorfulComponentsHeuristics, compute_mec_value, _clock
from cg_cc_mop import MOPReductionHeuristics

ROOT = os.path.dirname(os.path.abspath(__file__))
INSTANCE_SETS = {
    "BHK": os.path.join(ROOT, "../data", "BHK"),
    "RANDOM": os.path.join(ROOT, "../data", "RANDOM"),
    "DIMACS": os.path.join(ROOT, "../data", "DIMACS"),
}
DEFAULT_WEIGHT = {"MCC": "wCC", "MEC": "wCS", "MOP": "wEC"}
MIN_COMPONENT_SIZE = 8

COLUMNS = ["Date", "Processor", "Set", "Instance", "Component", "n", "m", "Colors", "n_c", "m_c",
           "Problem", "Weight", "Decontraction", "alpha", "beta", "lambda", "Time_Limit",
           "ECH_Timestamp", "ECH_Connected", "ECH_Colorful", "Reduction_Time", "ECH_Time",
           "ECH_MCC", "ECH_MEC", "ECH_MOP",
           "CGCC_Timestamp", "|S|", "CGCC_Iterations", "CGCC_Last_Update",
           "CGCC_Connected", "CGCC_Colorful", "CGCC_Time", "CGCC_MCC", "CGCC_MEC", "CGCC_MOP"]


def processor_info():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECH and CG-CC on a set of instances.")
    parser.add_argument("--instances", required=True, choices=list(INSTANCE_SETS),
                        help="Set of instances: BHK, RANDOM or DIMACS.")
    parser.add_argument("--problem", required=True, choices=["MCC", "MEC", "MOP"],
                        help="Problem to solve.")
    parser.add_argument("--decontraction", required=True, choices=["SD", "ESD"],
                        help="Decontraction operator of CG-CC: SD or ESD.")
    parser.add_argument("--weight", choices=["wCC", "wCS", "wEC"],
                        help="Edge selection strategy (default: wCC for MCC, wCS for MEC, wEC for MOP).")
    parser.add_argument("--alpha", type=int, required=True, help="CG-CC parameter alpha.")
    parser.add_argument("--beta", type=int, required=True,
                        help="CG-CC parameter beta, as a percentage (e.g. 30 for beta = 0.3).")
    parser.add_argument("--lambda", dest="ec_lambda", type=float, default=1.0,
                        help="Parameter lambda of wEC (default: 1.0). Ignored by wCC and wCS.")
    parser.add_argument("--time_limit", type=int, default=3600, help="Time limit in seconds (default: 3600).")
    parser.add_argument("--reduction_frac", type=float, default=0.75,
                        help="MOP only: maximum fraction of the time limit for the preprocessing (default: 0.75).")
    parser.add_argument("--instance", nargs="+",
                        help="Run only the given instance files of the set (file names, e.g. BB11001C1.graph).")
    parser.add_argument("--output", default=os.path.join(ROOT, "results"),
                        help="Output folder (default: results).")
    args = parser.parse_args()
    if args.weight is None:
        args.weight = DEFAULT_WEIGHT[args.problem]
    return args


def main():
    args = parse_args()
    use_wcc = args.weight == "wCC"
    esd = args.decontraction == "ESD"
    ec_lambda = args.ec_lambda if args.weight == "wEC" else None

    run_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    os.makedirs(args.output, exist_ok=True)
    results_file_path = os.path.join(
        args.output, f"{args.instances}_{args.problem}_{args.weight}_{args.decontraction}_{run_date}.csv")
    print(f"Results: {results_file_path}")

    instance_folder = INSTANCE_SETS[args.instances]
    files = sorted(f for f in os.listdir(instance_folder) if f.endswith(".graph"))
    if args.instance:
        missing = set(args.instance) - set(files)
        if missing:
            sys.exit(f"Instances not found in {instance_folder}: {sorted(missing)}")
        files = [f for f in files if f in args.instance]
    processor = processor_info()

    with open(results_file_path, "w", newline="") as file_out:
        writer = csv.writer(file_out)
        writer.writerow(COLUMNS)
        file_out.flush()

        for file in files:
            G, num_colors, node_color = read_instance(os.path.join(instance_folder, file))
            n = G.number_of_nodes()
            m = G.number_of_edges()
            for idx, c in enumerate(nx.connected_components(G)):
                if len(c) < MIN_COMPONENT_SIZE:
                    continue
                print(f"Solving instance {file}, component {idx}")
                G_C = nx.Graph(nx.induced_subgraph(G, c))
                n_c = G_C.number_of_nodes()
                m_c = G_C.number_of_edges()
                print(f"Number of nodes in the component {n_c}")

                if args.problem == "MOP":
                    CCH = MOPReductionHeuristics(G=G_C, node_color=node_color, num_colors=num_colors)
                    CCH.reduction_frac = args.reduction_frac
                else:
                    CCH = ColorfulComponentsHeuristics(G=G_C, node_color=node_color, num_colors=num_colors)
                CCH.weight = args.weight
                CCH.ec_lambda = args.ec_lambda

                ech_timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                start_time = _clock()
                component_index, index_component, components_color, contractions, contractions_components, \
                    CG, node_component, component_edge, edge_component, edge_weights, history_G = \
                    CCH.edge_contraction_heuristic(start_time=start_time, time_limit=args.time_limit, use_wcc=use_wcc)
                greedy_time = round(_clock() - start_time, 4)
                reduction_time = round(getattr(CCH, "reduction_time", 0.0), 4)
                ech_time = max(0.0, round(greedy_time - reduction_time, 4))
                ech_mcc = len(index_component)
                ech_mec = compute_mec_value(index_component)
                ech_mop = CCH.compute_mop_value(index_component)
                connected, colorful = CCH.feasibility_check([C for C in component_index.keys()])
                if not (connected and colorful):
                    raise RuntimeError(f"Infeasible ECH solution on {file}, component {idx}")
                print(f"ECH MCC {ech_mcc} --- ECH Time {ech_time} --- Feasible {connected} - {colorful}")

                opt = CCH.is_optimum(index_component=index_component, components_color=components_color,
                                     node_component=node_component, node_color=node_color,
                                     component_index=component_index)
                if ech_mcc > 1 and not opt:
                    cg_timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                    CCH.max_key = max(list(index_component.keys()))
                    start_time_CG = _clock()
                    index_component_complete, component_index_complete, cg_mcc, \
                        history, last_update, tot_iteration, _, len_initial_sol, \
                        index_component_complete_mec, component_index_complete_mec, cg_mec_value, history_mec, \
                        index_component_complete_mop, component_index_complete_mop, cg_mop, history_mop = \
                        CCH.carousel_greedy(component_index, index_component, components_color, contractions,
                                            contractions_components, component_edge, edge_component, CG,
                                            node_component, node_color=node_color, greedy_time=greedy_time,
                                            time_limit=args.time_limit, alpha=args.alpha, beta=args.beta,
                                            esd=esd, edge_weights=edge_weights, use_wcc=use_wcc)
                    cg_time = round(_clock() - start_time_CG, 2)

                    cg_connected, cg_colorful = CCH.feasibility_check([C for C in component_index_complete.keys()])
                    if not (cg_connected and cg_colorful):
                        raise RuntimeError(f"Infeasible CG-CC solution on {file}, component {idx}")
                    print(f"CG-CC MCC {cg_mcc} --- CG-CC Time {cg_time} --- Feasible {cg_connected} - {cg_colorful}")
                    cg_mec = compute_mec_value(index_component_complete_mec)

                else:
                    cg_timestamp = len_initial_sol = tot_iteration = last_update = "NA"
                    cg_connected = cg_colorful = cg_time = cg_mcc = cg_mec = cg_mop = "NA"

                writer.writerow([run_date, processor, args.instances, file, idx, n, m, num_colors, n_c, m_c,
                                 args.problem, args.weight, args.decontraction, args.alpha, args.beta, ec_lambda,
                                 args.time_limit,
                                 ech_timestamp, connected, colorful, reduction_time, ech_time,
                                 ech_mcc, ech_mec, ech_mop,
                                 cg_timestamp, len_initial_sol, tot_iteration, last_update,
                                 cg_connected, cg_colorful, cg_time, cg_mcc, cg_mec, cg_mop])
                file_out.flush()
                sys.stdout.flush()


if __name__ == "__main__":
    main()
