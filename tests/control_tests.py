
import argparse
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import population_functions as pop
import relationship_functions as relationship
import disease_functions


def reset_flu(G):
    for n, d in G.nodes(data=True):
        d['flu_infection_status'] = 'S'
        d['flu_infection_step'] = 0
        d['flu_recovered_step'] = 0
        d['flu_ever_infected'] = False
        d.pop('flu_became_infectious_step', None)
        d.pop('flu_waning_days', None)
    G.graph['flu_total_infections'] = 0


def reset_hiv(G):
    for n, d in G.nodes(data=True):
        d['hiv_infection_status'] = 'S'
        d['hiv_infection_step'] = 0
        d['hiv_ever_infected'] = False
        d.pop('hiv_recovery_step', None)
    G.graph['hiv_total_infections'] = 0


def seed_only_hiv(G, num_seeds=20, min_age=18):
    adult_nodes = [n for n, a in G.nodes(data=True) if a.get('age', 0) >= min_age]
    if not adult_nodes:
        return
    seeds = random.sample(adult_nodes, min(num_seeds, len(adult_nodes)))
    for s in seeds:
        G.nodes[s]['hiv_infection_status'] = 'I'
        G.nodes[s]['hiv_infection_step'] = 0
        G.nodes[s]['hiv_ever_infected'] = True


def seed_only_flu(G, num_seeds=20):
    nodes = list(G.nodes)
    if not nodes:
        return
    seeds = random.sample(nodes, min(num_seeds, len(nodes)))
    for s in seeds:
        G.nodes[s]['flu_infection_status'] = 'I'
        G.nodes[s]['flu_infection_step'] = 0
        G.nodes[s]['flu_ever_infected'] = True
        G.graph['flu_total_infections'] = G.graph.get('flu_total_infections', 0) + 1


def prepare_population(population_size, age_distribution, male_fraction, indigenous_fraction, mode, hiv_seeds=20, flu_seeds=20):
    G = pop.generate_population(population_size, age_distribution, male_fraction, indigenous_fraction)
    if mode == 'hiv_only':
        reset_flu(G); reset_hiv(G); seed_only_hiv(G, hiv_seeds)
    elif mode == 'flu_only':
        reset_hiv(G); reset_flu(G); seed_only_flu(G, flu_seeds)
    elif mode == 'both':
        reset_hiv(G); reset_flu(G); seed_only_hiv(G, hiv_seeds); seed_only_flu(G, flu_seeds)
    else:
        raise ValueError("mode must be one of: hiv_only, flu_only, both")
    return G


def run_simulation(mode='both',
                   total_days=730,
                   relationship_formation_prob=0.1429,
                   homophily=0.7,
                   min_age=16,
                   breakup_probability=0.02,
                   hiv_p_mtf=1/1234,
                   hiv_p_ftm=1/2380,
                   flu_edge_beta=0.15,
                   flu_hiv_multiplier=2.0,
                   flu_comm_contacts=5,
                   flu_comm_beta=0.1,
                   rng_seed=42):
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    #bourke population
    population_size = 2340
    age_distribution = [
        (0, 4,   178/population_size), (5, 9,   172/population_size),
        (10, 14, 165/population_size), (15, 19, 134/population_size),
        (20, 24, 134/population_size), (25, 29, 150/population_size),
        (30, 34, 168/population_size), (35, 39, 148/population_size),
        (40, 44, 122/population_size), (45, 49, 128/population_size),
        (50, 54, 173/population_size), (55, 59, 164/population_size),
        (60, 64, 158/population_size), (65, 69, 102/population_size),
        (70, 74, 110/population_size), (75, 79,  67/population_size),
        (80, 84,  46/population_size), (85, 120, 34/population_size),
    ]
    male_fraction = 1164 / population_size
    indigenous_fraction = 708 / population_size

    G = prepare_population(population_size, age_distribution, male_fraction, indigenous_fraction, mode)

    infected_hiv_counts = [sum(1 for _, a in G.nodes(data=True) if a.get("hiv_infection_status") == "I")]
    infected_flu_counts = [sum(1 for _, a in G.nodes(data=True) if a.get("flu_infection_status") == "I")]
    new_hiv_cases_per_day = []
    new_flu_cases_per_day = []

    for day in range(total_days):
        if mode in ('both', 'flu_only'):
            disease_functions.progress_flu(G, day, incubation_period=4, infectious_period=7)

        relationship.start_relationship(G, relationship_formation_prob, homophily, day, min_age, None)

        if mode in ('both', 'flu_only'):
            flu_new_today = disease_functions.transmit_flu(
                G, day,
                edge_beta=flu_edge_beta,
                hiv_multiplier=flu_hiv_multiplier if mode == 'both' else 1.0,
                community_contacts=flu_comm_contacts,
                community_beta=flu_comm_beta
            )
            new_flu_cases_per_day.append(flu_new_today)

        if mode in ('both', 'hiv_only'):
            hiv_new_today = disease_functions.transmit_hiv(
                G,
                transmission_probability_ftm=hiv_p_ftm,
                transmission_probability_mtf=hiv_p_mtf,
                current_step=day
            )
            new_hiv_cases_per_day.append(hiv_new_today)

        relationship.breakup(G, breakup_probability=breakup_probability)
        pop.apply_recovery(G, day)

        infected_hiv_counts.append(sum(1 for _, a in G.nodes(data=True) if a.get("hiv_infection_status") == "I"))
        infected_flu_counts.append(sum(1 for _, a in G.nodes(data=True) if a.get("flu_infection_status") == "I"))

    deg = dict(G.degree()); nx.set_node_attributes(G, deg, "degree_now")
    nx.write_graphml(G, f"bourke_{mode}_final_{total_days}d.graphml")

    return {
        "G": G,
        "infected_hiv_counts": infected_hiv_counts,
        "infected_flu_counts": infected_flu_counts,
        "new_hiv_cases_per_day": new_hiv_cases_per_day,
        "new_flu_cases_per_day": new_flu_cases_per_day,
        "total_days": total_days,
        "population_size": G.number_of_nodes()
    }


def _series_to_array(series_list, expected_len):
    arr = []
    for s in series_list:
        if len(s) < expected_len:
            s = s + [s[-1] if s else 0]*(expected_len-len(s))
        elif len(s) > expected_len:
            s = s[:expected_len]
        arr.append(s)
    return np.asarray(arr, dtype=float)


def aggregate_runs(all_results, key, total_days, popN):
    if not all_results:
        return None
    series = [r[key] for r in all_results]
    expected_len = total_days + 1 if "counts" in key else total_days
    arr = _series_to_array(series, expected_len)
    mean = np.nanmean(arr, axis=0)
    low = np.nanpercentile(arr, 2.5, axis=0)
    high = np.nanpercentile(arr, 97.5, axis=0)
    return mean, low, high


def plot_with_ci(x, mean, low, high, title, ylabel):
    plt.figure(figsize=(9,4))
    plt.plot(x, mean, linestyle='-')
    plt.fill_between(x, low, high, alpha=0.2)
    plt.xlabel("Day"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


def save_csv_series(prefix, x, mean, low, high, units):
    import csv
    fn = f"{prefix}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", f"mean_{units}", f"ci2.5_{units}", f"ci97.5_{units}"])
        for xi, m, l, h in zip(x, mean, low, high):
            w.writerow([xi, m, l, h])
    return fn


def run_batch(mode, seeds, total_days=730, rng_seed=None):
    all_results = [run_simulation(mode=mode, total_days=total_days, rng_seed=s) for s in seeds]
    popN = all_results[0]["population_size"] if all_results else 0

    #prevalence & incidence aggregation
    hiv_prev = aggregate_runs(all_results, "infected_hiv_counts", total_days, popN)
    flu_prev = aggregate_runs(all_results, "infected_flu_counts", total_days, popN)
    hiv_inc  = aggregate_runs(all_results, "new_hiv_cases_per_day", total_days, popN)
    flu_inc  = aggregate_runs(all_results, "new_flu_cases_per_day", total_days, popN)

    #cumulative & per-1,000 // not really used
    hiv_cum_series, flu_cum_series, hiv_cum_1k_series, flu_cum_1k_series = [], [], [], []
    for r in all_results:
        hiv_cum = np.cumsum(r["new_hiv_cases_per_day"]) if r["new_hiv_cases_per_day"] else np.zeros(total_days)
        flu_cum = np.cumsum(r["new_flu_cases_per_day"]) if r["new_flu_cases_per_day"] else np.zeros(total_days)
        hiv_cum_series.append(hiv_cum.tolist()); flu_cum_series.append(flu_cum.tolist())
        hiv_cum_1k_series.append((hiv_cum / popN * 1000.0).tolist())
        flu_cum_1k_series.append((flu_cum / popN * 1000.0).tolist())

    hiv_cum   = aggregate_runs([{"cum": s} for s in hiv_cum_series], "cum", total_days, popN)
    flu_cum   = aggregate_runs([{"cum": s} for s in flu_cum_series], "cum", total_days, popN)
    hiv_cum_1k= aggregate_runs([{"cum": s} for s in hiv_cum_1k_series], "cum", total_days, popN)
    flu_cum_1k= aggregate_runs([{"cum": s} for s in flu_cum_1k_series], "cum", total_days, popN)

    #plots
    days_prev = list(range(0, total_days + 1))
    days_inc  = list(range(1, total_days + 1))
    if hiv_prev and np.any(hiv_prev[0]):
        plot_with_ci(days_prev, *hiv_prev, title=f"HIV Prevalence — mean & 95% CI ({mode}, n={len(seeds)})", ylabel="Number of HIV-infected")
    if flu_prev and np.any(flu_prev[0]):
        plot_with_ci(days_prev, *flu_prev, title=f"Influenza Prevalence — mean & 95% CI ({mode}, n={len(seeds)})", ylabel="Number of people Infected with Flu")
    if hiv_inc and np.any(hiv_inc[0]):
        plot_with_ci(days_inc, *hiv_inc, title=f"HIV Daily Incidence — mean & 95% CI ({mode}, n={len(seeds)})", ylabel="Number of new HIV cases")
    if flu_inc and np.any(flu_inc[0]):
        plot_with_ci(days_inc, *flu_inc, title=f"Influenza Daily Incidence — mean & 95% CI ({mode}, n={len(seeds)})", ylabel="New flu cases")
    if hiv_cum and np.any(hiv_cum[0]):
        plot_with_ci(days_inc, *hiv_cum, title=f"HIV Cumulative Incidence — mean & 95% CI ({mode}, n={len(seeds)})", ylabel="Number of people infected with HIV")
    if flu_cum and np.any(flu_cum[0]):
        plot_with_ci(days_inc, *flu_cum, title=f"Influenza Cumulative Incidence — mean & 95% CI ({mode}, n={len(seeds)})", ylabel="Number of people infected with Influenza")
    if hiv_cum_1k and np.any(hiv_cum_1k[0]):
        plot_with_ci(days_inc, *hiv_cum_1k, title=f"HIV Cumulative Incidence per 1,000 — mean & 95% CI ({mode}, n={len(seeds)})", ylabel="Per 1,000")
    if flu_cum_1k and np.any(flu_cum_1k[0]):
        plot_with_ci(days_inc, *flu_cum_1k, title=f"Influenza Cumulative Incidence per 1,000 — mean & 95% CI ({mode}, n={len(seeds)})", ylabel="Per 1,000")

    #CSV exports for figures
    saved = []
    if hiv_prev:   saved.append(save_csv_series(f"batch_{mode}_hiv_prevalence", days_prev, *hiv_prev, units="count"))
    if flu_prev:   saved.append(save_csv_series(f"batch_{mode}_flu_prevalence", days_prev, *flu_prev, units="count"))
    if hiv_inc:    saved.append(save_csv_series(f"batch_{mode}_hiv_incidence", days_inc, *hiv_inc, units="count"))
    if flu_inc:    saved.append(save_csv_series(f"batch_{mode}_flu_incidence", days_inc, *flu_inc, units="count"))
    if hiv_cum:    saved.append(save_csv_series(f"batch_{mode}_hiv_cumulative", days_inc, *hiv_cum, units="count"))
    if flu_cum:    saved.append(save_csv_series(f"batch_{mode}_flu_cumulative", days_inc, *flu_cum, units="count"))
    if hiv_cum_1k: saved.append(save_csv_series(f"batch_{mode}_hiv_cumulative_per1000", days_inc, *hiv_cum_1k, units="per1000"))
    if flu_cum_1k: saved.append(save_csv_series(f"batch_{mode}_flu_cumulative_per1000", days_inc, *flu_cum_1k, units="per1000"))
    return {"csv_files": saved, "n_runs": len(seeds), "population_size": popN}


def plot_results(mode, results):
    days = list(range(0, results["total_days"] + 1))
    popN = results["population_size"]
    if results["infected_hiv_counts"] and any(results["infected_hiv_counts"]):
        plt.figure(figsize=(8,4))
        plt.plot(days, results["infected_hiv_counts"], linestyle='-')
        plt.xlabel("Day"); plt.ylabel("Number of HIV-infected individuals")
        plt.title(f"HIV Prevalence over Time — {mode}"); plt.grid(True); plt.tight_layout(); plt.show()
        if results["new_hiv_cases_per_day"]:
            d = list(range(1, results["total_days"] + 1))
            cum = np.cumsum(results["new_hiv_cases_per_day"])
            per_1000 = (cum / popN) * 1000.0
            plt.figure(figsize=(9,4)); plt.plot(d, cum, linestyle='-'); plt.xlabel("Day"); plt.ylabel("People ever infected")
            plt.title(f"Cumulative HIV incidence — {mode}"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
            plt.figure(figsize=(9,4)); plt.plot(d, per_1000, linestyle='-'); plt.xlabel("Day"); plt.ylabel("Cumulative incidence per 1,000")
            plt.title(f"Cumulative HIV incidence per 1,000 — {mode}"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    if results["infected_flu_counts"] and any(results["infected_flu_counts"]):
        plt.figure(figsize=(8,4))
        plt.plot(days, results["infected_flu_counts"], linestyle='-')
        plt.xlabel("Day"); plt.ylabel("Number of Flu-infected individuals")
        plt.title(f"Influenza Prevalence over Time — {mode}"); plt.grid(True); plt.tight_layout(); plt.show()
        if results["new_flu_cases_per_day"]:
            d = list(range(1, results["total_days"] + 1))
            cum = np.cumsum(results["new_flu_cases_per_day"])
            per_1000 = (cum / popN) * 1000.0
            plt.figure(figsize=(9,4)); plt.plot(d, results["new_flu_cases_per_day"], linestyle='-')
            plt.xlabel("Day"); plt.ylabel("Number of new flu cases (incidence)"); plt.title(f"Daily influenza incidence — {mode}")
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
            plt.figure(figsize=(9,4)); plt.plot(d, cum, linestyle='-'); plt.xlabel("Day"); plt.ylabel("Number of people infected with HIV")
            plt.title(f"Cumulative influenza incidence — {mode}"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
            plt.figure(figsize=(9,4)); plt.plot(d, per_1000, linestyle='-'); plt.xlabel("Day"); plt.ylabel("Cumulative incidence per 1,000")
            plt.title(f"Cumulative influenza incidence per 1,000 — {mode}"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run Bourke simulation in control modes or batch mode.")
    parser.add_argument("--mode", choices=["hiv_only", "flu_only", "both"], default="both")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42, help="Single-run seed (ignored in --batch if --seeds provided)")
    parser.add_argument("--batch", action="store_true", help="Run batch (multi-seed) mode")
    parser.add_argument("--seeds", type=str, default="", help="Comma list or range like 0-99 for batch mode")
    args = parser.parse_args()

    if args.batch:
        if args.seeds:
            if "-" in args.seeds:
                a, b = args.seeds.split("-", 1)
                seeds = list(range(int(a), int(b) + 1))
            else:
                seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        else:
            seeds = list(range(0, 30))  # default n=30

        summary = run_batch(args.mode, seeds, total_days=args.days)
        print(f"Batch completed: mode={args.mode}, runs={summary['n_runs']}, population={summary['population_size']}")
        if summary["csv_files"]:
            print("CSV outputs:")
            for p in summary["csv_files"]:
                print(" -", p)
    else:
        results = run_simulation(mode=args.mode, total_days=args.days, rng_seed=args.seed)
        print(f"Mode: {args.mode}")
        print(f"Days: {args.days}")
        print(f"Nodes: {results['population_size']}")
        print(f"HIV total infections: {results['G'].graph.get('hiv_total_infections', 0)}")
        print(f"Flu total infections: {results['G'].graph.get('flu_total_infections', 0)}")
        print("Saving GraphML to:", f"bourke_{args.mode}_final_{args.days}d.graphml")
        plot_results(args.mode, results)


if __name__ == "__main__":
    main()
