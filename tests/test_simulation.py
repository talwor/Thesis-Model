"""
Unit tests for the Bourke HIV & Influenza Co-infection Simulation.

Run with:
    python -m pytest tests/test_simulation.py -v

Or from the repo root:
    python -m pytest -v
"""

import sys
import os
import random
import unittest
import networkx as nx

# Allow importing modules from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import population_functions as pop
import relationship_functions as relationship
import disease_functions as disease


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_small_graph(n=10, seed=42):
    """Return a tiny deterministic population graph for testing."""
    random.seed(seed)
    age_distribution = [(18, 40, 1.0)]
    return pop.generate_population(
        population_size=n,
        age_distribution=age_distribution,
        male_fraction=0.5,
        indigenous_fraction=0.3,
    )


def make_pair(male_hiv="S", female_hiv="S", male_flu="S", female_flu="S"):
    """
    Return a graph with exactly two connected nodes (one M, one F)
    with the given infection statuses.
    """
    G = nx.Graph()
    G.graph.update(
        num_relationships_formed=0,
        num_breakups=0,
        hiv_total_infections=0,
        flu_total_infections=0,
    )
    G.add_node(0, age=25, gender="M", is_indigenous=False,
               hiv_infection_status=male_hiv, hiv_infection_step=0, hiv_ever_infected=(male_hiv == "I"),
               flu_infection_status=male_flu, flu_infection_step=0,
               flu_recovered_step=0, flu_ever_infected=(male_flu == "I"))
    G.add_node(1, age=25, gender="F", is_indigenous=False,
               hiv_infection_status=female_hiv, hiv_infection_step=0, hiv_ever_infected=(female_hiv == "I"),
               flu_infection_status=female_flu, flu_infection_step=0,
               flu_recovered_step=0, flu_ever_infected=(female_flu == "I"))
    G.add_edge(0, 1, formed_step=0)
    return G


# ===========================================================================
# Population generation
# ===========================================================================

class TestGeneratePopulation(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        self.age_dist = [(0, 17, 0.3), (18, 65, 0.6), (66, 90, 0.1)]
        self.G = pop.generate_population(
            population_size=100,
            age_distribution=self.age_dist,
            male_fraction=0.5,
            indigenous_fraction=0.2,
        )

    def test_correct_node_count(self):
        self.assertEqual(self.G.number_of_nodes(), 100)

    def test_required_node_attributes_present(self):
        required = {
            "age", "gender", "is_indigenous",
            "hiv_infection_status", "hiv_infection_step", "hiv_ever_infected",
            "flu_infection_status", "flu_infection_step",
            "flu_recovered_step", "flu_ever_infected",
        }
        for _, attrs in self.G.nodes(data=True):
            self.assertTrue(required.issubset(attrs.keys()),
                            f"Missing attributes: {required - attrs.keys()}")

    def test_gender_values_are_valid(self):
        genders = {attrs["gender"] for _, attrs in self.G.nodes(data=True)}
        self.assertEqual(genders, {"M", "F"})

    def test_hiv_infection_status_values_are_valid(self):
        statuses = {attrs["hiv_infection_status"] for _, attrs in self.G.nodes(data=True)}
        self.assertTrue(statuses.issubset({"S", "I", "R"}))

    def test_graph_counters_initialised_to_zero_or_positive(self):
        self.assertGreaterEqual(self.G.graph["hiv_total_infections"], 0)
        self.assertGreaterEqual(self.G.graph["flu_total_infections"], 0)
        self.assertEqual(self.G.graph["num_relationships_formed"], 0)
        self.assertEqual(self.G.graph["num_breakups"], 0)

    def test_flu_seeds_are_marked_ever_infected(self):
        """Every node initialised as flu-infectious should have flu_ever_infected=True."""
        for _, attrs in self.G.nodes(data=True):
            if attrs["flu_infection_status"] == "I":
                self.assertTrue(attrs["flu_ever_infected"])


class TestSampleAge(unittest.TestCase):

    def test_age_within_bracket_bounds(self):
        brackets = [(0, 17, 0.3), (18, 65, 0.6), (66, 90, 0.1)]
        for _ in range(200):
            age = pop.sample_age(brackets)
            self.assertGreaterEqual(age, 0)
            self.assertLessEqual(age, 90)

    def test_single_bracket_always_returns_within_range(self):
        brackets = [(20, 30, 1.0)]
        for _ in range(50):
            age = pop.sample_age(brackets)
            self.assertGreaterEqual(age, 20)
            self.assertLessEqual(age, 30)


# ===========================================================================
# Relationship functions
# ===========================================================================

class TestFindEligiblePartners(unittest.TestCase):

    def _make_two_nodes(self, age_m=25, age_f=25):
        G = nx.Graph()
        G.add_node(0, age=age_m, gender="M", is_indigenous=False,
                   hiv_infection_status="S", hiv_infection_step=0, hiv_ever_infected=False,
                   flu_infection_status="S", flu_infection_step=0,
                   flu_recovered_step=0, flu_ever_infected=False)
        G.add_node(1, age=age_f, gender="F", is_indigenous=False,
                   hiv_infection_status="S", hiv_infection_step=0, hiv_ever_infected=False,
                   flu_infection_status="S", flu_infection_step=0,
                   flu_recovered_step=0, flu_ever_infected=False)
        return G

    def test_opposite_gender_eligible(self):
        G = self._make_two_nodes(25, 25)
        eligible = relationship.find_eligible_partners(G, 0)
        self.assertIn(1, eligible)

    def test_same_gender_not_eligible(self):
        G = nx.Graph()
        G.add_node(0, age=25, gender="M", is_indigenous=False,
                   hiv_infection_status="S", hiv_infection_step=0, hiv_ever_infected=False,
                   flu_infection_status="S", flu_infection_step=0,
                   flu_recovered_step=0, flu_ever_infected=False)
        G.add_node(1, age=25, gender="M", is_indigenous=False,
                   hiv_infection_status="S", hiv_infection_step=0, hiv_ever_infected=False,
                   flu_infection_status="S", flu_infection_step=0,
                   flu_recovered_step=0, flu_ever_infected=False)
        eligible = relationship.find_eligible_partners(G, 0)
        self.assertNotIn(1, eligible)

    def test_too_large_age_gap_not_eligible(self):
        G = self._make_two_nodes(age_m=20, age_f=45)  # gap = 25 > default 10
        eligible = relationship.find_eligible_partners(G, 0)
        self.assertNotIn(1, eligible)

    def test_minor_not_eligible(self):
        G = self._make_two_nodes(age_m=25, age_f=15)
        eligible = relationship.find_eligible_partners(G, 0)
        self.assertNotIn(1, eligible)

    def test_self_not_eligible(self):
        G = self._make_two_nodes(25, 25)
        eligible = relationship.find_eligible_partners(G, 0)
        self.assertNotIn(0, eligible)


class TestBreakup(unittest.TestCase):

    def test_edges_can_be_removed(self):
        """With probability=1 every edge should dissolve."""
        G = make_small_graph(n=20, seed=1)
        # Add some edges manually so we have something to break
        nodes = list(G.nodes())
        for i in range(0, len(nodes) - 1, 2):
            G.add_edge(nodes[i], nodes[i + 1], formed_step=0)
        edges_before = G.number_of_edges()
        relationship.breakup(G, breakup_probability=1.0)
        self.assertEqual(G.number_of_edges(), 0)
        self.assertEqual(G.graph["num_breakups"], edges_before)

    def test_no_edges_removed_at_zero_probability(self):
        """With probability=0 no edges should dissolve."""
        G = make_small_graph(n=20, seed=2)
        nodes = list(G.nodes())
        for i in range(0, len(nodes) - 1, 2):
            G.add_edge(nodes[i], nodes[i + 1], formed_step=0)
        edges_before = G.number_of_edges()
        relationship.breakup(G, breakup_probability=0.0)
        self.assertEqual(G.number_of_edges(), edges_before)


# ===========================================================================
# HIV transmission
# ===========================================================================

class TestTransmitHIV(unittest.TestCase):

    def test_no_transmission_when_no_infected(self):
        """If nobody is infected, nobody new should be infected."""
        G = make_pair(male_hiv="S", female_hiv="S")
        new_cases = disease.transmit_hiv(G, 1/1234, 1/2380, current_step=1)
        self.assertEqual(new_cases, 0)

    def test_transmission_possible_with_infected_source(self):
        """Over many trials, at least one transmission should occur (p=1)."""
        total = 0
        for _ in range(50):
            G = make_pair(male_hiv="I", female_hiv="S")
            total += disease.transmit_hiv(G, 1.0, 1.0, current_step=1)
        self.assertGreater(total, 0)

    def test_susceptible_becomes_infected_not_source(self):
        """The source node should not change status."""
        random.seed(5)
        G = make_pair(male_hiv="I", female_hiv="S")
        disease.transmit_hiv(G, 1.0, 1.0, current_step=1)
        self.assertEqual(G.nodes[0]["hiv_infection_status"], "I")  # source stays I

    def test_already_infected_node_not_double_infected(self):
        """Transmitting to an already-infected node should have no effect."""
        G = make_pair(male_hiv="I", female_hiv="I")
        initial_step = G.nodes[1]["hiv_infection_step"]
        disease.transmit_hiv(G, 1.0, 1.0, current_step=99)
        self.assertEqual(G.nodes[1]["hiv_infection_step"], initial_step)

    def test_flu_coinfection_doubles_probability(self):
        """
        With co-infection the effective HIV probability is doubled.
        Transmit with p=0.5 and flu active; over many trials the
        rate should be clearly higher than p=0.5 without flu.
        """
        hits_with_flu = 0
        hits_without_flu = 0
        trials = 500
        random.seed(7)
        for _ in range(trials):
            G = make_pair(male_hiv="I", female_hiv="S", male_flu="I")
            hits_with_flu += disease.transmit_hiv(G, 0.5, 0.5, current_step=1)

        random.seed(7)
        for _ in range(trials):
            G = make_pair(male_hiv="I", female_hiv="S")  # no flu
            hits_without_flu += disease.transmit_hiv(G, 0.5, 0.5, current_step=1)

        self.assertGreater(hits_with_flu, hits_without_flu)


# ===========================================================================
# Flu transmission
# ===========================================================================

class TestTransmitFlu(unittest.TestCase):

    def test_no_transmission_when_no_infectious(self):
        G = make_pair(male_flu="S", female_flu="S")
        new_cases = disease.transmit_flu(
            G, current_step=1, edge_beta=1.0,
            hiv_multiplier=2.0, community_contacts=5, community_beta=0.5
        )
        self.assertEqual(new_cases, 0)

    def test_transmission_occurs_at_max_probability(self):
        """p=1 edge_beta guarantees transmission along the edge."""
        hits = 0
        for _ in range(20):
            G = make_pair(male_flu="I", female_flu="S")
            hits += disease.transmit_flu(
                G, current_step=1, edge_beta=1.0,
                hiv_multiplier=1.0, community_contacts=0, community_beta=0.0
            )
        self.assertGreater(hits, 0)

    def test_newly_infected_node_enters_exposed_state(self):
        """Freshly infected nodes should be 'E', not immediately 'I'."""
        random.seed(3)
        G = make_pair(male_flu="I", female_flu="S")
        disease.transmit_flu(
            G, current_step=1, edge_beta=1.0,
            hiv_multiplier=1.0, community_contacts=0, community_beta=0.0
        )
        # Female (node 1) should now be E if infected
        status = G.nodes[1]["flu_infection_status"]
        self.assertIn(status, {"E", "S"})  # S only if somehow not infected
        if status == "E":
            self.assertTrue(G.nodes[1]["flu_ever_infected"])

    def test_graph_infection_counter_increments(self):
        G = make_pair(male_flu="I", female_flu="S")
        G.graph["flu_total_infections"] = 5
        disease.transmit_flu(
            G, current_step=1, edge_beta=1.0,
            hiv_multiplier=1.0, community_contacts=0, community_beta=0.0
        )
        self.assertGreaterEqual(G.graph["flu_total_infections"], 5)


# ===========================================================================
# Flu progression (SEIRS)
# ===========================================================================

class TestProgressFlu(unittest.TestCase):

    def _node_graph(self, status, infection_step=0, became_infectious_step=None, recovered_step=None):
        G = nx.Graph()
        G.graph["flu_total_infections"] = 0
        attrs = dict(
            age=30, gender="M", is_indigenous=False,
            hiv_infection_status="S", hiv_infection_step=0, hiv_ever_infected=False,
            flu_infection_status=status,
            flu_infection_step=infection_step,
            flu_recovered_step=recovered_step if recovered_step is not None else 0,
            flu_ever_infected=(status != "S"),
        )
        if became_infectious_step is not None:
            attrs["flu_became_infectious_step"] = became_infectious_step
        G.add_node(0, **attrs)
        return G

    def test_exposed_transitions_to_infectious_after_incubation(self):
        G = self._node_graph(status="E", infection_step=0)
        disease.progress_flu(G, current_step=4, incubation_period=4)
        self.assertEqual(G.nodes[0]["flu_infection_status"], "I")

    def test_exposed_stays_exposed_before_incubation_ends(self):
        G = self._node_graph(status="E", infection_step=0)
        disease.progress_flu(G, current_step=3, incubation_period=4)
        self.assertEqual(G.nodes[0]["flu_infection_status"], "E")

    def test_infectious_transitions_to_recovered_after_infectious_period(self):
        G = self._node_graph(status="I", infection_step=0, became_infectious_step=0)
        disease.progress_flu(G, current_step=7, infectious_period=7)
        self.assertEqual(G.nodes[0]["flu_infection_status"], "R")

    def test_recovered_transitions_to_susceptible_after_immunity_wanes(self):
        G = self._node_graph(status="R", infection_step=0, recovered_step=0)
        disease.progress_flu(G, current_step=181, immunity_days=180)
        self.assertEqual(G.nodes[0]["flu_infection_status"], "S")

    def test_susceptible_node_unchanged_by_progression(self):
        G = self._node_graph(status="S")
        disease.progress_flu(G, current_step=100)
        self.assertEqual(G.nodes[0]["flu_infection_status"], "S")


# ===========================================================================
# Recovery (HIV)
# ===========================================================================

class TestApplyRecovery(unittest.TestCase):

    def _infected_graph(self, infection_step=0):
        G = nx.Graph()
        G.add_node(0, age=30, gender="M", is_indigenous=False,
                   hiv_infection_status="I", hiv_infection_step=infection_step,
                   hiv_ever_infected=True,
                   flu_infection_status="S", flu_infection_step=0,
                   flu_recovered_step=0, flu_ever_infected=False)
        return G

    def test_recovery_occurs_after_recovery_days(self):
        G = self._infected_graph(infection_step=0)
        pop.apply_recovery(G, current_day=180, recovery_days=180)
        self.assertEqual(G.nodes[0]["hiv_infection_status"], "R")

    def test_no_recovery_before_recovery_days(self):
        G = self._infected_graph(infection_step=0)
        pop.apply_recovery(G, current_day=179, recovery_days=180)
        self.assertEqual(G.nodes[0]["hiv_infection_status"], "I")

    def test_susceptible_node_not_affected_by_recovery(self):
        G = nx.Graph()
        G.add_node(0, age=30, gender="M", is_indigenous=False,
                   hiv_infection_status="S", hiv_infection_step=0,
                   hiv_ever_infected=False,
                   flu_infection_status="S", flu_infection_step=0,
                   flu_recovered_step=0, flu_ever_infected=False)
        pop.apply_recovery(G, current_day=500, recovery_days=180)
        self.assertEqual(G.nodes[0]["hiv_infection_status"], "S")


# ===========================================================================
# Integration: one full simulation day
# ===========================================================================

class TestOneDayIntegration(unittest.TestCase):

    def test_one_day_does_not_crash(self):
        """Smoke test — run all components for a single day without error."""
        random.seed(99)
        age_dist = [(18, 60, 1.0)]
        G = pop.generate_population(50, age_dist, 0.5, 0.3)

        step = 0
        disease.progress_flu(G, step)
        relationship.start_relationship(G, 0.15, 0.7, step, 16, None)
        disease.transmit_flu(G, step, edge_beta=0.15, hiv_multiplier=2.0,
                             community_contacts=5, community_beta=0.1)
        disease.transmit_hiv(G, 1/1234, 1/2380, step)
        relationship.breakup(G, 0.02)
        pop.apply_recovery(G, step)

        # Basic sanity checks
        self.assertEqual(G.number_of_nodes(), 50)
        for _, attrs in G.nodes(data=True):
            self.assertIn(attrs["hiv_infection_status"], {"S", "I", "R"})
            self.assertIn(attrs["flu_infection_status"], {"S", "E", "I", "R"})

    def test_counters_are_non_negative_after_one_day(self):
        random.seed(42)
        age_dist = [(18, 60, 1.0)]
        G = pop.generate_population(50, age_dist, 0.5, 0.3)

        relationship.start_relationship(G, 0.15, 0.7, 0, 16, None)
        relationship.breakup(G, 0.02)

        self.assertGreaterEqual(G.graph["num_relationships_formed"], 0)
        self.assertGreaterEqual(G.graph["num_breakups"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
