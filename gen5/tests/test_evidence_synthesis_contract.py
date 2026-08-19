from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.evidence_synthesis import EVIDENCE_FILENAMES, synthesize_gen5_evidence


class EvidenceSynthesisContractTest(unittest.TestCase):
    def test_required_evidence_chain_is_complete(self) -> None:
        self.assertEqual(tuple(EVIDENCE_FILENAMES), (
            "phase44", "phase45", "phase46", "phase47", "phase48", "phase49",
            "milestone_a", "gen6", "gen7", "gen8", "gen9", "gen10", "gen11", "gen12", "gen13", "gen14", "gen15", "gen16", "gen17", "gen18", "gen19", "gen20",
        ))
        self.assertEqual(len(set(EVIDENCE_FILENAMES.values())), 22)

    def test_milestone_a_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["milestone_a_tcn_confirmation_accuracy"],
            0.5916985575507802,
        )
        self.assertEqual(result.metrics["milestone_a_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        readiness = claims[
            "The current Gen-5 architecture qualifies for hardware optimization"
        ]
        self.assertEqual(readiness["status"], "rejected")
        self.assertFalse(readiness["gate_passed"])

    def test_gen6_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(result.metrics["gen6_tcn_accuracy"], 0.5908154253753312)
        self.assertAlmostEqual(result.metrics["gen6_lif_accuracy"], 0.5901612533935171)
        self.assertAlmostEqual(
            result.metrics["gen6_lif_state_specificity_vs_shuffled"],
            -0.006574428417230882,
        )
        self.assertEqual(result.metrics["gen6_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims[
                "The Gen-6 shared residual LIF preserves TCN predictive accuracy"
            ]["status"],
            "supported",
        )
        self.assertEqual(
            claims[
                "The Gen-6 LIF correction is beneficially sample-specific"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims[
                "The Gen-6 successor qualifies for hardware optimization"
            ]["status"],
            "rejected",
        )

    def test_gen7_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen7_lif_gain_vs_tcn"],
            0.004170346384064438,
        )
        self.assertAlmostEqual(
            result.metrics["gen7_lif_future_alignment_margin"],
            0.2927859868889986,
        )
        self.assertAlmostEqual(
            result.metrics["gen7_lif_state_specificity_vs_shuffled"],
            -0.01022143721584401,
        )
        self.assertEqual(result.metrics["gen7_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-7 paired future prediction improves state alignment"][
                "status"
            ],
            "supported",
        )
        self.assertEqual(
            claims[
                "Gen-7 uses predictive state beneficially by sample identity and temporal order"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["The Gen-7 successor qualifies for hardware optimization"][
                "status"
            ],
            "rejected",
        )

    def test_gen8_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen8_pooled_lif_gain_vs_tcn"],
            0.015356687273084146,
        )
        self.assertAlmostEqual(
            result.metrics["gen8_analog_temporal_order_vs_reversed"],
            0.0056095247440552205,
        )
        self.assertAlmostEqual(
            result.metrics["gen8_analog_state_specificity_vs_shuffled"],
            0.0011775095672652187,
        )
        self.assertAlmostEqual(
            result.metrics["gen8_candidate_screen_validation_accuracy"],
            0.07266666666666667,
        )
        self.assertAlmostEqual(
            result.metrics["gen8_candidate_screen_spike_rate"],
            0.5065631499290466,
        )
        self.assertEqual(result.metrics["gen8_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims[
                "Gen-8 time-local analog binding introduces temporal-order sensitivity"
            ]["status"],
            "supported",
        )
        self.assertEqual(
            claims[
                "Gen-8 time-local analog binding uses the correct sample identity"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims[
                "The Gen-8 paired time-local LIF candidate is stable enough for confirmation"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["The Gen-8 successor qualifies for hardware optimization"][
                "status"
            ],
            "rejected",
        )

    def test_gen9_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen9_predictive_lif_screen_gap_vs_tcn"],
            -0.06466666666666665,
        )
        self.assertAlmostEqual(
            result.metrics["gen9_tcn_static_shift_drop"],
            0.09364471919667683,
        )
        self.assertAlmostEqual(
            result.metrics["gen9_tcn_readout_adaptation_gain"],
            0.05601347594282539,
        )
        self.assertAlmostEqual(
            result.metrics["gen9_tcn_full_adaptation_gain"],
            0.08461714584764339,
        )
        self.assertEqual(result.metrics["gen9_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-9 sensor damage creates a non-trivial distribution shift"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["The Gen-9 predictive LIF representation is source-competent"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["A frozen TCN representation adapts through a trainable readout"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-9 qualifies for STW/LTW memory experiments"]["status"],
            "rejected",
        )

    def test_gen10_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen10_dropout_clean_gain_vs_tcn"],
            0.026837405553920113,
        )
        self.assertAlmostEqual(
            result.metrics["gen10_dropout_damaged_gain_vs_tcn"],
            0.0876263369639878,
        )
        self.assertAlmostEqual(
            result.metrics["gen10_lif_screen_clean_gap"],
            -0.09199999999999997,
        )
        self.assertAlmostEqual(
            result.metrics["gen10_lif_screen_spike_rate"],
            0.11538150972127914,
        )
        self.assertEqual(result.metrics["gen10_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Sensor dropout improves conventional robustness in Gen-10"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["The Gen-10 masked residual LIF representation is source-competent"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-10 qualifies a spiking representation for adaptation"]["status"],
            "rejected",
        )

    def test_gen11_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen11_readout_adaptation_gain"],
            0.02330487685212442,
        )
        self.assertAlmostEqual(
            result.metrics["gen11_full_adaptation_gain"],
            0.03295391358388119,
        )
        self.assertAlmostEqual(
            result.metrics["gen11_lif_adaptation_gain"],
            0.00783370948222284,
        )
        self.assertAlmostEqual(
            result.metrics["gen11_lif_state_specificity"],
            0.00011448009681743383,
        )
        self.assertEqual(result.metrics["gen11_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-11 state adapters improve damaged-task accuracy by the preregistered margin"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-11 LIF adaptation depends on sample-specific spiking state"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-11 qualifies for synaptic STW/LTW consolidation"]["status"],
            "rejected",
        )

    def test_gen12_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen12_readout_adaptation_gain"],
            0.03563601870931865,
        )
        self.assertAlmostEqual(
            result.metrics["gen12_full_adaptation_gain"],
            0.04767278317469659,
        )
        self.assertAlmostEqual(
            result.metrics["gen12_spiking_adaptation_gain"],
            0.0027802309227095514,
        )
        self.assertAlmostEqual(
            result.metrics["gen12_spiking_association_specificity"],
            0.004170346384064365,
        )
        self.assertEqual(result.metrics["gen12_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-12 prototype memory provides useful fast adaptation"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-12 spiking memory depends on correct class associations"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-12 qualifies for context-free consolidation"]["status"],
            "rejected",
        )

    def test_gen13_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen13_readout_adaptation_gain"],
            0.020491937330324134,
        )
        self.assertAlmostEqual(
            result.metrics["gen13_full_adaptation_gain"],
            0.03269224479115559,
        )
        self.assertAlmostEqual(
            result.metrics["gen13_spiking_adaptation_gain"],
            0.004104929185882937,
        )
        self.assertAlmostEqual(
            result.metrics["gen13_spiking_class_specificity"],
            0.0046773296699702165,
        )
        self.assertAlmostEqual(
            result.metrics["gen13_spiking_activity"],
            0.20000001776588058,
        )
        self.assertEqual(result.metrics["gen13_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-13 local output plasticity provides useful adaptation"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-13 spiking fast weights are causally class-specific"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-13 qualifies for STW/LTW consolidation"]["status"],
            "rejected",
        )

    def test_gen14_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen14_oracle_final_fitness"],
            8.38111162185669,
        )
        self.assertAlmostEqual(
            result.metrics["gen14_spiking_final_fitness"],
            -0.10888889328473145,
        )
        self.assertAlmostEqual(
            result.metrics["gen14_spiking_margin_vs_static"],
            -0.7500000287675195,
        )
        self.assertAlmostEqual(
            result.metrics["gen14_spiking_margin_vs_shuffled"],
            -0.16111113027566007,
        )
        self.assertEqual(result.metrics["gen14_passed"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-14 embodied sensor-to-action mapping is solvable"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-14 baseline-to-evaluation improvement identifies local learning"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-14 spiking eligibility depends on correctly assigned reward"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-14 qualifies for reward-eligibility confirmation"]["status"],
            "rejected",
        )

    def test_gen15_reward_protocol_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(result.metrics["gen15_reinforce_gain"], 0.9922222627533807)
        self.assertAlmostEqual(
            result.metrics["gen15_reinforce_margin_vs_shuffled"],
            1.2666667252779005,
        )
        self.assertEqual(result.metrics["gen15_passed"], 1.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-15 identical-reset evaluation removes phase non-stationarity"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-15 delayed scalar reward supports conventional learning"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-15 conventional learning depends on agent-specific reward"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-15 validates an AMMC local-learning mechanism"]["status"],
            "not tested",
        )

    def test_gen16_local_credit_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(result.metrics["gen16_local_gain"], 0.18333334527495837)
        self.assertAlmostEqual(result.metrics["gen16_local_autograd_gap"], 0.0)
        self.assertAlmostEqual(
            result.metrics["gen16_maximum_gradient_error"],
            2.7939677238464355e-09,
        )
        self.assertEqual(result.metrics["gen16_passed"], 1.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-16 manual score-function gradient matches autograd"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-16 local reward credit is behaviorally equivalent to autograd"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-16 local learning depends on agent-specific reward"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-16 establishes sparse-spiking or structural continuous learning"]["status"],
            "not tested",
        )

    def test_gen17_sparse_translation_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(result.metrics["gen17_analog_gain"], 0.004444449312157095)
        self.assertAlmostEqual(result.metrics["gen17_spiking_gain"], -0.39111114210552644)
        self.assertAlmostEqual(
            result.metrics["gen17_spiking_margin_vs_shuffled"],
            -1.0522222932842042,
        )
        self.assertEqual(result.metrics["gen17_passed"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-17 sparse event generation and local gradient are operational"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-16 analog local-credit gain replicates on Gen-17 seeds"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-17 Bernoulli sparse translation preserves local learning"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-17 sparse local learning depends on correctly assigned reward"]["status"],
            "rejected",
        )

    def test_gen18_local_credit_program_is_closed_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(result.metrics["gen18_local_gain"], 0.7956667011603713)
        self.assertAlmostEqual(
            result.metrics["gen18_local_gain_ci95_lower"],
            -0.01628414509997178,
        )
        self.assertAlmostEqual(
            result.metrics["gen18_local_margin_vs_shuffled"],
            0.5100000280266006,
        )
        self.assertEqual(result.metrics["gen18_passed"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-18 stationary controls and manual-gradient implementation remain valid"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-16 analog local-credit behavior replicates across ten held-out seeds"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-18 local behavior depends reliably on correctly assigned reward"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["The tested local reward-credit program qualifies for further mechanism expansion"]["status"],
            "rejected",
        )

    def test_gen19_external_event_vision_replication_is_closed_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(result.metrics["gen19_conv_accuracy"], 0.9686)
        self.assertAlmostEqual(result.metrics["gen19_residual_lif_accuracy"], 0.9631666666666666)
        self.assertAlmostEqual(
            result.metrics["gen19_state_contribution_vs_direct_only"],
            0.15210000000000004,
        )
        self.assertAlmostEqual(
            result.metrics["gen19_state_specificity_vs_shuffled"],
            -0.022999999999999982,
        )
        self.assertEqual(result.metrics["gen19_passed"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-19 establishes a learnable parameter-matched N-MNIST benchmark"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Residual LIF state is causally used on N-MNIST"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Residual LIF state is beneficially sample-specific on N-MNIST"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["The event-audio residual-state result generalizes to event vision"]["status"],
            "rejected",
        )

    def test_gen20_spiking_translation_is_closed_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen20_teacher_screen_accuracy"],
            0.9911651941990332,
        )
        self.assertAlmostEqual(
            result.metrics["gen20_multiscale_screen_accuracy"],
            0.9636606101016836,
        )
        self.assertAlmostEqual(
            result.metrics["gen20_multiscale_gap_to_gate"],
            -0.011339389898316399,
        )
        self.assertAlmostEqual(
            result.metrics["gen20_distillation_gain_vs_multiscale"],
            -0.0003333888981496935,
        )
        self.assertAlmostEqual(
            result.metrics["gen20_multiscale_ops_reduction_vs_teacher"],
            74.37058850781366,
        )
        self.assertEqual(result.metrics["gen20_promoted_arm_count"], 0.0)
        self.assertEqual(result.metrics["gen20_passed"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims[
                "Gen-20 multiscale residual PLIF closes the N-MNIST representation gap"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims[
                "Gen-20 proposed arms maintain sparse activity and a low operation proxy"
            ]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-20 establishes causal temporal state use on N-MNIST"][
                "status"
            ],
            "not tested",
        )


if __name__ == "__main__":
    unittest.main()
