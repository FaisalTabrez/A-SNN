from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ammc_gen5.evidence1_followup_controls import within_class_derangement, component_ablation, require_registered_followup


class FollowupControlTests(unittest.TestCase):
    def test_derangement_preserves_class_and_has_no_fixed_points(self):
        labels = torch.tensor([0, 1, 0, 2, 1, 2, 2])
        for seed in range(10):
            partners = within_class_derangement(labels, seed)
            self.assertTrue(torch.equal(labels, labels[partners]))
            self.assertEqual(sorted(partners.tolist()), list(range(len(labels))))
            self.assertFalse((partners == torch.arange(len(labels))).any())
            self.assertTrue(torch.equal(partners, within_class_derangement(labels, seed)))

    def test_singletons_are_not_silently_exchanged_across_classes(self):
        for labels in [torch.tensor([0, 0, 1]), torch.tensor([0.1, 0.1]), torch.zeros(2, 2)]:
            with self.assertRaises(ValueError): within_class_derangement(labels, 310)

    def test_component_controls_isolate_membrane_and_spikes_without_mutation(self):
        state = torch.arange(24.).reshape(2, 12); original = state.clone()
        spikes = component_ablation(state, 3, 'zero_spike_trace')
        membrane = component_ablation(state, 3, 'zero_final_membrane')
        self.assertTrue(torch.equal(spikes[:, -3:], state[:, -3:]))
        self.assertEqual(float(spikes[:, :-3].abs().sum()), 0.)
        self.assertTrue(torch.equal(membrane[:, :-3], state[:, :-3]))
        self.assertEqual(float(membrane[:, -3:].abs().sum()), 0.)
        self.assertTrue(torch.equal(state, original))
        with self.assertRaises(ValueError): component_ablation(state, 5, 'zero_both')

    def test_pending_or_partial_decisions_cannot_authorize_experiments(self):
        for decision in [None, {}, {'evidence1_lock_reviewed': True}]:
            with self.assertRaises(ValueError): require_registered_followup(decision)


if __name__ == '__main__': unittest.main()
