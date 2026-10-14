"""Dormant follow-up controls. Not part of the registered Evidence-1 runner."""
import torch


def within_class_derangement(labels, seed):
    """Exchange samples only inside their class; refuse singleton classes."""
    if labels.ndim != 1 or labels.numel() < 2:
        raise ValueError('Need one-dimensional labels with at least two samples')
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('Class labels must be integers')
    labels = labels.cpu()
    generator = torch.Generator().manual_seed(seed)
    partners = torch.empty_like(labels, dtype=torch.long)
    for label in torch.unique(labels).tolist():
        members = torch.nonzero(labels == label, as_tuple=False).flatten()
        if len(members) < 2: raise ValueError('Singleton class cannot be deranged without changing class')
        order = members[torch.randperm(len(members), generator=generator)]
        partners[order] = order.roll(1)
    assert torch.equal(labels, labels[partners])
    assert not (partners == torch.arange(len(labels))).any()
    return partners


def component_ablation(state, hidden_units, mode):
    """Separate pooled spikes from the last hidden-width analog membrane block."""
    if state.ndim != 2 or hidden_units < 1 or state.shape[1] <= hidden_units:
        raise ValueError('Expected pooled spike features followed by final membrane')
    if (state.shape[1] - hidden_units) % hidden_units:
        raise ValueError('State width does not match whole pooled hidden-unit blocks')
    if not torch.isfinite(state).all(): raise FloatingPointError('Nonfinite control state')
    result = state.clone()
    if mode == 'zero_spike_trace': result[:, :-hidden_units] = 0
    elif mode == 'zero_final_membrane': result[:, -hidden_units:] = 0
    elif mode == 'zero_both': result.zero_()
    else: raise ValueError('Unknown component intervention')
    return result


def require_registered_followup(decision):
    """Human artifacts must explicitly authorize a separate preregistered study."""
    required = ['evidence1_lock_reviewed', 'followup_protocol_registered', 'untouched_holdout_selected']
    if not isinstance(decision, dict) or any(decision.get(key) is not True for key in required):
        raise ValueError('Follow-up is not authorized; Evidence-1 remains active')
    if not decision.get('reviewer') or not decision.get('protocol_sha256'):
        raise ValueError('Named human reviewer and immutable follow-up protocol identity required')
    return True
