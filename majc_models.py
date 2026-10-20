"""Causal shared features and equal-capacity marginal/direct/JC runtime.

Feature construction has no fitted normalization and never reads a future label.
The same FeatureCodec is used for saved pi_SL states and online candidates.
"""
import hashlib
import math
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # stdlib feature/label tests remain usable on the client.
    torch = None
    nn = None


CODEC_VERSION = 'causal84_budget8_v1'
FACT_NAMES = ([f'before_{i}' for i in range(12)]
              + [f'after_{i}' for i in range(12)]
              + [f'delta_{i}' for i in range(12)]
              + [f'before_known_{i}' for i in range(12)]
              + [f'after_known_{i}' for i in range(12)]
              + [f'delta_known_{i}' for i in range(12)]
              + ['parent_false', 'parent_true', 'parent_unknown',
                 'child_false', 'child_true', 'child_unknown']
              + [f'original_numeric_parameter_{i}' for i in range(6)])
BUDGET_NAMES = ['t_fraction', 'remaining_fraction', 'log_t', 'log_remaining',
                'log_cap', 'guard_empty', 'parent_unknown_fraction', 'child_unknown_fraction']


def signed_log(value):
    value = float(value)
    assert math.isfinite(value), 'Non-finite causal feature'
    return math.copysign(math.log1p(abs(value)), value)


def numeric_parameters(value):
    """Keep actual numeric values, with deterministic public-parameter order.

    Strings are not normalized into numbers: checker parameter types stay intact.
    Nothing here changes native checker inputs or their scoring semantics.
    """
    if type(value) in (int, float):
        return [value]
    if isinstance(value, dict):
        return [n for key in sorted(value) for n in numeric_parameters(value[key])]
    if isinstance(value, (list, tuple)):
        return [n for item in value for n in numeric_parameters(item)]
    return []


def _matrix(value, rows, default=0.):
    if value is None:
        return [[default] * 12 for _ in range(rows)]
    if hasattr(value, 'tolist'):
        value = value.tolist()
    assert len(value) == rows and all(len(row) == 12 for row in value), 'Causal facts must be [all_atoms,12]'
    return value


def _shape(value):
    if hasattr(value, 'shape'):
        return tuple(value.shape)
    return (len(value), len(value[0]) if value else 0)


class FeatureCodec:
    version = CODEC_VERSION
    fact_names = FACT_NAMES
    budget_names = BUDGET_NAMES

    @staticmethod
    def prepare(contract, rule_embeddings, feature_raw):
        """Return one causal state; h/embeddings remain tensors when supplied.

        Required raw fields: h, action_embedding, facts_before/after,
        fact_masks_before/after, parent_v/child_v, t, remaining. ``cap`` is
        optional because it is exactly t+remaining, never actual answer length.
        """
        n = len(contract['atom_texts'])
        native_n = int(contract['native_atoms'])
        semantic_n = int(contract.get('semantic_atoms', n - native_n))
        assert n == native_n + semantic_n
        event_map = list(contract['event_map'])
        event_n = int(contract['event_count'])
        assert len(event_map) == n and all(type(i) is int and 0 <= i < event_n for i in event_map)
        weights = list(contract['event_weights'])
        assert len(weights) == event_n and sum(weights) == n
        assert weights == [event_map.count(i) for i in range(event_n)], 'Alias weights disagree'
        emb_shape = _shape(rule_embeddings)
        assert emb_shape[0] == n, 'Missing requirement embeddings'
        h, action = feature_raw['h'], feature_raw['action_embedding']
        hidden = int(h.numel()) if hasattr(h, 'numel') else len(h)
        action_hidden = int(action.numel()) if hasattr(action, 'numel') else len(action)
        assert hidden == action_hidden and (n == 0 or emb_shape[1] == hidden)
        before = _matrix(feature_raw.get('facts_before'), n)
        after = _matrix(feature_raw.get('facts_after'), n)
        mb = _matrix(feature_raw.get('fact_masks_before'), n, False)
        ma = _matrix(feature_raw.get('fact_masks_after'), n, False)
        parent = list(feature_raw.get('parent_v', []))
        child = list(feature_raw.get('child_v', []))
        unique_n = int(contract.get('unique_supported', len(parent)))
        assert len(parent) == len(child) == unique_n, 'Reachability vector does not match contract'
        assert all(v is None or type(v) in (bool, int) and v in (0, 1) for v in parent + child)
        aliases = {item['native_atom']: item['event'] for item in contract['exact_alias_map']}
        rules = {r['atom']: r for r in contract['rules']}
        facts, types = [], []
        for atom in range(n):
            bmask, amask = [bool(v) for v in mb[atom]], [bool(v) for v in ma[atom]]
            dmask = [x and y for x, y in zip(bmask, amask)]
            b = [signed_log(v) if known else 0. for v, known in zip(before[atom], bmask)]
            a = [signed_log(v) if known else 0. for v, known in zip(after[atom], amask)]
            d = [signed_log(float(y) - float(x)) if known else 0.
                 for x, y, known in zip(before[atom], after[atom], dmask)]
            idx = aliases.get(atom)
            pv, cv = (parent[idx], child[idx]) if idx is not None else (None, None)
            reach = [float(pv == 0), float(pv == 1), float(pv is None),
                     float(cv == 0), float(cv == 1), float(cv is None)]
            rule = rules.get(atom, {})
            params = numeric_parameters(rule.get('original_parameters'))
            assert len(params) <= 6, 'More than six numeric parameters; no silent truncation'
            params = [signed_log(v) for v in params] + [0.] * (6 - len(params))
            row = b + a + d + list(map(float, bmask + amask + dmask)) + reach + params
            assert len(row) == 84
            facts.append(row)
            typ = str(contract['dataset']) + ':' + str(rule.get('type', 'SEMANTIC'))
            types.append(63 if atom >= native_n else 1 + int(hashlib.sha256(typ.encode()).hexdigest()[:8], 16) % 62)
        t, remaining = int(feature_raw['t']), int(feature_raw['remaining'])
        cap = int(feature_raw.get('cap', t + remaining))
        assert t >= 0 and remaining >= 1 and cap == t + remaining
        if 'prefix_ids' in feature_raw:
            assert len(feature_raw['prefix_ids']) == t, 'Prefix position mismatch'
        budget = [t / cap, remaining / cap, math.log1p(t) / 10,
                  math.log1p(remaining) / 10, math.log1p(cap) / 10,
                  float(bool(feature_raw.get('guard_empty', False))),
                  sum(v is None for v in parent) / max(1, unique_n),
                  sum(v is None for v in child) / max(1, unique_n)]
        # An explicit null attention atom prevents all-padding attention NaNs.
        # It is not a model event and contributes neither labels nor weights.
        atom_slots, event_slots = max(1, n), max(1, event_n)
        if not n:
            facts, types = [[0.] * 84], [0]
        grouping = [[0.] * atom_slots for _ in range(event_slots)]
        for atom, event in enumerate(event_map):
            grouping[event][atom] = 1. / weights[event]
        return dict(h=h, action_embedding=action, rule_embeddings=rule_embeddings,
                    facts=facts, types=types, budget=budget, atom_mask=[True] * atom_slots,
                    event_mask=[i < event_n for i in range(event_slots)],
                    event_pool=grouping, atom_event_index=event_map or [0],
                    event_weights=weights or [0.], atom_count=n, event_count=event_n,
                    hidden_size=hidden, codec_version=CODEC_VERSION)

    @staticmethod
    def pack(examples, device='cpu'):
        if torch is None:
            raise ImportError('PyTorch required for batch packing/model execution')
        assert examples and len({x['hidden_size'] for x in examples}) == 1
        b, h = len(examples), examples[0]['hidden_size']
        na = max(len(x['atom_mask']) for x in examples)
        ne = max(len(x['event_mask']) for x in examples)
        out = dict(h=torch.zeros(b, h), action_embedding=torch.zeros(b, h),
                   rule_embeddings=torch.zeros(b, na, h), facts=torch.zeros(b, na, 84),
                   budget=torch.zeros(b, 8), types=torch.zeros(b, na, dtype=torch.long),
                   atom_mask=torch.zeros(b, na, dtype=torch.bool),
                   event_mask=torch.zeros(b, ne, dtype=torch.bool),
                   event_pool=torch.zeros(b, ne, na),
                   atom_event_index=torch.zeros(b, na, dtype=torch.long),
                   event_weights=torch.zeros(b, ne))
        for i, x in enumerate(examples):
            a, e = len(x['atom_mask']), len(x['event_mask'])
            out['h'][i] = torch.as_tensor(x['h']).detach().float().cpu().reshape(h)
            out['action_embedding'][i] = torch.as_tensor(x['action_embedding']).detach().float().cpu().reshape(h)
            if x['atom_count']:
                out['rule_embeddings'][i, :a] = torch.as_tensor(x['rule_embeddings']).detach().float().cpu()
            for key in ('facts', 'types', 'atom_mask', 'atom_event_index'):
                out[key][i, :a] = torch.as_tensor(x[key])
            for key in ('event_mask', 'event_weights'):
                out[key][i, :e] = torch.as_tensor(x[key])
            out['event_pool'][i, :e, :a] = torch.as_tensor(x['event_pool'])
            out['budget'][i] = torch.as_tensor(x['budget'])
        return {k: v.to(device) for k, v in out.items()}


def canonical_labels(contract, label):
    """Unknown labels remain unknown; a full-task audit never becomes an atom."""
    native, semantic = label['native'], label['semantic']
    assert len(native) == contract['native_atoms'] and len(semantic) == contract.get('semantic_atoms', 0)
    atoms = list(native) + list(semantic)
    assert all(v is None or type(v) in (int, bool) and v in (0, 1) for v in atoms)
    assert label['U'] is None or type(label['U']) in (int, bool) and label['U'] in (0, 1)
    assert label['J'] is None or type(label['J']) in (int, bool) and label['J'] in (0, 1)
    events, conflicts = [], []
    for event in range(contract['event_count']):
        values = {int(v) for i, v in enumerate(atoms) if contract['event_map'][i] == event and v is not None}
        if len(values) > 1:
            conflicts.append(event)
            events.append(None)
        else:
            events.append(next(iter(values)) if values else None)
    z = None
    if label['U'] == 1:
        if any(v == 0 for v in atoms):
            z = 0
        elif all(v == 1 for v in atoms) and label['J'] == 1 and not conflicts:
            z = 1
        # In particular, all atom flags=1 plus full audit failure/unknown is
        # not a positive and is not silently relabeled with a fake extra event.
    return dict(U=label['U'], J=label['J'], events=events, Z=z, alias_conflicts=conflicts,
                all_atom_positive_but_not_confirmed=all(v == 1 for v in atoms) and z != 1)


class MarginalModel(nn.Module if nn else object):
    def __init__(self, hidden_size=4096, type_count=64):
        if nn is None:
            raise ImportError('PyTorch required')
        super().__init__()
        self.h = nn.Linear(hidden_size, 128)
        self.a = nn.Linear(hidden_size, 128)
        self.rule = nn.Linear(hidden_size, 128)
        self.types = nn.Embedding(type_count, 128)
        self.facts = nn.Linear(84, 128)
        self.budget = nn.Linear(8, 128)
        layer = nn.TransformerEncoderLayer(128, 4, 256, .1, batch_first=True, norm_first=True)
        self.rules = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(128, 4, .1, batch_first=True)
        self.q = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1))
        self.u = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1))

    def forward(self, batch):
        mask = batch['atom_mask'].bool()
        x = self.rule(batch['rule_embeddings'].float()) + self.types(batch['types']) + self.facts(batch['facts'].float())
        x = self.rules(x, src_key_padding_mask=~mask)
        query = self.h(batch['h'].float()) + self.a(batch['action_embedding'].float()) + self.budget(batch['budget'].float())
        attended, _ = self.cross(query[:, None], x, x, key_padding_mask=~mask, need_weights=False)
        state = query + attended[:, 0]
        atoms = torch.cat([x, state[:, None].expand_as(x)], -1)
        event_repr = torch.bmm(batch['event_pool'].float(), atoms)
        q = self.q(event_repr).squeeze(-1).sigmoid().masked_fill(~batch['event_mask'].bool(), 0)
        pooled = (x * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        u = self.u(torch.cat([state, pooled], -1)).squeeze(-1).sigmoid()
        return dict(u=u, q=q, state=state, rule_repr=x)


def marginal_loss(pred, U, labels, U_known, label_known, event_mask, event_weights=None):
    usable = U_known.bool() & (U == 1)
    mask = label_known.bool() & usable[:, None] & event_mask.bool()
    weight = mask.float() if event_weights is None else mask.float() * event_weights.float()
    raw = nn.functional.binary_cross_entropy(pred['q'].clamp(1e-6, 1-1e-6), labels.float(), reduction='none')
    qloss = (raw * weight).sum(1) / weight.sum(1).clamp_min(1)
    qloss = qloss[mask.any(1)].mean() if bool(mask.any()) else pred['q'].sum() * 0
    uloss = (nn.functional.binary_cross_entropy(pred['u'][U_known.bool()].clamp(1e-6, 1-1e-6), U[U_known.bool()].float())
             if bool(U_known.any()) else pred['u'].sum() * 0)
    return uloss + qloss


def frechet(q, event_mask):
    mask, q = event_mask.bool(), q.double()
    n = mask.sum(-1)
    lo = ((q * mask).sum(-1) - (n - 1)).clamp_min(0)
    hi = q.masked_fill(~mask, 1).amin(-1)
    ind = torch.exp((q.clamp(1e-6, 1-1e-6).log() * mask).sum(-1))
    ones = torch.ones_like(lo)
    return torch.where(n == 0, ones, lo), torch.where(n == 0, ones, hi), torch.where(n == 0, ones, ind)


class JointHead(nn.Module if nn else object):
    """Identical trainable architecture; only the output mapping differs."""
    def __init__(self, coherent):
        if nn is None:
            raise ImportError('PyTorch required')
        super().__init__()
        self.coherent = bool(coherent)
        self.prob = nn.Linear(2, 128)
        self.usable = nn.Linear(1, 128)
        layer = nn.TransformerEncoderLayer(128, 4, 256, .1, batch_first=True, norm_first=True)
        self.rules = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(128, 4, .1, batch_first=True)
        self.g = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 1))

    def forward(self, frozen, event_mask, atom_mask, atom_event_index):
        u, q = frozen['u'].detach(), frozen['q'].detach()
        state, x = frozen['state'].detach(), frozen['rule_repr'].detach()
        atomq = q.gather(1, atom_event_index)
        probs = torch.stack([atomq, u[:, None].expand_as(atomq)], -1)
        x = self.rules(x + self.prob(probs), src_key_padding_mask=~atom_mask.bool())
        query = state + self.usable(u[:, None])
        attended, _ = self.cross(query[:, None], x, x, key_padding_mask=~atom_mask.bool(), need_weights=False)
        logit = self.g(torch.cat([query, attended[:, 0]], -1)).squeeze(-1)
        lo, hi, ind = frechet(q, event_mask)
        gamma = logit.double().sigmoid()
        eta = lo + (hi - lo) * gamma if self.coherent else gamma
        if self.coherent:
            eta = torch.where((hi-lo) < 1e-7, lo.detach(), eta)
        eta = torch.where(event_mask.sum(-1) == 0, torch.ones_like(eta), eta)
        return dict(eta=eta, J=u.double()*eta, L=lo, H=hi, eta_ind=ind, logit=logit)


class Predictor:
    def __init__(self, marginal, direct, jc, device='cpu', metadata=None):
        self.device = device
        self.codec = FeatureCodec
        self.marginal = marginal.to(device).eval()
        self.direct = direct.to(device).eval()
        self.jc = jc.to(device).eval()
        self.metadata = metadata or {}
        for model in (self.marginal, self.direct, self.jc):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def predict_batch(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            frozen = self.marginal(batch)
            args = (batch['event_mask'], batch['atom_mask'], batch['atom_event_index'])
            direct, jc = self.direct(frozen, *args), self.jc(frozen, *args)
            weights = batch['event_weights'].double() * batch['event_mask']
            mean = (frozen['q'].double()*weights).sum(-1)/weights.sum(-1).clamp_min(1)
            mean = torch.where(weights.sum(-1) == 0, torch.ones_like(mean), mean)
            clipped = torch.maximum(jc['L'], torch.minimum(jc['H'], direct['eta']))
            return dict(u=frozen['u'], q=frozen['q'], L=jc['L'], H=jc['H'],
                        eta_ind=jc['eta_ind'], eta_direct=direct['eta'], eta_JC=jc['eta'],
                        Jind=frozen['u'].double()*jc['eta_ind'], Jdirect=direct['J'], JJC=jc['J'],
                        M=frozen['u'].double()*mean, eta_direct_clipped=clipped,
                        Jdirect_clipped=frozen['u'].double()*clipped)

    def predict_prepared(self, examples):
        pred = self.predict_batch(FeatureCodec.pack(examples, self.device))
        rows = []
        for i, example in enumerate(examples):
            row = {key: float(value[i]) for key, value in pred.items() if key != 'q'}
            row['q'] = pred['q'][i, :example['event_count']].detach().cpu().tolist()
            row['prediction_status'] = 'TRAINED_FIXED_FINAL_STEP_EXPLORATORY'
            rows.append(row)
        return rows

    def predict(self, feature_raw, contract, rule_embeddings):
        return self.predict_prepared([FeatureCodec.prepare(contract, rule_embeddings, feature_raw)])[0]


def load_predictor(path, device='cpu'):
    if torch is None:
        raise ImportError('PyTorch required')
    path = Path(path)
    checkpoint = torch.load(path / 'MODELS.pt' if path.is_dir() else path, map_location='cpu', weights_only=True)
    assert checkpoint['codec_version'] == CODEC_VERSION
    marginal, direct, jc = MarginalModel(checkpoint['hidden_size']), JointHead(False), JointHead(True)
    marginal.load_state_dict(checkpoint['marginal'])
    direct.load_state_dict(checkpoint['direct'])
    jc.load_state_dict(checkpoint['JC'])
    return Predictor(marginal, direct, jc, device, checkpoint.get('metadata'))


def load_marginal(path, device='cpu'):
    """Load the final shared marginal checkpoint for an independent ABC runtime."""
    if torch is None:
        raise ImportError('PyTorch required')
    path = Path(path)
    checkpoint = torch.load(path / 'MARGINAL_FINAL.pt' if path.is_dir() else path,
                            map_location='cpu', weights_only=True)
    assert checkpoint['codec_version'] == CODEC_VERSION
    model = MarginalModel(checkpoint['hidden_size'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pack_frozen(examples, frozen, device='cpu'):
    """Pack cached OOF representations without allocating unused Qwen vectors.

    Return (frozen_dict, masks_dict), suitable for JointHead or ABC trainers.
    """
    if torch is None:
        raise ImportError('PyTorch required')
    assert len(examples) == len(frozen) and examples
    n = len(examples)
    na = max(len(x['atom_mask']) for x in examples)
    ne = max(len(x['event_mask']) for x in examples)
    masks = dict(atom_mask=torch.zeros(n, na, dtype=torch.bool),
                 event_mask=torch.zeros(n, ne, dtype=torch.bool),
                 atom_event_index=torch.zeros(n, na, dtype=torch.long),
                 event_weights=torch.zeros(n, ne))
    pred = dict(u=torch.zeros(n), q=torch.zeros(n, ne), state=torch.zeros(n, 128),
                rule_repr=torch.zeros(n, na, 128))
    for i, (example, value) in enumerate(zip(examples, frozen)):
        a, e = len(example['atom_mask']), len(example['event_mask'])
        for key in ('atom_mask', 'atom_event_index'):
            masks[key][i, :a] = torch.as_tensor(example[key])
        for key in ('event_mask', 'event_weights'):
            masks[key][i, :e] = torch.as_tensor(example[key])
        pred['u'][i] = value['u']
        pred['q'][i, :e] = value['q']
        pred['state'][i] = value['state']
        pred['rule_repr'][i, :a] = value['rule_repr']
    return ({k: v.to(device) for k, v in pred.items()},
            {k: v.to(device) for k, v in masks.items()})
