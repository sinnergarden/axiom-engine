"""Feature semantic identity excludes deployment and unrelated catalog contents."""
from copy import deepcopy
from ..core import FeaturePlan, required_history, validate_plan
from ..core.plan import CS
from ..core.operator_semantics import OPERATOR_SEMANTIC_REVISIONS
from .codec import digest, require


def feature_definitions(plan, *, input_semantics):
    """Bind each output's exact DAG/schema and explicit input interpretation.

    input_semantics maps used fact names to their unit/precision/adjustment/clock
    contracts, not data revisions. Revisions and reference members belong to
    the source adapter's scoped dependency receipts. No commit is a cache key.
    """
    p = validate_plan(plan, execution=True)
    require(type(input_semantics) is dict, 'explicit input semantics required')
    nodes = {n['name']: n for n in p['nodes']}
    schema = {c['name']: c for c in p['input_schema']}
    result = {}
    for output in p['outputs']:
        used, leaves = set(), set()
        def include(name):
            if name in nodes and name not in used:
                used.add(name)
                for parent in nodes[name]['inputs']: include(parent)
            elif name in schema:
                leaves.add(name)
        include(output['node'])
        require(leaves <= set(input_semantics), 'missing used input semantic contract')
        closure = [deepcopy(n) for n in p['nodes'] if n['name'] in used]
        fact_inputs = {name: {'schema': schema[name], 'semantics': input_semantics[name]} for name in sorted(leaves)}
        # A cached raw output is a virtual fact for CS execution. Expand its
        # immutable semantic receipt for identity only, so storing that layer
        # does not change the full DAG's mathematical feature identity.
        expanded = []; aliases = {}
        for name in sorted(leaves):
            derived = input_semantics[name].get('derived_feature')
            if derived is None: continue
            require(derived['abi'] == p['abi'] and derived['semantics'] == p['semantics'] and
                    derived['history_policy'] == p['history_policy'] and
                    derived['observation_domain'] == p['observation_domain'], 'derived Feature semantic profile mismatch')
            require(all(derived['output']['column'][key] == schema[name][key] for key in ('dtype','unit','missing')),
                    'derived Feature fact schema mismatch')
            aliases[name] = derived['output']['node']
            del fact_inputs[name]
            for field, contract in derived['inputs'].items():
                require(field not in fact_inputs or fact_inputs[field] == contract, 'conflicting derived input semantics')
                fact_inputs[field] = deepcopy(contract)
            for n in derived['nodes']:
                require(derived['operator_semantics'][n['op']]==OPERATOR_SEMANTIC_REVISIONS[n['op']],
                        'derived raw semantic revision is stale; rebuild affected output')
                previous = next((old for old in expanded if old['name'] == n['name']), None)
                require(previous is None or previous == n, 'conflicting derived DAG node')
                if previous is None: expanded.append(deepcopy(n))
        for n in closure:
            n['inputs'] = [aliases.get(a,a) for a in n['inputs']]
        closure = expanded + closure
        semantic_output = deepcopy(output)
        semantic_output['node'] = aliases.get(semantic_output['node'], semantic_output['node'])
        semantic = {'contract_version': 'feature_semantics_v1', 'abi': p['abi'],
            'semantics': p['semantics'], 'observation_domain': p['observation_domain'],
            'history_policy': p['history_policy'], 'nodes': closure,
            'operator_semantics':{n['op']:OPERATOR_SEMANTIC_REVISIONS[n['op']] for n in closure},
            'inputs': fact_inputs, 'output': semantic_output}
        expanded_plan = deepcopy(p)
        expanded_plan.update(nodes=closure, input_schema=[v['schema'] for v in fact_inputs.values()],
                             outputs=[semantic_output])
        depth = required_history(FeaturePlan.from_dict(expanded_plan))[output['column']['name']]
        name = output['column']['name']
        result[name] = {'semantic_ref': digest(semantic), 'semantic': semantic,
            'schema': deepcopy(output['column']), 'inputs': sorted(leaves),
            'reference_dependent': any(n['op'] in CS for n in closure),
            'history_sessions': depth + 1}
    return result


def select_feature_plan(plan, columns):
    """Project the original Core DAG to requested outputs, without translation."""
    p = validate_plan(plan, execution=True)
    outputs = {o['column']['name']: o for o in p['outputs']}
    require(type(columns) is list and bool(columns) and len(set(columns)) == len(columns)
            and set(columns) <= set(outputs), 'ordered unique plan output subset required')
    nodes = {n['name']: n for n in p['nodes']}; needed = set(); leaves=set()
    facts={c['name'] for c in p['input_schema']}
    def include(name):
        if name in nodes and name not in needed:
            needed.add(name)
            for parent in nodes[name]['inputs']: include(parent)
        elif name in facts: leaves.add(name)
    for name in columns: include(outputs[name]['node'])
    p['nodes'] = [n for n in p['nodes'] if n['name'] in needed]
    p['input_schema'] = [c for c in p['input_schema'] if c['name'] in leaves]
    streams={n['params']['stream'] for n in p['nodes'] if n['op']=='asof'}
    p['event_schema']={name:schema for name,schema in p['event_schema'].items() if name in streams}
    p['outputs'] = [outputs[name] for name in columns]
    # This execution receipt may differ from the original full-plan receipt.
    # Feature identity above never uses this experiment-wide recipe hash.
    p['recipe_ref'] = digest({'original_recipe_ref': p['recipe_ref'], 'outputs': columns})
    return FeaturePlan.from_dict(p)


def select_security_plan(plan, securities):
    """Project frozen reference metadata only for a reference-independent DAG."""
    p=plan.to_dict()
    require(not any(n['op'] in CS for n in p['nodes']),
            'cannot project a cross-sectional Feature execution cohort')
    selected=set(securities)
    p['reference_members']={day:{s:group for s,group in members.items() if s in selected}
                            for day,members in p['reference_members'].items()}
    p['reference_ref']=digest({'reference_ref':p['reference_ref'],'security_ids':securities})
    return FeaturePlan.from_dict(p)
