"""Frozen Data-resolved actions and the existing ledger's daily equity phases.

Dates describe an explicit offline simulation, never an opening cash receipt.
No announcement-alias inference or factor-to-share calculation occurs here.
"""
from copy import deepcopy
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext

from ..core.contracts import Document, canonical, digest, fields, integer, require, session
from ..core.portfolio import decimal, minor
from ..core.stock_portfolio import instant

EQUITY_POLICY = 'registered_equity_v1'
SIMULATION = dict(contract_version='stock_equity_simulation_v1',
    tax_convention='gross_before_tax_no_personal_tax_model',
    availability='pay_and_listing_before_decision_date_proxy',
    non_session_dates='next_exchange_session', valuation='unadjusted_close_pending_shares',
    event_evidence='declared_vendor_assumption_with_actual_observation_clock',
    cost='zero_added_cost_dilute_at_listing', fractional_shares='reject_non_integer',
    attribution='record_origin_fifo_v1',
    limitation='EXPLICIT DAILY SIMULATION: payout/listing dates proxy pre-decision availability, not actual credited cash or delivered shares. Gross dividends exclude personal holding-period taxes.')


class StockActionFacts(Document):
    """Reference-bound Data identities, revisions and values; not a resolver."""


def equity_profile(*, execution_rules, fee_schedule, unknown_status_policy='block'):
    from .profiles import stock_daily_open_profile_v2
    profile = stock_daily_open_profile_v2(execution_rules=execution_rules, fee_schedule=fee_schedule,
                                         unknown_status_policy=unknown_status_policy)
    profile.update(contract_version='stock_daily_open_profile_v3', equity_simulation=deepcopy(SIMULATION))
    profile['limitation'] += ' ' + SIMULATION['limitation']
    return profile


def equity_request(request, *, profile_input, action_facts_artifact):
    """Explicit v8 fork of a frozen v7 request; saved predictions stay intact."""
    from .backtest import BacktestRequest
    from .stock_stream_contracts import logical_ref, validate_artifact_ref, validate_manifest
    require(isinstance(request, BacktestRequest), 'BacktestRequest required')
    plan = validate_manifest(request)
    require(plan['contract_version'] == 'backtest_request_v7', 'v7 frozen request required')
    validate_artifact_ref(action_facts_artifact)
    plan.update(contract_version='backtest_request_v8', stock_action_policy=EQUITY_POLICY,
                profile_input=deepcopy(profile_input), action_facts_artifact=deepcopy(action_facts_artifact))
    plan['request_ref'] = logical_ref(plan, 'request_ref')
    return BacktestRequest.from_dict(validate_manifest(plan))


def validate_facts(wire, *, universe, calendar, parent_refs):
    fields(wire, 'contract_version producer snapshot_id observed_at availability_basis universe calendar parent_native_refs supplemental_view_refs identity_policy_ref actions limitations')
    require(wire['contract_version'] == 'stock_action_facts_v1' and wire['producer'] == 'axiom-data',
            'Data-resolved stock action facts required')
    require(type(wire['snapshot_id']) is str and bool(wire['snapshot_id']) and
            wire['universe'] == universe and wire['calendar'] == calendar and
            wire['parent_native_refs'] == list(parent_refs), 'equity facts scope/parent mismatch')
    digest(wire['identity_policy_ref'])
    require(type(wire['supplemental_view_refs']) is list and bool(wire['supplemental_view_refs']) and
        len(set(wire['supplemental_view_refs']))==len(wire['supplemental_view_refs']), 'new corporate-action ViewRefs required')
    for ref in wire['supplemental_view_refs']: digest(ref)
    instant(wire['observed_at'])
    require(wire['availability_basis']=='declared_vendor_assumption', 'explicit supplemental event clock basis required')
    require(type(wire['actions']) is list and type(wire['limitations']) is list and
            all(type(x) is str for x in wire['limitations']), 'explicit action catalog required')
    ids, aliases = set(), set()
    for a in wire['actions']:
        fields(a, 'economic_event_id revision_ref identity_status security_id report_period record_date ex_date payment_date stock_listing_date cash_dividend_before_tax_per_share stock_distribution_shares_per_share bonus_shares_per_share capital_transfer_shares_per_share available_at first_observed_at raw_batch_ids source_refs aliases')
        require(type(a['economic_event_id']) is str and bool(a['economic_event_id']), 'Data economic event identity required'); digest(a['revision_ref'])
        require(a['economic_event_id'] not in ids, 'duplicate economic action identity')
        ids.add(a['economic_event_id'])
        require(a['identity_status'] in ('RESOLVED', 'UNRESOLVED') and a['security_id'] in universe,
                'invalid action identity status/security')
        session(a['report_period']); instant(a['available_at'])
        require(instant(a['first_observed_at'])<=instant(wire['observed_at']), 'action receipt after supplemental observation')
        require(type(a['raw_batch_ids']) is list and bool(a['raw_batch_ids']) and
            all(type(b) is str and b.startswith('b_') for b in a['raw_batch_ids']), 'original action Raw receipts required')
        require(instant(a['available_at']) <= instant(calendar[-1]+'T20:30:00+08:00'), 'future action evidence')
        for k in ('record_date', 'ex_date', 'payment_date', 'stock_listing_date'):
            if a[k] is not None: session(a[k])
        if a['record_date'] is not None and a['ex_date'] is not None:
            require(a['record_date'] < a['ex_date'], 'action record must precede EX')
        for k in ('payment_date', 'stock_listing_date'):
            require(a[k] is None or a['ex_date'] is None or a[k] >= a['ex_date'], 'action settlement precedes EX')
        for k in ('cash_dividend_before_tax_per_share', 'stock_distribution_shares_per_share', 'bonus_shares_per_share', 'capital_transfer_shares_per_share'):
            if a[k] is not None:
                require(type(a[k]) is str, 'exact decimal action rate required')
                decimal(a[k], minimum=0)
        require(type(a['source_refs']) is list and bool(a['source_refs']) and
                len(set(a['source_refs'])) == len(a['source_refs']), 'action source refs required')
        for ref in a['source_refs']: digest(ref)
        require(bool(set(a['source_refs']) & set(wire['supplemental_view_refs'])), 'event lacks supplemental ViewRef binding')
        require(type(a['aliases']) is list and bool(a['aliases']), 'Data action aliases required')
        for alias in a['aliases']:
            fields(alias, 'security_id report_period announcement_date process_status')
            require(alias['security_id'] == a['security_id'] and alias['report_period'] == a['report_period'],
                    'action alias identity mismatch')
            session(alias['announcement_date'])
            require(type(alias['process_status']) is str, 'alias status required')
            key = canonical(alias)
            require(key not in aliases, 'alias assigned more than once')
            aliases.add(key)
    return wire


def mapped_session(day, calendar):
    return None if day is None else next((s for s in calendar if s >= day), None)


def cash_action(action, calendar):
    return dict(contract_version='stock_cash_action_v1', event_id=action['economic_event_id'],
        security_id=action['security_id'], record_session=action['record_date'],
        ex_session=action['ex_date'], pay_session=mapped_session(action['payment_date'], calendar),
        cash_before_tax_per_share=action['cash_dividend_before_tax_per_share'],
        tax_convention=SIMULATION['tax_convention'], available_at=action['available_at'],
        source_refs=action['source_refs'])


def _journal(ledger, a, day, phase, **values):
    ledger.sequence += 1
    row = dict(sequence=ledger.sequence, session=day,
        security_id=a['security_id'], quantity_delta=0, sellable_delta=0, cost_delta_minor=0,
        reason='EQUITY_'+phase, source_event_id=a['economic_event_id'], revision_ref=a['revision_ref'])
    row.update(values)
    ledger._append_output('position_ledger', row)


def register(ledger, action, day):
    key = action['economic_event_id']
    payload = canonical(action)
    if instant(action['available_at']) > instant(day+'T20:30:00+08:00'): return
    if key in ledger.equity_entitlements:
        require(ledger.equity_entitlements[key]['payload'] == payload, 'conflicting equity registration')
        return
    quantity = ledger.positions.get(action['security_id'], {}).get('quantity', 0)
    pending_quantity = pending(ledger).get(action['security_id'], 0)
    _journal(ledger, action, day, 'REGISTER', record_quantity=quantity,
             record_pending_quantity=pending_quantity)
    ledger.equity_entitlements[key] = dict(payload=payload, security_id=action['security_id'],
        record_quantity=quantity, record_pending_quantity=pending_quantity,
        record_sequence=ledger.sequence, pending_quantity=0,
        ex_applied=False, cash_paid=False, shares_listed=False)


def gaps(ledger, actions, day, decision_time):
    result = []
    for a in actions:
        state = ledger.equity_entitlements.get(a['economic_event_id'])
        if state is None:
            # Missing record date cannot manufacture an entitlement. Retain
            # uncertainty only for this account's observed ownership history.
            first_owned = ledger.equity_owned.get(a['security_id'])
            quantity = (ledger.equity_history.get((a['record_date'], a['security_id']), 0)
                        if a['record_date'] is not None else int(first_owned is not None and
                            (a['ex_date'] is None or first_owned < a['ex_date'])))
            if not quantity: continue
            state = {'record_quantity': quantity}
        elif not state['record_quantity'] and not state.get('record_pending_quantity'): continue
        if a['ex_date'] is not None and day < a['ex_date']: continue
        reason = None; unrounded = None
        if a['identity_status'] != 'RESOLVED': reason = 'UNRESOLVED_ACTION_IDENTITY'
        elif state.get('record_pending_quantity'): reason = 'UNRESOLVED_REGISTERED_SHARE_BASIS'
        elif a['record_date'] is None or a['ex_date'] is None: reason = 'MISSING_EQUITY_DATE'
        elif instant(a['available_at']) > instant(a['record_date']+'T20:30:00+08:00'): reason = 'LATE_ENTITLEMENT_FACT'
        elif instant(a['available_at']) > instant(day+'T'+decision_time): reason = 'UNAVAILABLE_EQUITY_FACT'
        elif a['cash_dividend_before_tax_per_share'] is None: reason = 'UNKNOWN_CASH_ENTITLEMENT'
        elif a['stock_distribution_shares_per_share'] is None: reason = 'UNKNOWN_SHARE_ENTITLEMENT'
        else:
            with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
                grant = Decimal(state['record_quantity'])*decimal(a['stock_distribution_shares_per_share'])
            if grant != grant.to_integral_value():
                reason = 'UNSUPPORTED_FRACTIONAL_ENTITLEMENT'; unrounded = str(grant)
            elif decimal(a['cash_dividend_before_tax_per_share']) > 0 and a['payment_date'] is None: reason = 'MISSING_PAY_DATE'
            elif grant > 0 and a['stock_listing_date'] is None: reason = 'MISSING_LIST_DATE'
        if reason:
            gap=dict(economic_event_id=a['economic_event_id'], revision_ref=a['revision_ref'],
                security_id=a['security_id'], reason=reason, source_refs=a['source_refs'])
            if unrounded is not None: gap['unrounded_share_quantity']=unrounded
            result.append(gap)
    return result


def advance(ledger, actions, day):
    """One shared phase path; caller prevalidates all qualified gaps atomically."""
    for a in sorted(actions, key=lambda a:a['economic_event_id']):
        state = ledger.equity_entitlements.get(a['economic_event_id'])
        if state is None or not state['record_quantity']: continue
        if a['ex_date'] == day and not state['ex_applied']:
            with localcontext(Context(prec=40, rounding=ROUND_HALF_UP)):
                grant = Decimal(state['record_quantity'])*decimal(a['stock_distribution_shares_per_share'])
            require(grant == grant.to_integral_value(), 'fractional entitlement unsupported')
            if decimal(a['cash_dividend_before_tax_per_share']) > 0:
                ledger.dividend(cash_action(a, ledger.calendar), 'EX', state['record_quantity'])
            state['pending_quantity'] = int(grant); state['ex_applied'] = True
            _journal(ledger, a, day, 'EX', pending_share_delta=int(grant))
    for a in sorted(actions, key=lambda a:a['economic_event_id']):
        state = ledger.equity_entitlements.get(a['economic_event_id'])
        if state is None or not state['record_quantity'] or not state['ex_applied']: continue
        if mapped_session(a['payment_date'], ledger.calendar) == day and not state['cash_paid'] and decimal(a['cash_dividend_before_tax_per_share']) > 0:
            ledger.dividend(cash_action(a, ledger.calendar), 'PAY', state['record_quantity'])
            state['cash_paid'] = True
    for a in sorted(actions, key=lambda a:a['economic_event_id']):
        state = ledger.equity_entitlements.get(a['economic_event_id'])
        if state is None or not state['record_quantity'] or not state['ex_applied']: continue
        if mapped_session(a['stock_listing_date'], ledger.calendar) == day and not state['shares_listed']:
            grant = state['pending_quantity']
            if grant:
                p = ledger.positions.setdefault(a['security_id'], dict(quantity=0, sellable_quantity=0, cost_minor=0))
                p['quantity'] += grant; p['sellable_quantity'] += grant
            state['pending_quantity'] = 0; state['shares_listed'] = True
            _journal(ledger, a, day, 'LISTING', quantity_delta=grant, sellable_delta=grant, pending_share_delta=-grant)


def pending(ledger):
    result = {}
    for state in ledger.equity_entitlements.values():
        security = state['security_id']
        result[security] = result.get(security, 0)+state['pending_quantity']
    return {s:q for s,q in result.items() if q}


def verify_saved(wire, rows, profile):
    """Replay saved accounting facts only; no planner, market source or broker."""
    from .accounting import AccountLedger
    require(profile == equity_profile(execution_rules=profile['stock_execution_rules'],
        fee_schedule=profile['stock_fee_schedule'], unknown_status_policy=profile['unknown_status_policy']),
        'saved equity simulation profile mismatch')
    scope = wire['request_manifest']['scope']; calendar = scope['calendar']
    actions = wire['account_events']['equity_facts']['actions']
    ledger = AccountLedger(cash_minor=wire['initial_nav_minor'], calendar=calendar,
        settlement_sessions=profile['settlement_sessions'])
    nav = {p['session']:p for p in rows['nav']}
    require(len(nav) == len(rows['nav']), 'duplicate saved equity NAV')
    days = [d for d in calendar if scope['start_session'] <= d <= scope['end_session']]
    require(list(nav) == days[:len(nav)], 'saved equity NAV is not a committed prefix')
    if wire['status'] == 'COMPLETE': require(list(nav) == days, 'incomplete saved equity account')
    else:
        require(len(nav) < len(days) and wire['stopped']['session'] == days[len(nav)], 'invalid equity stopped prefix')
    phases = {p['session']:p for p in rows['session_phases']}
    expected_days = list(nav) + ([] if wire['status']=='COMPLETE' else [wire['stopped']['session']])
    require(len(phases) == len(rows['session_phases']) and list(phases) == expected_days, 'equity phase coverage mismatch')
    fills_by_day = {d:[] for d in expected_days}
    for fill in rows['fills']:
        require(fill['session'] in fills_by_day, 'equity fill outside committed scope')
        fills_by_day[fill['session']].append(fill)
    for day in expected_days:
        ledger.advance(day)
        missing = gaps(ledger, actions, day, profile['decision_time_utc'])
        stopped = day not in nav
        if stopped and wire['stopped']['reason'] == 'HELD_EQUITY_FACT_GAP':
            require(missing == wire['stopped']['gaps'] and not fills_by_day[day], 'saved equity stop does not match qualified gap')
        else:
            require(not missing, 'saved equity account advanced through qualified gap')
            advance(ledger, actions, day)
            if not stopped:
                for fill in fills_by_day[day]: ledger.apply_fill(fill)
                for security, position in ledger.positions.items():
                    if position['quantity']: ledger.equity_owned.setdefault(security, day)
                for a in actions:
                    if a['record_date'] == day:
                        ledger.equity_history[day,a['security_id']] = ledger.positions.get(a['security_id'],{}).get('quantity',0)
                        register(ledger,a,day)
                ledger.sequence += 1
                point = nav[day]; shares = pending(ledger)
                snapshots = {p['security_id']:p for p in rows['positions'] if p['session']==day}
                require(len(snapshots) == sum(p['session']==day for p in rows['positions']), 'duplicate equity position snapshot')
                expected = {s for s,p in ledger.positions.items() if p['quantity']} | set(shares)
                require(set(snapshots) == expected, 'saved equity positions omit owned rights')
                value = 0
                for security,p in snapshots.items():
                    position = ledger.positions.get(security,dict(quantity=0,sellable_quantity=0,cost_minor=0))
                    require(all(p[k]==position[k] for k in position) and
                        p['pending_share_quantity']==shares.get(security,0) and p['committed_sequence']==ledger.sequence,
                        'saved equity position/rights/cost mismatch')
                    require(not shares.get(security) or p['mark_session']==day, 'stale pending-share mark')
                    with localcontext(Context(prec=40,rounding=ROUND_HALF_UP)):
                        amount=minor(decimal(p['mark_price'],minimum=0)*(position['quantity']+shares.get(security,0))*100)
                    require(amount==p['market_value_minor'],'saved original-price equity valuation mismatch')
                    value += amount
                receivable=sum(ledger.receivables.values())
                require(point['cash_minor']==ledger.cash and point['receivable_minor']==receivable and
                    point['market_value_minor']==value and point['nav_minor']==ledger.cash+receivable+value and
                    point['committed_sequence']==ledger.sequence,'saved equity NAV does not conserve value')
                with localcontext(Context(prec=40,rounding=ROUND_HALF_UP)):
                    require(decimal(point['nav_index'])==Decimal(point['nav_minor'])/wire['initial_nav_minor'],'saved equity index mismatch')
        marker=phases[day]
        require(marker['phase']==('STOPPED_BEFORE_NAV' if stopped else 'SESSION_COMMITTED') and
            marker['committed_sequence']==ledger.sequence,'saved equity phase watermark mismatch')
    require(ledger.fills==rows['fills'] and ledger.cash_ledger==rows['cash_ledger'] and
        ledger.position_ledger==rows['position_ledger'],'saved equity journals do not reconcile')
    require(wire['final_account']==dict(cash_minor=ledger.cash,receivable_minor=sum(ledger.receivables.values()),
        positions=ledger.positions,committed_sequence=ledger.sequence,equity_entitlements=ledger.equity_entitlements),
        'saved final equity account mismatch')
    require(wire['committed_sequence']==ledger.sequence,'saved equity final watermark mismatch')
    if wire['status']=='BLOCKED':
        require(wire['stopped']['committed_sequence']==ledger.sequence,'saved equity stop watermark mismatch')
    with localcontext(Context(prec=40,rounding=ROUND_HALF_UP)):
        peak=wire['initial_nav_minor']; drawdown=Decimal(0)
        for point in rows['nav']:
            peak=max(peak,point['nav_minor'])
            drawdown=min(drawdown,Decimal(point['nav_minor'])/peak-1)
        expected_return=None if wire['status']=='BLOCKED' else str(Decimal(rows['nav'][-1]['nav_minor'])/wire['initial_nav_minor']-1)
    require(wire['metrics']['total_return']==expected_return and wire['metrics']['max_drawdown']==
        (None if wire['status']=='BLOCKED' else str(drawdown)), 'saved equity return/drawdown mismatch')
