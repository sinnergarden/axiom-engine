"""Record-origin attribution of saved rights; never changes account execution.

Shares are assigned to the record holders. Mixed holdings are disposed FIFO by
original episode entry. Rational allocations are exact; integer cash uses
largest remainder, with entry order breaking ties, conserving every saved fen.
"""
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
from fractions import Fraction

from ..core.contracts import require
from .episode_evaluation import _new_episode, _actions, summarize_episodes


def _money(amount, weights):
    total=sum(weights,Fraction(0))
    require(total>0,'positive attribution weights required')
    if amount<0: return [-x for x in _money(-amount,weights)]
    exact=[Fraction(amount)*w/total for w in weights]
    result=[x.numerator//x.denominator for x in exact]
    for i in sorted(range(len(weights)),key=lambda i:(-(exact[i]-result[i]),i))[:amount-sum(result)]:
        result[i]+=1
    return result


def _quantity(value):
    if value.denominator==1: return value.numerator
    with localcontext(Context(prec=40,rounding=ROUND_HALF_UP)):
        return str(Decimal(value.numerator)/value.denominator)


def evaluate_equity_episodes(run, scope, binding):
    from .stock_equity import SIMULATION
    require(run['plan']['profile']['equity_simulation']==SIMULATION,'frozen equity attribution required')
    require(run['plan']['initial_account']['positions']=={},'equity episodes require observed entry')
    # Same-event scope contradictions are still rejected. Supplemental facts,
    # rather than an evaluation-only correction, define this account's rights.
    observed=_actions(run,scope)
    facts=run['plan']['market_replay']['equity_facts']['actions']
    actions={a['economic_event_id']:a for a in facts}
    require(all(a['event_id'] in actions for a in observed),'evaluation cannot add a v8 account action')
    episodes=[]; active={}; registered={}; ex_seen=set(); paid=set()
    events=[(f['sequence'],'FILL',f) for f in run['fills']]
    events += [(r['sequence'],r['reason'],r) for r in run['position_ledger'] if r['reason'].startswith('EQUITY_')]
    events += [(r['sequence'],r['reason'],r) for r in run['cash_ledger'] if r['reason'] in ('DIVIDEND_EX','DIVIDEND_PAY')]
    require(len({s for s,_,_ in events})==len(events),'duplicate equity attribution sequence')
    cash_events={(r['reason'],r['source_event_id']):r for r in run['cash_ledger']}
    for _,kind,row in sorted(events,key=lambda e:e[0]):
        key=row.get('source_event_id'); security=row.get('security_id')
        if kind=='FILL':
            require(cash_events.get(('FILL',row['fill_id']),{}).get('cash_delta_minor')==row['cash_delta_minor'],
                'equity attribution requires saved fill cash')
            if row['side']=='BUY':
                episode=active.get(security)
                if episode is None or episode['final_quantity']==0:
                    episode=_new_episode(binding,security,row)
                    episode.update(final_quantity=Fraction(0),pending_share_quantity=Fraction(0),
                        rights=[],fill_allocations=[])
                    episodes.append(episode); active[security]=episode
                allocation=[(episode,Fraction(row['quantity']))]
                episode['buy_cost_minor']-=row['cash_delta_minor']
                episode['final_quantity']+=row['quantity']
            else:
                remaining=Fraction(row['quantity']); allocation=[]
                for e in episodes:
                    if e['security_id']!=security or not e['final_quantity']: continue
                    quantity=min(remaining,e['final_quantity'])
                    allocation.append((e,quantity)); e['final_quantity']-=quantity; remaining-=quantity
                    if e['final_quantity']==0:
                        e.update(exit_session=row['session'],exit_sequence=row['sequence'])
                    if not remaining: break
                require(remaining==0,'saved sell exceeds record-origin holdings')
                proceeds=_money(row['cash_delta_minor'],[q for _,q in allocation])
                for (e,_),value in zip(allocation,proceeds): e['sell_proceeds_minor']+=value
                if active.get(security) is not None and active[security]['final_quantity']==0: del active[security]
            fees=_money(row['fee_minor'],[q for _,q in allocation])
            for (e,q),fee in zip(allocation,fees):
                e['fees_minor']+=fee
                e['fill_refs'].append(dict(fill_id=row['fill_id'],sequence=row['sequence']))
                e['fill_allocations'].append(dict(fill_id=row['fill_id'],sequence=row['sequence'],
                    side=row['side'],quantity_fraction=dict(numerator=q.numerator,denominator=q.denominator),fee_minor=fee))
        elif kind=='EQUITY_REGISTER':
            owners=[(e,e['final_quantity']) for e in episodes if e['security_id']==security and e['final_quantity']]
            require(sum((q for _,q in owners),Fraction(0))==row['record_quantity'],'record owners disagree with saved registration')
            registered[key]=owners
            for e,q in owners:
                e['rights'].append(dict(economic_event_id=key,record_sequence=row['sequence'],
                    record_quantity_fraction=dict(numerator=q.numerator,denominator=q.denominator),
                    revision_ref=actions[key]['revision_ref']))
        elif kind=='EQUITY_EX':
            owners=registered.get(key,[]); a=actions[key]
            grants=[q*Fraction(a['stock_distribution_shares_per_share']) for _,q in owners]
            require(sum(grants,Fraction(0))==row['pending_share_delta'],'saved share rights disagree with record origin')
            for (e,_),grant in zip(owners,grants): e['pending_share_quantity']+=grant
            ex_seen.add(key)
        elif kind=='EQUITY_LISTING':
            owners=registered.get(key,[]); a=actions[key]
            for e,q in owners:
                grant=q*Fraction(a['stock_distribution_shares_per_share'])
                e['pending_share_quantity']-=grant; e['final_quantity']+=grant
                if grant: e.update(exit_session=None,exit_sequence=None)
        elif kind=='DIVIDEND_EX':
            owners=registered.get(key,[])
            require(bool(owners),'saved dividend has no record origin')
            allocations=_money(row['receivable_delta_minor'],[q for _,q in owners])
            for (e,q),value in zip(owners,allocations):
                e['dividend_income_minor']+=value; e['receivable_minor']+=value
                a=actions[key]
                e['dividends'].append(dict(event_id=key,record_session=a['record_date'],ex_session=a['ex_date'],
                    pay_session=next((s for s in run['plan']['market_replay']['calendar'] if a['payment_date'] is not None and s>=a['payment_date']),None),
                    entitlement_quantity=_quantity(q),recognition_sequence=row['sequence'],payment_sequence=None,
                    recognized_minor=value,pending_minor=0,receivable_minor=value,source_refs=a['source_refs'],
                    payment_status='PENDING',tax_convention=SIMULATION['tax_convention']))
        elif kind=='DIVIDEND_PAY':
            owners=registered.get(key,[]); total=0
            for e,_ in owners:
                d=next(d for d in e['dividends'] if d['event_id']==key)
                total+=d['receivable_minor']; e['receivable_minor']-=d['receivable_minor']
                d.update(receivable_minor=0,payment_sequence=row['sequence'],payment_status='PAID')
            require(total==row['cash_delta_minor'],'saved payment disagrees with origin income'); paid.add(key)
    final=run['final_account']['positions']; snapshots={p['security_id']:p for p in run['positions'] if p['session']==run['plan']['end_session']}
    marks={}
    for security in set(final)|{e['security_id'] for e in episodes}:
        owners=[e for e in episodes if e['security_id']==security]
        require(sum((e['final_quantity'] for e in owners),Fraction(0))==final.get(security,{}).get('quantity',0),
            'record-origin episodes disagree with final physical shares')
        weights=[e['final_quantity']+e['pending_share_quantity'] for e in owners]
        if sum(weights,Fraction(0)):
            point=snapshots.get(security); require(point is not None,'owned rights lack final saved mark')
            require(sum(weights,Fraction(0))==point['quantity']+point['pending_share_quantity'],'episode rights disagree with final NAV')
            for e,value in zip(owners,_money(point['market_value_minor'],weights)): marks[e['episode_id']]=value
    for e in episodes:
        awaiting=any(r['economic_event_id'] not in ex_seen for r in e['rights'])
        future_grant=any(r['economic_event_id'] not in ex_seen and actions[r['economic_event_id']]['stock_distribution_shares_per_share'] is not None and
            Fraction(actions[r['economic_event_id']]['stock_distribution_shares_per_share'])>0 for r in e['rights'])
        open_position=bool(e['final_quantity'] or e['pending_share_quantity'] or future_grant)
        e['status']='OPEN' if open_position else 'CLOSED'
        reasons=(['OPEN_POSITION'] if open_position else [])+(['KNOWN_INCOME_PENDING_EX'] if awaiting else [])
        pnl=e['sell_proceeds_minor']+e['dividend_income_minor']-e['buy_cost_minor']
        e.update(statistics_eligible=not reasons,exclusion_reasons=reasons,income_status='PENDING_EX' if awaiting else
            ('RECOGNIZED' if e['rights'] else ('NO_OBSERVED_ENTITLEMENT' if scope is not None else 'COVERAGE_UNKNOWN')),
            pending_dividend_minor=None if awaiting else 0,net_pnl_minor=None if reasons else pnl,
            marked_pnl_minor=pnl+marks.get(e['episode_id'],0) if open_position and not awaiting else None,
            return_denominator_minor=e['buy_cost_minor'],net_return=str(Decimal(pnl)/e['buy_cost_minor']) if not reasons and e['buy_cost_minor'] else None,
            attribution_policy=SIMULATION['attribution'],final_quantity_fraction=dict(numerator=e['final_quantity'].numerator,denominator=e['final_quantity'].denominator),
            pending_share_quantity_fraction=dict(numerator=e['pending_share_quantity'].numerator,denominator=e['pending_share_quantity'].denominator))
        e['final_quantity']=_quantity(e['final_quantity']); e['pending_share_quantity']=_quantity(e['pending_share_quantity'])
    result,metrics=summarize_episodes(run,scope,episodes)
    metrics.update(attribution_policy=SIMULATION['attribution'],cash_allocation='largest_remainder_entry_order',
        pending_share_episode_count=sum(bool(e['pending_share_quantity']) for e in episodes))
    return result,metrics
