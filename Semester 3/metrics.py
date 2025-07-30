import numpy as np

_EPS = 1e-12

def _safe_var(x, ddof=1):
    v = np.var(x, ddof=ddof)
    return v if v > _EPS else np.nan

def bullwhip_ratio(orders, demand, *, use_cv=False, ddof=1):
    """
    BW = Var(orders) / Var(demand)   (default)
    If use_cv=True, BW = (CV_orders^2) / (CV_demand^2) (used in the literature).
    Returns NaN when either variance (or mean for CV) is ~0 so downstream code can handle it.
    """
    orders = np.asarray(orders, dtype=float)
    demand = np.asarray(demand, dtype=float)

    if use_cv:
        mu_o, mu_d = np.mean(orders), np.mean(demand)
        v_o, v_d = _safe_var(orders, ddof), _safe_var(demand, ddof)
        if (np.isnan(v_o) or np.isnan(v_d) or abs(mu_o) < _EPS or abs(mu_d) < _EPS):
            return np.nan
        return (v_o / (mu_o**2)) / (v_d / (mu_d**2))

    v_o, v_d = _safe_var(orders, ddof), _safe_var(demand, ddof)
    if np.isnan(v_o) or np.isnan(v_d):
        return np.nan
    return v_o / v_d

def ripple_ratio(orders, shipments, ddof=1):
    orders = np.asarray(orders, dtype=float)
    shipments = np.asarray(shipments, dtype=float)
    v_o, v_s = _safe_var(orders, ddof), _safe_var(shipments, ddof)
    if np.isnan(v_o) or np.isnan(v_s):
        return np.nan
    return v_s / v_o

def bullwhip_ripple_index(orders, demand, shipments, **kwargs):
    bw = bullwhip_ratio(orders, demand, **kwargs)
    re = ripple_ratio(orders, shipments, **kwargs)
    return bw * re if (np.isfinite(bw) and np.isfinite(re)) else np.nan

def total_costs(inventory_levels, backorders, holding_cost, backorder_cost):
    inv = np.asarray(inventory_levels, dtype=float)
    bo = np.asarray(backorders, dtype=float)
    return float(np.sum(
        holding_cost * np.maximum(inv, 0.0) +
        backorder_cost * np.maximum(bo, 0.0)
    ))

def evaluate_run(history, holding_cost, backorder_cost):
    orders     = history['orders']
    demand     = history['demand']
    shipments  = history['shipments']
    inventory  = history['inventory_levels']
    backorders = history['backorders']

    return {
        'BW':   bullwhip_ratio(orders, demand),
        'RE':   ripple_ratio(orders, shipments),
        'BRI':  bullwhip_ripple_index(orders, demand, shipments),
        'Cost': total_costs(inventory, backorders, holding_cost, backorder_cost)
    }
def evaluate_phases(history, disruption_info, holding_cost, backorder_cost):
    """
    Calculate KPIs for pre/during/post disruption phases (for RQ2).
    
    Parameters:
        history: dict with time series data
        disruption_info: dict with 'start' and 'duration'
        holding_cost: per-unit holding cost
        backorder_cost: per-unit backorder cost
    
    Returns:
        dict with phase-specific KPIs
    """
    start = disruption_info['start']
    duration = disruption_info['duration']
    total_periods = len(history['orders'])
    
    # Define phase boundaries
    pre_end = start
    during_end = start + duration
    
    # Initialize phase data
    phase_kpis = {}
    
    # Calculate KPIs for each phase
    for phase_name, phase_start, phase_end in [
        ('pre', 0, pre_end),
        ('during', start, during_end),
        ('post', during_end, total_periods)
    ]:
        # Skip if phase is empty
        if phase_start >= phase_end or phase_start >= total_periods:
            phase_kpis[phase_name] = {
                'BW': np.nan,
                'RE': np.nan,
                'BRI': np.nan,
                'Cost': 0.0
            }
            continue
            
        # Extract phase-specific data
        phase_history = {
            'orders': history['orders'][phase_start:phase_end],
            'demand': history['demand'][phase_start:phase_end],
            'shipments': history['shipments'][phase_start:phase_end],
            'inventory_levels': history['inventory_levels'][phase_start:phase_end],
            'backorders': history['backorders'][phase_start:phase_end]
        }
        
        # Calculate phase KPIs
        phase_kpis[phase_name] = {
            'BW': bullwhip_ratio(phase_history['orders'], phase_history['demand']),
            'RE': ripple_ratio(phase_history['orders'], phase_history['shipments']),
            'BRI': bullwhip_ripple_index(phase_history['orders'], 
                                       phase_history['demand'], 
                                       phase_history['shipments']),
            'Cost': total_costs(phase_history['inventory_levels'], 
                              phase_history['backorders'],
                              holding_cost, 
                              backorder_cost)
        }
    
    return phase_kpis