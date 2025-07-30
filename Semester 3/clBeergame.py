#!/usr/bin/env python
# coding: utf-8

import random
import time
import numpy as np
from collections import deque
import scipy.stats as stats

from BGAgent import Agent
from env.disruption import DisruptionProcess
from env.warning import EarlyWarningSignal
from metrics import evaluate_run, evaluate_phases

class clBeerGame:
    """
    Single-ordering-agent Beer Game environment: plays one retailer with Sterman, DQN, and EWS phases.
    """
    def __init__(self, config):
        self.config = config
        # Initialize one retailer agent
        self.player = Agent(
            agentNum=0,
            ILInit=config.ILInit,
            AOInit=config.AOInit,
            ASInit=config.ASInit,
            c_h=config.c_h,
            c_p=config.c_p,
            eta=config.eta,
            compuType=config.agentTypes[0],
            config=config
        )
        
        # Disruption process
        self.disruption = DisruptionProcess(config)
        
        # Time trackers
        self.curTime = 0
        self.T = 0

    def resetGame(self, demand, playType):
        """
        Prepare a new episode with given demand sequence and play type.
        playType: 'train' or 'test'
        """
        self.demand = demand
        self.playType = playType
        self.curTime = 0
        
        # Reset disruption process
        self.disruption.reset_episode()
        Ts, D = self.disruption.Ts, self.disruption.D
        
        # Determine episode length
        if playType == 'train':
            self.T = random.randint(self.config.TLow, self.config.TUp)
        else:
            self.T = self.config.Ttest
            
        # Validate disruption timing for sufficient recovery
        min_recovery_periods = 20
        if Ts + D > self.T - min_recovery_periods:
            # Adjust disruption start to ensure minimum recovery time
            max_allowed_start = self.T - min_recovery_periods - D
            if max_allowed_start >= 10:  # Still respect minimum pre-disruption time
                old_Ts = Ts
                self.disruption.Ts = min(Ts, max_allowed_start)
                if self.disruption.Ts != old_Ts:
                    print(f"Adjusted disruption: was periods {old_Ts}-{old_Ts+D-1}, now {self.disruption.Ts}-{self.disruption.Ts+D-1}")
            else:
                # Can't satisfy both constraints, prioritize pre-disruption time
                print(f"Warning: Cannot ensure {min_recovery_periods} recovery periods with current episode length")
        
        # Create early warning signal with (possibly adjusted) timing
        self.warning = EarlyWarningSignal(self.config, self.disruption.Ts, D)
        
        # Inject warning signal into agent
        self.player.setWarning(self.warning)
            
        # Initialize shipment pipeline (array-based)
        self.arriving_shipments = np.zeros(self.T + self.config.leadRecItem + 10)
        # Set initial shipments in pipeline
        for i in range(self.config.leadRecItem):
            self.arriving_shipments[i] = self.config.ASInit
            
        # Reset agent's internal state
        self.player.resetPlayer(self.T)

    def playGame(self, demand, playType, return_history=False):
        """
        Run one episode.
        If return_history=True, returns a dict of lists:
            'orders', 'demand', 'shipments', 'inventory_levels', 'backorders'
        Otherwise returns cumulative reward.
        """
        self.resetGame(demand, playType)

        if return_history:
            history = {
                'orders': [],
                'demand': [],
                'shipments': [],
                'inventory_levels': [],
                'backorders': []
            }

        while self.curTime < self.T:
            # 1) Agent chooses action
            action = self.player.getAction(self.curTime, playType)

            # Decode actual order quantity
            idx = int(np.argmax(action))
            order_qty = self.config.actionList[idx]

            if self.disruption.in_window(self.curTime - self.config.leadRecItem):
                current_demand = self.demand[self.curTime]
                if order_qty > current_demand:
                    order_qty = current_demand

            # 2) Place order (updates agent's OO and AO pipeline)
            self.player.placeOrder(order_qty, self.curTime)
            
            # 3) Apply disruption to the order and schedule shipment
            # The wholesaler receives the order and ships (with possible disruption)
            # Disruption affects orders placed at time (t - leadRecItem)
            shipped_qty = self.disruption.apply(order_qty, self.curTime)
            
            # Schedule the shipment to arrive after leadRecItem periods
            if self.curTime + self.config.leadRecItem < len(self.arriving_shipments):
                self.arriving_shipments[self.curTime + self.config.leadRecItem] = shipped_qty

            # 4) Receive shipment that was scheduled to arrive today
            received = self.arriving_shipments[self.curTime]
            self.player.receiveItems(received)
            
            # 5) Serve customer demand
            current_demand = self.demand[self.curTime]
            self.player.serveDemand(current_demand)

            # 6) Update inventory and compute reward
            self.player.updateInventory(self.curTime)

            # 7) Record metrics if needed
            if return_history:
                history['orders'].append(order_qty)
                history['demand'].append(current_demand)
                history['shipments'].append(received)
                inv = self.player.IL
                history['inventory_levels'].append(inv)
                # backorders = positive part of -inv
                history['backorders'].append(max(0, -inv))

            # 8) Prepare next state
            self.player.prepareNext(self.curTime)

            # 9) Update DQN state (for both train and test)
            if hasattr(self.player, 'brain'):
                terminal = (self.curTime == self.T - 1)
                self.player.trainStep(terminal, playType)

            # Advance time
            self.curTime += 1

        if return_history:
            return history
        else:
            # Return cumulative reward of the agent
            return self.player.cumReward

    def doTestMid(self, demandList, return_results=False):
        """
        Evaluate the agent on multiple test sequences with clean summary output.
        """
        kpi_list = []
        histories = []
        disruption_info = None
        all_orders = []
        all_inventory = []

        for seq in demandList:
            history = self.playGame(seq, 'test', return_history=True)
            kpi = evaluate_run(
                history,
                holding_cost=self.config.c_h,
                backorder_cost=self.config.c_p
            )
            kpi_list.append(kpi)
            histories.append(history)
            
            # Collect orders and inventory for summary stats
            all_orders.extend(history['orders'])
            all_inventory.extend(history['inventory_levels'])
            
            # Capture disruption info from first run
            if disruption_info is None:
                disruption_info = {
                    'start': self.disruption.Ts,
                    'duration': self.disruption.D
                }

        # Compute mean of each KPI across runs
        avg_kpi = {}
        ci_kpis = {}
        
        for key in ['BW', 'RE', 'BRI', 'Cost']:
            values = [run[key] for run in kpi_list if not np.isnan(run[key]) and not np.isinf(run[key])]
            
            if values:
                avg_kpi[key] = np.mean(values)
                
                # Calculate 95% confidence interval
                if len(values) > 1:
                    mean = np.mean(values)
                    sem = stats.sem(values)
                    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=sem)
                    ci_kpis[key] = {'mean': mean, 'ci_lower': ci[0], 'ci_upper': ci[1]}
                else:
                    ci_kpis[key] = {'mean': avg_kpi[key], 'ci_lower': np.nan, 'ci_upper': np.nan}
            else:
                avg_kpi[key] = 0.0
                ci_kpis[key] = {'mean': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan}

        # CLEAN SUMMARY OUTPUT
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        agent_type = self.config.agentTypes[0]
        
        print(f"\n{'='*60}")
        print(f"TEST RESULTS | {timestamp} | Agent: {agent_type}")
        print(f"{'='*60}")
        
        # Performance metrics with confidence intervals
        print(f"Performance Metrics:")
        print(f"  BW:   {avg_kpi['BW']:.2f} (95% CI: [{ci_kpis['BW']['ci_lower']:.2f}, {ci_kpis['BW']['ci_upper']:.2f}])")
        print(f"  RE:   {avg_kpi['RE']:.2f} (95% CI: [{ci_kpis['RE']['ci_lower']:.2f}, {ci_kpis['RE']['ci_upper']:.2f}])")
        print(f"  BRI:  {avg_kpi['BRI']:.2f} (95% CI: [{ci_kpis['BRI']['ci_lower']:.2f}, {ci_kpis['BRI']['ci_upper']:.2f}])")
        print(f"  Cost: {avg_kpi['Cost']:.0f} (95% CI: [{ci_kpis['Cost']['ci_lower']:.0f}, {ci_kpis['Cost']['ci_upper']:.0f}])")
        
        # Behavioral summary
        avg_order = np.mean(all_orders)
        order_std = np.std(all_orders)
        avg_inventory = np.mean(all_inventory)
        inventory_std = np.std(all_inventory)
        order_variance = np.var(all_orders)
        
        print(f"\nBehavioral Summary:")
        print(f"  Average Order:     {avg_order:.1f} ± {order_std:.1f}")
        print(f"  Order Variance:    {order_variance:.3f}")
        print(f"  Average Inventory: {avg_inventory:.1f} ± {inventory_std:.1f}")
        
        # Policy assessment
        if order_variance < 0.01:
            print(f"  🔴 WARNING: Constant ordering detected!")
            unique_orders = np.unique(all_orders)
            print(f"  Unique orders: {unique_orders}")
        elif order_variance < 1.0:
            print(f"  🟡 Low order variability")
        else:
            print(f"  🟢 Dynamic ordering behavior")
        
        print(f"{'='*60}")
        
        if return_results:
            # Calculate phase-specific KPIs for the first test run
            phase_kpis = evaluate_phases(
                histories[0],  # Use first test run as representative
                disruption_info,
                self.config.c_h,
                self.config.c_p
            )
            
            return {
                'avg_kpi': avg_kpi,
                'ci_kpis': ci_kpis,
                'phase_kpis': phase_kpis,
                'all_kpis': kpi_list,
                'histories': histories,
                'disruption_info': disruption_info,
                'agent_type': agent_type
            }
        
        return avg_kpi