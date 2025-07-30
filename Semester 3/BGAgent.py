#!/usr/bin/env python
# coding: utf-8

import numpy as np
from SRDQN import DQN, EWADQN

class Agent:
    """
    Beer Game agent with stable reward engineering.
    """
    def __init__(self, agentNum, ILInit, AOInit, ASInit, c_h, c_p, eta, compuType, config):
        # Identity & initial state
        self.agentNum    = agentNum
        self.ILInitial   = ILInit
        self.IL          = ILInit
        self.OO          = 0            # outstanding orders
        self.ASInitial   = ASInit
        self.AOInitial   = AOInit

        # Config & costs
        self.config = config
        self.c_h    = c_h
        self.c_p    = c_p
        self.eta    = eta

        # Sterman parameters
        self.alpha_b = config.alpha_b
        self.betta_b = config.betta_b
        if config.demandDistribution == 0:
            self.a_b = np.mean((config.demandUp, config.demandLow))
        else:
            self.a_b = config.demandMu
        # Lead time is just 2 (shipment lead time)
        self.b_b = self.a_b * self.config.leadRecItem

        # Pipelines
        self.AS = None
        self.AO = None

        # Rewards & trajectory
        self.curReward = 0
        self.cumReward = 0
        self.cumCost = 0
        self.Trajectory = []

        # Policy settings
        self.compTypeTrain = compuType
        self.compTypeTest  = compuType
        self.action        = None
        
        # Track performance for reward scaling
        self.episode_costs = []
        self.good_performance_threshold = 2000  # Based on my test runs, below this is "good"

        # DQN brain
        if compuType == 'srdqn':
            self.brain = DQN(agentNum, config)
        elif compuType == 'ewa':
            self.brain = EWADQN(agentNum, config)

        # Warning signal placeholder
        self.warning = None
        
        # Track warning history for context
        self.warning_history = []
        self.time_since_warning = 0

    def setWarning(self, warning_signal):
        """Set the warning signal object from the environment."""
        self.warning = warning_signal

    def resetPlayer(self, T):
        """Initialize inventories and pipelines at the start of an episode."""
        self.IL = self.ILInitial
        self.OO = 0
        # Build pipeline arrays
        horizon = T + self.config.leadRecItem + 10
        self.AS = np.zeros(horizon)
        self.AO = np.zeros(horizon)
        # Initialize pipelines
        for i in range(self.config.leadRecItem):
            self.AS[i] = self.ASInitial
        # Unlike in Oroojlooyjadid, there is no need for AO initialization since leadRecOrder = leadRecItem 

        # Reset rewards and history
        self.curReward = 0
        self.cumReward = 0
        self.cumCost = 0
        self.Trajectory = []
        
        # Reset warning tracking
        self.warning_history = []
        self.time_since_warning = 0

        # Set initial DQN state
        state0 = self.getCurState(0)
        if hasattr(self, 'brain'):
            self.brain.setInitState(state0)

    def getCurState(self, t):
        """
        State vector with normalization
        """
        # Basic state components
        backorder = -min(0, self.IL)
        on_hand   =  max(0, self.IL)
        oo        =  self.OO
        as_t      =  self.AS[t] if t < len(self.AS) else 0
        ao_t      =  self.AO[t] if t < len(self.AO) else 0
        
        # For EWA agent, include normalized warning information
        if self.compTypeTrain == 'ewa' and self.warning is not None:
            wflag = self.warning.get_flag(t)
            # Update warning tracking
            if wflag == 1:
                self.time_since_warning = 0
            else:
                self.time_since_warning = min(self.time_since_warning + 1, 10)
            
            return np.array([
                backorder/10.0, 
                on_hand/20.0, 
                oo/20.0,         
                as_t/10.0, 
                ao_t/10.0,      
                wflag,           
                self.time_since_warning / 10.0
            ])
        else:
            # For standard DQN
            return np.array([
                backorder/10.0, 
                on_hand/20.0, 
                oo/20.0, 
                as_t/10.0, 
                ao_t/10.0,  
                self.a_b / 10.0  # demand estimate
            ])
        # Normalization constants (10,20) were chosen by Claude Opus 4 to ensure learning stability


    def compute_action(self, t, playType):
        """Compute and return a one‑hot action according to the current policy."""
        comp = self.compTypeTrain if playType == 'train' else self.compTypeTest

        if comp == 'Strm':
            # Corrected Sterman formula (with proper lead time)
            expected_arrivals = self.AS[t] if t < len(self.AS) else 0
            raw = (expected_arrivals +                                    
                   self.alpha_b * (self.a_b - self.IL) +          
                   self.betta_b * (self.b_b - self.OO))           
            
            order = max(0, round(raw))
            
            # Map to action space
            opt = np.array(self.config.actionListOpt)
            idx = np.argmin(np.abs(opt - order))
            action = np.zeros(self.config.actionListLenOpt)
            action[idx] = 1

        elif comp in ('srdqn', 'ewa'):
            action = self.brain.getDNNAction(playType)

        elif comp == 'rnd':
            action = np.zeros(self.config.actionListLen)
            idx = np.random.randint(self.config.actionListLen)
            action[idx] = 1

        else:
            raise ValueError(f"Unknown policy type: {comp}")

        self.action = action
        return action

    def getAction(self, t, playType):
        """Return a one‑hot action according to current policy."""
        return self.compute_action(t, playType)

    def placeOrder(self, order_qty, t):
        """Place an order and update the pipelines."""
        self.OO += order_qty
        # Schedule arrival at t + leadRecItem (which is 2)
        if t + self.config.leadRecItem < len(self.AS):
            self.AS[t + self.config.leadRecItem] += order_qty

    def receiveItems(self, qty):
        """Add incoming shipment to inventory and reduce outstanding orders."""
        self.IL += qty
        self.OO -= qty
        self.OO = max(0, self.OO)

    def serveDemand(self, demand):
        """Serve customer demand and update inventory."""
        self.IL -= demand

    def updateInventory(self, t):
        """Compute reward with improved EWA-aware shaping."""
        # Base economic costs
        holding_cost = self.c_h * max(0, self.IL)
        backorder_cost = self.c_p * max(0, -self.IL)
        base_cost = holding_cost + backorder_cost
        
        # Track cumulative cost
        self.cumCost += base_cost
        
        # Get actual order quantity
        order_qty = 0.0
        if self.action is not None:
            idx = int(np.argmax(self.action))
            order_qty = self.config.actionList[idx]
        
        # STABLE REWARD FUNCTION
        # Use fixed normalization to avoid catastrophic forgetting
        normalized_cost = base_cost / 100.0  # Fixed scale
        
        # Strong shaping to prevent collapse to zero ordering
        shaped_reward = 0.0
        
        # 1. CRITICAL: Heavy penalty for chronic zero ordering when backlogged
        if self.IL < -5 and order_qty == 0:
            shaped_reward -= 0.5  # Strong penalty
        elif self.IL < -10 and order_qty < 2:
            shaped_reward -= 0.3  # Still penalize very low orders
        
        # 2. Reward maintaining reasonable inventory
        if -2 <= self.IL <= 10:
            shaped_reward += 0.1  # Bonus for good inventory
        
        # 3. Penalty for extreme positions
        if self.IL > 30:
            shaped_reward -= 0.1 * (self.IL - 30) / 30.0
        elif self.IL < -20:
            shaped_reward -= 0.2 * (-20 - self.IL) / 20.0
        
        # 4. Bonus for reasonable ordering (not too high, not zero)
        if 2 <= order_qty <= 8 and -10 <= self.IL <= 20:
            shaped_reward += 0.05
        
        # 5. EWA Reward Shaping
        if self.compTypeTrain == 'ewa' and self.warning is not None:
            warning_flag = self.warning.get_flag(t)
            
            if warning_flag == 1:
                # Check if we're actually IN disruption vs just warned
                if self.time_since_warning == 0:  # First warning
                    # During WARNING (pre-disruption): reward strategic preparation
                    if self.IL < 8:  # Low inventory
                        if 4 <= order_qty <= 8:  # Building inventory
                            shaped_reward += 0.3  # Strong reward for preparation
                        elif order_qty < 2:
                            shaped_reward -= 0.2  # Penalty for not preparing
                    elif self.IL > 20:  # Already high inventory
                        if order_qty > 6:
                            shaped_reward -= 0.2  # Penalty for over-ordering
                        elif 0 <= order_qty <= 3:
                            shaped_reward += 0.1  # Reward conservative approach
                    else:  # Moderate inventory (8-20)
                        if 2 <= order_qty <= 6:
                            shaped_reward += 0.15  # Reward balanced approach
                else:
                    # DURING disruption (warning continues): be conservative
                    if order_qty > 6:
                        shaped_reward -= 0.3  # Penalty for high orders during disruption
                    elif 2 <= order_qty <= 5:
                        shaped_reward += 0.1  # Small reward for moderate ordering
                    elif order_qty < 2 and self.IL < -5:
                        shaped_reward -= 0.2  # Still penalize zero ordering if backlogged
            else:
                # No warning but track recent warning behavior
                if self.time_since_warning <= 3:  # Recent warning
                    # Post-disruption recovery period
                    if self.IL < -5:  # Backlogged after disruption
                        if 4 <= order_qty <= 8:
                            shaped_reward += 0.2  # Reward recovery efforts
                    elif self.IL > 15:  # Overstocked after disruption
                        if order_qty <= 3:
                            shaped_reward += 0.1  # Reward restraint
        
        # Final reward combines normalized cost and shaping
        self.curReward = -normalized_cost + shaped_reward
        
        # Clip reward to prevent extreme values
        self.curReward = np.clip(self.curReward, -2.0, 1.0)
        
        # Update cumulative
        self.cumReward = self.config.gamma * self.cumReward + self.curReward
        self.Trajectory.append((t, self.IL, self.curReward, base_cost))

    def prepareNext(self, t):
        """Compute next state for DQN training."""
        self.nextState = self.getCurState(t+1)

    def trainStep(self, terminal, playType='train'):
        """Perform a DQN training step if applicable."""
        if hasattr(self, 'brain'):
            self.brain.train(self.nextState, self.action, self.curReward, terminal, playType)