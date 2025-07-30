#!/usr/bin/env python
# coding: utf-8

import os
import random
import tensorflow as tf
import numpy as np
from collections import deque
import heapq

class PrioritizedReplayBuffer:
    """Simple prioritized experience replay to focus on important transitions."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        # Priority based on reward magnitude (important transitions)
        priority = abs(reward) + 0.1
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        # Mix prioritized and uniform sampling
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        
        # 70% prioritized, 30% uniform
        n_prioritized = int(0.7 * batch_size)
        n_uniform = batch_size - n_prioritized
        
        # Prioritized sampling
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities / priorities.sum()
        prioritized_indices = np.random.choice(len(self.buffer), n_prioritized, p=probs)
        
        # Uniform sampling
        uniform_indices = np.random.choice(len(self.buffer), n_uniform)
        
        # Combine and return
        all_indices = np.concatenate([prioritized_indices, uniform_indices])
        return [self.buffer[i] for i in all_indices]
    
    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, agentNum, config, state_dim_override=None):
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(1)
            self.agentNum = agentNum
            self.global_step = tf.Variable(0, trainable=False)
            self.config = config

            # --- Decide the state dimension ONCE ---
            if state_dim_override is not None:
                self.state_dim = int(state_dim_override)
            elif hasattr(self.config, 'stateDim'):
                self.state_dim = int(self.config.stateDim)
            else:
                # Fallback if neither override nor config.stateDim is available
                self.state_dim = 6

            modelNumber = 'model' + str(agentNum + 1)
            self.address = os.path.join(self.config.model_dir, modelNumber)
            self.addressName = self.address + '/network-'

            # Initialize counters
            self.timeStep = 0
            self.episodes_trained = 0
            self.steps_since_good_performance = 0

            # Epsilon schedule
            self.epsilon = self.config.epsilonBeg
            self.epsilonRed = self._build_epsilon_decay()

            # Derived input size from the decided dimension
            self.inputSize = self.state_dim * self.config.multPerdInpt

            # Prioritized replay memory
            self.replayMemory = PrioritizedReplayBuffer(self.config.maxReplayMem)

            # Keep track of recent performance
            self.recent_episode_costs = deque(maxlen=10)

            # Build networks using the decided dimension
            self._create_inputs(self.state_dim)
            we, be = [], []
            if self.config.ifUsePreviousModel:
                we, be = self._load_pretrained_weights()

            self.QValue, self.W_fc, self.b_fc = self._build_network('Q', we, be, self.state_dim)
            self.QValueT, self.W_fcT, self.b_fcT = self._build_network('TQ', None, None, self.state_dim)
            self.copyTargetOp = self._copy_target_op()

            # Training operators
            self._create_training_method()

            # Initialize session
            self.saver = tf.train.Saver(max_to_keep=3)  # Keep multiple checkpoints
            tf_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=self.config.gpu_memory_fraction,
                    allow_growth=True
                ),
                intra_op_parallelism_threads=self.config.number_cpu_active
            )
            self.session = tf.Session(graph=self.graph, config=tf_config)
            self.session.run(tf.global_variables_initializer())

    def _build_epsilon_decay(self):
        """Adaptive epsilon decay based on performance."""
        if self.config.maxEpisodesTrain != 0:
            # Slower decay to maintain exploration
            decay_episodes = self.config.maxEpisodesTrain * 0.8
            return (self.config.epsilonBeg - self.config.epsilonEnd) / decay_episodes
        return 0

    def _create_inputs(self, state_dim):
        with tf.name_scope('input'):
            self.stateInput = tf.placeholder(
                tf.float32,
                [None, self.config.multPerdInpt, state_dim],
                name='state'
            )
        with tf.name_scope('input_flat'):
            self.stateInputFlat = tf.reshape(
                self.stateInput, [-1, self.inputSize]
            )

    def _load_pretrained_weights(self):
        """Load weights from a previously saved model if it exists."""
        we = []
        be = []
        
        checkpoint_path = tf.train.latest_checkpoint(self.address)
        if checkpoint_path:
            try:
                # Restore directly to current session
                self.saver = tf.train.Saver()
                self.saver.restore(self.session, checkpoint_path)
                print(f"Successfully restored model from {checkpoint_path}")
                return we, be
            except Exception as e:
                print(f"Could not restore model: {e}")
        
        return we, be

    def _build_network(self, name, init_w=None, init_b=None, state_dim=None):
        """Build feed-forward network with dropout for regularization."""
        W, b = [], []
        layer = []
        
        actual_state_dim = state_dim if state_dim else self.config.stateDim
        
        for j in range(self.config.NoHiLayer + 1):
            layer_name = f'{name}-layer{j+1}'
            input_tensor = self.stateInputFlat if j == 0 else layer[j-1]
            in_dim = (actual_state_dim * self.config.multPerdInpt 
                     if j == 0 else self.config.nodes[j])
            out_dim = (self.config.baseActionSize 
                      if j == self.config.NoHiLayer else self.config.nodes[j+1])
            
            w_init = init_w[j] if init_w and j < len(init_w) else None
            b_init = init_b[j] if init_b and j < len(init_b) else None

            hidden, w_var, b_var = self._fc_layer(
                input_tensor, in_dim, out_dim, layer_name, j,
                init_w=w_init, init_b=b_init
            )
            layer.append(hidden)
            W.append(w_var)
            b.append(b_var)

        return layer[-1], W, b

    def _fc_layer(self, inp, in_d, out_d, name, idx, init_w=None, init_b=None):
        """Fully connected layer."""
        def _weight_variable(shape):
            if init_w is not None:
                init = tf.constant(init_w)
            else:
                # Xavier/He initialization
                if idx == self.config.NoHiLayer:  # Output layer
                    init = tf.random.truncated_normal(shape, stddev=0.01)
                else:
                    init = tf.random.truncated_normal(shape, stddev=np.sqrt(2.0/in_d))
            trainable = not (self.config.iftl and idx < self.config.NoFixedLayer)
            return tf.Variable(init, trainable=trainable, name=name+'_W')

        def _bias_variable(shape):
            if init_b is not None:
                init = tf.constant(init_b)
            else:
                init = tf.constant(0.0, shape=shape)
            trainable = not (self.config.iftl and idx < self.config.NoFixedLayer)
            return tf.Variable(init, trainable=trainable, name=name+'_b')

        with tf.name_scope(name):
            w = _weight_variable([in_d, out_d])
            b = _bias_variable([out_d])
            pre = tf.matmul(inp, w) + b
            
            # Output layer: linear, hidden layers: ReLU
            if idx == self.config.NoHiLayer:
                act = tf.identity(pre)
            else:
                act = tf.nn.relu(pre)
            return act, w, b

    def _copy_target_op(self):
        ops = []
        for i in range(self.config.NoHiLayer + 1):
            ops.append(self.W_fcT[i].assign(self.W_fc[i]))
            ops.append(self.b_fcT[i].assign(self.b_fc[i]))
        return ops

    def _create_training_method(self):
        self.actionInput = tf.placeholder(
            tf.float32, [None, self.config.actionListLen], name='action'
        )
        self.yInput = tf.placeholder(
            tf.float32, [None], name='target'
        )
        
        Q_act = tf.reduce_sum(
            tf.multiply(self.QValue, self.actionInput), axis=1
        )
        
        with tf.name_scope('cost'):
            # Huber loss for stability
            error = self.yInput - Q_act
            huber_loss = tf.where(
                tf.abs(error) <= 1.0,
                0.5 * tf.square(error),
                tf.abs(error) - 0.5
            )
            self.cost = tf.reduce_mean(huber_loss)
            
        # Adam optimizer with lower learning rate
        optimizer = tf.train.AdamOptimizer(self.config.lr0)
        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)  # Aggressive clipping
        self.trainStep = optimizer.apply_gradients(
            zip(gradients, variables), global_step=self.global_step
        )

    def trainQNetwork(self):
        """Train with prioritized experience replay."""
        # Sample from prioritized buffer
        minibatch = self.replayMemory.sample(self.config.batchSize)
        
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        terminal_batch = [data[4] for data in minibatch]

        # Double DQN
        QValue_next_main = self.QValue.eval(
            feed_dict={self.stateInput: nextState_batch},
            session=self.session
        )
        QValue_next_target = self.QValueT.eval(
            feed_dict={self.stateInput: nextState_batch},
            session=self.session
        )
        
        y_batch = []
        for i in range(len(minibatch)):
            if terminal_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                best_action = np.argmax(QValue_next_main[i])
                target_q = QValue_next_target[i][best_action]
                y_batch.append(reward_batch[i] + self.config.gamma * target_q)

        # Train network
        feed_dict = {
            self.yInput: y_batch,
            self.actionInput: action_batch,
            self.stateInput: state_batch
        }
        _, cost = self.session.run([self.trainStep, self.cost], feed_dict)

        # Save periodically and when performing well
        if (self.timeStep + 1) % self.config.saveInterval == 0:
            os.makedirs(self.address, exist_ok=True)
            self.saver.save(self.session, self.addressName, global_step=self.timeStep)
            
            # Extra save if performing well
            if len(self.recent_episode_costs) > 0 and np.mean(self.recent_episode_costs) < 1000:
                best_path = os.path.join(self.address, 'best-model')
                self.saver.save(self.session, best_path)
                print(f"Saved best model with avg cost: {np.mean(self.recent_episode_costs):.0f}")

        # Update target network
        if self.timeStep % self.config.dnnUpCnt == 0:
            self.copyTargetQNetwork()

    def train(self, nextObservation, action, reward, terminal, playType='train'):
        """Store experience and train."""
        newState = np.append(self.currentState[1:, :], [nextObservation], axis=0)

        if playType == 'train':
            # Store in prioritized buffer
            self.replayMemory.push(self.currentState, action, reward, newState, terminal)
            
            # Train if ready
            if len(self.replayMemory) >= self.config.minReplayMem:
                self.trainQNetwork()
                self.timeStep += 1

            # Episode end processing
            if terminal:
                self.epsilonReduce()
                self.episodes_trained += 1

        self.currentState = newState

    def getDNNAction(self, playType='test'):
        """Get action with anti-collapse mechanisms."""
        action = np.zeros(self.config.actionListLen)
        
        if playType == 'train':
            # Check current state for emergency override
            current_inventory = self.currentState[-1, 1] * 20.0 - self.currentState[-1, 0] * 10.0
            
            # EMERGENCY OVERRIDE: Force ordering if heavily backlogged
            if current_inventory < -15 and random.random() < 0.3:
                # Force a reasonable order
                action_index = random.randint(4, 8)
            elif self.episodes_trained < 200:
                # Heavy exploration early
                action_index = random.randrange(self.config.actionListLen)
            elif random.random() <= self.epsilon:
                # Smart exploration
                if current_inventory < -5:
                    # Bias toward ordering when backlogged
                    action_index = random.randint(3, 10)
                elif current_inventory > 20:
                    # Bias toward low orders when overstocked
                    action_index = random.randint(0, 4)
                else:
                    # Normal range
                    action_index = random.randint(2, 8)
            else:
                # Exploitation
                QValue = self.QValue.eval(
                    feed_dict={self.stateInput: [self.currentState]},
                    session=self.session
                )[0]
                action_index = np.argmax(QValue)
                
                # SAFETY CHECK: Prevent persistent zero ordering
                if action_index == 0 and current_inventory < -10:
                    # Override with minimum order
                    if random.random() < 0.2:
                        action_index = random.randint(2, 5)
        else:
            # Test mode
            QValue = self.QValue.eval(
                feed_dict={self.stateInput: [self.currentState]},
                session=self.session
            )[0]
            action_index = np.argmax(QValue)

        action[action_index] = 1
        return action

    def setInitState(self, observation):
        """Set initial state."""
        self.currentState = np.stack(
            [observation for _ in range(self.config.multPerdInpt)], 
            axis=0
        )

    def epsilonReduce(self):
        """Adaptive epsilon reduction."""
        # Slower reduction if performance is poor
        if len(self.recent_episode_costs) > 5:
            avg_cost = np.mean(self.recent_episode_costs)
            if avg_cost > 5000:  # Poor performance
                # Increase exploration
                self.epsilon = min(0.3, self.epsilon + 0.01)
            else:
                # Normal reduction
                if self.epsilon > self.config.epsilonEnd:
                    self.epsilon -= self.epsilonRed
        else:
            # Normal reduction
            if self.epsilon > self.config.epsilonEnd:
                self.epsilon -= self.epsilonRed

    def copyTargetQNetwork(self):
        """Copy Q network weights to target network."""
        self.session.run(self.copyTargetOp)
    
    def update_performance(self, episode_cost):
        """Track recent performance for adaptive mechanisms."""
        self.recent_episode_costs.append(episode_cost)


class EWADQN(DQN):
    """
    Early-Warning-Aware DQN with enhanced architecture and learning.
    FIXED: Properly handle state dimensions.
    """
    def __init__(self, agentNum, config):
        # EWA uses 7 features instead of 6
        ewa_state_dim = 7
        
        # Modify architecture for warning processing
        original_nodes = config.nodes.copy()
        config.nodes[1] = config.nodes[1] + 16  # Moderate increase for warning processing
        config.nodes[2] = config.nodes[2] + 8   # Additional processing in second layer
        
        # Call parent with explicit state dimension
        super().__init__(agentNum, config, state_dim_override=ewa_state_dim)
        
        # Restore original config nodes
        config.nodes = original_nodes
        
        # Track warning-specific performance
        self.warning_episode_rewards = deque(maxlen=20)
        self.ewa_state_dim = ewa_state_dim
    
    def getDNNAction(self, playType='test'):
        """EWA-specific action selection with learned warning response."""
        action = np.zeros(self.config.actionListLen)
        
        if playType == 'train':
            # Extract state components (accounting for EWA's extended state)
            current_inventory = self.currentState[-1, 1] * 20.0 - self.currentState[-1, 0] * 10.0
            current_warning = self.currentState[-1, 5] * 2.0  # Denormalize warning
            time_since_warning = self.currentState[-1, 6] * 10.0  # Denormalize time
            
            # Early training - heavy exploration
            if self.episodes_trained < 300:
                action_index = random.randrange(self.config.actionListLen)
            elif random.random() <= self.epsilon:
                # Intelligent exploration based on state
                if current_warning > 0.5:  # Warning active
                    # Explore strategic responses to warnings
                    if current_inventory < 5:
                        # Low inventory + warning = explore building stock
                        action_index = random.randint(4, 10)
                    elif current_inventory > 15:
                        # High inventory + warning = explore conservative
                        action_index = random.randint(0, 5)
                    else:
                        # Moderate inventory = balanced exploration
                        action_index = random.randint(2, 8)
                else:
                    # No warning - use inventory-based exploration
                    if current_inventory < -5:
                        action_index = random.randint(3, 10)
                    elif current_inventory > 20:
                        action_index = random.randint(0, 4)
                    else:
                        action_index = random.randint(1, 8)
            else:
                # Exploitation - let network decide
                QValue = self.QValue.eval(
                    feed_dict={self.stateInput: [self.currentState]},
                    session=self.session
                )[0]
                action_index = np.argmax(QValue)
                
                # Safety override for extreme situations
                if action_index == 0 and current_inventory < -10:
                    if random.random() < 0.15:
                        action_index = random.randint(3, 6)
        else:
            # Test mode - pure exploitation
            QValue = self.QValue.eval(
                feed_dict={self.stateInput: [self.currentState]},
                session=self.session
            )[0]
            action_index = np.argmax(QValue)
        
        action[action_index] = 1
        return action
    
    def train(self, nextObservation, action, reward, terminal, playType='train'):
        """Enhanced training for EWA with warning-aware experience replay."""
        # Track cumulative reward for this episode
        if not hasattr(self, 'episode_reward'):
            self.episode_reward = 0
        self.episode_reward += reward
        
        # Call parent train method
        super().train(nextObservation, action, reward, terminal, playType)
        
        # Track warning-specific performance
        if playType == 'train' and terminal:
            # Store episode reward for warning analysis
            if hasattr(self, 'warning_episode_rewards'):
                self.warning_episode_rewards.append(self.episode_reward)
                
                # Adjust exploration based on warning performance
                if len(self.warning_episode_rewards) > 10:
                    recent_warning_perf = np.mean(list(self.warning_episode_rewards)[-10:])
                    # If performing poorly during warnings, increase exploration
                    if recent_warning_perf < -0.5:  # Poor warning response (normalized scale)
                        self.epsilon = min(self.epsilon + 0.02, 0.3)
            
            # Reset episode reward for next episode
            self.episode_reward = 0