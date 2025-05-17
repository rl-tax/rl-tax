import numpy as np

class EconomicEnv:
    """AI Economist simulation environment with agents and a social planner."""

    def __init__(self, num_agents=4, grid_size=(25, 25), tax_period=100):
        """Initialize the environment with a grid, agents, market, and tax system."""
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.tax_period = tax_period
        self.timestep = 0
        self.prev_swf = 0.0
        self.new_orders = []

        # Grid constants
        self.EMPTY, self.WATER, self.STONE, self.WOOD, self.HOUSE, self.AGENT = 0, 1, 2, 3, 4, 5

        # Set up grid and regeneration masks
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.is_stone_regen = np.zeros(self.grid_size, dtype=bool)
        self.is_wood_regen = np.zeros(self.grid_size, dtype=bool)
        self._setup_grid()

        # Initialize agents
        self.agents_build_skill = [10, 15, 20, 25]
        self.agents = self._init_agents()

        # Initialize market
        self.market = {'stone': {'bids': [], 'asks': []}, 'wood': {'bids': [], 'asks': []}}

        # Tax system
        self.tax_rates = np.zeros(7)  # 7 brackets, initialized to 0%
        self.brackets = [0, 9, 39, 84, 160, 204, 510, np.inf]
        self.prev_incomes = [0.0] * num_agents
        self.prev_sorted_incomes = [0.0] * num_agents

    def _setup_grid(self):
        """Set up the grid with water barriers and resource regeneration cells."""
        mid = self.grid_size[0] // 2
        # Water barriers
        self.grid[mid, :] = self.WATER
        self.grid[:, mid] = self.WATER
        # Passages
        self.grid[mid, mid // 2] = self.EMPTY
        self.grid[mid, mid + mid // 2] = self.EMPTY
        self.grid[mid // 2, mid] = self.EMPTY
        self.grid[mid + mid // 2, mid] = self.EMPTY

        # Define quadrants and their resources
        quadrants = [
            ((0, mid), (0, mid), ['stone', 'wood']),  # Top-left: both
            ((0, mid), (mid + 1, self.grid_size[1]), ['stone']),  # Top-right: stone
            ((mid + 1, self.grid_size[0]), (0, mid), ['wood']),  # Bottom-left: wood
            ((mid + 1, self.grid_size[0]), (mid + 1, self.grid_size[1]), [])  # Bottom-right: none
        ]

        # Set regeneration cells
        for (i_start, i_end), (j_start, j_end), resources in quadrants:
            mask = (self.grid[i_start:i_end, j_start:j_end] == self.EMPTY)
            coords = np.where(mask)
            coords = list(zip(coords[0] + i_start, coords[1] + j_start))
            if coords and 'stone' in resources:
                selected = np.random.choice(len(coords), min(10, len(coords)), replace=False)
                for idx in selected:
                    self.is_stone_regen[coords[idx]] = True
            if coords and 'wood' in resources:
                selected = np.random.choice(len(coords), min(10, len(coords)), replace=False)
                for idx in selected:
                    self.is_wood_regen[coords[idx]] = True

    def _init_agents(self):
        """Initialize agents with random positions and attributes."""
        agents = []
        occupied = set()
        for i in range(self.num_agents):
            while True:
                pos = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
                if self.grid[pos] == self.EMPTY and pos not in occupied:
                    occupied.add(pos)
                    break
            agents.append({
                'pos': pos,
                'inventory': {'stone': 0, 'wood': 0, 'coin': 0.0},
                # 'build_skill': np.random.uniform(10, 30),
                'build_skill': self.agents_build_skill[i], # temp
                'labor': 0.0,
                'orders': [],
                'coin_start': 0.0,
                'prev_utility': 0.0
            })
        return agents

    def _process_agent_action(self, agent_id, action):
        """Process an agent's action: move, build, or trade."""
        agent = self.agents[agent_id]
        if action == 0:  # NO-OP
            return
        elif 1 <= action <= 4:  # Move (up, down, left, right)
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action - 1]
            new_pos = (agent['pos'][0] + dx, agent['pos'][1] + dy)
            if (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1] and
                self.grid[new_pos] not in [self.WATER, self.HOUSE] and
                all(a['pos'] != new_pos for a in self.agents if a != agent)):
                agent['pos'] = new_pos
                agent['labor'] += 0.21
                if self.grid[new_pos] in [self.STONE, self.WOOD]:
                    resource = 'stone' if self.grid[new_pos] == self.STONE else 'wood'
                    agent['inventory'][resource] += 1
                    self.grid[new_pos] = self.EMPTY
                    agent['labor'] += 0.21
        elif action == 5:  # Build
            if (agent['inventory']['stone'] >= 1 and agent['inventory']['wood'] >= 1 and
                self.grid[agent['pos']] == self.EMPTY):
                self.grid[agent['pos']] = self.HOUSE
                agent['inventory']['stone'] -= 1
                agent['inventory']['wood'] -= 1
                agent['inventory']['coin'] += agent['build_skill']
                agent['labor'] += 2.1
        elif 6 <= action <= 49:  # Trade
            trade_idx = action - 6
            trade_type = trade_idx // 11  # 0: bid stone, 1: ask stone, 2: bid wood, 3: ask wood
            price = trade_idx % 11
            resource = 'stone' if trade_type in [0, 1] else 'wood'
            order_type = 'bid' if trade_type % 2 == 0 else 'ask'
            if ((order_type == 'bid' and agent['inventory']['coin'] >= price) or
                (order_type == 'ask' and agent['inventory'][resource] >= 1)):
                order = {'agent_id': agent_id, 'type': order_type, 'resource': resource,
                         'price': price, 'timestamp': self.timestep}
                self.new_orders.append(order)
                agent['labor'] += 0.05

    def _update_market(self):
        """Update the market by matching and executing trades."""
        # Remove expired orders
        for resource in ['stone', 'wood']:
            for key in ['bids', 'asks']:
                self.market[resource][key] = [o for o in self.market[resource][key]
                                            if self.timestep - o['timestamp'] <= 50]

        # Process new orders
        np.random.shuffle(self.new_orders)
        for order in self.new_orders:
            resource = order['resource']
            if order['type'] == 'bid':
                asks = sorted(self.market[resource]['asks'], key=lambda x: (x['price'], x['timestamp']))
                for ask in asks[:]:
                    if ask['price'] <= order['price']:
                        trade_price = ask['price'] if ask['timestamp'] < order['timestamp'] else order['price']
                        buyer, seller = self.agents[order['agent_id']], self.agents[ask['agent_id']]
                        buyer['inventory']['coin'] -= trade_price
                        buyer['inventory'][resource] += 1
                        seller['inventory']['coin'] += trade_price
                        seller['inventory'][resource] -= 1
                        self.market[resource]['asks'].remove(ask)
                        break
                else:
                    self.market[resource]['bids'].append(order)
            else:  # 'ask'
                bids = sorted(self.market[resource]['bids'], key=lambda x: (-x['price'], x['timestamp']))
                for bid in bids[:]:
                    if bid['price'] >= order['price']:
                        trade_price = bid['price'] if bid['timestamp'] < order['timestamp'] else order['price']
                        buyer, seller = self.agents[bid['agent_id']], self.agents[order['agent_id']]
                        buyer['inventory']['coin'] -= trade_price
                        buyer['inventory'][resource] += 1
                        seller['inventory']['coin'] += trade_price
                        seller['inventory'][resource] -= 1
                        self.market[resource]['bids'].remove(bid)
                        break
                else:
                    self.market[resource]['asks'].append(order)
        self.new_orders.clear()

    def _respawn_resources(self):
        """Respawn resources on regeneration cells with probability 0.01."""
        for resource, mask, value in [('stone', self.is_stone_regen, self.STONE),
                                    ('wood', self.is_wood_regen, self.WOOD)]:
            regen_cells = np.where(mask & (self.grid == self.EMPTY))
            if regen_cells[0].size > 0:
                spawn = np.random.rand(regen_cells[0].size) < 0.01
                self.grid[regen_cells[0][spawn], regen_cells[1][spawn]] = value

    def _apply_taxes_and_redistribution(self):
        """Apply taxes and redistribute wealth at the end of a tax period."""
        incomes = [max(0, agent['inventory']['coin'] - agent['coin_start']) for agent in self.agents]
        taxes = [self._compute_tax(income) for income in incomes]
        total_tax = sum(taxes)
        redistribution = total_tax / self.num_agents
        for agent, tax in zip(self.agents, taxes):
            agent['inventory']['coin'] += -tax + redistribution
            agent['coin_start'] = agent['inventory']['coin']
        self.prev_incomes = incomes
        self.prev_sorted_incomes = sorted(incomes)

    def _compute_tax(self, income):
        """Compute tax for a given income based on tax brackets and rates."""
        tax = 0.0
        for i in range(len(self.brackets) - 1):
            low, high = self.brackets[i], self.brackets[i + 1]
            if income > low:
                taxable = min(income, high) - low
                tax += self.tax_rates[i] * taxable
            if income <= high:
                break
        return tax

    def _get_agent_obs(self, agent_id):
        """Generate observation for an agent."""
        agent = self.agents[agent_id]
        x, y = agent['pos']
        half = 5
        spatial = np.full((11, 11), -1, dtype=int)
        i_start, i_end = max(0, x - half), min(self.grid_size[0], x + half + 1)
        j_start, j_end = max(0, y - half), min(self.grid_size[1], y + half + 1)
        spatial[i_start - (x - half):11 - (x + half + 1 - i_end),
                j_start - (y - half):11 - (y + half + 1 - j_end)] = self.grid[i_start:i_end, j_start:j_end]
        for a in self.agents:
            if a['pos'][0] >= i_start and a['pos'][0] < i_end and a['pos'][1] >= j_start and a['pos'][1] < j_end:
                spatial[a['pos'][0] - (x - half), a['pos'][1] - (y - half)] = self.AGENT

        inventory = agent['inventory'].copy()
        market = self._get_market_obs(agent_id)
        income = max(0, agent['inventory']['coin'] - agent['coin_start'])
        bracket_idx = next((i for i, b in enumerate(self.brackets) if income <= b), len(self.tax_rates) - 1)
        marginal_rate = self.tax_rates[bracket_idx - 1] if bracket_idx > 0 else 0
        tax_info = {
            'current_tax_rates': self.tax_rates.tolist(),
            'time_in_period': self.timestep % self.tax_period,
            'prev_incomes': self.prev_sorted_incomes,
            'marginal_rate': marginal_rate
        }
        return {'spatial': spatial, 'inventory': inventory, 'build_skill': agent['build_skill'],
                'market': market, 'tax_info': tax_info}

    def _get_market_obs(self, agent_id):
        """Get market observation for an agent."""
        market_obs = {}
        for resource in ['stone', 'wood']:
            own_bids = [0] * 11
            own_asks = [0] * 11
            others_bids = [0] * 11
            others_asks = [0] * 11
            for bid in self.market[resource]['bids']:
                if bid['agent_id'] == agent_id:
                    own_bids[bid['price']] += 1
                else:
                    others_bids[bid['price']] += 1
            for ask in self.market[resource]['asks']:
                if ask['agent_id'] == agent_id:
                    own_asks[ask['price']] += 1
                else:
                    others_asks[ask['price']] += 1
            market_obs[resource] = {'own_bids': own_bids, 'own_asks': own_asks,
                                  'others_bids': others_bids, 'others_asks': others_asks}
        return market_obs

    def _get_planner_obs(self):
        """Generate observation for the planner."""
        inventories = [a['inventory'].copy() for a in self.agents]
        market = {r: {'bids': [o.copy() for o in self.market[r]['bids']],
                     'asks': [o.copy() for o in self.market[r]['asks']]} for r in ['stone', 'wood']}
        tax_info = {
            'current_tax_rates': self.tax_rates.tolist(),
            'time_in_period': self.timestep % self.tax_period,
            'prev_incomes': self.prev_incomes
        }
        return {'inventories': inventories, 'market': market, 'tax_info': tax_info}

    def _compute_agent_reward(self, agent_id):
        """Compute reward as the change in utility for an agent."""
        agent = self.agents[agent_id]
        eta = 0.23
        C_t = max(0, agent['inventory']['coin'])
        L_t = agent['labor']
        u_t = (C_t ** (1 - eta) - 1) / (1 - eta) - L_t
        reward = u_t - agent['prev_utility']
        agent['prev_utility'] = u_t
        return reward

    def _compute_planner_reward(self):
        """Compute planner reward as change in social welfare (equality * productivity)."""
        C_t = [max(0, a['inventory']['coin']) for a in self.agents]
        prod = sum(C_t)
        # Simplified Gini coefficient
        n = self.num_agents
        sorted_C = sorted(C_t)
        gini = sum(abs(c1 - c2) for i, c1 in enumerate(sorted_C) for c2 in sorted_C[i + 1:]) / (n * prod) if prod > 0 else 0
        eq = 1 - gini
        swf_t = eq * prod
        reward = swf_t - self.prev_swf
        self.prev_swf = swf_t
        return reward

    def _apply_planner_action(self, planner_action):
        """Apply planner's action to set tax rates."""
        return np.array([min(max(0, a), 20) * 0.05 for a in planner_action])  # 0% to 100% in 5% steps

    def step(self, agent_actions, planner_action=None):
        """Advance the simulation by one timestep."""
        if self.timestep % self.tax_period == 0 and planner_action is not None:
            self.tax_rates = self._apply_planner_action(planner_action)

        for i, action in enumerate(agent_actions):
            self._process_agent_action(i, action)

        self._update_market()
        self._respawn_resources()
        if self.timestep % self.tax_period == self.tax_period - 1:
            self._apply_taxes_and_redistribution()

        agent_obs = [self._get_agent_obs(i) for i in range(self.num_agents)]
        planner_obs = self._get_planner_obs()
        agent_rewards = [self._compute_agent_reward(i) for i in range(self.num_agents)]
        planner_reward = self._compute_planner_reward()

        self.timestep += 1
        done = self.timestep >= 1000  # Example termination condition

        return agent_obs, planner_obs, agent_rewards, planner_reward, done
    

    def reset(self):
        self.timestep = 0
        agent_obs = [{
            'spatial': np.full((11, 11), -1, dtype=int),  # 11x11 grid
            'inventory': {'stone': 0, 'wood': 0, 'coin': 0},
            'build_skill': self.agents_build_skill[i], # temp
            'market': {
                'stone': {'own_bids': [0]*11, 'own_asks': [0]*11, 'others_bids': [0]*11, 'others_asks': [0]*11},
                'wood': {'own_bids': [0]*11, 'own_asks': [0]*11, 'others_bids': [0]*11, 'others_asks': [0]*11}
            },
            'tax_info': {
                'current_tax_rates': [0.0]*7,
                'time_in_period': 0,
                'prev_incomes': [0.0]*self.num_agents,
                'marginal_rate': 0.0
            }
        } for i in range(self.num_agents)]
        planner_obs = {
            'inventories': [{'stone': 0, 'wood': 0, 'coin': 0} for _ in range(self.num_agents)],
            'market': {'stone': {'bids': [], 'asks': []}, 'wood': {'bids': [], 'asks': []}},
            'tax_info': {'current_tax_rates': [0.0]*7, 'time_in_period': 0, 'prev_incomes': [0.0]*self.num_agents}
        }
        return agent_obs, planner_obs

# Example usage
if __name__ == "__main__":
    env = EconomicEnv()
    agent_actions = [0] * env.num_agents  # NO-OP for all agents
    planner_action = [0] * 7  # Zero tax rates
    obs, planner_obs, rewards, planner_reward, done = env.step(agent_actions, planner_action)
    print("Agent observations:", obs[3]['spatial'])
    print("Planner reward:", planner_reward)