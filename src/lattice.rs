use std::collections::HashMap;
use std::cell::RefCell;
use option::OptionType::*;
use option::BarrierInOut::*;
use option::BarrierUpDown::*;
use option::*;

/// Structure to represent a recombining binomial tree.  Node (i, j) is the node
/// in the i-th period with j up moves from the initial asset price.  j can be
/// negative, in which case it means the number of down moves from the initial
/// price.  For example, in period 1, there are two nodes, (1, -1) and (1, 1).
/// In period 2, there are 3 nodes, (2, -2), (2, 0) and (2, 2).  In period 3,
/// there are 4 nodes (3, -3), (3, -1), (3, 1) and (3, 3).  In general, there
/// are i+1 nodes in the i-th period with the nodes ranging from (i, -i),
/// (i, -i+2), ..., (i, -i+2n), ...(i, i).  Other parameters are as follows:
///
/// # Fields
/// * `s0` - initial asset price at node 0 (user-supplied)
/// * `r` - interest rate (user-supplied)
/// * `q` - convenience yield (user-supplied)
/// * `v` - volatility p.a. (user-supplied)
/// * `t` - time to expiry (user-supplied)
/// * `dt` - time duration of one time step (calculated)
/// * `period` - number of periods (user-supplied)
/// * `up` - the up move factor to be applied on the asset price of the current
///   node to arrive at the asset price of the up node (calculated)
/// * `down` - the down move factor to be applied on the asset price of the
///   current node to arrive at the asset price of the down node (calculated)
/// * `p` - the risk-neutral probability of an up move (calculated)
/// * `asset_prices` - vector representing the asset price of each node.  The
///   index can be uniquely determined by (i, j) (calculated)
/// * `values` - vector of (ref cell of) hashmap with key being the state
///   (integer - can be negative) and the value is the value of that node which
///   is calculated either by the payoff function if it is a terminal node or
///   by backward induction, if it is a non-leaf node.  The payoff function
///   is an input to the price() method.  Any other node specific policies can
///   be specified by a node value function, which is the another paramater to
///   the price() method. (calculated)
/// * `state_transition` - the state transition function which is of type
///   StateTransitionFunc (user-supplied)
/// * `initial_state` - the starting state of node 0 (user-supplied)
#[allow(dead_code)]
pub struct Binomial {
    s0: f64,
    r: f64,
    q: f64,
    v: f64,
    t: f64,
    dt: f64,
    period: usize,
    num_nodes: usize,
    up: f64,
    down: f64,
    p: f64, // up probability
    asset_prices: Vec<f64>,
    values: Vec<RefCell<HashMap<isize, f64>>>,
    state_transition: StateTransitionFunc,
    initial_state: isize,
}

/// Type alias for state transition function in the grid.  This determines
/// the state to evolve to, given the current state k in node (i, j) and going
/// forward to node (i+1, to_j).  (i+1) is because we are always moving to a
/// node in the next period.  to_j can basically be going up or going down from
/// node (i, j), therefore, can take up either j+1 or j-1.  In short, this
/// represents the grid evolution function:
///
/// F(t+dt) = G(F(t), t, S(t+dt))
pub type StateTransitionFunc =
    Box<Fn(&Binomial, (isize, isize, isize), isize)->isize>;

/// The payoff function which is used to calculate the value of terminal nodes
/// based on the payoff of the product at expiry.  It has access to a reference
/// to the Binomial struct and also the node index (i, j, k) where i represents
/// the period, j represents the number of up moves from the initial price and k
/// is the state.  Each state represents one particular path possibility.  This
/// is a boxed closure in order for it to be passed around easily as both input
/// parameter to a function and return value from a function
pub type PayoffFunc = Box<Fn(&Binomial, (isize, isize, isize))->f64>;

/// The node value function is used to specify any early exercise policy or
/// any policies which depends on the state of the node and its induced value
/// which is calculated by discounting the expected value of the up node and
/// the down node in the corresponding states determined by the state transition
/// function, using the risk-neutral measure.  The induced value is calculated
/// by the backward_induce() method and supplied to the NodeValueFunc as a
/// parameter.  In addition, it is also passed a reference to the Binomial
/// struct and the node index (i, j, k) where i represents the period, j
/// represents the number of up moves in the node and k is the state.  Each
/// state represents one particular path possibility. This is a boxed closure in
/// order for it to be passed around easily as both input parameter to a
/// function and return value from a function
pub type NodeValueFunc = Box<Fn(&Binomial, (isize, isize, isize), f64)->f64>;

/// A default state transition function which always return a state 0 regardless
/// of the current node and the node to move to.  It can be used for non-path
/// dependent product, e.g. vanilla options.  It also serves as an example
/// state transition function
pub fn default_state_func() -> StateTransitionFunc {
    Box::new( |_, (_, _, _): (isize, isize, isize), _: isize| { 0 } )
}

/// A default node value function which always simply returns the backward
/// induced value that is passed to it.  It can be used for products which do
/// not have any node specific policies e.g. european options.  It also serves
/// as an example node value function
pub fn default_node_value() -> NodeValueFunc {
    Box::new( |_, (_, _, _), induced_value| {
        induced_value
    })
}

/// A payoff function for vanilla call and put options
/// # Arguments
/// * `opt_type` - OptionType, either Call or Put
/// * `strike` - strike price
///
/// # Example
/// ```
/// use payoffs::option::*;
/// use payoffs::option::OptionType::*;
/// use payoffs::lattice::*;
///
/// let s0 = 50.0;
/// let r = 0.05;
/// let q = 0.0;
/// let v = 0.3;
/// let t = 0.25;
/// let strike = 50.0;
/// let period = 4;
///
/// let mut grid =
///     Binomial::new(s0, r, q, v, t, period, default_state_func(), 0);
///
/// grid.price(&vanilla_payoff(Call, strike), &default_node_value());
/// ```
pub fn vanilla_payoff(opt_type: OptionType, strike: f64) -> PayoffFunc {
    Box::new(
        move |grid, (i, j, _)| {
            let underlying_price = grid.get_asset_price(i, j);
            let intrinsic = underlying_price - strike;
            match opt_type {
                Call => intrinsic.max(0.0),
                Put => (-intrinsic).max(0.0)
            }
        })
}

/// A payoff function for barrier option
/// # Arguments
/// * `barrier_inout` - either In or Out for knock-in and knock-out respectively
/// * `opt_type` - either Call or Put
/// * `strike` - strike price
pub fn barrier_payoff(barrier_inout: BarrierInOut, opt_type: OptionType,
    strike: f64) -> PayoffFunc {
        let vanilla_payoff = vanilla_payoff(opt_type, strike);
        Box::new( move |grid, (i, j, k)| {
            let triggered = if k == 0 { false } else { true };
            let vanilla_payoff = vanilla_payoff(&grid, (i, j, k));
            match barrier_inout {
                In => if triggered { vanilla_payoff } else { 0.0 },
                Out => if triggered { 0.0 } else { vanilla_payoff },
            }
        })
}

/// State transition function for barrier option with continuous observation
/// # Arguments
/// * `barrier_updown` - either Up or Down
/// * `barrier` - barrier price
///
/// # Example
/// ```
/// use payoffs::lattice::*;
/// use payoffs::option::OptionType::*;
/// use payoffs::option::BarrierInOut::*;
/// use payoffs::option::BarrierUpDown::*;
/// let s0 = 100.0;
/// let r = 0.02;
/// let q = 0.0;
/// let v = 0.4;
/// let t = 0.25;
/// let strike = 100.0;
/// let barrier = 125.0;
/// let period = 3000;
///
/// let mut grid = Binomial::new(s0, r, q, v, t, period,
///     barrier_state(Up, barrier), 0);
///
/// let price = grid.price(&barrier_payoff(Out, Call, strike),
///     &default_node_value());
///
/// println!("Price: {}", price);
/// ```
pub fn barrier_state(barrier_updown: BarrierUpDown, barrier: f64)
    -> StateTransitionFunc {

    Box::new( move |grid, (i, _, k), to_j| {
        let asset_price = grid.get_asset_price(i+1, to_j);
        // State 0 - not triggered; State 1 - triggered
        // if already triggered, stay triggered
        let triggered = ( k == 1 ) || match barrier_updown {
            Up => if asset_price > barrier { true } else { false },
            Down => if asset_price < barrier { true } else { false },
        };
        if triggered { 1 } else { 0 }
    })
}

impl Binomial {
    /// Calculates the number of nodes in the recombining Binomial tree given
    /// the number of periods.  A 1-period tree has 3 nodes, the initial node
    /// leading to an up node and a down node
    pub fn calc_num_nodes(period: usize) -> usize {
        (period+1)*(period+2)/2
    }

    /// Converts node index (i, j) to the index in the vector storing the
    /// node values
    pub fn to_index(i: isize, j: isize) -> usize {
        ((i*i + 2*i + j)/2) as usize
    }

    /// Iterates through the node indices of a binomial tree.  Each node index
    /// is then passed to a mutable closure.  It only traverses in the forward
    /// direction.
    ///
    /// * `start` - the start period
    /// * `end` - the end period (inclusive)
    pub fn iter_nodes<F>(start: isize, end: isize, f: &mut F)
    where F: FnMut((isize, isize))
    {
        for i in start..(end+1) {
            for ii in 0..(i+1) {
                // skip feature is unstable as of writing
                let j = -i + ii * 2;
                f((i, j));
            }
        }
    }

    /// Calculates the risk-neutral probability of moving up a node in the next
    /// period.  This is used in calculate the expected value of the current
    /// node using the risk-neutral measure.
    ///
    /// * `r` - interest rate
    /// * `q` - convenience yield
    /// * `v` - volatility p.a.
    /// * `dt` - the duration of time of one period
    pub fn calc_up_prob( r: f64, q: f64, v: f64, dt: f64) -> f64 {
        let u = (v*dt.sqrt()).exp();
        let d = 1.0/u;
        (((r-q)*dt).exp() - d)/(u-d)
    }

    /// Returns the underlying asset price at a particular node with index
    /// (i, j)
    pub fn get_asset_price(&self, i: isize, j: isize) -> f64 {
        self.asset_prices[Binomial::to_index(i, j)]
    }

    /// Constructs a binomial tree.  This method sets the various fields in the
    /// struct based on the input parameters as well as all the calculated
    /// fields.  It populates the asset price at each node and also runs the
    /// forward state transitions to populate all the possible states in each
    /// node, by calling the build() method
    ///
    /// # Arguments
    /// * `s0` - initial asset price at node 0 (user-supplied)
    /// * `r` - interest rate (user-supplied)
    /// * `q` - convenience yield (user-supplied)
    /// * `v` - volatility p.a. (user-supplied)
    /// * `t` - time to expiry (user-supplied)
    /// * `period` - number of periods (user-supplied)
    /// * `state_transition` - the state transition function which is of type
    ///   StateTransitionFunc
    ///
    /// # Example
    /// ```
    /// use payoffs::lattice::*;
    /// let s0 = 50.0;
    /// let r = 0.05;
    /// let q = 0.0;
    /// let v = 0.3;
    /// let t = 2.0;
    /// let period = 5;
    /// let grid =
    ///     Binomial::new(s0, r, q, v, t, period, default_state_func(), 0);
    /// ```
    pub fn new(s0: f64, r: f64, q: f64, v: f64, t: f64, period: usize,
        f: StateTransitionFunc, initial_state: isize) -> Binomial {
        let num_nodes = Binomial::calc_num_nodes(period);
        let dt = t/period as f64;
        let u = v * dt.sqrt();
        let up = u.exp();
        let down = 1.0/up;
        let mut grid =
            Binomial {
                s0: s0, r: r, q: q, v: v, t: t, period: period,
                num_nodes: num_nodes,
                dt: dt, up: up, down: down,
                p: Binomial::calc_up_prob(r, q, v, dt),
                asset_prices: Vec::with_capacity(num_nodes),
                values: Vec::with_capacity(num_nodes),
                state_transition: f, initial_state: initial_state,
            };
        grid.build();
        grid
    }

    /// Sets the value of state k at node (i, j)
    pub fn set_state_value(&mut self, (i, j, k): (isize, isize, isize), v: f64)
    {
        self.values[Binomial::to_index(i, j)].borrow_mut().insert(k, v);
    }

    /// Iterates through each of the states in node (i, j) and for each state
    /// run the state transition function to initialize the values of the
    /// possible states in the up node and down node it leads to in the next
    /// period with value 0.0.
    pub fn forward_states(&mut self, (i, j):(isize, isize))
    {
        let f = &self.state_transition;
        let current_index = Binomial::to_index(i, j);
        // i is the time index, always larger than 0, okay to cast to usize
        let down_index = current_index + i as usize + 1;
        let up_index = down_index + 1;
        for &k in self.values[current_index].borrow().keys() {
            let up_state = f(&self, (i, j, k), j+1);
            let down_state = f(&self, (i, j, k), j-1);
            self.values[down_index].borrow_mut().insert(down_state, 0.0);
            self.values[up_index].borrow_mut().insert(up_state, 0.0);
        }
    }

    /// Builds the binomial tree by setting the underlying asset price of each
    /// of the nodes and also set up the possible states of each node. It also
    /// initializes the starting state of the root node of the tree to be the
    /// value passed in through new()
    pub fn build(&mut self) {
        let n = self.period as isize;
        let initial_state = self.initial_state;
        let mut v: Vec<f64> = Vec::with_capacity(2*self.period+1);
        for i in (-n)..(n+1) {
            v.push( self.s0 * self.up.powi(i as i32) );
        }

        Binomial::iter_nodes(0, n, &mut |(_, j)| {
            self.asset_prices.push( v[(j+n) as usize] );
            self.values.push( RefCell::new(HashMap::new()) );
        });

        // initialize the default initial state value
        self.set_state_value((0, 0, initial_state), 0.0);

        // run forward-shooting
        Binomial::iter_nodes(0, n-1, &mut |(i, j)| {
            self.forward_states((i, j));
        });
    }

    /// Calculates the value of each of the states in all terminal nodes i.e.
    /// the nodes at expiry, given a PayoffFunc
    pub fn calc_terminal(&mut self, payoff: &PayoffFunc) {
        let n = self.period as isize;
        Binomial::iter_nodes(n, n, &mut |(i, j)| {
            let index = Binomial::to_index(i, j);
            for (&k, v) in self.values[index].borrow_mut().iter_mut() {
                *v = payoff(&self, (i, j, k));
            }
        });
    }

    /// Runs backward induction from the period of nodes which are one period
    /// before expiry.  In each of the nodes, it iterates through its states and
    /// for each state, it calculates the induced value by calculating the
    /// expected value based on the risk-neutral measure (which is calculated by
    /// the calc_up_prob() method) and the values of the states in the up
    /// node and down node according to the state transition function.  Finally,
    /// discount this expected value by the duration of one time period. i.e.
    /// for each k in node (i, j),
    ///
    /// V(i, j, k) = exp(-r dt) *  [ pu * V(i+1, j+1, s((i, j, k), j+1) +
    ///     pd * V(i+1, j-1, s((i, j, k), j-1)]
    ///
    /// where
    ///
    /// * pu is the risk neutral probability of going up one node,
    /// * pd is the risk neutral probability of going down one node and
    /// * s((i, j, k), j') is the state transition function which returns
    ///   the state to transition to, given current state k at node(i, j)
    ///   when moving to node (i+1, j')
    ///
    /// With the induced value of each node at each state k calculated, it then
    /// passes it to the NodeValueFunc which then returns the final value of the
    /// node for state k.  The NodeValueFunc takes care of any required
    /// manipulation of the induced value to arrive at the final value of the
    /// node at state k.  For example, an early exercise policy can be
    /// specified in the NodeValueFunc
    pub fn backward_induce(&mut self, node_value: &NodeValueFunc) {
        let n = self.period as isize;
        let state = &self.state_transition;
        let discount = (-self.r*self.dt).exp();
        // Note the caveat in using rev() below. The range (0..n) is from 0 to
        // n-1 inclusive.  Reversing it gives us (n-1) to 0 *inclusive*
        for i in (0..n).rev() {
            // use index as start and inc by 1 to save calling to_index, since
            // we know all nodes in the same period are consecutive
            let mut index = (i * (i+1) / 2) as usize;
            for ii in 0..(i+1) {
                let j = -i + ii * 2;
                for (&k, v) in self.values[index].borrow_mut().iter_mut() {
                    let down_index = index + i as usize + 1;
                    let up_index = down_index + 1;
                    let up_state = state(&self, (i, j, k), j+1);
                    let up_value = self.values[up_index].borrow()[&up_state];
                    let down_state = state(&self, (i, j, k), j-1);
                    let down_value =
                        self.values[down_index].borrow()[&down_state];
                    let induced_value = discount * (
                        self.p*up_value+(1.0 - self.p)*down_value);
                    *v = node_value(&self, (i, j, k), induced_value);
                }
                index += 1;
            }
        }
    }

    /// Prices the binomial tree by first calculating the values of each node
    /// at expiry, then run backward induction and returns the price of the
    /// initial state at the root node
    #[allow(dead_code)]
    pub fn price(&mut self, payoff: &PayoffFunc, node_value: &NodeValueFunc)
        -> f64 {
        self.calc_terminal(payoff);
        self.backward_induce(node_value);
        self.values[0].borrow()[&self.initial_state]
    }
}

#[cfg(test)]
mod test {
    extern crate time;
    use lattice::*;
    use std::cmp;
    use std::collections::HashMap;
    use option::OptionType::*;
    use option::black_scholes;
    use option::BarrierInOut::*;
    use option::BarrierUpDown::*;

    #[test]
    pub fn test_binomial_populate_and_get_asset_price() {

        fn expected_price(s0: f64, up: f64, j: isize) -> f64 {
            s0 * up.powi( j as i32 )
        }

        let s0 = 50.0;
        let r = 0.05;
        let q = 0.0;
        let v = 0.3;
        let t = 2.0;
        let period = 5;
        let grid =
            Binomial::new(s0, r, q, v, t, period, default_state_func(), 0);

        let n = grid.period as isize;
        for i in 0..(n+1) {
            for ii in 0..(i+1) {
                let j = -i + ii * 2;
                let expected = expected_price(s0, grid.up, j);
                let actual = grid.get_asset_price(i, j);
                println!("({}, {}), expected: {}, actual: {}", i, j, expected,
                    actual );
                assert_eq!( expected, actual );
            }
        }
        assert_eq!( s0, grid.get_asset_price(4, 0) );
    }

    #[test]
    pub fn test_binomial_populate_time() {

        let s0 = 50.0;
        let r = 0.05;
        let q = 0.0;
        let v = 0.3;
        let t = 2.0;
        let period = 3000;

        let now = time::precise_time_s();
        let mut _grid = Binomial::new(s0, r, q, v, t, period,
            default_state_func(), 0);
        println!("Creation time:  {}", time::precise_time_s() - now);

        // assert!( false );
    }

    #[test]
    pub fn test_state_transition() {
        let s0 = 50.0;
        let r = 0.05;
        let q = 0.0;
        let v = 0.3;
        let t = 2.0;
        let period = 4;

        let lookback: StateTransitionFunc = Box::new(
            |_, (_, _, k): (isize, isize, isize), to_j: isize| {
                cmp::max(to_j, k)
        } );

        let grid = Binomial::new(s0, r, q, v, t, period, lookback, 0);
        let (_, last_period_nodes) =
            grid.values.split_at(grid.num_nodes-grid.period-1);

        let mut expected: Vec<HashMap<isize, f64>> = Vec::new();
        for _ in 0..5 {
            expected.push(HashMap::new());
        }
        expected[0].insert(0, 0.0);
        expected[1].insert(0, 0.0);
        expected[1].insert(1, 0.0);
        expected[2].insert(0, 0.0);
        expected[2].insert(1, 0.0);
        expected[2].insert(2, 0.0);
        expected[3].insert(2, 0.0);
        expected[3].insert(3, 0.0);
        expected[4].insert(4, 0.0);

        let results: Vec<HashMap<isize, f64>> = last_period_nodes.iter().map(
            |cell| {
                cell.borrow().clone()
            }).collect();

        assert_eq!(results.as_slice(), expected.as_slice());
    }

    #[test]
    pub fn test_payoff_function() {
        let s0 = 50.0;
        let r = 0.05;
        let q = 0.0;
        let v = 0.3;
        let t = 0.25;
        let strike = 50.0;
        let period = 4;

        let mut grid =
            Binomial::new(s0, r, q, v, t, period, default_state_func(), 0);
        let up = grid.up;

        grid.price(&vanilla_payoff(Call, strike), &default_node_value());

        let (_, last_period_nodes) =
            grid.values.split_at(grid.num_nodes-grid.period-1);
        let mut expected: Vec<HashMap<isize, f64>> = Vec::new();
        for i in 0..5 {
            let j = (i-2)*2;
            let mut map = HashMap::new();
            map.insert(0, (s0 * up.powi( j as i32 ) - strike).max(0.0));
            expected.push(map);
        }

        let results: Vec<HashMap<isize, f64>> = last_period_nodes.iter().map(
            |cell| {
                cell.borrow().clone()
            }).collect();

        assert_eq!(results.as_slice(), expected.as_slice());
    }

    #[test]
    pub fn test_vanilla_call() {
        let s0 = 50.0;
        let r = 0.05;
        let q = 0.0;
        let v = 0.3;
        let t = 0.25;
        let strike = 50.0;
        let period = 1000;

        let now = time::precise_time_s();
        let mut grid =
            Binomial::new(s0, r, q, v, t, period, default_state_func(), 0);

        let price = grid.price(&vanilla_payoff(Call, strike),
            &default_node_value());
        println!("Time taken: {}", time::precise_time_s() - now);

        let expected = black_scholes(s0, r, q, v, t, Call, strike);
        assert!( (price-expected).abs() < 0.01 );
    }

    #[test]
    pub fn test_barrier() {
        let s0 = 100.0;
        let r = 0.02;
        let q = 0.0;
        let v = 0.4;
        let t = 0.25;
        let strike = 100.0;
        let barrier = 125.0;
        let period = 1000;

        let now = time::precise_time_s();
        let mut grid = Binomial::new(s0, r, q, v, t, period,
            barrier_state(Up, barrier), 0);

        let price = grid.price(&barrier_payoff(Out, Call, strike),
            &default_node_value());
        println!("Price: {}", price);
        println!("Time taken: {}", time::precise_time_s() - now);
    }

}
