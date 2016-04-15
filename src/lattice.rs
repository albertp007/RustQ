use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Binomial {
    s0: f64,
    r: f64,
    q: f64,
    v: f64,
    t: f64,
    dt: f64,
    period: usize,
    num_nodes: usize,
    asset_prices: Vec<f64>,
    up: f64,
    down: f64,
    values: Vec<HashMap<isize, f64>>,
}

pub type StateTransitionFunc = Box<Fn((isize, isize, isize), isize)->isize>;
pub type PayoffFunc = Box<Fn((isize, isize, isize))->f64>;

pub fn default_state_func() -> StateTransitionFunc {
    Box::new( |(_, _, _): (isize, isize, isize), _: isize| { 0 } )
}

impl Binomial {
    pub fn calc_num_nodes(period: usize) -> usize {
        (period+1)*(period+2)/2
    }

    pub fn to_index(i: isize, j: isize) -> usize {
        ((i*i + 2*i + j)/2) as usize
    }

    fn iter_nodes<F>(num_periods: usize, f: &mut F)
    where F: FnMut((isize, isize))
    {
        for i in 0..(num_periods as isize+1) {
            for ii in 0..(i+1) {
                // skip feature is unstable as of writing
                let j = -i + ii * 2;
                f((i, j));
            }
        }
    }

    pub fn get_asset_price(&self, i: isize, j: isize) -> f64 {
        self.asset_prices[Binomial::to_index(i, j)]
    }

    pub fn new(s0: f64, r: f64, q: f64, v: f64, t: f64, period: usize,
        f: &StateTransitionFunc) -> Binomial {
        let num_nodes = Binomial::calc_num_nodes(period);
        let dt = t/period as f64;
        let u = v * dt.sqrt();
        let up = u.exp();
        let down = 1.0/up;
        let mut grid =
            Binomial {
                s0: s0, r: r, q: q, v: v, t: t, period: period,
                num_nodes: num_nodes,
                asset_prices: Vec::with_capacity(num_nodes),
                dt: dt, up: up, down: down,
                values: Vec::with_capacity(num_nodes),
            };
        grid.build( f );
        grid
    }

    fn set_state_value(&mut self, (i, j, k): (isize, isize, isize), v: f64) {
        self.values[Binomial::to_index(i, j)].insert(k, v);
    }

    fn forward_states(&mut self, (i, j):(isize, isize), f: &StateTransitionFunc)
    {
        let current_index = Binomial::to_index(i, j);
        // i is the time index, always larger than 0, okay to cast to usize
        let down_index = current_index + i as usize + 1;
        let up_index = down_index + 1;
        let (left, mut right) = self.values.split_at_mut(current_index+1);
        for &k in left.last().unwrap().keys() {
            let up_state = f((i, j, k), j+1);
            let down_state = f((i, j, k), j-1);
            right[up_index-current_index-1].insert(up_state, 0.0);
            right[down_index-current_index-1].insert(down_state, 0.0);
        }
    }

    fn build(&mut self, f: &StateTransitionFunc) {
        let n = self.period as isize;
        let mut v: Vec<f64> = Vec::with_capacity(2*self.period+1);
        for i in (-n)..(n+1) {
            v.push( self.s0 * self.up.powi(i as i32) );
        }

        Binomial::iter_nodes(self.period, &mut |(_, j)| {
            self.asset_prices.push( v[(j+n) as usize] );
            self.values.push( HashMap::new() );
        });

        // initialize the default initial state value
        self.set_state_value((0, 0, 0), 0.0);

        // run forward-shooting
        Binomial::iter_nodes(self.period-1, &mut |(i, j)| {
            self.forward_states((i, j), f);
        });
    }

    #[allow(dead_code)]
    fn price(&mut self, payoff: &PayoffFunc) -> f64 {
        let mut j = -(self.period as isize);
        let (_, mut last_period_nodes) =
            self.values.split_at_mut(self.num_nodes-self.period-1);
        for mut states in last_period_nodes.iter_mut() {
            for (state, value) in states.iter_mut() {
                *value = payoff((self.period as isize, j, *state));
                j += 2;
            }
        }
        0.0
    }
}

#[cfg(test)]
mod test {
    extern crate time;
    use lattice::*;
    use std::cmp;
    use std::collections::HashMap;

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
        let grid = Binomial::new(s0, r, q, v, t, period, &default_state_func());

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
            &default_state_func());
        println!("Creation time:  {}", time::precise_time_s() - now);

        assert!( false );
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
            |(_, _, k): (isize, isize, isize), to_j: isize| {
                cmp::max(to_j, k)
        } );

        let grid = Binomial::new(s0, r, q, v, t, period, &lookback);
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

        assert_eq!(last_period_nodes, expected.as_slice());
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
            Binomial::new(s0, r, q, v, t, period, &default_state_func());
        let up = grid.up;

        let european_payoff: PayoffFunc = Box::new(
            move |(_, j, _)| {
                (s0 * up.powi( j as i32 ) - strike).max(0.0)
            }
        );

        grid.price(&european_payoff);

        let (_, last_period_nodes) =
            grid.values.split_at(grid.num_nodes-grid.period-1);
        let mut expected: Vec<HashMap<isize, f64>> = Vec::new();
        for i in 0..5 {
            let j = (i-2)*2;
            let mut map = HashMap::new();
            map.insert(0, (s0 * up.powi( j as i32 ) - strike).max(0.0));
            expected.push(map);
        }

        assert_eq!(last_period_nodes, expected.as_slice());
    }
}
