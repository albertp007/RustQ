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
    down: f64
}

impl Binomial {
    pub fn calc_num_nodes(period: usize) -> usize {
        (period+1)*(period+2)/2
    }

    pub fn to_index(i: isize, j: isize) -> usize {
        ((i*i + 2*i + j)/2) as usize
    }

    pub fn get_asset_price(&self, i: isize, j: isize) -> f64 {
        self.asset_prices[Binomial::to_index(i, j)]
    }

    pub fn new(s0: f64, r: f64, q: f64, v: f64, t: f64, period: usize)
        -> Binomial {
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
                dt: dt, up: up, down: down
            };
        grid.populate_grid();
        grid
    }

    fn populate_grid(&mut self) {
        let n = self.period as isize;
        for i in 0..(n+1) {
            for ii in 0..(i+1) {
                let j = -i + ii * 2;
                self.asset_prices.push( self.s0 * (j as f64*self.up).exp() );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use lattice::Binomial;

    #[test]
    pub fn test_binomial_populate_and_get_asset_price() {

        fn expected_price(s0: f64, up: f64, j: isize) -> f64 {
            s0 * (up * j as f64).exp()
        }

        let s0 = 50.0;
        let r = 0.05;
        let q = 0.0;
        let v = 0.3;
        let t = 2.0;
        let period = 5;
        let mut grid = Binomial::new(s0, r, q, v, t, period);

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
}
