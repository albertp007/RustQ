extern crate rand;

use self::rand::distributions::normal::Normal;
use self::rand::distributions::IndependentSample;
use option::OptionType;
use option::BarrierInOut;
use option::BarrierUpDown;

/// Type to specify the type of variance reduction scheme to be used when
/// generating paths
///
/// * None for no variance reduction
/// * ATV for Antithetic variates
///
pub enum VarianceReduction {
    None,
    ATV
}

/// Type to represent a stock path.
///
/// * An ordinary path is simply a vector of f64, each item representing a
/// price.
/// * An antithetic path is a pair of vectors, whereby the element-by-element
/// sum of the vector of normally distributed samples which are used to generate
/// the stock paths is equal to zero.  (Note that it is not the sum of the stock
/// paths that add up to zero, but the sample vectors which are used to generate
/// them)
pub enum Path<T> {
    Ordinary(Vec<T>),
    Antithetic((Vec<T>, Vec<T>))
}

pub type PayoffFunc = Box<Fn(&Vec<f64>)->f64>;

/// Draw samples from a normal distribution with the mean and standard deviation
/// specified in the arguments
///
/// # Argument
/// * `mean` - Mean
/// * `sd` - Standard deviation
/// * `n` - number of samples to draw
///
/// # Example
/// ```
/// let samples = payoffs::montecarlo::draw_normal( 0.0, 1.0, 100 );
/// println!( "{:?}", samples );
/// ```
pub fn draw_normal(mean: f64, sd: f64, n: u32) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(mean, sd);
    (0..n).map( |_| {
        normal.ind_sample(&mut rng)
    }).collect()
}

/// Given a vector of normally distributed samples, generate one stock path
/// using the geometric brownian motion model
///
/// #Argument
/// * `s0` - initial stock price at time 0
/// * `r` - interest rate
/// * `q` - convenience yield (absorbs any cost or yield of holding the asset)
/// * `v` - volatility e.g. 0.4 is 40 vol points (per annum)
/// * `t` - number of years e.g. 0.25 is quarter of a year
///
/// #Example
/// ```
/// let t = 0.25;
/// let n = 100;
/// let normal_samples = payoffs::montecarlo::draw_normal( 0.0,
///     (t/n as f64).sqrt(), n);
/// let samples = payoffs::montecarlo::gbm_samples(100.0, 0.02, 0.0, 0.4, 0.25,
///     &normal_samples);
/// println!("{:?}", samples);
/// ```
pub fn gbm_samples(s0: f64, r: f64, q: f64, v: f64, t: f64,
    normal_samples: &Vec<f64>) -> Vec<f64>
{
    let n = normal_samples.len();
    let dt = t/n as f64;
    let drift = (r - q - v * v/2.0) * dt;
    // println!("drift: {}", drift);
    normal_samples.iter()
        .map( |dwt| { drift + v * dwt })
        .scan(0.0, |acc, x| { *acc = *acc + x; Some(*acc) })
        .map( |x| { s0 * x.exp() })
        .collect()
}

/// Create the anti-thetic vector given a vector of normally distributed samples
///
/// #Argument
/// * `orig` - the original vector of normally distributed samples
///
/// #Example
/// ```
/// let normal_samples = payoffs::montecarlo::draw_normal( 0.0, 1.0, 100 );
/// let anti = payoffs::montecarlo::make_antithetic( &normal_samples );
/// println!("{:?}", anti);
/// ```
pub fn make_antithetic(orig: &Vec<f64>) -> Vec<f64> {
    orig.iter().map( |x| { -x }).collect()
}

/// Generate an instance of Path struct assuming geometric brownian motion
/// given the type of variance reduction scheme
///
/// #Argument
/// * `s0` - initial stock price at time 0
/// * `r` - interest rate
/// * `q` - convenience yield (absorbs any cost or yield of holding the asset)
/// * `v` - volatility e.g. 0.4 is 40 vol points (per annum)
/// * `t` - number of years e.g. 0.25 is quarter of a year
/// * `n` - number of points in the path
/// * `vr` - variance reduction scheme
///
/// #Example
/// ```
/// let single_path = payoffs::montecarlo::gbm_path(100.0, 0.02, 0.0, 0.4, 0.25,
///     100, payoffs::montecarlo::VarianceReduction::None);
/// let pair_of_paths = payoffs::montecarlo::gbm_path(100.0, 0.02, 0.0, 0.4,
///     0.25, 100, payoffs::montecarlo::VarianceReduction::ATV);
/// ```
pub fn gbm_path( s0: f64, r: f64, q: f64, v: f64, t: f64, n: u32,
    vr: VarianceReduction ) -> Path<f64> {
    let sd = (t/n as f64).sqrt();
    let normal_samples = draw_normal(0.0, sd, n);
    let path = gbm_samples(s0, r, q, v, t, &normal_samples);

    match vr {
        VarianceReduction::None => Path::Ordinary(path),
        VarianceReduction::ATV => {
            let anti_samples = make_antithetic(&normal_samples);
            let anti = gbm_samples(s0, r, q, v, t, &anti_samples);
            Path::Antithetic((path, anti))
        }
    }
}

/// A boxed closure representing the vanilla european payoff function
///
/// # Argument
/// * `opt_type` - option::OptionType, either call or put
/// * `r` - interest rate
/// * `t` - time to expiry in years
pub fn european_payoff(opt_type: OptionType, r: f64, t: f64, strike: f64)
    -> PayoffFunc
{
    Box::new( move |path| {
        let intrinsic: f64 = path[path.len()-1] - strike;
        let intrinsic = match opt_type {
            OptionType::Call => intrinsic,
            OptionType::Put => -intrinsic
        };
        let payoff = intrinsic.max(0.0);
        if payoff > 0.0 { (-r*t).exp() * payoff } else { 0.0 }
    } )
}

/// A boxed closure representing the payoff function of barrier option
///
/// # Argument
/// * `barrier_updown` - Up or down
/// * `barrier_inout` - In or Out
/// * `opt_type` - Call or put
/// * `r` - interest rate
/// * `t` - time to expiry in years
/// * `barrier` - barrier price
/// * `strike` - strike price
///
/// # Example
/// To price and up and out call with initial stock price 100.0, interest rate
/// 2%, zero yield, volatility 40% p.a., time to expiry 0.25 years (3 months),
/// barrier 125 and strike 100, with a monthly observation i.e. 3 observations
///
/// ```
/// let s0 = 100.0;
/// let r = 0.02;
/// let q = 0.0;
/// let v = 0.4;
/// let t = 0.25;
/// let k = 100.0;
/// let barrier = 125.0;
/// let m = 10000000;
/// let obs = 3;
///
/// let paths: Vec<payoffs::montecarlo::Path<f64>> =
///     (0..m).map(|_| payoffs::montecarlo::gbm_path(s0, r, q, v, t, obs,
///         payoffs::montecarlo::VarianceReduction::ATV)).collect();
///
/// let (estimate, _, _) =
///     payoffs::montecarlo::monte_carlo(&paths,
///         payoffs::montecarlo::barrier_payoff(
///             payoffs::option::BarrierUpDown::Up,
///             payoffs::option::BarrierInOut::Out,
///             payoffs::option::OptionType::Call, r, t, barrier, k));
/// ```
pub fn barrier_payoff(barrier_updown: BarrierUpDown,
    barirer_inout: BarrierInOut, opt_type: OptionType,
    r: f64, t: f64, barrier: f64, strike: f64) -> PayoffFunc
{
    Box::new( move |path| {
        let intrinsic = path[path.len()-1] - strike;
        let discount = (-r*t).exp();
        let payoff: f64  =
            match barirer_inout {
                BarrierInOut::In =>
                    match barrier_updown {
                        BarrierUpDown::Up =>
                            if path.iter().any( |&x| x > barrier )
                            { intrinsic } else { 0.0 },
                        BarrierUpDown::Down =>
                            if path.iter().any( |&x| x < barrier )
                            { intrinsic } else { 0.0 },
                    },
                BarrierInOut::Out =>
                    match barrier_updown {
                        BarrierUpDown::Up =>
                            if path.iter().any( |&x| x > barrier )
                            { 0.0 } else { intrinsic },
                        BarrierUpDown::Down =>
                            if path.iter().any( |&x| x < barrier )
                            { 0.0 } else { intrinsic },
                    }
            };
        // println!( "intrinsic: {}, payoff: {}", intrinsic, payoff.max(0.0));
        ( match opt_type {
            OptionType::Call => payoff.max(0.0),
            OptionType::Put => (-payoff).max(0.0)
        } ) * discount
    })
}

/// Run monte carlo simulation given an instance of Path struct and a boxed
/// closure which defines the payoff function given a path
///
/// # Argument
/// * `paths` - a vector of Path structs
/// * `f` - a boxed closure defining the payoff function given a path.  See
/// european_payoff for an example
///
/// # Example
/// ```
/// let s0 = 100.0;
/// let r = 0.02;
/// let q = 0.0;
/// let v = 0.4;
/// let t = 0.25;
/// let k = 100.0;
/// let m = 5000000;
///
/// let paths: Vec<payoffs::montecarlo::Path<f64>> = (0..m)
///     .map(|_| payoffs::montecarlo::gbm_path( s0, r, q, v, t, 1,
///         payoffs::montecarlo::VarianceReduction::ATV))
///     .collect();
///
/// let (estimate, _, _) =
///     payoffs::montecarlo::monte_carlo(&paths,
///         payoffs::montecarlo::european_payoff(
///         payoffs::option::OptionType::Call, r, t, k));
///
/// println!("Estimate: {}", estimate);
/// ```
pub fn monte_carlo(paths: &Vec<Path<f64>>, f: PayoffFunc) -> (f64, f64, f64)
{
    let payoffs: Vec<f64> = paths.iter().map( |path| {
        match path {
            &Path::Ordinary(ref ord) => f( &ord ),
            &Path::Antithetic((ref ord, ref anti)) => 0.5 * (f(&ord) + f(&anti))
        }
    } ).collect();

    let sum = payoffs.iter().fold( 0.0, |acc, x| acc + x );
    let sq_sum = payoffs.iter().fold( 0.0, |acc, x| acc + x * x );
    let n = paths.len();
    let estimate = sum/n as f64;
    let var = sq_sum/(n-1) as f64 - estimate*estimate;
    let stderr = (var/n as f64).sqrt();
    (estimate, var, stderr)
}

#[cfg(test)]
mod test {
    extern crate time;
    use option::OptionType;
    use option::black_scholes;
    use option::BarrierUpDown;
    use option::BarrierInOut;
    use montecarlo::monte_carlo;
    use montecarlo::VarianceReduction;
    use montecarlo::gbm_path;
    use montecarlo::Path;
    use montecarlo::draw_normal;
    use montecarlo::european_payoff;
    use montecarlo::barrier_payoff;
    use montecarlo::make_antithetic;
    use util::equal_within;

    #[test]
    fn mc_european_call() {
        let s0 = 100.0;
        let r = 0.02;
        let q = 0.0;
        let v = 0.4;
        let t = 0.25;
        let k = 100.0;
        let m = 5000000;

        let now = time::precise_time_s();
        let paths: Vec<Path<f64>> = (0..m).map(|_| gbm_path(s0, r, q, v, t, 1,
            VarianceReduction::ATV)).collect();

        let (estimate, _, err) =
            monte_carlo(&paths, european_payoff(OptionType::Call, r, t, k));

        println!("{} secs", time::precise_time_s() - now);

        let bs = black_scholes(s0, r, q, v, t, OptionType::Call, k);
        println!("estimate: {}, bs: {}", estimate, bs);
        assert_eq!( true, equal_within(bs, estimate, 2.0*err) );
    }

    #[test]
    fn test_draw_normal() {
        let mean = 0.0;
        let sd = 0.4;
        let m = 5000000;
        let samples: Vec<f64> = draw_normal(mean, sd, m);

        let sum = samples.iter().fold(0.0, |acc, x| acc + x );
        let sum_sq = samples.iter().fold(0.0, |acc, x| acc + x * x );
        let est = sum/m as f64;
        let var = sum_sq/(m-1) as f64 - est * est;
        let sd1 = var.sqrt();
        assert_eq!( true, equal_within(mean, est, 0.001) );
        assert_eq!( true, equal_within(sd, sd1, 0.001) );
    }

    #[test]
    fn test_make_antithetic() {
        let t = 0.25;
        let n = 10;
        let normal_samples = draw_normal(0.0, (t/n as f64).sqrt(), n);
        let anti = make_antithetic( &normal_samples );
        let sum_to_zero = anti.iter().zip( normal_samples.iter() )
            .map(|(x, y)| x + y)
            .all( |x| x == 0.0 );
        assert_eq!( true, sum_to_zero );
    }

    #[test]
    fn test_mc_up_out_call() {
        let s0 = 100.0;
        let r = 0.02;
        let q = 0.0;
        let v = 0.4;
        let t = 0.25;
        let k = 100.0;
        let barrier = 125.0;
        let m = 10000000;

        let now = time::precise_time_s();
        let paths: Vec<Path<f64>> = (0..m).map(|_| gbm_path(s0, r, q, v, t, 3,
            VarianceReduction::ATV)).collect();

        let (estimate, _, _) =
            monte_carlo(&paths, barrier_payoff(BarrierUpDown::Up,
                BarrierInOut::Out, OptionType::Call, r, t, barrier, k));
        println!("{} secs", time::precise_time_s() - now);

        // No easy way to test this, hard-coding a value of 3.30 for now which
        // is obtained from the F# deep dives book
        assert_eq!( true, equal_within(estimate, 3.30, 0.01) );
    }
}
