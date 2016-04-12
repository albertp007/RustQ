extern crate rand;

use self::rand::distributions::normal::Normal;
use self::rand::distributions::IndependentSample;
use option::OptionType;
use option::OptionType::*;
use option::BarrierInOut;
use option::BarrierInOut::*;
use option::BarrierUpDown;
use option::BarrierUpDown::*;

/// Type to specify the type of variance reduction scheme to be used when
/// generating paths
///
/// * None for no variance reduction
/// * ATV for Antithetic variates
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
use montecarlo::VarianceReduction::*;

pub trait PathGenerator<T> {
    fn generate_paths(&self, n: usize) -> Vec<Path<T>>;
}

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
/// use payoffs::montecarlo::draw_normal;
///
/// let samples = draw_normal( 0.0, 1.0, 100 );
/// println!( "{:?}", samples );
/// ```
pub fn draw_normal(mean: f64, sd: f64, n: usize) -> Vec<f64> {
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
/// use payoffs::montecarlo::draw_normal;
/// use payoffs::montecarlo::gbm_samples;
///
/// let t = 0.25;
/// let n = 100;
/// let normal_samples = draw_normal( 0.0, (t/n as f64).sqrt(), n);
/// let samples = gbm_samples(100.0, 0.02, 0.0, 0.4, 0.25, &normal_samples);
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
/// use payoffs::montecarlo::draw_normal;
/// use payoffs::montecarlo::make_antithetic;
///
/// let normal_samples = draw_normal( 0.0, 1.0, 100 );
/// let anti = make_antithetic( &normal_samples );
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
/// use payoffs::montecarlo::gbm_path;
/// use payoffs::montecarlo::VarianceReduction::*;
///
/// let single_path = gbm_path(100.0, 0.02, 0.0, 0.4, 0.25, 100, None);
/// let pair_of_paths = gbm_path(100.0, 0.02, 0.0, 0.4, 0.25, 100, ATV);
/// ```
pub fn gbm_path( s0: f64, r: f64, q: f64, v: f64, t: f64, n: usize,
    vr: VarianceReduction ) -> Path<f64> {
    let sd = (t/n as f64).sqrt();
    let normal_samples = draw_normal(0.0, sd, n);
    let path = gbm_samples(s0, r, q, v, t, &normal_samples);

    match vr {
        None => Path::Ordinary(path),
        ATV => {
            let anti_samples = make_antithetic(&normal_samples);
            let anti = gbm_samples(s0, r, q, v, t, &anti_samples);
            Path::Antithetic((path, anti))
        }
    }
}

/// Generate a vector of Path assuming geometric brownian motion
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
/// * `num_paths` - number of paths to generate
///
/// #Example
/// ```
/// use payoffs::montecarlo::gbm_path;
/// use payoffs::montecarlo::VarianceReduction::*;
///
/// let paths = gbm_path(100.0, 0.02, 0.0, 0.4, 0.25, 100, None);
/// ```
pub fn gbm_paths( s0: f64, r: f64, q: f64, v: f64, t: f64, n: usize,
    vr: VarianceReduction, num_paths: usize ) -> Vec<Path<f64>> {

    (0..num_paths).map(|_| gbm_path(s0, r, q, v, t, n, vr)).collect()

}

#[derive(Clone, Copy, Debug)]
pub struct GbmPathGenerator {
    s0: f64,
    r: f64,
    q: f64,
    v: f64,
    t: f64,
    num_points: usize,
    vr: VarianceReduction,
}

impl PathGenerator<f64> for GbmPathGenerator {
    fn generate_paths(&self, n: usize) -> Vec<Path<f64>> {
        gbm_paths( self.s0, self.r, self.q, self.v, self.t, self.num_points,
            self.vr, n)
    }
}

impl GbmPathGenerator {
    pub fn new( s0: f64, r: f64, q: f64, v: f64, t: f64, n: usize,
        vr: VarianceReduction ) -> GbmPathGenerator {

        GbmPathGenerator { s0: s0, r: r, q: q, v: v, t: t, num_points: n,
            vr: vr }
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
            Call => intrinsic,
            Put => -intrinsic
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
/// use payoffs::montecarlo::*;
/// use payoffs::montecarlo::VarianceReduction::*;
/// use payoffs::option::OptionType::*;
/// use payoffs::option::BarrierInOut::*;
/// use payoffs::option::BarrierUpDown::*;
///
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
/// let (estimate, _, _) =
///     monte_carlo(GbmPathGenerator::new(s0, r, q, v, t, obs, ATV), m,
///         barrier_payoff(Up, Out, Call, r, t, barrier, k));
/// ```
pub fn barrier_payoff(barrier_updown: BarrierUpDown,
    barrier_inout: BarrierInOut, opt_type: OptionType,
    r: f64, t: f64, barrier: f64, strike: f64) -> PayoffFunc
{
    Box::new( move |path| {
        let intrinsic = path[path.len()-1] - strike;
        let discount = (-r*t).exp();
        let triggered = match barrier_updown {
            Up => path.iter().any( |&x| x > barrier ),
            Down => path.iter().any( |&x| x < barrier )
        };
        let payoff: f64 = match barrier_inout {
            In => if triggered { intrinsic } else { 0.0 },
            Out => if triggered { 0.0 } else {intrinsic}
        };
        ( match opt_type {
            Call => payoff.max(0.0),
            Put => (-payoff).max(0.0)
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
/// use payoffs::montecarlo::Path;
/// use payoffs::montecarlo::gbm_path;
/// use payoffs::montecarlo::VarianceReduction::*;
/// use payoffs::montecarlo::monte_carlo_0;
/// use payoffs::montecarlo::european_payoff;
/// use payoffs::option::OptionType::*;
///
/// let s0 = 100.0;
/// let r = 0.02;
/// let q = 0.0;
/// let v = 0.4;
/// let t = 0.25;
/// let k = 100.0;
/// let m = 5000000;
///
/// let paths: Vec<Path<f64>> = (0..m)
///     .map(|_| gbm_path( s0, r, q, v, t, 1, ATV))
///     .collect();
///
/// let (estimate, _, _) = monte_carlo_0(&paths,
///     european_payoff( Call, r, t, k));
///
/// println!("Estimate: {}", estimate);
/// ```
pub fn monte_carlo_0(paths: &Vec<Path<f64>>, f: PayoffFunc) -> (f64, f64, f64)
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

pub fn monte_carlo<G>( path_gen: G, n: usize, f: PayoffFunc) -> (f64, f64, f64)
where G: PathGenerator<f64>
{
    let paths = path_gen.generate_paths( n );
    monte_carlo_0( &paths, f )
}

#[cfg(test)]
mod test {
    extern crate time;
    extern crate num_cpus;
    use option::OptionType::*;
    use option::black_scholes;
    use option::BarrierUpDown::*;
    use option::BarrierInOut::*;
    use montecarlo::*;
    use montecarlo::VarianceReduction::*;
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

        let (estimate, _, err) =
            monte_carlo(GbmPathGenerator::new(s0, r, q, v, t, 1, ATV),
                m, european_payoff(Call, r, t, k));

        println!("{} secs", time::precise_time_s() - now);

        let bs = black_scholes(s0, r, q, v, t, Call, k);
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

        let (estimate, _, _) =
            monte_carlo(
                GbmPathGenerator::new(s0, r, q, v, t, 3, ATV),
                m, barrier_payoff(Up, Out, Call, r, t, barrier, k));

        println!("{} secs", time::precise_time_s() - now);

        // No easy way to test this, hard-coding a value of 3.30 for now which
        // is obtained from the F# deep dives book
        assert_eq!( true, equal_within(estimate, 3.30, 0.01) );
    }

    #[test]
    fn test_mc_up_out_call_mt() {
        use std::thread;
        let s0 = 100.0;
        let r = 0.02;
        let q = 0.0;
        let v = 0.4;
        let t = 0.25;
        let k = 100.0;
        let barrier = 125.0;
        let m = 10000000;
        let num_threads = 100;

        let now = time::precise_time_s();
        let mut threads = vec![];
        for _ in 0..num_threads {
            threads.push(thread::spawn( move || {
                let num_paths = m/num_threads;
                let (estimate, _, _) =
                    monte_carlo(
                        GbmPathGenerator::new( s0, r, q, v, t, 3, ATV ),
                        num_paths,
                        barrier_payoff(Up, Out, Call, r, t, barrier, k)
                    );
                (estimate * num_paths as f64, num_paths)
            }));
        }

        let (total, num_paths) = threads.into_iter().fold((0.0, 0), |acc, t| {
            let (running_total, running_num_paths) = acc;
            let (t, n) = t.join().unwrap();
            (running_total + t, running_num_paths + n) } );
        println!("{} secs", time::precise_time_s() - now);
        let estimate = total/num_paths as f64;
        println!("result: {}", estimate);
        println!("number of cores: {}", num_cpus::get());
        assert_eq!( true, equal_within(estimate, 3.30, 0.01) );
    }
}
