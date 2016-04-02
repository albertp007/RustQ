extern crate rand;

use self::rand::distributions::normal::Normal;
use self::rand::distributions::IndependentSample;
use option::OptionType;

pub enum VarianceReduction {
    None,
    ATV
}

pub enum Path<T> {
    Ordinary(Vec<T>),
    Antithetic((Vec<T>, Vec<T>))
}

pub fn draw_normal(mean: f64, sd: f64, n: u32) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(mean, sd);
    (0..n).map( |_| {
        normal.ind_sample(&mut rng)
    }).collect()
}

pub fn gbm_samples(s0: f64, r: f64, q: f64, v: f64, t: f64,
    normal_samples: &Vec<f64>) -> Vec<f64>
{
    let n = normal_samples.len();
    let dt = t/n as f64;
    let drift = (r - q - v * v/2.0) * dt;
    // println!("drift: {}", drift);
    normal_samples.iter()
        .map( |dwt| { drift + v * dwt })
        .scan(0.0, |acc, x| { Some(*acc + x) })
        .map( |x| { s0 * x.exp() })
        .collect()
}

fn make_antithetic(orig: &Vec<f64>) -> Vec<f64> {
    orig.iter().map( |x| { -x }).collect()
}

pub fn gbm_path( s0: f64, r: f64, q: f64, v: f64, t: f64, n: u32,
    vr: VarianceReduction ) -> Path<f64> {
    let sd = (t/n as f64).sqrt();
    let normal_samples = draw_normal(0.0, sd, n);
    let path = gbm_samples(s0, r, q, v, t, &normal_samples);

    // println!("{:?}", path);
    match vr {
        VarianceReduction::None => Path::Ordinary(path),
        VarianceReduction::ATV => {
            let anti_samples = make_antithetic(&normal_samples);
            let anti = gbm_samples(s0, r, q, v, t, &anti_samples);
            Path::Antithetic((path, anti))
        }
    }
}

pub fn monte_carlo(paths: &Vec<Path<f64>>, f: Box<Fn(&Vec<f64>)->f64>)
    -> (f64, f64, f64)
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
    println!("n: {}", n);
    let estimate = sum/n as f64;
    let var = sq_sum/(n-1) as f64 - estimate*estimate;
    let stderr = (var/n as f64).sqrt();
    (estimate, var, stderr)
}

pub fn european_payoff(opt_type: OptionType, r: f64, t: f64, strike: f64)
    -> Box<Fn(&Vec<f64>)->f64>
{
    Box::new( move |path: &Vec<f64>| {
        let intrinsic: f64 = path[path.len()-1] - strike;
        let intrinsic = match opt_type {
            OptionType::Call => intrinsic,
            OptionType::Put => -intrinsic
        };
        let payoff = intrinsic.max(0.0);
        if payoff > 0.0 { (-r*t).exp() * payoff } else { 0.0 }
    } )
}

#[cfg(test)]
mod test {
    extern crate time;
    use option::OptionType;
    use option::black_scholes;
    use montecarlo::monte_carlo;
    use montecarlo::VarianceReduction;
    use montecarlo::gbm_path;
    use montecarlo::gbm_samples;
    use montecarlo::Path;
    use montecarlo::draw_normal;
    use montecarlo::european_payoff;
    use montecarlo::make_antithetic;
    use util::equal_within;
    use std::cmp;

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

        let (estimate, var, err) =
            monte_carlo(&paths, european_payoff(OptionType::Call, r, t, k));

        println!("{} secs", time::precise_time_s() - now);

        let bs = black_scholes(s0, r, q, v, t, OptionType::Call, s0);
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
}
