extern crate probability;

use self::probability::prelude::*;

pub enum OptionType {
    Call,
    Put
}

pub enum ExerciseType {
    European,
    American
}

pub enum BarrierUpDown {
    Up,
    Down
}

pub enum BarrierInOut {
    In,
    Out
}

/// Calculate vanilla option price using the Black-Scholes equation
/// #Argument
/// * `s0` - initial stock price at time 0
/// * `r` - interest rate
/// * `q` - convenience yield (absorbs any cost or yield of holding the asset)
/// * `v` - volatility e.g. 0.4 is 40 vol points (per annum)
/// * `t` - number of years e.g. 0.25 is quarter of a year
/// * `opt_type` - option type
/// * `k` - strike price
///
/// #Example
/// ```
/// let call = payoffs::option::black_scholes( 100.0, 0.02, 0.0, 0.4, 0.25,
///     payoffs::option::OptionType::Call, 100.0);
/// println!( "price: {}", call);
/// ```
pub fn black_scholes(s0:f64, r:f64, q:f64, v: f64, t:f64, opt_type:OptionType,
    k:f64) -> f64 {
    let normal = Gaussian::new(0.0, 1.0);
    let f = s0 * ((r-q)*t).exp();
    let d1 = 1.0 / v / t.sqrt() * ( (s0/k).ln() + (r-q+0.5*v*v)*t);
    let d2 = d1 - v*t.sqrt();
    match opt_type {
        OptionType::Call => (-r*t).exp()*(f*normal.cdf(d1)-k*normal.cdf(d2)),
        OptionType::Put => (-r*t).exp()*(k*normal.cdf(-d2)-f*normal.cdf(-d1))
    }
}

#[cfg(test)]
mod test {
    extern crate time;
    use option::black_scholes;
    use option::OptionType;
    use util::equal_within;

    #[test]
    fn put_call_parity() {
        let s0 = 100.0;
        let r = 0.02;
        let q = 0.0;
        let v = 0.4;
        let t = 0.25;
        let k = s0;
        let call = black_scholes(s0, r, q, v, t, OptionType::Call, k);
        let put = black_scholes(s0, r, q, v, t, OptionType::Put, k);
        println!("Call: {}, Put: {}", call, put);
        assert_eq!(true, equal_within(put + s0, call + k * (-r*t as f64).exp(),
            0.0000001));
    }
}
