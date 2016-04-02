pub enum OptionType {
    Call,
    Put
}

pub enum ExerciseType {
    European,
    American
}

pub enum BarrierDirection {
    Up,
    Down
}

pub enum BarrierType {
    In,
    Out
}

fn phi(x: f64)->f64 {
    // constants
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    // Save the sign of x
    let sign = if x < 0.0 {-1} else {1};
    let x: f64 = x.abs()/(2.0 as f64).sqrt();

    // A&S formula 7.1.26
    let t = 1.0/(1.0 + p*x);
    let y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*(-x*x).exp();

    0.5*(1.0 + sign as f64*y)
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
    let f = s0 * ((r-q)*t).exp();
    let d1 = 1.0 / v / t.sqrt() * ( (s0/k).ln() + (r-q+0.5*v*v)*t);
    let d2 = d1 - v*t.sqrt();
    match opt_type {
        OptionType::Call => (-r*t).exp()*(f*phi(d1)-k*phi(d2)),
        OptionType::Put => (-r*t).exp()*(k*phi(-d2)-f*phi(-d1))
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
