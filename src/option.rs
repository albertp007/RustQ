//
// RustQ - library for pricing financial derivatives written in Rust
// Copyright (c) 2016 by Albert Pang <albert.pang@me.com>
// All rights reserved.
//
// This file is a part of RustQ
//
// RustQ is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// RustQ is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
extern crate probability;

use self::probability::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExerciseType {
    European,
    American
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierUpDown {
    Up,
    Down
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
/// let call = rustq::option::black_scholes( 100.0, 0.02, 0.0, 0.4, 0.25,
///     rustq::option::OptionType::Call, 100.0);
/// println!( "price: {}", call);
/// ```
pub fn black_scholes(s0:f64, r:f64, q:f64, v: f64, t:f64, opt_type:OptionType,
    k:f64) -> f64 {
    use option::OptionType::*;
    let normal = Gaussian::new(0.0, 1.0);
    let f = s0 * ((r-q)*t).exp();
    let d1 = 1.0 / v / t.sqrt() * ( (s0/k).ln() + (r-q+0.5*v*v)*t);
    let d2 = d1 - v*t.sqrt();
    match opt_type {
        Call => (-r*t).exp()*(f*normal.cdf(d1)-k*normal.cdf(d2)),
        Put => (-r*t).exp()*(k*normal.cdf(-d2)-f*normal.cdf(-d1))
    }
}

#[cfg(test)]
mod test {
    extern crate time;
    use option::black_scholes;
    use option::OptionType::*;
    use util::equal_within;

    #[test]
    fn put_call_parity() {
        let s0 = 100.0;
        let r = 0.02;
        let q = 0.0;
        let v = 0.4;
        let t = 0.25;
        let k = s0;
        let call = black_scholes(s0, r, q, v, t, Call, k);
        let put = black_scholes(s0, r, q, v, t, Put, k);
        println!("Call: {}, Put: {}", call, put);
        assert_eq!(true, equal_within(put + s0, call + k * (-r*t as f64).exp(),
            0.0000001));
    }
}
