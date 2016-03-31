pub mod option;
pub mod montecarlo;

#[cfg(test)]
mod test {
    use option::black_scholes;
    use option::OptionType;
    #[test]
    fn it_works() {
        let s0 = 100.0;
        let r = 0.01;
        let q = 0.0;
        let v = 0.4;
        let t = 0.25;
        let k = s0;
        let call = 0.0;
        let put = 0.0;
        let call = black_scholes(s0, r, q, v, t, OptionType::Call, k);
        let put = black_scholes(s0, r, q, v, t, OptionType::Put, k);
        assert_eq!(put + s0, call + k * (-r*t as f64).exp());
    }
}
