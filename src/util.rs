pub fn equal_within(x: f64, y:f64, e: f64) -> bool {
    (x-y).abs() < e
}
