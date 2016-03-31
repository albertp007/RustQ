extern crate rand;

use self::rand::distributions::normal::Normal;
use self::rand::distributions::IndependentSample;

pub enum VarianceReduction {
    None,
    ATV
}

pub enum Path<T> {
    Ordinary(Vec<T>),
    Antithetic((Vec<T>, Vec<T>))
}

fn draw_normal(mean: f64, sd: f64, n: u32) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(mean, sd);
    (0..n).map( |_| {
        normal.ind_sample(&mut rng)
    }).collect()
}

fn gbm_paths(s0: f64, r: f64, q: f64, v: f64, t: f64, randVec: &Vec<f64>)
    -> Vec<f64> {

    let n = randVec.len();
    let dt = t/n as f64;
    let drift = (r - v * v/2.0) * dt;
    randVec.iter()
        .map( |dwt| { drift + v * dwt })
        .scan(0.0, |acc, x| { Some(*acc + x) })
        .map( |x| { s0 * x.exp() })
        .collect()
}
