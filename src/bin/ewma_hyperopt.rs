extern crate csv;
extern crate rand;
extern crate wheedle;
extern crate wyrm;

use std::time::{Duration, Instant};

use rand::distributions::{IndependentSample, Range};
// use rand::{Rng, SeedableRng, XorShiftRng};

use wheedle::data::{user_based_split, CompressedInteractions, Interaction, Interactions};
use wheedle::models::ewma;
use wheedle::evaluation::mrr_score;

struct Result {
    mrr: f32,
    elapsed: Duration,
    hyperparameters: ewma::Hyperparameters,
}

fn load_movielens(path: &str) -> Interactions {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let interactions: Vec<Interaction> = reader.deserialize()
        .map(|x| x.unwrap())
        .collect();

    Interactions::from(interactions)
}

fn fit(
    train: &CompressedInteractions,
    hyper: ewma::Hyperparameters,
) -> ewma::ImplicitEWMAModel {
    let mut model = hyper.build();
    model.fit(train).unwrap();

    model
}

fn main() {
    let mut data = load_movielens("data.csv");
    let mut rng = rand::thread_rng();

    let mut results = Vec::new();

    for _ in 0..1000 {
        let hyper = ewma::Hyperparameters::random(data.num_items(), &mut rng);
        let (train, test) = user_based_split(&mut data, &mut rng, 0.2);

        let train = train.to_compressed();
        let test = test.to_compressed();

        let mut elapsed = Duration::new(0, 0);

        let mrr: f32 = (0..3)
            .map(|_| {
                let start = Instant::now();
                let model = fit(&train, hyper.clone());
                elapsed += start.elapsed();
                mrr_score(&model, &test).unwrap()
            })
            .sum::<f32>() / 3.0;

        println!(
            "MRR {} for hyperparams: {:#?} (elapsed {:#?})",
            mrr,
            &hyper,
            elapsed / 3
        );

        if mrr.is_normal() {
            results.push((mrr, hyper));
            results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        println!("Best result: {:#?}", results.last());
    }
}
