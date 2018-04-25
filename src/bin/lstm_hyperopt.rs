extern crate csv;
extern crate rand;
extern crate wheedle;
extern crate wyrm;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate serde_json;

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, Read};

use std::time::{Duration, Instant};

use rand::distributions::{IndependentSample, Range};
// use rand::{Rng, SeedableRng, XorShiftRng};

use wheedle::data::{user_based_split, CompressedInteractions, Interaction, Interactions};
use wheedle::evaluation::mrr_score;
use wheedle::models::lstm;

#[derive(Deserialize, Serialize)]
struct GoodbooksInteraction {
    user_id: usize,
    book_id: usize,
    rating: usize,
}

fn load_goodbooks(path: &str) -> Interactions {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let interactions: Vec<Interaction> = reader
        .deserialize::<GoodbooksInteraction>()
        .map(|x| x.unwrap())
        .enumerate()
        .map(|(i, x)| Interaction::new(x.user_id, x.book_id, i))
        //.take(100_000)
        .collect();

    Interactions::from(interactions)
}

#[derive(Debug, Serialize, Deserialize)]
struct Result {
    test_mrr: f32,
    train_mrr: f32,
    elapsed: Duration,
    hyperparameters: lstm::Hyperparameters,
}

fn load_movielens(path: &str) -> Interactions {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let interactions: Vec<Interaction> = reader.deserialize().map(|x| x.unwrap()).collect();

    Interactions::from(interactions)
}

fn fit(train: &CompressedInteractions, hyper: lstm::Hyperparameters) -> lstm::ImplicitLSTMModel {
    let mut model = hyper.build();
    model.fit(train).unwrap();

    model
}

fn main() {
    let mut data = load_goodbooks("ratings.csv");
    // let mut data = load_movielens("data.csv");
    let mut rng = rand::thread_rng();

    let (train, test) = user_based_split(&mut data, &mut rng, 0.2);

    let train = train.to_compressed();
    let test = test.to_compressed();

    for _ in 0..1000 {
        let mut results: Vec<Result> = File::open("lstm_results.json")
            .map(|file| serde_json::from_reader(&file).unwrap())
            .unwrap_or(Vec::new());

        let hyper = lstm::Hyperparameters::random(data.num_items(), &mut rng);
        let start = Instant::now();
        let model = fit(&train, hyper.clone());
        let result = Result {
            train_mrr: mrr_score(&model, &train).unwrap(),
            test_mrr: mrr_score(&model, &test).unwrap(),
            elapsed: start.elapsed(),
            hyperparameters: hyper,
        };

        println!("{:#?}", result);

        if !result.test_mrr.is_nan() {
            results.push(result);
            results.sort_by(|a, b| a.test_mrr.partial_cmp(&b.test_mrr).unwrap());
        }

        println!("Best result: {:#?}", results.last());

        File::create("lstm_results.json")
            .map(|file| serde_json::to_writer_pretty(&file, &results).unwrap())
            .unwrap();
    }
}
