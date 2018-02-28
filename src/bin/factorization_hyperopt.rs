extern crate csv;
extern crate rand;
extern crate wheedle;
extern crate wyrm;

use std::time::{Duration, Instant};

use rand::distributions::{IndependentSample, Range};
// use rand::{Rng, SeedableRng, XorShiftRng};

use wheedle::data::{user_based_split, Interaction, Interactions, TripletInteractions};
use wheedle::models::factorization;
use wheedle::evaluation::mrr_score;

struct Result {
    mrr: f32,
    elapsed: Duration,
    num_epochs: usize,
    hyperparameters: factorization::Hyperparameters,
}

fn load_movielens(path: &str) -> Interactions {
    let mut reader = csv::Reader::from_path(path).unwrap();
    let interactions: Vec<Interaction> = reader.deserialize().map(|x| x.unwrap()).collect();

    Interactions::from(interactions)
}

fn fit(
    train: &TripletInteractions,
    hyper: factorization::Hyperparameters,
    num_epochs: usize,
) -> factorization::ImplicitFactorizationModel {
    let mut model = factorization::ImplicitFactorizationModel::new(hyper);
    model.fit(train, num_epochs).unwrap();

    model
}

fn main() {
    let mut data = load_movielens("data.csv");
    let mut rng = rand::thread_rng();

    let epochs = Range::new(10, 100);
    let fold_in_epochs = Range::new(10, 100);
    let learning_rates = Range::new(1.0, 4.0);
    let latent_dims = Range::new(16, 128);
    let minibatch_sizes = Range::new(1, 128);

    let mut results = Vec::new();

    for _ in 0..100 {
        let hyper = factorization::HyperparametersBuilder::default()
            .learning_rate((2.0_f32).powf(-learning_rates.ind_sample(&mut rng)))
            .fold_in_epochs(fold_in_epochs.ind_sample(&mut rng))
            .latent_dim(latent_dims.ind_sample(&mut rng))
            .minibatch_size(minibatch_sizes.ind_sample(&mut rng))
            .build()
            .unwrap();

        let num_epochs = epochs.ind_sample(&mut rng);

        let (train, test) = user_based_split(&mut data, &mut rng, 0.2);

        let mut elapsed = Duration::new(0, 0);

        let mrr: f32 = (0..3)
            .map(|_| {
                let start = Instant::now();
                let model = fit(&train.to_triplet(), hyper.clone(), num_epochs);
                elapsed += start.elapsed();
                mrr_score(&model, &test.to_compressed()).unwrap()
            })
            .sum::<f32>() / 3.0;

        println!(
            "MRR {} for hyperparams: {:#?} and epochs {} (elapsed {:#?})",
            mrr,
            &hyper,
            num_epochs,
            elapsed / 3
        );

        if !mrr.is_nan() {
            results.push((mrr, num_epochs, hyper));
            results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        println!("Best result: {:#?}", results.last());
    }
}
