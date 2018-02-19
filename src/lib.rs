#![feature(test)]

#[macro_use]
extern crate serde_derive;

#[macro_use]
extern crate itertools;

#[macro_use]
extern crate derive_builder;

extern crate csv;
extern crate ndarray;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate siphasher;
extern crate test;
extern crate wyrm;

use std::cmp::Ordering;

use ndarray::Axis;

use rayon::prelude::*;
use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng, SeedableRng};
use std::sync::Arc;
use std::hash::Hasher;

use siphasher::sip::SipHasher;

use wyrm::{Arr, DataInput};

mod data;

pub struct Interactions {
    num_users: usize,
    num_items: usize,
    interactions: Vec<Interaction>,
}

impl Interactions {
    pub fn new(num_users: usize, num_items: usize) -> Self {
        Interactions {
            num_users: num_users,
            num_items: num_items,
            interactions: Vec::new(),
        }
    }

    pub fn data(&self) -> &[Interaction] {
        &self.interactions
    }

    pub fn len(&self) -> usize {
        self.interactions.len()
    }

    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        rng.shuffle(&mut self.interactions);
    }

    fn split_at(&self, idx: usize) -> (Self, Self) {
        let head = Interactions {
            num_users: self.num_users,
            num_items: self.num_items,
            interactions: self.interactions[..idx].to_owned(),
        };
        let tail = Interactions {
            num_users: self.num_users,
            num_items: self.num_items,
            interactions: self.interactions[idx..].to_owned(),
        };

        (head, tail)
    }

    fn split_by<F: Fn(&Interaction) -> bool>(&self, func: F) -> (Self, Self) {
        let head = Interactions {
            num_users: self.num_users,
            num_items: self.num_items,
            interactions: self.interactions
                .iter()
                .filter(|x| func(x))
                .cloned()
                .collect(),
        };
        let tail = Interactions {
            num_users: self.num_users,
            num_items: self.num_items,
            interactions: self.interactions
                .iter()
                .filter(|x| !func(x))
                .cloned()
                .collect(),
        };

        (head, tail)
    }

    pub fn to_triplet(&self) -> TripletInteractions {
        TripletInteractions::from(self)
    }

    pub fn to_compressed(&self) -> CompressedInteractions {
        CompressedInteractions::from(self)
    }
}

impl From<Vec<Interaction>> for Interactions {
    fn from(data: Vec<Interaction>) -> Interactions {
        let num_users = data.iter().map(|x| x.user_id()).max().unwrap() + 1;
        let num_items = data.iter().map(|x| x.item_id()).max().unwrap() + 1;

        Interactions {
            num_users: num_users,
            num_items: num_items,
            interactions: data,
        }
    }
}

pub type UserId = usize;
pub type ItemId = usize;
pub type Timestamp = usize;

pub struct CompressedInteractions {
    num_users: usize,
    num_items: usize,
    user_pointers: Vec<usize>,
    item_ids: Vec<ItemId>,
    timestamps: Vec<Timestamp>,
}

fn cmp_timestamp(x: &Interaction, y: &Interaction) -> Ordering {
    let uid_comparison = x.user_id().cmp(&y.user_id());

    if uid_comparison == Ordering::Equal {
        x.timestamp().cmp(&y.timestamp())
    } else {
        uid_comparison
    }
}

impl<'a> From<&'a Interactions> for CompressedInteractions {
    fn from(interactions: &Interactions) -> CompressedInteractions {
        let mut data = interactions.data().to_owned();

        data.sort_by(cmp_timestamp);

        let mut user_pointers = vec![0; interactions.num_users + 1];
        let mut item_ids = Vec::with_capacity(data.len());
        let mut timestamps = Vec::with_capacity(data.len());

        for datum in &data {
            item_ids.push(datum.item_id());
            timestamps.push(datum.timestamp());

            user_pointers[datum.user_id() + 1] += 1;
        }

        for idx in 1..user_pointers.len() {
            user_pointers[idx] += user_pointers[idx - 1];
        }

        CompressedInteractions {
            num_users: interactions.num_users,
            num_items: interactions.num_items,
            user_pointers: user_pointers,
            item_ids: item_ids,
            timestamps: timestamps,
        }
    }
}

impl CompressedInteractions {
    pub fn iter_users(&self) -> CompressedInteractionsUserIterator {
        CompressedInteractionsUserIterator {
            interactions: &self,
            idx: 0,
        }
    }

    pub fn get_user(&self, user_id: UserId) -> Option<CompressedInteractionsUser> {
        if user_id >= self.num_users {
            return None;
        }

        let start = self.user_pointers[user_id];
        let stop = self.user_pointers[user_id + 1];

        Some(CompressedInteractionsUser {
            user_id: user_id,
            item_ids: &self.item_ids[start..stop],
            timestamps: &self.timestamps[start..stop],
        })
    }
}

pub struct CompressedInteractionsUserIterator<'a> {
    interactions: &'a CompressedInteractions,
    idx: usize,
}

#[derive(Debug)]
pub struct CompressedInteractionsUser<'a> {
    pub user_id: UserId,
    pub item_ids: &'a [ItemId],
    pub timestamps: &'a [Timestamp],
}

impl<'a> Iterator for CompressedInteractionsUserIterator<'a> {
    type Item = CompressedInteractionsUser<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.idx >= self.interactions.num_users {
            None
        } else {
            let start = self.interactions.user_pointers[self.idx];
            let stop = self.interactions.user_pointers[self.idx + 1];

            Some(CompressedInteractionsUser {
                user_id: self.idx,
                item_ids: &self.interactions.item_ids[start..stop],
                timestamps: &self.interactions.timestamps[start..stop],
            })
        };

        self.idx += 1;

        value
    }
}

#[derive(Debug)]
pub struct TripletInteractions {
    num_users: usize,
    num_items: usize,
    user_ids: Vec<UserId>,
    item_ids: Vec<ItemId>,
    timestamps: Vec<Timestamp>,
}

impl TripletInteractions {
    pub fn len(&self) -> usize {
        self.user_ids.len()
    }
    pub fn iter_minibatch(&self, minibatch_size: usize) -> TripletMinibatchIterator {
        TripletMinibatchIterator {
            interactions: &self,
            idx: 0,
            stop_idx: self.len(),
            minibatch_size: minibatch_size,
        }
    }
    pub fn iter_minibatch_partitioned(
        &self,
        minibatch_size: usize,
        num_partitions: usize,
    ) -> Vec<TripletMinibatchIterator> {
        let iterator = self.iter_minibatch(minibatch_size);
        let chunk_size = self.len() / num_partitions;

        (0..num_partitions)
            .map(|x| iterator.slice(x * chunk_size, (x + 1) * chunk_size))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct TripletMinibatchIterator<'a> {
    interactions: &'a TripletInteractions,
    idx: usize,
    stop_idx: usize,
    minibatch_size: usize,
}

impl<'a> TripletMinibatchIterator<'a> {
    pub fn slice(&self, start: usize, stop: usize) -> TripletMinibatchIterator<'a> {
        TripletMinibatchIterator {
            interactions: &self.interactions,
            idx: start,
            stop_idx: stop,
            minibatch_size: self.minibatch_size,
        }
    }
}

#[derive(Debug)]
pub struct TripletMinibatch<'a> {
    pub user_ids: &'a [UserId],
    pub item_ids: &'a [ItemId],
    pub timestamps: &'a [Timestamp],
}

impl<'a> TripletMinibatch<'a> {
    pub fn len(&self) -> usize {
        self.user_ids.len()
    }
}

impl<'a> Iterator for TripletMinibatchIterator<'a> {
    type Item = TripletMinibatch<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.idx + self.minibatch_size > self.stop_idx {
            None
        } else {
            let start = self.idx;
            let stop = self.idx + self.minibatch_size;

            Some(TripletMinibatch {
                user_ids: &self.interactions.user_ids[start..stop],
                item_ids: &self.interactions.item_ids[start..stop],
                timestamps: &self.interactions.timestamps[start..stop],
            })
        };

        self.idx += self.minibatch_size;

        value
    }
}

impl<'a> From<&'a Interactions> for TripletInteractions {
    fn from(interactions: &'a Interactions) -> Self {
        let user_ids = interactions.data().iter().map(|x| x.user_id()).collect();
        let item_ids = interactions.data().iter().map(|x| x.item_id()).collect();
        let timestamps = interactions.data().iter().map(|x| x.timestamp()).collect();

        TripletInteractions {
            num_users: interactions.num_users,
            num_items: interactions.num_items,
            user_ids: user_ids,
            item_ids: item_ids,
            timestamps: timestamps,
        }
    }
}

pub fn mrr_score(
    model: &ImplicitFactorizationModel,
    test: &CompressedInteractions,
    train: &CompressedInteractions,
) -> Result<f32, &'static str> {
    if test.num_users != train.num_users || test.num_items != train.num_items {
        return Err("Number of users or items in train and test sets don't match");
    }

    let mrrs: Vec<f32> = test.iter_users()
        .zip(train.iter_users())
        .filter_map(|(test_user, train_user)| {
            if test_user.item_ids.len() == 0 {
                return None;
            }

            let mut predictions = model.predict(test_user.user_id).unwrap();

            for &train_item_id in train_user.item_ids.iter() {
                predictions[train_item_id] = std::f32::MIN;
            }

            let test_scores: Vec<f32> = test_user
                .item_ids
                .iter()
                .map(|&idx| predictions[idx])
                .collect();
            let mut ranks: Vec<usize> = vec![0; test_user.item_ids.len()];

            for &prediction in &predictions {
                for (rank, &score) in ranks.iter_mut().zip(&test_scores) {
                    if prediction >= score {
                        *rank += 1;
                    }
                }
            }

            Some(ranks.iter().map(|&x| 1.0 / x as f32).sum::<f32>() / ranks.len() as f32)
        })
        .collect();

    Ok(mrrs.iter().sum::<f32>() / mrrs.len() as f32)
}

pub fn fold_in_mrr_score(
    model: &ImplicitFactorizationModel,
    test: &CompressedInteractions,
) -> Result<f32, &'static str> {
    let mrrs: Vec<f32> = test.iter_users()
        .filter_map(|test_user| {
            if test_user.item_ids.len() == 0 {
                return None;
            }

            let train_items = &test_user.item_ids[..test_user.item_ids.len() - 1];
            let test_item = *test_user.item_ids.last().unwrap();

            let user_embedding = model.fold_in_user(train_items).unwrap();

            let mut predictions = model.predict_user(&user_embedding).unwrap();

            for &train_item_id in train_items {
                predictions[train_item_id] = std::f32::MIN;
            }

            let test_scores = vec![predictions[test_item]];
            let mut ranks: Vec<usize> = vec![0];

            for &prediction in &predictions {
                for (rank, &score) in ranks.iter_mut().zip(&test_scores) {
                    if prediction >= score {
                        *rank += 1;
                    }
                }
            }

            Some(ranks.iter().map(|&x| 1.0 / x as f32).sum::<f32>() / ranks.len() as f32)
        })
        .collect();

    Ok(mrrs.iter().sum::<f32>() / mrrs.len() as f32)
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Interaction {
    user_id: UserId,
    item_id: ItemId,
    timestamp: Timestamp,
}

impl Interaction {
    pub fn new(user_id: UserId, item_id: ItemId, timestamp: Timestamp) -> Self {
        Interaction {
            user_id,
            item_id,
            timestamp,
        }
    }
}

impl Interaction {
    fn user_id(&self) -> UserId {
        self.user_id
    }
    fn item_id(&self) -> ItemId {
        self.item_id
    }
    fn weight(&self) -> f32 {
        1.0
    }
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }
}

pub fn train_test_split<R: Rng>(
    interactions: &mut Interactions,
    rng: &mut R,
    test_fraction: f32,
) -> (Interactions, Interactions) {
    interactions.shuffle(rng);

    let (test, train) = interactions.split_at((test_fraction * interactions.len() as f32) as usize);

    (train, test)
}

pub fn user_based_split<R: Rng>(
    interactions: &mut Interactions,
    rng: &mut R,
    test_fraction: f32,
) -> (Interactions, Interactions) {
    let denominator = 100000;
    let train_cutoff = (test_fraction * denominator as f32) as u64;

    let range = Range::new(0, std::u64::MAX);
    let (key_0, key_1) = (range.ind_sample(rng), range.ind_sample(rng));

    let is_train = |x: &Interaction| {
        let mut hasher = SipHasher::new_with_keys(key_0, key_1);
        let user_id = x.user_id();
        hasher.write_usize(user_id);
        hasher.finish() % denominator > train_cutoff
    };

    interactions.split_by(is_train)
}

fn embedding_init(rows: usize, cols: usize) -> wyrm::Arr {
    Arr::zeros((rows, cols)).map(|_| rand::random::<f32>() / (cols as f32).sqrt())
}

#[derive(Builder)]
pub struct Hyperparameters {
    #[builder(default = "16")]
    latent_dim: usize,
    #[builder(default = "10")]
    minibatch_size: usize,
    #[builder(default = "0.01")]
    learning_rate: f32,
    #[builder(default = "50")]
    fold_in_epochs: usize,
}

struct ModelData {
    num_users: usize,
    num_items: usize,
    user_embedding: Arc<wyrm::HogwildParameter>,
    item_embedding: Arc<wyrm::HogwildParameter>,
    item_biases: Arc<wyrm::HogwildParameter>,
}

pub struct ImplicitFactorizationModel {
    hyper: Hyperparameters,
    model: Option<ModelData>,
}

impl std::default::Default for ImplicitFactorizationModel {
    fn default() -> Self {
        ImplicitFactorizationModel {
            hyper: HyperparametersBuilder::default().build().unwrap(),
            model: None,
        }
    }
}

impl ImplicitFactorizationModel {
    pub fn new(hyper: Hyperparameters) -> Self {
        ImplicitFactorizationModel {
            hyper: hyper,
            model: None,
        }
    }

    pub fn num_users(&self) -> Option<usize> {
        match &self.model {
            &Some(ref model) => Some(model.num_users),
            _ => None,
        }
    }

    pub fn num_items(&self) -> Option<usize> {
        match &self.model {
            &Some(ref model) => Some(model.num_items),
            _ => None,
        }
    }

    pub fn predict(&self, user_id: UserId) -> Result<Vec<f32>, &'static str> {
        if let Some(ref model) = self.model {
            let user_embeddings = &model.user_embedding;
            let item_embeddings = &model.item_embedding;
            let item_biases = &model.item_biases;

            let user_embeddings = user_embeddings.value.borrow();
            let user_vector = user_embeddings.subview(Axis(0), user_id);
            let user_vector_slice = user_vector.as_slice().unwrap();

            let predictions: Vec<f32> = item_embeddings
                .value
                .borrow()
                .genrows()
                .into_iter()
                .zip(item_biases.value.borrow().as_slice().unwrap())
                .map(|(item_embedding, item_bias)| {
                    item_bias
                        + wyrm::simd_dot(user_vector_slice, item_embedding.as_slice().unwrap())
                })
                .collect();

            Ok(predictions)
        } else {
            Err("Model must be fitted first.")
        }
    }

    pub fn predict_user(&self, user_embedding: &[f32]) -> Result<Vec<f32>, &'static str> {
        if let Some(ref model) = self.model {
            let item_embeddings = &model.item_embedding;
            let item_biases = &model.item_biases;

            let predictions: Vec<f32> = item_embeddings
                .value
                .borrow()
                .genrows()
                .into_iter()
                .zip(item_biases.value.borrow().as_slice().unwrap())
                .map(|(item_embedding, item_bias)| {
                    item_bias + wyrm::simd_dot(user_embedding, item_embedding.as_slice().unwrap())
                })
                .collect();

            Ok(predictions)
        } else {
            Err("Model must be fitted first.")
        }
    }

    fn build_model(&self, num_users: usize, num_items: usize, latent_dim: usize) -> ModelData {
        let user_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            num_users,
            latent_dim,
        )));
        let item_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            num_items,
            latent_dim,
        )));

        let item_biases = Arc::new(wyrm::HogwildParameter::new(embedding_init(num_items, 1)));

        ModelData {
            num_users: num_users,
            num_items: num_items,
            user_embedding: user_embeddings,
            item_embedding: item_embeddings,
            item_biases: item_biases,
        }
    }

    pub fn fold_in_user(&self, interactions: &[ItemId]) -> Result<Vec<f32>, &'static str> {
        if self.model.is_none() {
            return Err("Model must be fitted before trying to fold-in users.");
        }

        let negative_item_range = Range::new(0, self.model.as_ref().unwrap().num_items);
        let minibatch_size = 1;

        let user_vector = wyrm::ParameterNode::new(embedding_init(1, self.hyper.latent_dim));

        let item_embeddings =
            wyrm::ParameterNode::shared(self.model.as_ref().unwrap().item_embedding.clone());
        let item_biases =
            wyrm::ParameterNode::shared(self.model.as_ref().unwrap().item_biases.clone());

        let positive_item_idx = wyrm::IndexInputNode::new(&vec![0; minibatch_size]);
        let negative_item_idx = wyrm::IndexInputNode::new(&vec![0; minibatch_size]);

        let positive_item_vector = item_embeddings.index(&positive_item_idx);
        let negative_item_vector = item_embeddings.index(&negative_item_idx);
        let positive_item_bias = item_biases.index(&positive_item_idx);
        let negative_item_bias = item_biases.index(&negative_item_idx);

        let positive_prediction =
            user_vector.vector_dot(&positive_item_vector) + positive_item_bias;
        let negative_prediciton =
            user_vector.vector_dot(&negative_item_vector) + negative_item_bias;

        let score_diff = positive_prediction - negative_prediciton;
        let mut loss = -score_diff.sigmoid();

        let mut optimizer = wyrm::Adagrad::new(self.hyper.learning_rate, vec![user_vector.clone()]);

        let mut rng = rand::XorShiftRng::from_seed(thread_rng().gen());

        for _ in 0..self.hyper.fold_in_epochs {
            for &item_id in interactions {
                positive_item_idx.set_value(item_id);
                negative_item_idx.set_value(negative_item_range.ind_sample(&mut rng));

                loss.forward();
                loss.backward(1.0);

                optimizer.step();
                loss.zero_gradient();
            }
        }

        let user_vec = user_vector.value();

        Ok(user_vec.as_slice().unwrap().to_owned())
    }

    pub fn fit(
        &mut self,
        interactions: &TripletInteractions,
        num_epochs: usize,
    ) -> Result<f32, &'static str> {
        let minibatch_size = self.hyper.minibatch_size;

        if self.model.is_none() {
            self.model = Some(self.build_model(
                interactions.num_users,
                interactions.num_items,
                self.hyper.latent_dim,
            ));
        }

        let negative_item_range = Range::new(0, interactions.num_items);

        let num_partitions = rayon::current_num_threads();

        let losses: Vec<f32> = interactions
            .iter_minibatch_partitioned(minibatch_size, num_partitions)
            .into_par_iter()
            .map(|data| {
                let user_embeddings = wyrm::ParameterNode::shared(
                    self.model.as_ref().unwrap().user_embedding.clone(),
                );
                let item_embeddings = wyrm::ParameterNode::shared(
                    self.model.as_ref().unwrap().item_embedding.clone(),
                );
                let item_biases =
                    wyrm::ParameterNode::shared(self.model.as_ref().unwrap().item_biases.clone());

                let user_idx = wyrm::IndexInputNode::new(&vec![0; minibatch_size]);
                let positive_item_idx = wyrm::IndexInputNode::new(&vec![0; minibatch_size]);
                let negative_item_idx = wyrm::IndexInputNode::new(&vec![0; minibatch_size]);

                let user_vector = user_embeddings.index(&user_idx);
                let positive_item_vector = item_embeddings.index(&positive_item_idx);
                let negative_item_vector = item_embeddings.index(&negative_item_idx);
                let positive_item_bias = item_biases.index(&positive_item_idx);
                let negative_item_bias = item_biases.index(&negative_item_idx);

                let positive_prediction =
                    user_vector.vector_dot(&positive_item_vector) + positive_item_bias;
                let negative_prediciton =
                    user_vector.vector_dot(&negative_item_vector) + negative_item_bias;

                let score_diff = positive_prediction - negative_prediciton;
                let mut loss = -score_diff.sigmoid();

                let mut optimizer = wyrm::Adagrad::new(self.hyper.learning_rate, loss.parameters());

                let mut batch_negatives = vec![0; minibatch_size];

                let mut rng = rand::XorShiftRng::from_seed(thread_rng().gen());
                let mut loss_value = 0.0;

                let mut num_observations = 0;

                for _ in 0..num_epochs {
                    for batch in data.clone() {
                        if batch.len() < minibatch_size {
                            break;
                        }

                        num_observations += batch.len();

                        for negative in batch_negatives.iter_mut() {
                            *negative = negative_item_range.ind_sample(&mut rng);
                        }

                        user_idx.set_value(batch.user_ids);
                        positive_item_idx.set_value(batch.item_ids);
                        negative_item_idx.set_value(batch_negatives.as_slice());

                        loss.forward();
                        loss.backward(1.0);

                        loss_value += loss.value().scalar_sum();

                        optimizer.step();
                        loss.zero_gradient();
                    }
                }

                loss_value / num_observations as f32
            })
            .collect();

        Ok(losses.into_iter().sum())
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use test::Bencher;

    fn load_movielens(path: &str) -> Interactions {
        let mut reader = csv::Reader::from_path(path).unwrap();
        let interactions: Vec<Interaction> = reader.deserialize().map(|x| x.unwrap()).collect();

        Interactions::from(interactions)
    }

    #[test]
    fn it_works() {
        let mut data = load_movielens("data.csv");

        let (train, test) =
            train_test_split(&mut data, &mut rand::XorShiftRng::new_unseeded(), 0.2);

        println!("Train: {}, test: {}", train.len(), test.len());

        let hyper = HyperparametersBuilder::default()
            .learning_rate(0.5)
            .latent_dim(32)
            .build()
            .unwrap();

        let num_epochs = 50;

        let mut model = ImplicitFactorizationModel::new(hyper);

        println!(
            "Loss: {}",
            model.fit(&train.to_triplet(), num_epochs).unwrap()
        );

        let train_mat = train.to_compressed();
        let test_mat = test.to_compressed();

        let mrr = mrr_score(&model, &test_mat, &train_mat).unwrap();

        println!("MRR {}", mrr);

        assert!(mrr > 0.09);
    }

    #[test]
    fn fold_in() {
        let mut data = load_movielens("data.csv");

        let (train, test) =
            user_based_split(&mut data, &mut rand::XorShiftRng::new_unseeded(), 0.2);

        println!("Train: {}, test: {}", train.len(), test.len());

        let hyper = HyperparametersBuilder::default()
            .learning_rate(0.5)
            .latent_dim(32)
            .build()
            .unwrap();

        let num_epochs = 50;

        let mut model = ImplicitFactorizationModel::new(hyper);

        println!(
            "Loss: {}",
            model.fit(&train.to_triplet(), num_epochs).unwrap()
        );

        let test_mat = test.to_compressed();

        let mrr = fold_in_mrr_score(&model, &test_mat).unwrap();

        println!("MRR {}", mrr);

        assert!(mrr > 0.09);
    }

    #[bench]
    fn bench_movielens(b: &mut Bencher) {
        let data = load_movielens("data.csv");
        let num_epochs = 2;

        let mut model = ImplicitFactorizationModel::default();

        let data = data.to_triplet();

        model.fit(&data, num_epochs).unwrap();

        b.iter(|| {
            model.fit(&data, num_epochs).unwrap();
        });
    }

    // #[bench]
    // fn bench_movielens_10m(b: &mut Bencher) {
    //     let data = load_movielens("/home/maciej/Downloads/data.csv");
    //     //let data = load_movielens("data.csv");
    //     let num_epochs = 1;

    //     let mut model = ImplicitFactorizationModel::default();
    //     println!("Num obs {}", data.len());

    //     model.fit(&data, 1).unwrap();

    //     let mut runs = 0;
    //     let mut elapsed = std::time::Duration::default();

    //     b.iter(|| {
    //         let start = std::time::Instant::now();
    //         println!("Loss: {}", model.fit(&data, num_epochs).unwrap());
    //         elapsed += start.elapsed();
    //         runs += 1;
    //         println!("Avg duration: {:#?}", elapsed / runs);
    //     });
    // }
}
