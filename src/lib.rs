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
extern crate test;

extern crate wyrm;

use ndarray::Axis;

use rayon::prelude::*;
use rand::distributions::{IndependentSample, Range};
use rand::{thread_rng, Rng, SeedableRng};
use std::sync::Arc;

use wyrm::{Arr, DataInput};

pub struct Interactions {
    num_users: usize,
    num_items: usize,
    user_ids: Vec<UserId>,
    item_ids: Vec<ItemId>,
    timestamps: Vec<Timestamp>,
}

impl Interactions {
    pub fn new<T>(interactions: &[T]) -> Result<Self, &'static str>
    where
        T: Interaction,
    {
        if interactions.len() == 0 {
            return Err("No interactions present.");
        }

        let num_users = interactions.iter().map(|x| x.user_id()).max().unwrap();
        let num_items = interactions.iter().map(|x| x.item_id()).max().unwrap();

        let user_ids = interactions.iter().map(|x| x.user_id()).collect();
        let item_ids = interactions.iter().map(|x| x.item_id()).collect();
        let timestamps = interactions.iter().map(|x| x.timestamp()).collect();

        Ok(Interactions {
            num_users: num_users,
            num_items: num_items,
            user_ids: user_ids,
            item_ids: item_ids,
            timestamps: timestamps,
        })
    }

    pub fn len(&self) -> usize {
        self.user_ids.len()
    }

    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        let mut i = self.len();

        while i >= 2 {
            i -= 1;
            let j = rng.gen_range(0, i + 1);

            self.user_ids.swap(i, j);
            self.item_ids.swap(i, j);
            self.timestamps.swap(i, j);
        }
    }

    pub fn split_at(&self, idx: usize) -> (Self, Self) {
        let head = Interactions {
            num_users: self.num_users,
            num_items: self.num_items,
            user_ids: self.user_ids[..idx].to_owned(),
            item_ids: self.item_ids[..idx].to_owned(),
            timestamps: self.timestamps[..idx].to_owned(),
        };
        let tail = Interactions {
            num_users: self.num_users,
            num_items: self.num_items,
            user_ids: self.user_ids[idx..].to_owned(),
            item_ids: self.item_ids[idx..].to_owned(),
            timestamps: self.timestamps[idx..].to_owned(),
        };

        (head, tail)
    }
}

impl<'a> IntoIterator for &'a Interactions {
    type Item = UnweightedInteraction;
    type IntoIter = InteractionsIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        InteractionsIterator {
            interactions: self,
            idx: 0
        }
    }
        
}

pub struct InteractionsIterator<'a> {
    interactions: &'a Interactions,
    idx: usize,
}

impl<'a> Iterator for InteractionsIterator<'a> {
    type Item = UnweightedInteraction;
    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.idx >= self.interactions.len() {
            None
        } else {
            Some(UnweightedInteraction::new(
                self.interactions.user_ids[self.idx],
                self.interactions.item_ids[self.idx],
                self.interactions.timestamps[self.idx],
            ))
        };

        self.idx += 1;

        value
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.interactions.len(), Some(self.interactions.len()))
    }
}

pub type UserId = usize;
pub type ItemId = usize;
pub type Timestamp = usize;

pub struct InteractionMatrix {
    num_users: usize,
    num_items: usize,
    rows: Vec<Vec<usize>>,
}

impl InteractionMatrix {
    pub fn new(num_users: usize, num_items: usize) -> Self {
        InteractionMatrix {
            num_users: num_users,
            num_items: num_items,
            rows: vec![Vec::new(); num_users],
        }
    }
    pub fn rows(&self) -> &[Vec<ItemId>] {
        &self.rows
    }
    pub fn from_interactions<T: Interaction>(
        num_users: usize,
        num_items: usize,
        interactions: &[T],
    ) -> Self {
        let mut mat = InteractionMatrix {
            num_users: num_users,
            num_items: num_items,
            rows: vec![Vec::new(); num_users],
        };

        for elem in interactions {
            mat.add(elem.user_id(), elem.item_id());
        }

        mat
    }
    pub fn add(&mut self, user_id: UserId, item_id: ItemId) {
        if let Err(idx) = self.rows[user_id].binary_search(&item_id) {
            self.rows[user_id].insert(idx, item_id);
        }
    }
    pub fn get(&self, user_id: UserId) -> &[ItemId] {
        &self.rows[user_id]
    }
}

pub fn mrr_score(
    model: &ImplicitFactorizationModel,
    test: &InteractionMatrix,
    train: &InteractionMatrix,
) -> f32 {
    let mrrs: Vec<f32> = test.rows()
        .par_iter()
        .zip(train.rows())
        .enumerate()
        .filter_map(|(user_id, (test_row, train_row))| {
            if test_row.len() == 0 {
                return None;
            }

            let mut predictions = model.predict(user_id).unwrap();

            for &train_idx in train_row.iter() {
                predictions[train_idx] = std::f32::MIN;
            }

            let test_scores: Vec<f32> = test_row.iter().map(|&idx| predictions[idx]).collect();
            let mut ranks: Vec<usize> = vec![0; test_row.len()];

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

    mrrs.iter().sum::<f32>() / mrrs.len() as f32
}

fn get_dimensions<T: Interaction>(data: &[T]) -> (usize, usize) {
    let max_user = data.iter().map(|x| x.user_id()).max().unwrap() + 1;
    let max_item = data.iter().map(|x| x.item_id()).max().unwrap() + 1;

    (max_user, max_item)
}

pub trait Interaction: Sync + Clone {
    fn user_id(&self) -> UserId;
    fn item_id(&self) -> ItemId;
    fn timestamp(&self) -> Timestamp;
    fn weight(&self) -> f32;
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct UnweightedInteraction {
    user_id: UserId,
    item_id: ItemId,
    timestamp: Timestamp,
}

impl UnweightedInteraction {
    pub fn new(user_id: UserId, item_id: ItemId, timestamp: Timestamp) -> Self {
        UnweightedInteraction {
            user_id,
            item_id,
            timestamp,
        }
    }
}

impl<'a> Interaction for UnweightedInteraction {
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

pub fn train_test_split<T: Interaction, R: Rng>(
    interactions: &mut Interactions,
    rng: &mut R,
    test_fraction: f32,
) -> (Interactions, Interactions) {
    interactions.shuffle(rng);

    interactions.split_at((test_fraction * interactions.len() as f32) as usize)
}

fn embedding_init(rows: usize, cols: usize) -> wyrm::Arr {
    Arr::zeros((rows, cols)).map(|_| rand::random::<f32>() / (cols as f32).sqrt())
}

#[derive(Builder)]
pub struct Hyperparameters {
    #[builder(default = "16")] latent_dim: usize,
    #[builder(default = "10")] minibatch_size: usize,
    #[builder(default = "0.01")] learning_rate: f32,
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

    pub fn fit(
        &mut self,
        interactions: &Interactions,
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
        let chunk_size = interactions.len() / num_partitions;

        let losses: Vec<f32> = (0..rayon::current_num_threads())
            .into_par_iter()
            .map(|partition_idx| {
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

                let mut optimizer = wyrm::SGD::new(
                    self.hyper.learning_rate,
                    vec![
                        user_embeddings.clone(),
                        item_embeddings.clone(),
                        item_biases.clone(),
                    ],
                );

                let mut batch_negatives = vec![0; minibatch_size];

                let mut rng = rand::XorShiftRng::from_seed(thread_rng().gen());
                let start = partition_idx * chunk_size;
                let stop = start + chunk_size;

                let mut loss_value = 0.0;

                for _ in 0..num_epochs {
                    for (user_ids, item_ids) in izip!(
                        interactions.user_ids.chunks(minibatch_size),
                        interactions.item_ids.chunks(minibatch_size)
                    ) {
                        if user_ids.len() < minibatch_size {
                            break;
                        }

                        for negative in batch_negatives.iter_mut() {
                            *negative = negative_item_range.ind_sample(&mut rng);
                        }

                        user_idx.set_value(user_ids);
                        positive_item_idx.set_value(item_ids);
                        negative_item_idx.set_value(batch_negatives.as_slice());

                        loss.forward();
                        loss.backward(1.0);

                        loss_value += loss.value().scalar_sum();

                        optimizer.step();
                        loss.zero_gradient();
                    }
                }

                loss_value / (num_epochs * (stop - start)) as f32
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
        let interactions: Vec<UnweightedInteraction> =
            reader.deserialize().map(|x| x.unwrap()).collect();

        Interactions::new(&interactions).unwrap()
    }

    #[test]
    fn it_works() {
        let mut data = load_movielens("data.csv");

        let (train, test) =
            train_test_split(&mut data, &mut rand::XorShiftRng::new_unseeded(), 0.2);

        println!("Train: {}, test: {}", train.len(), test.len());

        let hyper = HyperparametersBuilder::default()
            .learning_rate(0.1)
            .latent_dim(32)
            .build()
            .unwrap();

        let num_epochs = 50;

        let mut model = ImplicitFactorizationModel::new(hyper);
        println!("Loss: {}", model.fit(&train, num_epochs).unwrap());

        let train_mat = InteractionMatrix::from_interactions(
            model.num_users().unwrap(),
            model.num_items().unwrap(),
            &train,
        );
        let test_mat = InteractionMatrix::from_interactions(
            model.num_users().unwrap(),
            model.num_items().unwrap(),
            &test,
        );

        let mrr = mrr_score(&model, &test_mat, &train_mat);

        println!("MRR {}", mrr);

        assert!(mrr > 0.09);
    }

    #[bench]
    fn bench_movielens(b: &mut Bencher) {
        let data = load_movielens("data.csv");
        let num_epochs = 2;

        let mut model = ImplicitFactorizationModel::default();

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
