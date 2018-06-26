//! Model based on exponentially-weighted average (EWMA) of past embeddings.
//!
//! The model estimates three sets of parameters:
//!
//! - n-dimensional item embeddings
//! - item biases (capturing item popularity), and
//! - an n-dimensional `alpha` parameter, capturing the rate at which past interactions should be decayed.
//!
//! The representation of a user at time t is given by an n-dimensional vector u:
//! ```text
//! u_t = sigmoid(alpha) * i_{t-1} + (1.0 - sigmoid(alpha)) + i_t
//! ```
//! where `i_t` is the embedding of the item the user interacted with at time `t`.
use std::sync::Arc;

use rand;
use rand::distributions::{Distribution, Normal, Uniform};
use rand::{Rng, SeedableRng, XorShiftRng};
use rayon;

use ndarray::Axis;

use wyrm;
use wyrm::optim::Optimizers;
use wyrm::{Arr, BoxedNode, Variable};

use super::sequence_model::{fit_sequence_model, SequenceModel, SequenceModelParameters};
use super::{ImplicitUser, Loss, Optimizer, Parallelism};
use data::CompressedInteractions;
use {FittingError, ItemId, OnlineRankingModel, PredictionError};

fn embedding_init<T: Rng>(rows: usize, cols: usize, rng: &mut T) -> wyrm::Arr {
    let normal = Normal::new(0.0, 1.0 / cols as f64);
    Arr::zeros((rows, cols)).map(|_| normal.sample(rng) as f32)
}

fn dense_init<T: Rng>(rows: usize, cols: usize, rng: &mut T) -> wyrm::Arr {
    let normal = Normal::new(0.0, (2.0 / (rows + cols) as f64).sqrt());
    Arr::zeros((rows, cols)).map(|_| normal.sample(rng) as f32)
}

/// Hyperparameters describing the EWMA model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hyperparameters {
    num_items: usize,
    max_sequence_length: usize,
    item_embedding_dim: usize,
    learning_rate: f32,
    l2_penalty: f32,
    loss: Loss,
    optimizer: Optimizer,
    parallelism: Parallelism,
    rng: XorShiftRng,
    num_threads: usize,
    num_epochs: usize,
}

impl Hyperparameters {
    /// Build new hyperparameters.
    pub fn new(num_items: usize, max_sequence_length: usize) -> Self {
        Hyperparameters {
            num_items: num_items,
            max_sequence_length: max_sequence_length,
            item_embedding_dim: 16,
            learning_rate: 0.01,
            l2_penalty: 0.0,
            loss: Loss::BPR,
            optimizer: Optimizer::Adam,
            parallelism: Parallelism::Synchronous,
            rng: XorShiftRng::from_seed(rand::thread_rng().gen()),
            num_threads: rayon::current_num_threads(),
            num_epochs: 10,
        }
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the L2 penalty.
    pub fn l2_penalty(mut self, l2_penalty: f32) -> Self {
        self.l2_penalty = l2_penalty;
        self
    }

    /// Set the embedding dimensionality.
    pub fn embedding_dim(mut self, embedding_dim: usize) -> Self {
        self.item_embedding_dim = embedding_dim;
        self
    }

    /// Set the number of epochs to run per each `fit` call.
    pub fn num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    /// Set the loss function.
    pub fn loss(mut self, loss: Loss) -> Self {
        self.loss = loss;
        self
    }

    /// Set number of threads to be used.
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Set the type of paralellism.
    pub fn parallelism(mut self, parallelism: Parallelism) -> Self {
        self.parallelism = parallelism;
        self
    }

    /// Set the random number generator.
    pub fn rng(mut self, rng: XorShiftRng) -> Self {
        self.rng = rng;
        self
    }

    /// Set the random number generator from seed.
    pub fn from_seed(mut self, seed: [u8; 16]) -> Self {
        self.rng = XorShiftRng::from_seed(seed);
        self
    }

    /// Set the optimizer type.
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set hyperparameters randomly: useful for hyperparameter search.
    pub fn random<R: Rng>(num_items: usize, rng: &mut R) -> Self {
        Hyperparameters {
            num_items: num_items,
            max_sequence_length: 2_usize.pow(Uniform::new(4, 8).sample(rng)),
            item_embedding_dim: 2_usize.pow(Uniform::new(4, 8).sample(rng)),
            learning_rate: (10.0_f32).powf(Uniform::new(-3.0, 0.5).sample(rng)),
            l2_penalty: (10.0_f32).powf(Uniform::new(-7.0, -3.0).sample(rng)),
            loss: if Uniform::new(0.0, 1.0).sample(rng) < 0.5 {
                Loss::BPR
            } else {
                Loss::Hinge
            },
            optimizer: if Uniform::new(0.0, 1.0).sample(rng) < 0.5 {
                Optimizer::Adam
            } else {
                Optimizer::Adagrad
            },
            parallelism: if Uniform::new(0.0, 1.0).sample(rng) < 0.5 {
                Parallelism::Asynchronous
            } else {
                Parallelism::Synchronous
            },
            rng: XorShiftRng::from_seed(rand::thread_rng().gen()),
            num_threads: Uniform::new(1, rayon::current_num_threads() + 1).sample(rng),
            num_epochs: 2_usize.pow(Uniform::new(3, 7).sample(rng)),
        }
    }

    fn build_params(mut self) -> Parameters {
        let item_embeddings = Arc::new(wyrm::HogwildParameter::new(embedding_init(
            self.num_items,
            self.item_embedding_dim,
            &mut self.rng,
        )));

        let item_biases = Arc::new(wyrm::HogwildParameter::new(Arr::zeros((self.num_items, 1))));
        let alpha = Arc::new(wyrm::HogwildParameter::new(Arr::zeros((
            1,
            self.item_embedding_dim,
        ))));
        let fc1 = Arc::new(wyrm::HogwildParameter::new(dense_init(
            self.item_embedding_dim,
            self.item_embedding_dim,
            &mut self.rng,
        )));
        let fc2 = Arc::new(wyrm::HogwildParameter::new(dense_init(
            self.item_embedding_dim,
            self.item_embedding_dim,
            &mut self.rng,
        )));

        Parameters {
            hyper: self,
            item_embedding: item_embeddings,
            item_biases: item_biases,
            alpha: alpha,
            fc1: fc1,
            fc2: fc2,
        }
    }

    /// Build the implicit EWMA model.
    pub fn build(self) -> ImplicitEWMAModel {
        let params = self.build_params();

        ImplicitEWMAModel { params: params }
    }
}

#[derive(Debug)]
struct Parameters {
    hyper: Hyperparameters,
    item_embedding: Arc<wyrm::HogwildParameter>,
    item_biases: Arc<wyrm::HogwildParameter>,
    alpha: Arc<wyrm::HogwildParameter>,
    fc1: Arc<wyrm::HogwildParameter>,
    fc2: Arc<wyrm::HogwildParameter>,
}

impl Clone for Parameters {
    fn clone(&self) -> Self {
        Parameters {
            hyper: self.hyper.clone(),
            item_embedding: Arc::new(self.item_embedding.as_ref().clone()),
            item_biases: Arc::new(self.item_biases.as_ref().clone()),
            alpha: Arc::new(self.alpha.as_ref().clone()),
            fc1: Arc::new(self.alpha.as_ref().clone()),
            fc2: Arc::new(self.alpha.as_ref().clone()),
        }
    }
}

impl SequenceModelParameters for Parameters {
    type Output = Model;
    fn max_sequence_length(&self) -> usize {
        self.hyper.max_sequence_length
    }
    fn num_threads(&self) -> usize {
        self.hyper.num_threads
    }
    fn rng(&mut self) -> &mut XorShiftRng {
        &mut self.hyper.rng
    }
    fn optimizer(&self) -> Optimizers {
        match self.hyper.optimizer {
            Optimizer::Adagrad => Optimizers::Adagrad(
                wyrm::optim::Adagrad::new()
                    .learning_rate(self.hyper.learning_rate)
                    .l2_penalty(self.hyper.l2_penalty),
            ),

            Optimizer::Adam => Optimizers::Adam(
                wyrm::optim::Adam::new()
                    .learning_rate(self.hyper.learning_rate)
                    .l2_penalty(self.hyper.l2_penalty),
            ),
        }
    }
    fn parallelism(&self) -> &Parallelism {
        &self.hyper.parallelism
    }
    fn loss(&self) -> &Loss {
        &self.hyper.loss
    }
    fn num_epochs(&self) -> usize {
        self.hyper.num_epochs
    }
    fn build(&self) -> Model {
        let item_embeddings = wyrm::ParameterNode::shared(self.item_embedding.clone());
        let item_biases = wyrm::ParameterNode::shared(self.item_biases.clone());
        let alpha = wyrm::ParameterNode::shared(self.alpha.clone());

        let inputs: Vec<_> = (0..self.hyper.max_sequence_length)
            .map(|_| wyrm::IndexInputNode::new(&vec![0; 1]))
            .collect();
        let outputs: Vec<_> = (0..self.hyper.max_sequence_length)
            .map(|_| wyrm::IndexInputNode::new(&vec![0; 1]))
            .collect();
        let negatives: Vec<_> = (0..self.hyper.max_sequence_length)
            .map(|_| wyrm::IndexInputNode::new(&vec![0; 1]))
            .collect();

        let input_embeddings: Vec<_> = inputs
            .iter()
            .map(|input| item_embeddings.index(input))
            .collect();
        let negative_embeddings: Vec<_> = negatives
            .iter()
            .map(|negative| item_embeddings.index(negative))
            .collect();
        let output_embeddings: Vec<_> = outputs
            .iter()
            .map(|output| item_embeddings.index(output))
            .collect();
        let output_biases: Vec<_> = outputs
            .iter()
            .map(|output| item_biases.index(output))
            .collect();
        let negative_biases: Vec<_> = negatives
            .iter()
            .map(|negative| item_biases.index(negative))
            .collect();

        let alpha = alpha.sigmoid();
        let one_minus_alpha = 1.0 - alpha.clone();

        let mut states = Vec::with_capacity(self.hyper.max_sequence_length);
        let initial_state = input_embeddings.first().unwrap().clone().boxed();
        states.push(initial_state);
        for input in &input_embeddings[1..] {
            let previous_state = states.last().unwrap().clone();
            states.push(
                (alpha.clone() * previous_state + one_minus_alpha.clone() * input.clone()).boxed(),
            );
        }

        let positive_predictions: Vec<_> =
            izip!(states.iter(), output_embeddings.iter(), output_biases)
                .map(|(state, output_embedding, output_bias)| {
                    state.vector_dot(output_embedding) + output_bias
                })
                .collect();
        let negative_predictions: Vec<_> =
            izip!(states.iter(), negative_embeddings.iter(), negative_biases)
                .map(|(state, negative_embedding, negative_bias)| {
                    state.vector_dot(negative_embedding) + negative_bias
                })
                .collect();

        let losses: Vec<_> = positive_predictions
            .into_iter()
            .zip(negative_predictions.into_iter())
            .map(|(pos, neg)| match self.hyper.loss {
                Loss::BPR => (neg - pos).sigmoid().boxed(),
                Loss::Hinge | Loss::WARP => (1.0 + neg - pos).relu().boxed(),
            })
            .collect();

        let mut summed_losses = Vec::with_capacity(losses.len());
        summed_losses.push(losses[0].clone());

        for loss in &losses[1..] {
            let loss = (summed_losses.last().unwrap().clone() + loss.clone()).boxed();
            summed_losses.push(loss);
        }

        Model {
            inputs: inputs,
            outputs: outputs,
            negatives: negatives,
            hidden_states: states,
            summed_losses: summed_losses,
        }
    }
    fn predict_single(&self, user: &[f32], item_idx: usize) -> f32 {
        let item_embeddings = &self.item_embedding;
        let item_biases = &self.item_biases;

        let embeddings = item_embeddings.value();
        let biases = item_biases.value();

        let embedding = embeddings.subview(Axis(0), item_idx);
        let bias = biases[(item_idx, 0)];
        let dot = wyrm::simd_dot(user, embedding.as_slice().unwrap());

        bias + dot
    }
}

struct Model {
    inputs: Vec<Variable<wyrm::IndexInputNode>>,
    outputs: Vec<Variable<wyrm::IndexInputNode>>,
    negatives: Vec<Variable<wyrm::IndexInputNode>>,
    hidden_states: Vec<Variable<BoxedNode>>,
    summed_losses: Vec<Variable<BoxedNode>>,
}

impl SequenceModel for Model {
    fn state(
        &self,
    ) -> (
        &[Variable<wyrm::IndexInputNode>],
        &[Variable<wyrm::IndexInputNode>],
        &[Variable<wyrm::IndexInputNode>],
        &[Variable<BoxedNode>],
    ) {
        (
            &self.inputs,
            &self.outputs,
            &self.negatives,
            &self.hidden_states,
        )
    }
    fn losses(&mut self) -> &mut [Variable<BoxedNode>] {
        &mut self.summed_losses
    }
    fn hidden_states(&mut self) -> &mut [Variable<BoxedNode>] {
        &mut self.hidden_states
    }
}

/// Implicit EWMA model.
#[derive(Debug, Clone)]
pub struct ImplicitEWMAModel {
    params: Parameters,
}

impl ImplicitEWMAModel {
    /// Fit the EWMA model.
    pub fn fit(&mut self, interactions: &CompressedInteractions) -> Result<f32, FittingError> {
        fit_sequence_model(interactions, &mut self.params)
    }
}

impl OnlineRankingModel for ImplicitEWMAModel {
    type UserRepresentation = ImplicitUser;
    fn user_representation(
        &self,
        item_ids: &[ItemId],
    ) -> Result<Self::UserRepresentation, PredictionError> {
        self.params.user_representation(item_ids)
    }

    fn predict(
        &self,
        user: &Self::UserRepresentation,
        item_ids: &[ItemId],
    ) -> Result<Vec<f32>, PredictionError> {
        self.params.predict(user, item_ids)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use data::{user_based_split, Interactions};
    use datasets::download_movielens_100k;
    use evaluation::mrr_score;

    fn run_test(mut data: Interactions, hyperparameters: Hyperparameters) -> (f32, f32) {
        let mut rng = rand::XorShiftRng::from_seed([42; 16]);

        let (train, test) = user_based_split(&mut data, &mut rng, 0.2);
        let train_mat = train.to_compressed();
        let test_mat = test.to_compressed();

        let mut model = hyperparameters.rng(rng).build();

        let start = Instant::now();
        let loss = model.fit(&train_mat).unwrap();
        let elapsed = start.elapsed();
        let train_mrr = mrr_score(&model, &train_mat).unwrap();
        let test_mrr = mrr_score(&model, &test_mat).unwrap();

        println!(
            "Train MRR {} at loss {} and test MRR {} (in {:?})",
            train_mrr, loss, test_mrr, elapsed
        );

        (test_mrr, train_mrr)
    }

    #[test]
    fn mrr_test_single_thread() {
        let data = download_movielens_100k().unwrap();

        let hyperparameters = Hyperparameters::new(data.num_items(), 128)
            .embedding_dim(32)
            .learning_rate(0.16)
            .l2_penalty(0.0004)
            .loss(Loss::Hinge)
            .optimizer(Optimizer::Adagrad)
            .num_epochs(10)
            .num_threads(1);

        let (test_mrr, _) = run_test(data, hyperparameters);

        let expected_mrr = match ::std::env::var("MKL_CBWR") {
            Ok(ref val) if val == "AVX" => 0.091,
            _ => 0.11,
        };

        assert!(test_mrr > expected_mrr)
    }

    #[test]
    fn mrr_test_warp() {
        let data = download_movielens_100k().unwrap();

        let hyperparameters = Hyperparameters::new(data.num_items(), 128)
            .embedding_dim(32)
            .learning_rate(0.16)
            .l2_penalty(0.0004)
            .loss(Loss::WARP)
            .optimizer(Optimizer::Adagrad)
            .num_epochs(10)
            .num_threads(1);

        let (test_mrr, _) = run_test(data, hyperparameters);

        let expected_mrr = match ::std::env::var("MKL_CBWR") {
            Ok(ref val) if val == "AVX" => 0.089,
            _ => 0.14,
        };

        assert!(test_mrr > expected_mrr)
    }
}
