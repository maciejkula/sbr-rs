use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::{Rng, SeedableRng, XorShiftRng};
use rayon::prelude::*;

use wyrm;
use wyrm::optim::{Optimizer as Optim, Optimizers, Synchronizable};
use wyrm::{BoxedNode, DataInput, Variable};

use super::{ImplicitUser, Loss, Parallelism};
use data::CompressedInteractions;
use {FittingError, ItemId, OnlineRankingModel, PredictionError};

pub trait SequenceModelParameters {
    type Output;
    fn max_sequence_length(&self) -> usize;
    fn num_threads(&self) -> usize;
    fn rng(&mut self) -> &mut XorShiftRng;
    fn optimizer(&self) -> Optimizers;
    fn parallelism(&self) -> &Parallelism;
    fn loss(&self) -> &Loss;
    fn num_epochs(&self) -> usize;
    fn build(&self) -> Self::Output;
    fn predict_single(&self, user: &[f32], item_idx: usize) -> f32;
}

/// Trait expressing a sequence model.
pub trait SequenceModel {
    /// Return the sequence losses of the model.
    fn losses(&mut self) -> &mut [Variable<BoxedNode>];
    /// Return the inner state of the model. These are:
    /// - inputs
    /// - targets
    /// - negatives
    /// - hidden states.
    fn state(
        &self,
    ) -> (
        &[Variable<wyrm::IndexInputNode>],
        &[Variable<wyrm::IndexInputNode>],
        &[Variable<wyrm::IndexInputNode>],
        &[Variable<BoxedNode>],
    );
    fn hidden_states(&mut self) -> &mut [Variable<BoxedNode>];
}

fn sample_warp_negative<U: SequenceModel, T: SequenceModelParameters<Output = U>>(
    parameters: &T,
    hidden_state: &[f32],
    positive_idx: usize,
    negative_item_range: &Uniform<usize>,
    thread_rng: &mut XorShiftRng,
) -> usize {
    let pos_prediction = parameters.predict_single(hidden_state, positive_idx);

    let mut negative_idx = 0;

    for _ in 0..5 {
        negative_idx = negative_item_range.sample(thread_rng);
        let neg_prediction = parameters.predict_single(hidden_state, negative_idx);

        if 1.0 - pos_prediction + neg_prediction > 0.0 {
            break;
        }
    }

    negative_idx
}

pub fn fit_sequence_model<U: SequenceModel, T: SequenceModelParameters<Output = U> + Sync>(
    interactions: &CompressedInteractions,
    parameters: &mut T,
) -> Result<f32, FittingError> {
    let negative_item_range = Uniform::new(0, interactions.num_items());

    let mut subsequences: Vec<_> = interactions
        .iter_users()
        .flat_map(|user| {
            user.chunks(parameters.max_sequence_length())
                .map(|(item_ids, _)| item_ids)
                .filter(|item_ids| item_ids.len() > 2)
        })
        .collect();
    parameters.rng().shuffle(&mut subsequences);

    if subsequences.len() == 0 {
        return Err(FittingError::NoInteractions);
    }

    let optimizer = parameters.optimizer();
    let num_chunks = subsequences.len() / parameters.num_threads();
    let sync_optim = optimizer.synchronized(parameters.num_threads());

    let mut partitions: Vec<_> = subsequences
        .chunks_mut(num_chunks)
        .zip(sync_optim.into_iter())
        .map(|(chunk, optim)| (chunk, XorShiftRng::from_seed(parameters.rng().gen()), optim))
        .collect();

    let loss = partitions
        .par_iter_mut()
        .map(|(partition, ref mut thread_rng, sync_optim)| {
            let mut model = parameters.build();

            let mut loss_value = 0.0;
            let mut examples = 0;

            for _ in 0..parameters.num_epochs() {
                thread_rng.shuffle(partition);

                for &item_ids in partition.iter() {
                    {
                        let (inputs, outputs, negatives, hidden_states) = model.state();

                        for (&input_idx, &output_idx, input, output, negative, hidden) in izip!(
                            item_ids,
                            item_ids.iter().skip(1),
                            inputs,
                            outputs,
                            negatives,
                            hidden_states
                        ) {
                            input.set_value(input_idx);

                            let negative_idx = if parameters.loss() == &Loss::WARP {
                                hidden.forward();
                                let hidden_state = hidden.value();

                                sample_warp_negative(
                                    parameters,
                                    hidden_state.as_slice().unwrap(),
                                    output_idx,
                                    &negative_item_range,
                                    thread_rng,
                                )
                            } else {
                                negative_item_range.sample(thread_rng)
                            };

                            output.set_value(output_idx);
                            negative.set_value(negative_idx);
                        }
                    }

                    // Get the loss at the end of the sequence.
                    let loss_idx = item_ids.len().saturating_sub(2);

                    // We need to clear the graph if the loss is WARP
                    // in order for backpropagation to trigger correctly.
                    // This is because by calling forward we've added the
                    // resulting nodes to the graph.
                    if parameters.loss() == &Loss::WARP {
                        &model.hidden_states()[loss_idx].clear();
                    }

                    let loss = &mut model.losses()[loss_idx];
                    loss_value += loss.value().scalar_sum();
                    examples += loss_idx + 1;

                    loss.forward();
                    loss.backward(1.0);

                    if parameters.num_threads() > 1
                        && parameters.parallelism() == &Parallelism::Synchronous
                    {
                        sync_optim.step(loss.parameters());
                    } else {
                        optimizer.step(loss.parameters());
                    }
                }
            }

            loss_value / (1.0 + examples as f32)
        })
        .sum();

    Ok(loss)
}

impl<U: SequenceModel, T: SequenceModelParameters<Output = U> + Sync> OnlineRankingModel for T {
    type UserRepresentation = ImplicitUser;
    fn user_representation(
        &self,
        item_ids: &[ItemId],
    ) -> Result<Self::UserRepresentation, PredictionError> {
        let model = self.build();

        let item_ids = &item_ids[item_ids.len().saturating_sub(self.max_sequence_length())..];

        let (inputs, _, _, hidden_states) = model.state();

        for (&input_idx, input) in izip!(item_ids, inputs) {
            input.set_value(input_idx);
        }

        // Get the loss at the end of the sequence.
        let loss_idx = item_ids.len().saturating_sub(1);

        // Select the hidden state after ingesting all the inputs.
        let hidden_state = &hidden_states[loss_idx];

        // Run the network forward up to that point.
        hidden_state.forward();

        // Get the value.
        let representation = hidden_state.value();

        Ok(ImplicitUser {
            user_embedding: representation.as_slice().unwrap().to_owned(),
        })
    }

    fn predict(
        &self,
        user: &Self::UserRepresentation,
        item_ids: &[ItemId],
    ) -> Result<Vec<f32>, PredictionError> {
        let user_slice = &user.user_embedding;

        item_ids
            .iter()
            .map(|&item_idx| {
                let prediction = self.predict_single(user_slice, item_idx);

                if prediction.is_finite() {
                    Ok(prediction)
                } else {
                    Err(PredictionError::InvalidPredictionValue)
                }
            })
            .collect()
    }
}
