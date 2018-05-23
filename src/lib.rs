// #![deny(missing_docs)]
//! # sbr-rs
//!
//! `sbr` implements efficient recommender algorithms which operate on
//! sequences of items: given previous items a user has interacted with,
//! the model will recommend the items the user is likely to interact with
//! in the future.

#[macro_use]
extern crate serde_derive;
extern crate serde_json;

#[macro_use]
extern crate itertools;

extern crate csv;
#[macro_use]
extern crate failure;
extern crate ndarray;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate siphasher;

extern crate wyrm;

pub mod data;
pub mod evaluation;
pub mod models;

/// Alias for user indices.
pub type UserId = usize;
/// Alias for item indices.
pub type ItemId = usize;
/// Alias for timestamps.
pub type Timestamp = usize;

/// Precition error types.
#[derive(Debug, Fail)]
pub enum PredictionError {
    /// Failed prediction due to numerical issues.
    #[fail(display = "Invalid prediction value: non-finite or not a number.")]
    InvalidPredictionValue,
}

/// Trait describing models that can compute predictions given
/// a user's sequences of past interactions.
pub trait OnlineRankingModel {
    /// The representation the model computes from past interactions.
    type UserRepresentation: std::fmt::Debug;
    /// Compute a user representation from past interactions.
    fn user_representation(
        &self,
        item_ids: &[ItemId],
    ) -> Result<Self::UserRepresentation, PredictionError>;
    /// Given a user representation, rank `item_ids` according
    /// to how likely the user is to interact with them in the future.
    fn predict(
        &self,
        user: &Self::UserRepresentation,
        item_ids: &[ItemId],
    ) -> Result<Vec<f32>, PredictionError>;
}
