//! Models module.
// pub mod ewma;
pub mod lstm;
mod sequence_model;

/// The user representation used by implicit sequence models.
#[derive(Clone, Debug)]
pub struct ImplicitUser {
    user_embedding: Vec<f32>,
}

/// The loss used for training the model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Loss {
    /// Bayesian Personalised Ranking.
    BPR,
    /// Pairwise hinge loss.
    Hinge,
    /// WARP
    WARP,
}

/// Optimizer user to train the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Optimizer {
    /// Adagrad.
    Adagrad,
    /// Adam.
    Adam,
}

/// Type of parallelism used to train the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Parallelism {
    /// Multiple threads operate in parallel without any locking.
    Asynchronous,
    /// Multiple threads synchronise parameters between minibatches.
    Synchronous,
}
