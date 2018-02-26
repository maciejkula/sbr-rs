use std;
use std::cmp::Ordering;
use std::hash::Hasher;

use rand::distributions::{IndependentSample, Range};
use rand::Rng;

use siphasher::sip::SipHasher;

use super::{ItemId, Timestamp, UserId};

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
    let denominator = 100_000;
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

    pub fn split_at(&self, idx: usize) -> (Self, Self) {
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

    pub fn split_by<F: Fn(&Interaction) -> bool>(&self, func: F) -> (Self, Self) {
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

    pub fn num_users(&self) -> usize {
        self.num_users
    }

    pub fn num_items(&self) -> usize {
        self.num_items
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.num_users, self.num_items)
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

fn cmp_timestamp(x: &Interaction, y: &Interaction) -> Ordering {
    let uid_comparison = x.user_id().cmp(&y.user_id());

    if uid_comparison == Ordering::Equal {
        x.timestamp().cmp(&y.timestamp())
    } else {
        uid_comparison
    }
}

pub struct CompressedInteractions {
    num_users: usize,
    num_items: usize,
    user_pointers: Vec<usize>,
    item_ids: Vec<ItemId>,
    timestamps: Vec<Timestamp>,
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

    pub fn num_users(&self) -> usize {
        self.num_users
    }

    pub fn num_items(&self) -> usize {
        self.num_items
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.num_users, self.num_items)
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
    pub fn num_users(&self) -> usize {
        self.num_users
    }

    pub fn num_items(&self) -> usize {
        self.num_items
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.num_users, self.num_items)
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
