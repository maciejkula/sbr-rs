use std;

use super::ImplicitFactorizationModel;
use data::CompressedInteractions;

pub fn mrr_score(
    model: &ImplicitFactorizationModel,
    test: &CompressedInteractions,
    train: &CompressedInteractions,
) -> Result<f32, &'static str> {
    if test.shape() != train.shape() {
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
