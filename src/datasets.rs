//! Built-in datasets for easy testing and experimentation.
use std::env;
use std::fs::{create_dir_all, rename, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use csv;
use failure;
use reqwest;

use data::{Interaction, Interactions};

/// Dataset error types.
#[derive(Debug, Fail)]
pub enum DatasetError {
    /// Can't find the home directory.
    #[fail(display = "Cannot find home directory.")]
    NoHomeDir,
}

fn create_data_dir() -> Result<PathBuf, failure::Error> {
    let path = env::home_dir()
        .ok_or_else(|| DatasetError::NoHomeDir)?
        .join(".sbr-rs");

    if !path.exists() {
        create_dir_all(&path)?;
    }

    Ok(path)
}

fn download(url: &str, dest_filename: &Path) -> Result<Interactions, failure::Error> {
    let data_dir = create_data_dir()?;
    let desired_filename = data_dir.join(dest_filename);
    let temp_filename = env::temp_dir().join(dest_filename);

    if !desired_filename.exists() {
        let file = File::create(&temp_filename)?;
        let mut writer = BufWriter::new(file);

        let mut response = reqwest::get(url)?;
        response.copy_to(&mut writer)?;

        rename(temp_filename, &desired_filename)?;
    }

    let mut reader = csv::Reader::from_path(desired_filename)?;
    let interactions: Vec<Interaction> = reader.deserialize().collect::<Result<Vec<_>, _>>()?;

    Ok(Interactions::from(interactions))
}

/// Download the Movielens 100K dataset and return it.
///
/// The data is stored in `~/.sbr-rs/`.
pub fn download_movielens_100k() -> Result<Interactions, failure::Error> {
    download(
        "https://github.com/maciejkula/sbr-rs/raw/master/data.csv",
        Path::new("movielens_100K.csv"),
    )
}
