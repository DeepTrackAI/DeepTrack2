from tkinter.tix import Tree
import requests
import os
import zipfile

import math

# Dataset ids stored on google drive.
# (dataset_id, folder name, model)
_ID = {
    "CellCounting": (
        "18Afk9Fwe4y3FVLPYd7fr4sfNIW59KEGR",
        "CellCounting",
        "1SCfYpesI2KMasiZalIJ7lgGV3By8qe9v",
    ),
    "MNIST": (
        "1UePQAYNp-ja9userMwTprIwGWjwCu3Tf",
        "MNIST",
        "1w7uat5Vshve9OSaqvGo4EtLiqixHk2yy",
    ),
    "QuantumDots": (
        "1naaoxIaAU1F_rBaI-I1pB1K4Sp6pq_Jv",
        "QuantumDots",
        "131H6xdyC5gyTMQcnzb1ozztBH2nelWLg",
    ),
    "ParticleTracking": (
        "1eA9F_GjJbErkJu2TizE_CHjpqD6WePqy",
        "ParticleTracking",
        "17sHytcOEmQaxkLJ7mPwX7KRKijOSEOAL",
    ),
    "ParticleSizing": (
        "1FaygrzmEDnXjVe_W3yVfqNTM0Ir5jFkR",
        "ParticleSizing",
        "1k3wf9c6BZd6HhdpOaOFd8p6m_c9FXdrk",
    ),
    "3DTracking": (
        "1Ez5z4rVlJ7islra2oEoewdY9FrX1zcex",
        "3DTracking",
        "1owS49f8hyFTOEvFNgjSwv3TCusznzEWc",
    ),
    "MitoGAN": (
        "13fzMUXz3QSJPjXOf9p-To72K8Z0XGKyU",
        "MitoGAN",
        "1Qf5vIWEksKPHJ1CBK6GPEhsSCvFFHFrg",
    ),
    "CellData": ("1CJW7msDiI7xq7oMce4l9tRkNN6O5eKtj", "CellData", ""),
    "CellMigData": ("1vRsWcxjbTz6rffCkrwOfs_ezPvUjPwGw", "CellMigData", ""),
    "BFC2Cells": ("1lHgJdG5I3vRnU_DRFwTr_c69nx1Xkd3X", "BFC2Cells", ""),
}


def load(key):
    """Downloads a dataset from google drive.
    One of: "CellCounting", "MNIST", "QuantumDots", "ParticleTracking",
    "ParticleSizing", "3DTracking", "MitoGAN", "CellData", "CellMigData", "BFC2Cells"

    Data will be stored in the "datasets" folder.

    Returns:
        None

    """
    if key not in _ID:
        raise ValueError("Invalid dataset key: {}".format(key))

    # Get the dataset id and folder name.
    dataset_id, folder_name, model_id = _ID[key]

    # Create the datasets folder if it doesn't exist.
    if not os.path.exists("datasets"):
        os.mkdir("datasets")

    # Check if dataset is already downloaded.
    if os.path.exists("datasets/{}".format(folder_name)):
        # Check if folder content is non-empty.
        if os.listdir("datasets/{}".format(folder_name)):
            print("Dataset already downloaded.")
            return

    # Create the folder for the dataset if it doesn't exist.
    if not os.path.exists("datasets/" + folder_name):
        os.mkdir("datasets/" + folder_name)

    # Download zip file.
    print(f"Downloading {key}...")
    url = f"https://drive.google.com/uc?export=download&id={dataset_id}"
    response = requests.get(url, stream=True, params={"confirm": "true"})
    # Check that the response is ok.

    if response.status_code != 200:
        raise ValueError(
            "Error downloading dataset.",
            response.status_code,
            response.reason,
            response.text,
        )

    save_response_content(response, f"datasets/{key}.zip")

    # Extract zip file.
    print(f"Extracting {key}...")
    zip_ref = zipfile.ZipFile(f"datasets/{key}.zip", "r")
    zip_ref.extractall("datasets")
    zip_ref.close()

    # Delete zip file.
    os.remove(f"datasets/{key}.zip")

    # If the extracted folder is another folder with the same name, move it.
    if os.path.isdir(f"datasets/{folder_name}/{folder_name}"):
        os.rename(f"datasets/{folder_name}/{folder_name}", f"datasets/{folder_name}")


def load_model(key):
    """Downloads a model from google drive.
    One of: "CellCounting", "MNIST", "QuantumDots", "ParticleTracking",
    "ParticleSizing", "3DTracking", "MitoGAN", "CellData", "CellMigData", "BFC2Cells"

    Data will be stored in the "models" folder.

    Returns:
        path to the model : str

    """
    if key not in _ID:
        raise ValueError("Invalid dataset key: {}".format(key))

    # Get the dataset id and folder name.
    dataset_id, folder_name, model_id = _ID[key]

    # Create the datasets folder if it doesn't exist.
    if not os.path.exists("models"):
        os.mkdir("models")

    # Check if dataset is already downloaded.
    if os.path.exists("models/{}".format(folder_name)):
        # Check if folder content is non-empty.
        if os.listdir("models/{}".format(folder_name)):
            print("Model already downloaded.")
            return

    # Create the folder for the dataset if it doesn't exist.
    if not os.path.exists("models/" + folder_name):
        os.mkdir("models/" + folder_name)

    # Download zip file.
    print(f"Downloading {key}...")
    url = f"https://drive.google.com/uc?export=download&id={model_id}"
    response = requests.get(url, stream=True, params={"confirm": "true"})
    # Check that the response is ok.

    if response.status_code != 200:
        raise ValueError(
            "Error downloading model.",
            response.status_code,
            response.reason,
            response.text,
        )

    save_response_content(response, f"models/{key}.zip")

    # Extract zip file.
    print(f"Extracting {key}...")
    zip_ref = zipfile.ZipFile(f"models/{key}.zip", "r")
    zip_ref.extractall("models")
    zip_ref.close()

    # Delete zip file.
    os.remove(f"models/{key}.zip")

    # If the extracted folder is another folder with the same name, move it.
    if os.path.isdir(f"models/{folder_name}/{folder_name}"):
        os.rename(f"models/{folder_name}/{folder_name}", f"models/{folder_name}")

    return f"models/{folder_name}"


def save_response_content(response, destination):
    CHUNK_SIZE = 32768 * 4
    idx = 0
    with open(destination, "wb") as f:

        for idx, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            download_counter = convert_size(CHUNK_SIZE * idx)
            if idx % 40 == 0:
                print("Downloading file:", download_counter, end="\r")

            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
        print("")
        print("Download Complete!")


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s\t%s" % (s, size_name[i])