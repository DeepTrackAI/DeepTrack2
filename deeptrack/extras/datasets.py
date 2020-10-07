import requests
import os
import zipfile

import deeptrack as dt


# (dataset, folder name, model)
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
}


def load(id, force_overwrite=False):

    root = os.path.abspath("./datasets/")

    try:
        os.mkdir("datasets")
    except FileExistsError:
        pass

    if id not in _ID:
        print(
            "Dataset",
            id,
            "not recognized. Available datasets are:",
            ", ".join(_ID.keys()),
        )
        return

    tag, folder, _ = _ID[id]

    if not force_overwrite and os.path.exists(os.path.join(root, folder)):
        print(
            id,
            "already downloaded! Use force_overwrite=True to redownload the dataset.",
        )
        return

    destination = os.path.join(root, tag + ".zip")
    download_file_from_google_drive(tag, destination)

    if os.path.exists(destination):
        with zipfile.ZipFile(destination) as file:
            print("Extracting files...")
            file.extractall(root)
            print("Done")
        print("Cleaning up...")
        os.remove(destination)
        print("...OK!")

    else:
        print("Unable to download dataset")


def load_model(id, force_overwrite=False):

    root = os.path.abspath("./models/")

    try:
        os.mkdir("models")
    except FileExistsError:
        pass

    _, folder, tag = _ID[id]

    destination = os.path.join(root, folder + ".h5")

    if not force_overwrite and os.path.exists(destination):
        print(
            id, "already downloaded! Use force_overwrite=True to redownload the model."
        )
        return destination

    download_file_from_google_drive(tag, destination)

    return destination


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)

    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


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


import math


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s\t%s" % (s, size_name[i])
