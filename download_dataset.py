from pathlib import Path

from audio_processing.utilities.paths import DEFAULT_DATA_DIR
from audio_processing.download import mimii, mimii_due

if __name__ == "__main__":

    dataset_name = "mimii_due"
    entity_types = ["fan"]
    #noise_level = [6] # signal to noise ratio

    print(f"Downloading dataset {dataset_name}, entity_types {entity_types}.")
    print(f"Default data dir is {DEFAULT_DATA_DIR}")

    downloader = mimii_due.MimiiDueDownloader(entity_types=entity_types)
    downloader.download(destination=DEFAULT_DATA_DIR)
    is_downloaded = downloader.is_downloaded(destination=DEFAULT_DATA_DIR)
    print(f"is_downloaded is {is_downloaded}")

    preprocessor = mimii_due.MimiiDuePreprocessor()

    # the output dir is the audio_data dir
    output_dir = Path("audio_data/mimii_due").absolute()
    print(f"output_dir is {output_dir}. Starting preprocessing.")
    preprocessor.preprocess(raw_dir=DEFAULT_DATA_DIR, output_dir=output_dir)