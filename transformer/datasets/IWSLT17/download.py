import os
import pandas as pd
import zipfile
import requests
import re
import shutil

src_lang = "en"
tgt_lang = "de"

download_path = f"./{src_lang}-{tgt_lang}.zip"
train_path = "./train/train.parquet"
valid_path = "./valid/valid.parquet"
test_path = "./test/test.parquet"
url = f"https://hf-mirror.com/datasets/IWSLT/iwslt2017/resolve/main/data/2017-01-trnted/texts/{src_lang}/{tgt_lang}/{src_lang}-{tgt_lang}.zip?download=true"


def download(path):
    if not os.path.exists(path):
        print(f"Start downloading IWSLT17 {src_lang}-{tgt_lang} datasets...")
        response = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"\nDownload complete.")
    else:
        print(f"{path} already exists. Skipping download.")

    with zipfile.ZipFile(path) as f:
        f.extractall()
        namelist = f.namelist()

    return namelist


def prepare(raw_path, split_config_dict):
    raw_dir, _ = os.path.split(raw_path)
    for split in split_config_dict.keys():
        print(f"Preparing {split} set...")
        save_path = split_config_dict[split]["save_path"]
        dir, _ = os.path.split(save_path)
        os.makedirs(dir, exist_ok=True)

        filter = split_config_dict[split]["filter"]
        extractor = split_config_dict[split]["extractor"]
        src_sentences, tgt_sentences = [], []

        for src, tgt in zip(
            split_config_dict[split][src_lang], split_config_dict[split][tgt_lang]
        ):
            with open(
                os.path.join(raw_dir, src), mode="r", encoding="utf-8"
            ) as src_file, open(
                os.path.join(raw_dir, tgt), mode="r", encoding="utf-8"
            ) as tgt_file:
                src_sentences += [
                    extractor(line).strip()
                    for line in src_file.readlines()
                    if filter(line.strip())
                ]
                tgt_sentences += [
                    extractor(line).strip()
                    for line in tgt_file.readlines()
                    if filter(line.strip())
                ]

        assert len(src_sentences) == len(tgt_sentences)

        df = pd.DataFrame(
            {
                "translation": [
                    {src_lang: src_sentence, tgt_lang: tgt_sentence}
                    for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences)
                ]
            }
        )

        df.to_parquet(save_path)


if __name__ == "__main__":
    namelist = download(download_path)
    testset_pattern = re.compile("<seg .*?>(.*?)</seg>")
    valid_years = [2010]
    test_years = [2010, 2011, 2012, 2013, 2014, 2015]
    prepare(
        download_path,
        {
            "train": {
                "save_path": train_path,
                src_lang: [f"train.tags.{src_lang}-{tgt_lang}.{src_lang}"],
                tgt_lang: [f"train.tags.{src_lang}-{tgt_lang}.{tgt_lang}"],
                "filter": lambda s: not s.startswith("<"),
                "extractor": lambda s: s,
            },
            "valid": {
                "save_path": valid_path,
                src_lang: [
                    f"IWSLT17.TED.dev{year}.{src_lang}-{tgt_lang}.{src_lang}.xml"
                    for year in valid_years
                ],
                tgt_lang: [
                    f"IWSLT17.TED.dev{year}.{src_lang}-{tgt_lang}.{tgt_lang}.xml"
                    for year in valid_years
                ],
                "filter": lambda s: s.startswith("<seg"),
                "extractor": lambda s: re.findall(testset_pattern, s)[0],
            },
            "test": {
                "save_path": test_path,
                src_lang: [
                    f"IWSLT17.TED.tst{year}.{src_lang}-{tgt_lang}.{src_lang}.xml"
                    for year in test_years
                ],
                tgt_lang: [
                    f"IWSLT17.TED.tst{year}.{src_lang}-{tgt_lang}.{tgt_lang}.xml"
                    for year in test_years
                ],
                "filter": lambda s: s.startswith("<seg"),
                "extractor": lambda s: re.findall(testset_pattern, s)[0],
            },
        },
    )

    for file in namelist:
        if os.path.exists(file):
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)
