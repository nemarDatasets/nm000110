README_CONTENT = """## Introduction

The CHB-MIT Scalp EEG Database consists of EEG recordings from pediatric subjects with intractable seizures. This dataset was collected at the Children's Hospital Boston and includes recordings from 22 subjects (5 males, ages 3-22; and 17 females, ages 1.5-19) with epilepsy. The recordings contain 198 annotated seizures and were originally collected to characterize seizures and assess patients' candidacy for surgical intervention.

## Overview of the experiment

Subjects were monitored for up to several days following withdrawal of anti-seizure medication in a controlled hospital environment. The purpose was to capture and characterize their seizure patterns using continuous scalp EEG monitoring. Each case (subject) contains between 9 and 42 continuous EEG recording files. All signals were sampled at 256 samples per second with 16-bit resolution. Most files contain 23 EEG signals recorded using the International 10-20 system of EEG electrode positions and nomenclature. The recordings use bipolar montages, where each channel represents the potential difference between two electrode sites. Hardware limitations resulted in gaps between consecutively-numbered files, typically 10 seconds or less, during which signals were not recorded. Most recording files contain exactly one hour of digitized EEG signals, though some cases contain two-hour or four-hour recordings. Additional signals such as ECG and vagal nerve stimulus (VNS) were recorded in some cases.

## Description of the preprocessing if any

The original .edf files from PhysioNet have been converted to BIDS format. Channel names have been standardized to match the standard 10-05 montage naming convention. Bipolar channel pairs are represented in the format "Electrode1-Electrode2" (e.g., "FP1-F7"). Non-EEG channels such as ECG are preserved with appropriate BIDS channel types. Channels that did not match expected formats or could not be mapped to the standard montage were marked as "misc" type. All protected health information (PHI) in the original files has been replaced with surrogate information. Dates have been replaced with surrogate dates while preserving time relationships between files. Subject birthdates are calculated based on age at recording time when available.

## Description of the event values if any

The events.tsv files contain seizure onset and offset annotations. Each seizure event has:
- onset: Time in seconds from the beginning of the recording when the seizure starts
- duration: Duration of the seizure in seconds
- value: "seizure" - indicating a seizure event
- sample: Sample number at onset

The seizure annotations were originally marked with '[' for onset and ']' for offset in the .seizures annotation files and have been converted to BIDS-compliant event format. In total, the dataset contains 198 seizure events across all subjects (182 in the original 23 cases, plus 16 additional seizures from case chb24 added in December 2010).

## Citation

When using this dataset, please cite:

1. Ali Shoeb. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology, September 2009. http://hdl.handle.net/1721.1/54669

2. Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2K01R

3. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

**Data curators:**
Pierre Guetschel (BIDS conversion)

Original data collection team:
- Jack Connolly, REEGT (Children's Hospital Boston)
- Herman Edwards, REEGT (Children's Hospital Boston)
- Blaise Bourgeois, MD (Children's Hospital Boston)
- S. Ted Treves, MD (Children's Hospital Boston)
- Ali Shoeb, PhD (Massachusetts Institute of Technology)
- Professor John Guttag (Massachusetts Institute of Technology)
"""

DATASET_NAME = "CHB-MIT"


from pathlib import Path
import re
import datetime
import shutil
import warnings

from mne_bids import BIDSPath, write_raw_bids, make_dataset_description, make_report
import mne
from mne import Annotations
from mne.io.constants import FIFF
from mne.channels import make_standard_montage
import pandas as pd
import numpy as np
import wfdb
import tqdm

CH_NAME_REGEX = r"(?P<ch1>[a-zA-Z0-9]+)-(?P<ch2>[a-zA-Z0-9]+)(?P<extra>.*)"
strange_CH_NAME_REGEX = r"-*[0-9]+"


def _get_records(source_root: Path):
    records_path = source_root / "RECORDS"

    # example record subject 1, run 2
    # chb01/chb01_02.edf
    record_regex = r"(?P<subject>chb\d{2})/\1(?P<letter_suffix>[a-z]?)_(?P<run>\d{2})(?P<plus>[+]?)\.edf"

    with open(records_path, "r") as f:
        records = f.read().splitlines()

    for record in records:
        match = re.match(record_regex, record)
        assert match is not None, f"Record {record} does not match expected format"

        subject = match.group("subject")
        run = int(match.group("run"))
        plus = match.group("plus")
        letter_suffix = match.group("letter_suffix")

        # subject 2 has an extra run (16 and 16+)
        plus_subject = "chb02"
        if plus == "+" or (subject == plus_subject and run > 16):
            assert subject == plus_subject
            run += 1

        # subject 17 has strange a/b/c suffixes
        if letter_suffix:
            assert subject == "chb17", subject
            if letter_suffix == "a":  # 3 to 8
                run += -2
            elif letter_suffix == "b":  # 57 to 69
                run += 6 - 56
            elif letter_suffix == "c":  # 2 to 13
                run += 19 - 1
            else:
                raise ValueError(f"Unexpected letter suffix {letter_suffix}")

        # source_path = source_root / record
        bids_path = BIDSPath(
            subject=f"{subject}",
            run=run,
            task="rest",
            suffix="eeg",
            datatype="eeg",
            extension=".edf",
        )
        yield record, bids_path


def _get_seizure_records(source_root: Path):
    records_path = source_root / "RECORDS-WITH-SEIZURES"

    with open(records_path, "r") as f:
        records = f.read().splitlines()

    return set(records)


def main(
    source_root: Path,
    bids_root: Path,
    overwrite: bool = False,
    finalize_only: bool = False,
):
    """Convert the CHB-MIT dataset to BIDS format.

    Parameters
    ----------
    source_root : Path
        Path to the root folder of the CHB-MIT dataset.
        Downloaded from https://physionet.org/content/chbmit/1.0.0/
    bids_root : Path
        Path to the root of the BIDS dataset to create.
    overwrite : bool
        If True, overwrite existing BIDS files.
    """
    source_root = Path(source_root).expanduser()
    bids_root = Path(bids_root).expanduser()

    records = list(_get_records(source_root))

    montage = make_standard_montage("standard_1005")
    upper_ch_names = {ch.upper(): ch for ch in montage.ch_names}

    # File inconsistent (chb07/chb07_18.edf has no seizure annotation file)
    # seizure_records = _get_seizure_records(source_root)

    # Add bids root:
    bids_root.mkdir(parents=True, exist_ok=True)
    for _, bids_path in records:
        bids_path = bids_path.update(root=bids_root)

    # sanity check: no duplicate bids paths
    bids_paths = [bids_path.fpath for _, bids_path in records]
    assert len(bids_paths) == len(set(bids_paths)), "Duplicate BIDS paths found"

    if finalize_only:
        _finalize_dataset(bids_root, overwrite=overwrite)
        return

    # Read subject info
    subject_info = pd.read_csv(
        source_root / "SUBJECT-INFO", sep="\t", header=0, skip_blank_lines=True
    )
    subject_info["Age (years)"] = subject_info["Age (years)"].astype(float)

    # Case chb24 was added to this collection in December 2010, and is not currently included in SUBJECT-INFO.)
    # So we add a row with NaN values for missing info
    subject_info = pd.concat(
        [subject_info, pd.DataFrame({"Case": ["chb24"]})], ignore_index=True
    )

    # Adjust subject info
    subject_info = subject_info.set_index("Case")
    subject_info["sex"] = (
        subject_info["Gender"].map({"M": 1, "F": 2, np.nan: 0}).astype(int)
    )
    subject_info["age"] = subject_info["Age (years)"].apply(
        lambda x: datetime.timedelta(days=x * 365) if not np.isnan(x) else None
    )

    bad_records = set()
    # Link the files to the new BIDS structure:
    seizure_files_counter = 0
    for record, bids_path in tqdm.tqdm(records, unit="record"):
        if not overwrite and bids_path.fpath.exists():
            continue

        source_path = source_root / record
        raw = mne.io.read_raw(source_path)

        # bipolar channels
        ch_names_map = {}
        ch_types_map = {}
        for ch in raw.info["chs"]:
            # # NOT SAVED WITH BIDS!
            # ch["coil_type"] = FIFF.FIFFV_COIL_EEG_BIPOLAR

            # default to misc
            ch_types_map[ch["ch_name"]] = "misc"

            if ch["ch_name"] == "ECG":
                ch_types_map[ch["ch_name"]] = "ecg"
                continue
            if ch["ch_name"] in "VNS" or re.fullmatch(
                strange_CH_NAME_REGEX, ch["ch_name"]
            ):
                continue

            match = re.fullmatch(CH_NAME_REGEX, ch["ch_name"])

            if match is None:
                warnings.warn(
                    f"Channel name '{ch['ch_name']}' does not match expected format"
                )
                bad_records.add(bids_path.fpath)
                continue

            ch1 = match.group("ch1")
            ch2 = match.group("ch2")
            extra = match.group("extra")

            if ch1 not in upper_ch_names or ch2 not in upper_ch_names:
                warnings.warn(f"Channel '{ch1}-{ch2}' not found in montage")
                bad_records.add(bids_path.fpath)
                ch_types_map[ch["ch_name"]] = "eeg"
                continue

            new_ch_name = f"{upper_ch_names[ch1]}-{upper_ch_names[ch2]}{extra}"
            if new_ch_name != ch["ch_name"]:
                ch_names_map[ch["ch_name"]] = new_ch_name

            ch_types_map[ch["ch_name"]] = "eeg"
        raw.set_channel_types(ch_types_map)
        raw.rename_channels(ch_names_map)
        print(ch_names_map)
        print(ch_types_map)
        print(raw.info["ch_names"])

        # Annotations
        # https://github.com/mne-tools/mne-python/issues/12660
        # https://mne.discourse.group/t/how-to-load-edf-seizure-file-of-chb-mit-dataset/5829/6
        # if record in seizure_records:
        seizures_file = source_path.with_suffix(".edf.seizures")
        if seizures_file.exists():
            print(f"Found seizure annotation file {seizures_file}")
            seizure_files_counter += 1
            # annot_wfdb = wfdb.rdann(str(source_path).split(".")[0], "edf.seizures")
            annot_wfdb = wfdb.rdann(str(source_path), "seizures")
            sample = annot_wfdb.sample
            symbol = annot_wfdb.symbol
            assert sample is not None
            assert symbol is not None
            assert symbol == ["[", "]"] * (len(symbol) // 2)
            assert len(sample) == len(symbol)
            onset_sample = sample[::2]
            duration_sample = sample[1::2] - sample[::2]
            sfreq = raw.info["sfreq"]
            annot_mne = Annotations(
                onset=onset_sample / sfreq,
                duration=duration_sample / sfreq,
                description=["seizure"] * len(onset_sample),
            )
            assert raw.first_time == 0
            raw.set_annotations(annot_mne)

        # Subject info
        subject_id = bids_path.subject
        subject_row = subject_info.loc[subject_id]
        raw.info["subject_info"] = {
            "his_id": subject_id,
            "sex": int(subject_row["sex"]),
        }
        if subject_row["age"] is not None:
            bday = (
                raw.info["meas_date"]
                - subject_row["age"]
                - datetime.timedelta(days=70 * 365)
            )
            if not np.isnan(bday.year):
                raw.info["subject_info"]["birthday"] = datetime.datetime(
                    int(bday.year),
                    int(bday.month),
                    int(bday.day),
                    tzinfo=datetime.timezone.utc,
                )

        # Write BIDS file
        write_raw_bids(
            raw,
            bids_path=bids_path,
            overwrite=True,
            symlink=False,
            # Remove 70 years to all meas_date
            # Reason: Anonymized dates are too far in the future. Raises an error when saving BIDS
            anonymize={
                "daysback": 70 * 365,
                "keep_his": True,
            },
        )

    assert seizure_files_counter == 141

    if bad_records:
        warnings.warn(f"Some records had issues: {bad_records}")

    _finalize_dataset(bids_root, overwrite=overwrite)


def _finalize_dataset(bids_root: Path, overwrite: bool = False):
    # save script
    script_path = Path(__file__)
    script_dest = bids_root / "code" / script_path.name
    script_dest.parent.mkdir(exist_ok=True)
    shutil.copy2(script_path, script_dest)
    description_file = bids_root / "dataset_description.json"
    if description_file.exists() and overwrite:
        description_file.unlink()
    make_dataset_description(
        path=bids_root,
        name=DATASET_NAME,
        dataset_type="derivative",
        references_and_links=[
            "https://github.com/bernia/chb-mit-scalp",
        ],
        source_datasets=[
            {"URL": "https://physionet.org/content/chbmit/1.0.0/"},
        ],
        authors=["Pierre Guetschel"],
        overwrite=overwrite,
    )

    # cleanup macos hidden files
    for macos_file in bids_root.rglob("._*"):
        macos_file.unlink()

    report_str = make_report(bids_root)
    print(report_str)

    # overwrite README (include automatic report)
    readme_path = bids_root / "README.md"
    with open(readme_path, "w") as f:
        f.write(
            f"# {DATASET_NAME}\n\n{README_CONTENT}\n\n---\n\n"
            f"## Automatic report\n\n*Report automatically generated by `mne_bids.make_report()`.*\n\n> {report_str}"
        )

    # Remove participants.json if it exists
    participants_json = bids_root / "participants.json"
    if participants_json.exists():
        participants_json.unlink()
        print(f"Removed {participants_json}")

    # Clean up participants.tsv by removing columns where all values are "n/a"
    participants_tsv = bids_root / "participants.tsv"
    if participants_tsv.exists():
        df = pd.read_csv(participants_tsv, sep="\t")
        # Find columns where all non-participant_id values are "n/a"
        cols_to_drop = []
        for col in df.columns:
            if col != "participant_id" and (df[col] == "n/a").all():
                cols_to_drop.append(col)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            df.to_csv(participants_tsv, sep="\t", index=False)
            print(
                f"Removed columns with all 'n/a' values from {participants_tsv}: {cols_to_drop}"
            )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
    # python bids_maker/datasets/shoeb2009.py --source_root ~/data/chbmit/ --bids_root ~/data/bids/chbmit/ --overwrite
