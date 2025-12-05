import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import nibabel as nib
import re
import logging

sys.path.append(str(Path(__file__).parent.parent))
from src.preprocessing.mri_processor import MRIPreprocessor
from src.utils.logger import setup_logger


def find_subjects(raw_dirs):
    subjects = []
    for raw_dir in raw_dirs:
        if raw_dir.exists():
            subjects.extend([
                d for d in raw_dir.iterdir()
                if d.is_dir() and d.name.startswith('OAS2_')
            ])
    return sorted(subjects)


def find_mri_files(subject_dir):
    raw_dir = subject_dir / 'RAW'
    if not raw_dir.exists():
        return []
    img_files = sorted(raw_dir.glob('mpr-*.nifti.img'))
    return [f for f in img_files if f.with_suffix('.hdr').exists()]


def load_nifti_pair(img_path):
    try:
        img = nib.load(str(img_path))
        return img.get_fdata()
    except Exception:
        return None


def extract_subject_info(subject_dir):
    """
    subject_dir.name example: OAS2_0001_MR2
    returns subject_id 'OAS2_0001', visit 2
    """
    m = re.match(r"(OAS2_\d+)_MR(\d+)", subject_dir.name)
    if m:
        return m.group(1), int(m.group(2))
    else:
        raise ValueError(f"Invalid folder name: {subject_dir.name}")


def preprocess_oasis2(raw_dirs, output_dir, config):
    logger = setup_logger('preprocess', 'logs/preprocess.log')
    subjects = find_subjects(raw_dirs)

    if not subjects:
        logger.error("No subjects found")
        return None, None

    logger.info(f"Found {len(subjects)} subject directories")

    preprocessor = MRIPreprocessor(
        target_shape=tuple(config['preprocessing']['target_shape']),
        normalize=config['preprocessing']['normalize']
    )

    processed_scans = []
    metadata = []

    for subject_dir in tqdm(subjects):
        try:
            subject_id, visit = extract_subject_info(subject_dir)
        except Exception as e:
            logger.error(e)
            continue

        img_files = find_mri_files(subject_dir)
        if not img_files:
            continue

        img_file = img_files[0]

        try:
            raw_data = load_nifti_pair(img_file)
            if raw_data is None:
                continue

            temp_nii = output_dir / 'temp' / f"{subject_dir.name}.nii"
            temp_nii.parent.mkdir(parents=True, exist_ok=True)

            nii_img = nib.Nifti1Image(raw_data, affine=np.eye(4))
            nib.save(nii_img, str(temp_nii))

            mri_data = preprocessor.preprocess(str(temp_nii))
            temp_nii.unlink()

            processed_scans.append(mri_data)
            metadata.append({
                'subject_id': subject_id,   # e.g. OAS2_0001
                'visit': visit,            # int visit number
                'folder': subject_dir.name,
                'path': str(img_file)
            })

        except Exception as e:
            logger.error(f"Failed {subject_dir.name}: {e}")

    X = np.array(processed_scans)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'mri_scans.npy', X)
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)

    logger.info(f"Saved {len(X)} scans")
    return X, metadata_df


# -------- Demographics helpers --------

def _find_column(df_columns, candidates):
    for c in candidates:
        if c in df_columns:
            return c
    return None


def load_demographics(path='data/raw/oasis_longitudinal_demographics.xlsx'):
    xlsx_path = Path(path)
    if not xlsx_path.exists():
        print(f"Demographics not found at {xlsx_path}")
        return None

    df = pd.read_excel(xlsx_path)
    df.columns = df.columns.str.strip()

    id_candidates = ['ID', 'RID', 'Subject', 'OASISID', 'Subject ID', 'SUBJECT_ID']
    visit_candidates = ['Visit', 'visit', 'VISIT', 'MRI', 'MR', 'MR_ID']

    id_col = _find_column(df.columns, id_candidates)
    visit_col = _find_column(df.columns, visit_candidates)

    if id_col is None:
        logging.getLogger('preprocess').warning(
            f"No ID column found in demographics. Available columns: {list(df.columns)}"
        )
    else:
        df[id_col] = df[id_col].astype(str).str.strip()

    if visit_col is None:
        logging.getLogger('preprocess').warning(
            f"No Visit column found in demographics. Available columns: {list(df.columns)}"
        )
    else:
        try:
            df[visit_col] = df[visit_col].astype(int)
        except Exception:
            df[visit_col] = df[visit_col].astype(str).str.extract(r'(\d+)').astype(float).astype('Int64')

    if 'CDR' not in df.columns:
        df['CDR'] = 0.0

    df['label'] = df['CDR'].apply(lambda x: 1 if pd.notna(x) and float(x) >= 0.5 else 0)

    df.attrs['id_col'] = id_col
    df.attrs['visit_col'] = visit_col

    return df


def match_labels(metadata_df, demographics_df):

    if demographics_df is None:
        return np.zeros(len(metadata_df), dtype=int)

    id_col = demographics_df.attrs.get('id_col', None)
    visit_col = demographics_df.attrs.get('visit_col', None)

    meta = metadata_df.copy()
    meta['subject_num'] = meta['subject_id'].str.replace("OAS2_", "", regex=False).astype(int)

    if id_col is not None:
        demo = demographics_df.copy()

        sample_vals = demo[id_col].dropna().astype(str).head(20).tolist()

        has_oasis_prefix = any(s.startswith('OAS2_') for s in sample_vals)
        all_numeric = all(re.fullmatch(r'\d+', s) for s in sample_vals if s is not None and s != '')

        if has_oasis_prefix:
            if visit_col is not None:
                merged = meta.merge(
                    demo,
                    left_on=['subject_id', 'visit'],
                    right_on=[id_col, visit_col],
                    how='left'
                )
            else:
                logging.getLogger('preprocess').warning("No visit column in demographics: merging by subject only")
                merged = meta.merge(demo, left_on='subject_id', right_on=id_col, how='left')
        elif all_numeric:
            demo[id_col] = demo[id_col].astype(int)
            if visit_col is not None:
                merged = meta.merge(
                    demo,
                    left_on=['subject_num', 'visit'],
                    right_on=[id_col, visit_col],
                    how='left'
                )
            else:
                logging.getLogger('preprocess').warning("No visit column in demographics: merging by subject_num only")
                merged = meta.merge(demo, left_on='subject_num', right_on=id_col, how='left')
        else:
            try:
                if visit_col is not None:
                    merged = meta.merge(demo, left_on=['subject_id', 'visit'], right_on=[id_col, visit_col], how='left')
                else:
                    merged = meta.merge(demo, left_on='subject_id', right_on=id_col, how='left')
            except KeyError as e:
                raise KeyError(f"Could not merge using detected columns: {e}. Demographics columns: {list(demo.columns)}")
    else:
        raise KeyError(f"No ID-like column found in demographics. Available columns: {list(demographics_df.columns)}")

    labels = merged.get('label', pd.Series(0, index=merged.index)).fillna(0).astype(int).values

    logger = logging.getLogger('preprocess')
    logger.info(f"Matched labels: {len(labels)} (should equal number of MRI scans)")
    logger.info(f"Label counts -> Normal: {(labels==0).sum()}, Dementia: {(labels==1).sum()}")

    return labels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/processed')
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    raw_dirs = [
        Path('data/raw/OAS2_RAW_PART1'),
        Path('data/raw/OAS2_RAW_PART2')
    ]

    print("Processing OASIS-2 dataset...")
    X, metadata_df = preprocess_oasis2(raw_dirs, Path(args.output_dir), config)

    if X is None:
        print("No data processed")
        return

    demographics_df = load_demographics()
    labels = match_labels(metadata_df, demographics_df)

    np.save(Path(args.output_dir) / 'labels.npy', labels)

    print(f"\nComplete: {len(X)} samples")
    print(f"Normal: {sum(labels == 0)}, Dementia: {sum(labels == 1)}")


if __name__ == '__main__':
    main()
