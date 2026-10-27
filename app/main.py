"""Main module."""

import flywheel
import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.format import simplify_label
try:
    from app.session_level import flatten_session_level, write_outputs as write_session_outputs
except ImportError:  # running outside the gear package layout
    from session_level import flatten_session_level, write_outputs as write_session_outputs

"""
Tag files for visual QC reader tasks in Flywheel
See README.md for usage & license

Author: Niall Bourke & Hajer Karoui
"""


log = logging.getLogger(__name__)
work_dir = Path('/flywheel/v0/work', platform='auto')
out_dir = Path('/flywheel/v0/output', platform='auto')


# --- Mappings ---

ANSWER_TO_QC_STATUS = {
    0: 'pass',
    1: 'unclear',
    2: 'fail',
    "0" : 'pass',
    "1" : 'unclear',
    "2" : 'fail',
}

QC_STATUS_TO_TAG = {
    'pass':    'QC-passed',
    'unclear': 'QC-unclear',
    'fail':    'QC-failed',
}

SESSION_TAG_MAP = {
    'pass':    'T2w_QC_passed',
    'unclear': 'T2w_QC_unclear',
    'fail':    'T2w_QC_failed',
}

# The session-based "Clinical QC" protocol and the acquisition-based "QC" protocol can both be
# run against the SAME acquisitions/session (e.g. one reviewer does clinical reporting + visual
# QC via "Clinical QC", another does visual-QC-only via "QC"). If both wrote the exact same tag
# strings, the second one to run would silently overwrite/clash with the first. Tags coming from
# the session-based protocol are therefore prefixed with "Clinical-" / "Clinical_" so the two
# protocols' tags never collide and both remain visible on the acquisition/session.
CLINICAL_QC_STATUS_TO_TAG = {
    'pass':    'Clinical-QC-passed',
    'unclear': 'Clinical-QC-unclear',
    'fail':    'Clinical-QC-failed',
}

CLINICAL_SESSION_TAG_MAP = {
    'pass':    'Clinical_T2w_QC_passed',
    'unclear': 'Clinical_T2w_QC_unclear',
    'fail':    'Clinical_T2w_QC_failed',
}

# Maps question suffix → orientation key used in the rest of the code
QUESTION_TO_ORIENTATION = {
    'quality_t2_axi':        'axi',
    'quality_t2_cor':        'cor',
    'quality_t2_sag':        'sag',
    'quality_t2_axi_repeat': 'axi_repeat',
    'quality_t1_axi':        't1_axi',
}

# Protocol name constants
PROTOCOL_SESSION_BASED     = "Clinical QC" # "clinical-reporting QC" #'Clinical QC'
PROTOCOL_ACQUISITION_BASED = 'QC'


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def apply_qc_tag(fw, obj, qc_tag: str, context: str) -> None:
    """Delete the 'read' tag and apply a QC tag to a Flywheel object."""
    try:
        #log.info(f"ID OF OBJ {obj._id} {type(obj)}")
        obj = fw.get(obj._id)
        obj = obj.reload()
        if 'read' in obj.tags:
            obj.delete_tag('read')
        obj.add_tag(qc_tag)
    except flywheel.ApiException as e:
        log.error(f'Error adding tag to {context}: {e}')

def get_nii_file(fw, acquisition):
    """Return the Flywheel file object for the first .nii.gz file in an acquisition."""
    for f in acquisition.files:
        if f.name.endswith('nii.gz'):
            print(f"Found .nii.gz file: {f.name} {f._id} in acquisition {acquisition.label}")
            return f #fw.files.find_first(f"id={f._id}")
        
    return None


def resolve_session_qc(axi: str, cor: str, sag: str) :
    """
    Determine the overall session QC status from the three orientation results.
    Priority: fail > unclear > pass (all-pass required for pass).
    """
    orientations = (axi, cor, sag)
    if all(o == 'pass' for o in orientations):
        return 'pass'
    if any(o == 'fail' for o in orientations):
        return 'fail'
    if any(o == 'unclear' for o in orientations):
        return 'unclear'
    return None


def get_orientation(label: str):
    """Infer orientation key from an acquisition label."""
    label_upper = label.upper()
    if 'MAPPING' not in label_upper and 'AXI' in label_upper:
        return 'axi'
    if 'COR' in label_upper:
        return 'cor'
    if 'SAG' in label_upper:
        return 'sag'
    return None


def tag_acquisition(fw, acquisition, qc_status: str, label_hint: str,
                     status_to_tag: dict = QC_STATUS_TO_TAG) -> None:
    """Apply QC tag to an acquisition and its .nii.gz file.

    `status_to_tag` lets callers use a different tag vocabulary (e.g. the "Clinical-" prefixed
    tags for the session-based protocol) so tags from different protocols never clash.
    """
    qc_tag = status_to_tag[qc_status]
    apply_qc_tag(fw, acquisition, qc_tag, f'acquisition ({label_hint})')

    nii_file = get_nii_file(fw, acquisition)
    if nii_file:
        try:
            nii_file.add_tag(qc_tag)
            #apply_qc_tag(fw, nii_file, qc_tag, f'file ({label_hint})')
        except flywheel.ApiException as e:
            log.error(f'Error adding tag to .nii.gz file in acquisition "{label_hint}": {e}')
    else:
        log.warning(f'No .nii.gz file found in acquisition "{label_hint}"')


# ---------------------------------------------------------------------------
# Session-based helpers  (protocol: "clinical-reporting QC")
# ---------------------------------------------------------------------------

def parse_form_responses(task_df) -> dict:
    """
    Pivot the long-form question/answer rows into a flat dict for easy lookup.

    Returns something like:
    {
        'quality_t2_axi':                2,
        'quality_t2_cor':                0,
        'quality_t2_sag':                0,
        'quality_t2_axi_repeat':         0,
        'quality_t1_axi':                0,
        'repeat_t2_axi':                 1,   # 1 = yes there is a repeat
        'repeat_t2_cor':                 0,
        'repeat_t2_sag':                 0,
        'repeat_t1_axi':                 0,
        'reported_series_t2_axi':        1,   # series index of the *primary* scan
        'reported_series_number_t2_axi': 10,  # actual DICOM series number
        'artifacts_t2_axi':              ['motion', 'zipper'],
        ...
    }
    """
    responses = {}
    for _, row in task_df.iterrows():
        responses[row['Question']] = row['Answer']
    return responses



# def get_dicom_series_number(fw, dicom_header) :
#     """
#     Read the SeriesNumber DICOM tag from the first .dcm (or DICOM zip) file
#     attached to an acquisition.

#     Strategy:
#       1. Find the first file whose type is 'dicom' (or name ends with '.dcm').
#       2. Call fw.get_file_zip_info() to list zip members.
#       3. Read the first member with fw.read_file_zip_member() and parse with pydicom.
#       4. Return ds.SeriesNumber as an int, or None on any failure.
#     """
    
#     dicom_header.info.get('SeriesNumber', "0")
    
#     import pydicom
#     import io

#     # Locate the DICOM file on the acquisition
#     dicom_file = None
#     for f in acquisition.files:
#         if f.type == 'dicom' or f.name.lower().endswith('.dcm'):
#             dicom_file = f
#             break

#     if dicom_file is None:
#         log.warning(f'No DICOM file found on acquisition "{acquisition.label}"')
#         return None

#     try:
#         zip_info = fw.get_file_zip_info(dicom_file.file_id)
#         if not zip_info.members:
#             log.warning(f'DICOM zip for "{acquisition.label}" has no members')
#             return None

#         # Read just the first slice — enough to get SeriesNumber
#         first_member = zip_info.members[0].path
#         raw = fw.read_file_zip_member(dicom_file.file_id, first_member)
#         ds = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True)
#         return int(ds.SeriesNumber)

#     except Exception as e:
#         log.error(f'Error reading DICOM SeriesNumber for "{acquisition.label}": {e}')
#         return None


def _label_matches_tokens(label: str, tokens: list[str]) -> bool:
    """
    Return True if ALL tokens are present in the acquisition label (case-insensitive).

    This is more robust than a single-string match because acquisition labels
    can vary in word order and separator style (e.g. "AXI_T2_Fast", "T2w AXI").

    Example:
        _label_matches_tokens("T2w_AXI_Fast", ["T2", "AXI"])  → True
        _label_matches_tokens("T1_AXI",       ["T2", "AXI"])  → False
    """
    upper = label.upper()
    return all(token.upper() in upper for token in tokens)

def find_acquisition_by_pattern(fw, session_acquisitions, tokens: list[str], series_number: int):
    """
    Search a list of Flywheel acquisition objects for one whose label contains
    ALL tokens in `tokens` (case-insensitive).

    If `series_number` is given AND multiple acquisitions match, disambiguate
    by reading the SeriesNumber from each acquisition's DICOM file.

    Returns the first matching acquisition, or None.
    """
    candidates = [
        acq for acq in session_acquisitions
        if _label_matches_tokens(acq.label, tokens)
    ]

    if not candidates:
        log.warning(f'No acquisitions found matching tokens {tokens}')
        return None

    # Only one match — no disambiguation needed
    if series_number is None or len(candidates) == 1:
        return candidates[0]

    # Multiple acquisitions match the label — disambiguate via DICOM SeriesNumber
    log.info(
        f'Multiple acquisitions match {tokens} '
        f'({[a.label for a in candidates]}); disambiguating by SeriesNumber={series_number}'
    )
    for acq in candidates:
        for file_obj in acq.files: # get the files in the acquisition
            # Screen file object information & download the desired file
            if file_obj['type'] == 'dicom':
                dicom_header = fw._fw.get_acquisition_file_info(acq.id, file_obj.name)
                acq_series = dicom_header.info.get('SeriesNumber', None)
                
        if acq_series is not None and acq_series == int(series_number):
            return acq

    log.warning(
        f'Could not match SeriesNumber={series_number} among candidates '
        f'{[a.label for a in candidates]}. Falling back to first match.'
    )
    return candidates[0]


def find_repeat_acquisition(fw, session_acquisitions, tokens: list[str], primary_series_number: int):
    """
    Among all acquisitions whose label contains ALL tokens, return the one
    whose DICOM SeriesNumber does NOT match `primary_series_number`.

    This isolates the repeat scan once the primary has been identified by its
    series number.
    """
    candidates = [
        acq for acq in session_acquisitions
        if _label_matches_tokens(acq.label, tokens)
    ]

    if len(candidates) <= 1:
        log.warning(
            f'Expected multiple acquisitions for repeat lookup of {tokens}, '
            f'found {len(candidates)}.'
        )
        return None

    for acq in candidates:
        for file_obj in acq.files: # get the files in the acquisition
            # Screen file object information & download the desired file
            if file_obj['type'] == 'dicom':
                dicom_header = fw._fw.get_acquisition_file_info(acq.id, file_obj.name)
                acq_series = dicom_header.info.get('SeriesNumber', None)
                
        if acq_series is not None and int(acq_series) != int(primary_series_number):
            return acq

    log.warning(f'Could not identify repeat acquisition for tokens {tokens}.')
    return None



# ---------------------------------------------------------------------------
# Protocol implementations
# ---------------------------------------------------------------------------

# Configuration for the session-based protocol orientations.
# Extend this dict to support additional orientations or modalities.
ORIENTATION_CONFIG = {
    'axi': {
        'quality_key':        'quality_t2_axi',
        'repeat_key':         'repeat_t2_axi',
        'repeat_quality_key': 'quality_t2_axi_repeat',
        'series_key':         'reported_series_number_t2_axi',
        'label_tokens': ['T2', 'AXI']  # matches if label contains BOTH "T2" AND "AXI", in any order/format
    },
    'cor': {
        'quality_key':        'quality_t2_cor',
        'repeat_key':         'repeat_t2_cor',
        'repeat_quality_key': None,
        'series_key':         None,
        'label_tokens': ['T2', 'COR']  # matches if label contains BOTH "T2" AND "AXI", in any order/format
    },
    'sag': {
        'quality_key':        'quality_t2_sag',
        'repeat_key':         'repeat_t2_sag',
        'repeat_quality_key': None,
        'series_key':         None,
        'label_tokens': ['T2', 'SAG']  # matches if label contains BOTH "T2" AND "AXI", in any order/format
    },
    
    't1_axi': {
        'quality_key':        'quality_t1_axi',
        'repeat_key':         'repeat_t1_axi',
        'repeat_quality_key': None,
        'series_key':         None,
        'label_tokens':       ['T1', 'AXI'],
    },
}


def _process_task_session_based(task_df, fw):
    """
    Process a session-scoped QC form (protocol: "clinical-reporting QC").

    The CSV format has:
      - No acquisition.id — the task is linked to a session only.
      - One row per question/answer pair (long format).
      - Quality questions named like: quality_t2_axi, quality_t2_cor, quality_t2_sag,
        quality_t2_axi_repeat, quality_t1_axi.
      - Repeat flags: repeat_t2_axi / repeat_t2_cor / repeat_t2_sag (1 = repeat exists).
      - reported_series_number_t2_axi: the DICOM series number to disambiguate acquisitions.
    """
    first_row     = task_df.iloc[0]
    session_id    = first_row['session.id']
    subject_label = first_row['Subject Label']
    session_label = first_row['Session Label']

    log.info(f"[session-based] Processing session {session_label} for subject {subject_label}")

    responses = parse_form_responses(task_df)

    try:
        session              = fw.get(session_id)
        session_acquisitions = session.acquisitions()
    except flywheel.ApiException as e:
        log.error(f'Error fetching session {session_id}: {e}')
        return

    orientations = {'axi': None, 'cor': None, 'sag': None}

    for orientation, cfg in ORIENTATION_CONFIG.items():
        quality_answer = responses.get(cfg['quality_key'])
        if quality_answer is None:
            log.warning(f"No quality answer found for {cfg['quality_key']} — skipping.")
            continue

        qc_status = ANSWER_TO_QC_STATUS.get(quality_answer)
        if qc_status is None:
            log.warning(f"Unknown answer value {quality_answer!r} for {cfg['quality_key']} — skipping.")
            continue

        orientations[orientation] = qc_status
        has_repeat    = bool(responses.get(cfg['repeat_key'], 0))
        series_number = responses.get(cfg['series_key']) if cfg['series_key'] else None

        # --- Tag the primary acquisition ---
        primary_acq = find_acquisition_by_pattern(
            fw, session_acquisitions, cfg['label_tokens'], series_number
        )
        if primary_acq:
            log.info(f"[{orientation}] Primary acquisition: {primary_acq.label} → {qc_status}")
            tag_acquisition(fw, primary_acq, qc_status, primary_acq.label,
                             status_to_tag=CLINICAL_QC_STATUS_TO_TAG)
        else:
            log.warning(f"[{orientation}] No primary acquisition found for tokens '{cfg['label_tokens']}'")

        # --- Tag the repeat acquisition (if present) ---
        if has_repeat and cfg['repeat_quality_key']:
            repeat_answer = responses.get(cfg['repeat_quality_key'])
            repeat_status = ANSWER_TO_QC_STATUS.get(repeat_answer)
            if repeat_status is None:
                log.warning(f"No valid repeat quality answer for {cfg['repeat_quality_key']} — skipping repeat.")
            else:
                # The repeat is the *other* T2 AXI — the one whose SeriesNumber
                # does NOT match the primary's reported series number.
                repeat_acq = find_repeat_acquisition(
                    fw, session_acquisitions, cfg['label_tokens'], series_number
                )
                if repeat_acq:
                    log.info(f"[{session_label} | {subject_label}] [{orientation}] Repeat acquisition: {repeat_acq.label} → {repeat_status}")
                    tag_acquisition(fw, repeat_acq, repeat_status, repeat_acq.label,
                                     status_to_tag=CLINICAL_QC_STATUS_TO_TAG)
                else:
                    log.warning(f"[{session_label} | {subject_label}] [{orientation}] No repeat acquisition found for token '{cfg['label_tokens']}'")

    # --- Session-level tag ---
    # Uses the "Clinical_" prefixed tag vocabulary (CLINICAL_SESSION_TAG_MAP) - and only ADDS a
    # tag, never deletes existing ones - so this never clashes with / wipes out a session tag set
    # by the separate acquisition-based "QC" protocol handler for the same session.
    session_qc = resolve_session_qc(
        orientations['axi'], orientations['cor'], orientations['sag']
    )
    if session_qc:
        session_tag = CLINICAL_SESSION_TAG_MAP[session_qc]
        log.info(f"Session QC → {session_tag} for subject {subject_label}")
        try:
            session.add_tag(session_tag)
        except flywheel.ApiException as e:
            log.error(f'Error adding session tag: {e}')
    else:
        log.warning(f"Could not determine session QC status for {subject_label} — not all orientations resolved.")

    print("***")
    print(f"Visual QC report | subject: {subject_label} | session: {session_label}")
    print(f"  T2w_axi: {orientations['axi']}  T2w_cor: {orientations['cor']}  T2w_sag: {orientations['sag']}  T2w_all: {session_qc}")
    print("***")


def _process_task_acquisition_based(task_df, fw):
    """
    Process the legacy per-acquisition QC form (protocol: "QC").

    The CSV format has:
      - One row per acquisition, with acquisition.id present.
      - A single 'Question' == "quality" column with Answer 0/1/2.
    """
    axi = cor = sag = t2_qc = None

    for _, row in task_df.iterrows():
        if row['Question'] != 'quality':
            continue

        acquisition_id = row['acquisition.id']
        try:
            acquisition = fw.get(acquisition_id)
        except flywheel.ApiException as e:
            log.error(f'Error getting acquisition {acquisition_id}: {e}')
            continue

        qc_status = ANSWER_TO_QC_STATUS.get(row['Answer'])
        if qc_status is None:
            log.warning(f"Unknown answer {row['Answer']!r} for acquisition {acquisition_id} — skipping.")
            continue

        log.info(f"QC-{qc_status} {row['Subject Label']} {row['Acquisition Label']} {acquisition_id}")

        orientation = get_orientation(row['Acquisition Label'])
        if orientation:
            if orientation == 'axi':
                axi = qc_status
            elif orientation == 'cor':
                cor = qc_status
            elif orientation == 'sag':
                sag = qc_status

        tag_acquisition(fw, acquisition, qc_status, row['Acquisition Label'])

    # --- Session-level tag ---
    session_qc = resolve_session_qc(axi, cor, sag)
    if session_qc:
        session_tag = SESSION_TAG_MAP[session_qc]
        last_row      = task_df.iloc[-1]
        subject_label = last_row['Subject Label']
        session_label = last_row['Session Label']
        log.info(f"Session QC → {session_tag} for subject {subject_label}")
        try:
            session = fw.get(last_row['session.id'])
            # Only remove this protocol's OWN previous QC tags (not e.g. the "Clinical_" tags
            # that the separate session-based "Clinical QC" protocol may have set on this same
            # session) - otherwise running both protocols against one session would make
            # whichever one runs last wipe out the other's result.
            for tag in list(session.tags):
                if tag in SESSION_TAG_MAP.values():
                    log.info(f"Removing tag {tag} from session {session.id}")
                    session.delete_tag(tag)
            session.add_tag(session_tag)
        except flywheel.ApiException as e:
            log.error(f'Error adding session tag: {e}')

        print("***")
        print(f"Visual QC report | subject: {subject_label} | session: {session_label}")
        print(f"  T2w_axi: {axi}  T2w_cor: {cor}  T2w_sag: {sag}  T2w_all: {session_qc}")
        print("***")


# ---------------------------------------------------------------------------
# Dispatcher — single public entry point
# ---------------------------------------------------------------------------

PROTOCOL_HANDLERS = {
    PROTOCOL_SESSION_BASED:     _process_task_session_based,
    PROTOCOL_ACQUISITION_BASED: _process_task_acquisition_based,
}


def process_task(task_df, fw):
    """
    Route a task dataframe to the correct processing function based on protocolName.

    Supported protocols
    -------------------
    'clinical-reporting QC'  →  session-scoped, long-form questions, no acquisition.id
    'QC'                     →  legacy per-acquisition rows with acquisition.id
    """
    protocol = task_df.iloc[0].get('Protocol Name', '')
    handler  = PROTOCOL_HANDLERS.get(protocol)

    if handler is None:
        raise ValueError(
            f"Unknown protocolName '{protocol}'. "
            f"Expected one of: {list(PROTOCOL_HANDLERS.keys())}"
        )

    handler(task_df, fw)


def run_tagger(context, api_key):

    """Run the algorithm defined in this gear.

    Args:
        context (GearContext): The gear context.
        api_key (str): The API key generated for this gear run.

    Returns:
        int: The exit code.
    """

    fw = flywheel.Client(api_key=api_key)

    # Get the destination container and project ID
    destination_container = context.client.get_analysis(context.destination["id"])
    # destination_container = context.client.get(context.destination["id"]) # Change in SDK 18.3.0
    dest_proj_id = destination_container.parents["project"]
    project_container = context.client.get(dest_proj_id)
    project_label = project_container.label

    project  = fw.projects.find_one(f'label={project_label}')
    project = project.reload()

    # Specify the gear name to search for
    gear='form-and-annotations-exporter'

    # Get the project object
    project_container = context.client.get(dest_proj_id)
    project = project_container.reload()
    analyses = project.analyses

    # Filter the analyses by the gear name
    gear_to_find = gear.strip()  # Assuming 'gear' is the gear name you're looking for
  
    filtered_gear_runs =[]
    file_object = None
    for run in analyses:
        try:
            if run is not None and run.get('gear_info') is None and gear in run.label:
                filtered_gear_runs.append(run)

            if run is not None and run.get('gear_info', {}).get('name', '').strip().casefold() == gear_to_find.casefold() and run.reload().job.get('state')=="complete":
                filtered_gear_runs.append(run)
        except Exception as e:
            log.error(f'Exception caught  {e} {run}')

    # Get the latest gear run
    latest_gear_run = None
    if filtered_gear_runs:
        latest_gear_run =  filtered_gear_runs[-1]
    

    if latest_gear_run is not None:
        file_object = latest_gear_run.files
        #log.info(f"File object:  {file_object[0].name}")
    else:
        log.error(f"{gear} was not run. Please run the {gear} first before running form-parser.")
        return 0

    # Create a work directory in our local "home" directory
    work_dir = Path('/flywheel/v0/work', platform='auto')
    out_dir = Path('/flywheel/v0/output', platform='auto')

    # If it doesn't exist, create it
    if not work_dir.exists():
        work_dir.mkdir(parents = True)

    # If it doesn't exist, create it
    if not out_dir.exists():
        out_dir.mkdir(parents = True)


    download_path = work_dir/file_object[0].name
    file_object[0].download(download_path)
    df = pd.read_csv(download_path)

    df = pd.read_csv(download_path)

    # A single session can have form responses from BOTH protocols (e.g. one reviewer doing
    # clinical reporting + visual QC via "Clinical QC", another doing visual-QC-only via "QC").
    # Grouping by Session Label alone would mix both protocols' rows into one task_df, and
    # process_task() only looks at the FIRST row's 'Protocol Name' to decide which handler to
    # use - silently processing just one of the two and dropping the other's answers entirely.
    # Group by (Session Label, Protocol Name) instead so each protocol's rows for a session are
    # always handled separately.
    if 'Protocol Name' in df.columns:
        group_cols = ['Session Label', 'Protocol Name']
    else:
        group_cols = ['Session Label']

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _, task_df in df.groupby(group_cols, sort=False):
            futures.append(executor.submit(process_task, task_df.copy(), fw))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log.error(f"Error in parallel execution: {e}")
    
    return 1


def run_csv_parser(context, api_key):

    """Run the algorithm defined in this gear.

    Args:
        context (GearContext): The gear context.
        api_key (str): The API key generated for this gear run.

    Returns:
        int: The exit code.
    """

    # Create a work directory in our local "home" directory
    work_dir = Path('/flywheel/v0/work', platform='auto')
    out_dir = Path('/flywheel/v0/output', platform='auto')

    # If it doesn't exist, create it
    if not work_dir.exists():
        work_dir.mkdir(parents = True)

    # If it doesn't exist, create it
    if not out_dir.exists():
        out_dir.mkdir(parents = True)

    

    # Get the destination container and project ID

    destination_container = context.client.get_analysis(context.destination["id"])
    dest_proj_id = destination_container.parents["project"]

    # Specify the gear name to search for
    gear='form-and-annotations-exporter'

    # Get the project object
    project_container = context.client.get(dest_proj_id)
    project = project_container.reload()
    analyses = project.analyses
    
    
    # Filter the analyses by the gear name
    gear_to_find = gear.strip()  # Assuming 'gear' is the gear name you're looking for
    filtered_gear_runs =[]

    for run in analyses:
        try:
            if run.get('gear_info', {}).get('name', '').strip() == gear_to_find:
                filtered_gear_runs.append(run)
                
            # #take into account analysis containers not just gears
            # if run is not None and run.get('gear_info') is None and gear in run.label:
            #      filtered_gear_runs.append(run)
            # if run is not None and run.get('gear_info', {}).get('name', '').strip().casefold() == gear_to_find.casefold() and run.reload().job.get('state')=="complete":
            #     filtered_gear_runs.append(run)
        except Exception as e:
            log.info(f'Exception caught {e}')

    # Get the latest gear run
    latest_gear_run = None
    if filtered_gear_runs:
        latest_gear_run =  filtered_gear_runs[-1]
        #print(latest_gear_run)

    if latest_gear_run is not None:
        latest_gear_run = latest_gear_run.reload()
        file_object = latest_gear_run.files
        log.info(f"File object: {file_object[0].name}")

        download_path = work_dir/file_object[0].name
        try:
            file_object[0].download(download_path)
            log.info(f"Downloaded the file {download_path}")
        except Exception as e:
            log.info(f"Caught exception {e}")
        raw_df = pd.read_csv(download_path)
        
        log.info(f"Unique subjecst: {raw_df['Subject Label'].nunique()}")

        # Step 2: Select columns by name
        df = raw_df[['Project Label', 'Subject Label', 'Session Label', 'Session Timestamp','Acquisition Label', 'Question', 'Answer']]
    
    else:
        log.error(f"{gear_to_find} was not run.")
        log.info("No file was downloaded, getting all acquisitions in the current project.")
        # Get all sessions in the project
        sessions = project_container.sessions()
        acquisitions = []

        for session in sessions:
            for acquisition in session.acquisitions():
                acquisitions.append({
                    'Project Label': project_container.label,
                    'Subject Label': session.subject.label,
                    'Session Label': session.label,
                    'Session Timestamp': session.timestamp,
                    'Acquisition Label': acquisition.label,
                    'Question': '',
                    'Answer': ''
                })

        df = pd.DataFrame(acquisitions)
        df.to_csv('/flywheel/v0/work/filtered_file.csv', index=False)

        return (0, os.path.join(context.output_dir, 'filtered_file.csv'))

    
        
    # Step 3: Save the filtered DataFrame to a new CSV (optional)
    df.to_csv('/flywheel/v0/work/filtered_file.csv', index=False)

    # --- Session-level ("Clinical QC") form responses: flatten separately -------------------
    # These rows have no acquisition label, so the file-level pivot below cannot handle them.
    session_df = None
    if 'Protocol Name' in raw_df.columns:
        session_mask = raw_df['Protocol Name'] == PROTOCOL_SESSION_BASED
        if session_mask.any():
            log.info(f"Flattening {raw_df.loc[session_mask, 'Session Label'].nunique()} session-level QC form responses.")
            session_df, session_issues = flatten_session_level(raw_df.loc[session_mask])
            # The clinical free-text answers make pandas read the whole 'Answer' column as text; the
            # file-level relabelling below expects numbers (0/1/2), so convert the remaining rows back.
            df = df.loc[~session_mask].copy()
            df['Answer'] = pd.to_numeric(df['Answer'], errors='coerce')
            if not session_issues.empty:
                issues_path = os.path.join(context.output_dir, 'parse_issues.csv')
                session_issues.to_csv(issues_path, index=False)
                log.warning(f"{len(session_issues)} parsing issues written to {issues_path}")
            if df.empty:  # only session-level data in this project
                if session_df.empty:
                    return (1, "No data to export. The session-level DataFrame is empty.")
                out_path, _ = write_session_outputs(session_df, None, context.output_dir)
                log.info(f'Parsed QC data saved to: {out_path}')
                return (0, out_path)

        
    # Apply the function to simplify acquisition labels using .loc
    #df.loc[:, 'Acquisition Label'] = df['Acquisition Label'].apply(simplify_label)
    # After — NaN rows are left as-is, only valid labels are simplified:
    mask = df['Acquisition Label'].notna()
    df.loc[mask, 'Acquisition Label'] = df.loc[mask, 'Acquisition Label'].apply(simplify_label)

    # 'Session Timestamp' is sometimes missing for a session in the export. groupby/pivot_table
    # drop any group whose key is NaN by default, which silently removes that whole session from
    # the output even though it has real QC answers. Use a placeholder while pivoting and restore
    # blanks afterwards so sessions without a timestamp are not dropped.
    TS_PLACEHOLDER = '__NO_TIMESTAMP__'
    df['Session Timestamp'] = df['Session Timestamp'].fillna(TS_PLACEHOLDER)

    # Step 4: Filter for 'quality' questions and pivot
    quality_df = df[df['Question'] == 'quality']
    pivot_quality_df = quality_df.pivot_table(index=['Subject Label', 'Session Label','Session Timestamp'], 
                                            columns='Acquisition Label', 
                                            values='Answer', 
                                            aggfunc='first').reset_index()

    # Add prefix to the columns to identify them as quality checks
    pivot_quality_df.columns = [f'quality_{col}' if isinstance(col, str) else col for col in pivot_quality_df.columns]

    # NOTE: explaination of the code above
    # [] denotes a list comprehension, which is a concise way to create lists in Python
    # for col in pivot_quality_df.columns iterates over the columns in the DataFrame
    # if isinstance(col, str) checks if the column name is a string
    # f'quality_{col}' is an f-string, which allows for string interpolation (adds the prefix 'quality_' to the column name)
    # else col is used to keep the original column name if it's not a string


    # Step 5: Filter for additional questions and pivot
    expected_columns = ['banding', 'contrast', 'fov', 'motion', 'noise', 'other', 'zipper']

    additional_questions_df = df[df['Question'] != 'quality']
    additional_questions_df = additional_questions_df.pivot_table(index=['Subject Label', 'Session Label', 'Session Timestamp','Acquisition Label'], 
                                                                columns='Question', 
                                                                values='Answer', 
                                                                aggfunc='first').reset_index()

    
    available_values = [col for col in expected_columns if col in additional_questions_df.columns]

    for col in expected_columns:
        if col not in available_values:
            additional_questions_df[col] = np.nan * len(additional_questions_df)

    # Any question beyond the known artifact types (e.g. a free-text or categorical
    # question added to the file-level form) is still exported, as its own column per
    # acquisition, instead of being silently dropped by restricting `values` to
    # `expected_columns` only.
    id_cols_aq = ['Subject Label', 'Session Label', 'Session Timestamp', 'Acquisition Label']
    extra_question_cols = [c for c in additional_questions_df.columns
                            if c not in id_cols_aq and c not in expected_columns]
    value_columns = expected_columns + extra_question_cols

    additional_questions_wide = additional_questions_df.pivot(
        index=['Subject Label', 'Session Label','Session Timestamp'], 
        columns=['Acquisition Label'], 
        values=value_columns
    ).reset_index()

    
    #additional_questions_wide = additional_questions_wide.fillna(0)
    # # Flatten the multi-level columns for additional questions
    # additional_questions_wide = additional_questions_df.pivot(index=['Subject Label', 'Session Label'], 
    #                                                         columns=['Acquisition Label'], 
    #                                                         values=expected_columns).reset_index()

    # Flatten the multi-level columns
    additional_questions_wide.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in additional_questions_wide.columns]
    #print(pivot_quality_df.columns)
    #print(additional_questions_wide.columns)

    # Step 6: Merge the quality checks with additional questions
    merged_df = pd.merge(pivot_quality_df, additional_questions_wide, left_on=['quality_Subject Label', 'quality_Session Label','quality_Session Timestamp'], right_on=['Subject Label_', 'Session Label_','Session Timestamp_'], how='left')

    # Rename columns for clarity
    merged_df.rename(columns={'quality_Subject Label': 'Subject Label', 'quality_Session Label': 'Session Label','quality_Session Timestamp':'Session Timestamp'}, inplace=True)

    # Remove the redundant 'Subject Label_' column
    merged_df.drop(columns=['Subject Label_', 'Session Label_','Session Timestamp_'], inplace=True)

    # Step 7: Relabel the values
    # for col in merged_df.columns[2:]:
    #     merged_df[col] = merged_df[col].astype('Int64') 

    relabel_map = {0: 'good', 1: 'unsure', 2: 'bad', 
               0.0: 'good', 1.0: 'unsure', 2.0: 'bad'}
    merged_df.replace(relabel_map, inplace=True)

    
    #merged_df = merged_df.dropna(subset=[col for col in merged_df.columns if col.startswith("quality")], how="any")

    # Step 8: Check if answers to quality_AXI, quality_COR, quality_SAG are missing, all 'good', or any 'unsure', and set QC_all
    contrasts = ["T1","T2"]
    for contrast in contrasts:
        filters= f'quality_{contrast}|quality_AXI_{contrast}|quality_SAG_{contrast}|quality_COR_{contrast}'
        df_filtered = merged_df.filter(regex=filters)
        qc_columns = df_filtered.columns.tolist()

        # Function is defined here to determine the QC status
        def determine_qc_status(row):
            if row[qc_columns].isnull().any() and contrast  == "T2":
                return 'incomplete'
            elif all(val == 'good' for val in row[qc_columns].fillna('good')):
                return 'passed'
            elif any(val == 'unsure' for val in row[qc_columns].fillna('good')) and all(val in ['good', 'unsure'] for val in row[qc_columns].fillna('good')):
                return 'unsure'
            elif any(val == 'good' for val in row[qc_columns])  and contrast == "T1":
                return 'passed'
            else:
                return 'failed'

        # Function is applied here & creates a new column 'QC_all' with the status

        try:
            if qc_columns:
                merged_df[f'QC_all_{contrast}'] = df_filtered.apply(determine_qc_status, axis=1)
        except Exception as e:
            log.error(f'Error applying QC_all_{contrast} function: {e}')
  
    # Get the current date and time
    now = datetime.now(pytz.utc)
    formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    if session_df is not None and not session_df.empty:
        # A single session can have answers from BOTH the file-level 'QC' task (acquisition
        # quality ratings, in merged_df) and the session-level 'Clinical QC' task (clinical
        # findings, reviewer type, etc., in session_df). It can ALSO have MORE THAN ONE
        # session-level row, because the session-level form can be filled in independently by
        # more than one reviewer (e.g. one reviewer doing clinical reporting + visual QC, another
        # doing visual-QC-only) - flatten_session_level() now gives each distinct 'reviewer_type'
        # its own row. A plain groupby(...).first() would collapse those distinct reviewer rows
        # back into a single row and silently throw away one reviewer's answers again, so instead
        # we enrich EACH session-level/reviewer row with the file-level answers (filling only the
        # columns it left blank), while keeping every reviewer's row separate. Sessions that only
        # have file-level data (no session-level form response at all) are kept as-is.
        file_level_repr = (
            merged_df.groupby(['Subject Label', 'Session Label'], as_index=False, dropna=False)
            .first()
        )
        common_cols = [c for c in file_level_repr.columns
                       if c in session_df.columns and c not in ('Subject Label', 'Session Label')]

        session_enriched = pd.merge(
            session_df, file_level_repr,
            on=['Subject Label', 'Session Label'], how='left', suffixes=('', '_filelevel')
        )
        for col in common_cols:
            fl_col = f'{col}_filelevel'
            if fl_col in session_enriched.columns:
                session_enriched[col] = session_enriched[col].combine_first(session_enriched[fl_col])
                session_enriched.drop(columns=[fl_col], inplace=True)

        sessions_with_session_level = set(map(tuple, session_df[['Subject Label', 'Session Label']].values))
        file_only_mask = ~merged_df[['Subject Label', 'Session Label']].apply(tuple, axis=1).isin(
            sessions_with_session_level)
        merged_df = pd.concat([merged_df[file_only_mask], session_enriched], ignore_index=True, sort=False)

    # Restore the placeholder used to protect sessions with a missing timestamp from being
    # dropped by groupby/pivot_table (NaN keys are dropped by default).
    if 'Session Timestamp' in merged_df.columns:
        merged_df['Session Timestamp'] = merged_df['Session Timestamp'].replace(TS_PLACEHOLDER, np.nan)

    # Step 9: Export the results to a CSV file
    output_filename = f"parsed_qc_annotations_{formatted_date}.csv"
    
    if not merged_df.empty:
        log.info("Exporting annotations to CSV file.")
        #output_filename = f"parsed_qc_annotations_{formatted_date}.csv"
        merged_df.to_csv(context.output_dir / output_filename, index=False)
        log.info(f'Parsed QC data saved to: {context.output_dir / output_filename}')
        return (0, os.path.join(context.output_dir,output_filename))

    return (1, "No data to export. The merged DataFrame is empty.")
