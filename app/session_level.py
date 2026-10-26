"""Flatten session-level ("Clinical QC") form responses into the wide
``parsed_qc_annotations_*.csv`` format used by the file-level (acquisition) QC.

Input  (long format, one row per question/answer, no acquisition id):
    Subject Label | Session Label | Session Timestamp | Question | Answer ...
    quality_t2_axi -> 0/1/2, artifacts_t2_axi -> "['motion']", repeat_t2_axi -> 0/1, ...

Output (wide format, one row per session) - same columns/values as the file-level parser:
    Subject Label, Session Label, Session Timestamp,
    quality_AXI_T2, ..., banding_AXI_T2, ..., zipper_SAG_T2, QC_all_T1, QC_all_T2,
    clinical_findings_description
    values: good / unsure / bad   and   passed / unsure / failed / incomplete

Can be used from the gear (see ``flatten_session_level``) or standalone:
    python session_level.py form-responses-*.csv -o ./out [--repeat-policy primary|best]
"""
import argparse
import ast
import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

log = logging.getLogger(__name__)

# --- Vocabulary (identical to the file-level parser) -------------------------
ANSWER_TO_LABEL = {0: 'good', 1: 'unsure', 2: 'bad'}
SEVERITY = {'good': 0, 'unsure': 1, 'bad': 2}
ARTIFACT_TYPES = ['banding', 'contrast', 'fov', 'motion', 'noise', 'other', 'zipper']
ORIENTATIONS = {'axi': 'AXI', 'cor': 'COR', 'sag': 'SAG'}
CONTRASTS = {'t1': 'T1', 't2': 'T2'}
ID_COLS = ['Subject Label', 'Session Label', 'Session Timestamp']

# Free-text / extra questions that, if present, are carried straight through to the
# output as their own column (value = the raw Answer text), instead of being ignored.
EXTRA_QUESTIONS = ['clinical_findings_description']

# --- Question names in the session-level form --------------------------------
_QUALITY_RE = re.compile(r'^quality_(t1|t2)_(axi|cor|sag)(_repeat)?$')
_ARTIFACT_RE = re.compile(r'^artifacts_(t1|t2)_(axi|cor|sag)(_repeat)?$')
_REPEAT_FLAG_RE = re.compile(r'^repeat_(t1|t2)_(axi|cor|sag)$')
# Known questions that are not part of the image-QC table (deliberately not exported)
_IGNORED_RE = re.compile(r'^(reviewer_type|clinical_classification|reported_series.*)$')


def determine_qc_status(row, qc_columns, contrast):
    """Same rules as the file-level parser's QC_all_<contrast> logic."""
    vals = row[qc_columns]
    filled = vals.fillna('good')
    if vals.isnull().any() and contrast == 'T2':
        return 'incomplete'
    if all(v == 'good' for v in filled):
        return 'passed'
    if any(v == 'unsure' for v in filled) and all(v in ('good', 'unsure') for v in filled):
        return 'unsure'
    if any(v == 'good' for v in vals) and contrast == 'T1':
        return 'passed'
    return 'failed'


def _to_label(answer):
    """'0'/'1'/'2' (str/int/float) -> good/unsure/bad, else None."""
    try:
        return ANSWER_TO_LABEL.get(int(float(str(answer).strip())))
    except (ValueError, TypeError):
        return None


def _is_blank(answer):
    return answer is None or (isinstance(answer, float) and np.isnan(answer)) \
        or str(answer).strip().lower() in ('', 'nan', 'none')


def _parse_artifacts(answer):
    """"['motion', 'zipper']" -> ['motion', 'zipper'] (also tolerates a;b / a,b / a|b)."""
    if _is_blank(answer):
        return []
    s = str(answer).strip()
    try:
        v = ast.literal_eval(s)
        items = list(v) if isinstance(v, (list, tuple, set)) else [v]
    except (ValueError, SyntaxError):
        items = re.split(r'[;,|]', s.strip('[]'))
    return [str(i).strip().strip('\'"').lower() for i in items if str(i).strip()]


def flatten_session_level(raw_df, repeat_policy='primary'):
    """Convert long-format session-level form responses to the wide QC table.

    Args:
        raw_df: DataFrame from the form-and-annotations exporter, session-level rows only.
        repeat_policy: how repeat scans feed QC_all_*:
            'primary' - only the primary (reported) scan counts; matches the session tagging
                        in main.py (resolve_session_qc).
            'best'    - the better of primary and repeat counts per orientation.
            The repeat's own rating is always exported as quality_<ORI>_<CON>_repeat.

    Returns:
        (wide_df, issues_df)
    """
    if repeat_policy not in ('primary', 'best'):
        raise ValueError("repeat_policy must be 'primary' or 'best'")

    issues = []

    def issue(subj, sess, question, kind, detail=''):
        issues.append({'Subject Label': subj, 'Session Label': sess,
                       'Question': question, 'Issue': kind, 'Detail': detail})
        log.warning(f'[{subj} | {sess}] {question}: {kind} {detail}')

    df = raw_df.copy()
    session_key = 'session.id' if 'session.id' in df.columns else 'Session Label'
    records, keys_seen = [], set()

    for _, g in df.groupby(session_key, sort=False):
        first = g.iloc[0]
        subj, sess = first.get('Subject Label'), first.get('Session Label')

        if _is_blank(subj) or _is_blank(sess):
            issue(subj, sess, '', 'missing subject/session label', 'row skipped')
            continue

        # If a session has several form responses, keep the most recently modified one
        if 'formresponse.id' in g.columns and g['formresponse.id'].nunique() > 1:
            sort_col = 'Modified' if 'Modified' in g.columns else 'Created'
            latest = g.sort_values(sort_col).iloc[-1]['formresponse.id']
            issue(subj, sess, '', 'multiple form responses for session',
                  f'{g["formresponse.id"].nunique()} found, using latest ({latest})')
            g = g[g['formresponse.id'] == latest]

        info = {}  # (CONTRAST, ORI) -> {'primary', 'repeat', 'arts', 'flag'}
        extras = {}  # question -> raw answer, for EXTRA_QUESTIONS
        for _, r in g.iterrows():
            q, ans = str(r['Question']).strip(), r['Answer']
            m = _QUALITY_RE.match(q)
            if m:
                key = (CONTRASTS[m.group(1)], ORIENTATIONS[m.group(2)])
                label = _to_label(ans)
                if label is None and not _is_blank(ans):
                    issue(subj, sess, q, 'unknown quality answer', repr(ans))
                info.setdefault(key, {})['repeat' if m.group(3) else 'primary'] = label
                continue
            m = _ARTIFACT_RE.match(q)
            if m:
                if not m.group(3):  # artifacts of repeat scans are not exported
                    key = (CONTRASTS[m.group(1)], ORIENTATIONS[m.group(2)])
                    info.setdefault(key, {})['arts'] = _parse_artifacts(ans)
                continue
            m = _REPEAT_FLAG_RE.match(q)
            if m:
                key = (CONTRASTS[m.group(1)], ORIENTATIONS[m.group(2)])
                info.setdefault(key, {})['flag'] = str(ans).strip() in ('1', '1.0')  # 1 => repeat exists
                continue
            if q in EXTRA_QUESTIONS:
                if not _is_blank(ans):
                    extras[q] = ans
                continue
            if not _IGNORED_RE.match(q):
                issue(subj, sess, q, 'unrecognised question', 'ignored')

        rec = {c: first.get(c) for c in ID_COLS}
        rec.update(extras)
        for (con, ori), d in info.items():
            keys_seen.add((con, ori))
            primary, repeat, arts = d.get('primary'), d.get('repeat'), d.get('arts', [])
            qname = f'quality_{ori}_{con}'
            if primary is None:
                issue(subj, sess, qname, 'missing quality answer', 'left blank')
            rec[qname] = primary

            # repeat consistency
            if d.get('flag') and repeat is None:
                issue(subj, sess, qname, 'repeat flagged but no repeat quality answer')
            if repeat is not None and not d.get('flag'):
                issue(subj, sess, qname, 'repeat quality given but repeat not flagged')
            if repeat is not None:
                rec[f'{qname}_repeat'] = repeat

            # Artifacts: file-level form rates every artifact for unsure/bad scans and leaves
            # them blank for good scans. Listed artifact -> 'bad', not listed -> 'good'.
            arts_norm = []
            for a in arts:
                if a in ARTIFACT_TYPES:
                    arts_norm.append(a)
                else:
                    issue(subj, sess, f'artifacts_{con.lower()}_{ori.lower()}',
                          'unknown artifact type', f'{a!r} mapped to "other"')
                    arts_norm.append('other')
            if primary in ('unsure', 'bad') or arts_norm:
                if primary == 'good':
                    issue(subj, sess, qname, 'artifacts listed but quality is good', 'artifacts kept')
                for a in ARTIFACT_TYPES:
                    rec[f'{a}_{ori}_{con}'] = 'bad' if a in arts_norm else 'good'
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=ID_COLS), pd.DataFrame(issues)

    out = pd.DataFrame(records)

    # Make sure every quality/artifact column exists for every orientation seen (file-level parity)
    for con, ori in keys_seen:
        for col in [f'quality_{ori}_{con}'] + [f'{a}_{ori}_{con}' for a in ARTIFACT_TYPES]:
            if col not in out.columns:
                out[col] = np.nan

    # QC_all per contrast
    for con in ('T1', 'T2'):
        qcols = [f'quality_{o}_{con}' for o in ORIENTATIONS.values() if f'quality_{o}_{con}' in out.columns]
        if not qcols:
            continue
        eff = out[qcols].copy()
        if repeat_policy == 'best':
            for c in qcols:
                rc = f'{c}_repeat'
                if rc in out.columns:
                    both = pd.concat([out[c], out[rc]], axis=1)
                    eff[c] = both.apply(
                        lambda r: min((v for v in r if isinstance(v, str)), key=SEVERITY.get, default=np.nan),
                        axis=1)
        out[f'QC_all_{con}'] = eff.apply(determine_qc_status, axis=1, qc_columns=qcols, contrast=con)

    # Column order = file-level layout: ids, quality_*, artifact columns by type, QC_all_*, then extras
    quality_cols = sorted(c for c in out.columns if c.startswith('quality_') and not c.endswith('_repeat'))
    art_cols = [c for a in ARTIFACT_TYPES for c in sorted(out.columns) if c.startswith(f'{a}_')]
    qc_all_cols = [c for c in ('QC_all_T1', 'QC_all_T2') if c in out.columns]
    repeat_cols = sorted(c for c in out.columns if c.endswith('_repeat'))
    extra_cols = [c for c in EXTRA_QUESTIONS if c in out.columns]
    out = out[ID_COLS + quality_cols + art_cols + qc_all_cols + repeat_cols + extra_cols]
    out = out.sort_values(ID_COLS).reset_index(drop=True)

    issues_df = pd.DataFrame(issues, columns=['Subject Label', 'Session Label', 'Question', 'Issue', 'Detail'])
    return out, issues_df


def write_outputs(wide_df, issues_df, out_dir):
    """Write parsed_qc_annotations_<ts>.csv (+ parse_issues_<ts>.csv if there are issues)."""
    ts = datetime.now(pytz.utc).strftime('%Y-%m-%d_%H-%M-%S')
    out_path = os.path.join(out_dir, f'parsed_qc_annotations_{ts}.csv')
    wide_df.to_csv(out_path, index=False)
    issues_path = None
    if issues_df is not None and not issues_df.empty:
        issues_path = os.path.join(out_dir, f'parse_issues_{ts}.csv')
        issues_df.to_csv(issues_path, index=False)
    return out_path, issues_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('form_csv')
    ap.add_argument('-o', '--out-dir', default='.')
    ap.add_argument('--repeat-policy', choices=['primary', 'best'], default='primary')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    raw = pd.read_csv(args.form_csv)
    wide, iss = flatten_session_level(raw, args.repeat_policy)
    os.makedirs(args.out_dir, exist_ok=True)
    p, ip = write_outputs(wide, iss, args.out_dir)
    print(f'{len(wide)} sessions -> {p}')
    print(f'{len(iss)} issues' + (f' -> {ip}' if ip else ''))
