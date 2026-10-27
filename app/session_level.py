"""Flatten session-level ("Clinical QC") form responses into the wide
``parsed_qc_annotations_*.csv`` format used by the file-level (acquisition) QC.

Input  (long format, one row per question/answer, no acquisition id):
    Subject Label | Session Label | Session Timestamp | Question | Answer ...
    quality_t2_axi -> 0/1/2, artifacts_t2_axi -> "['motion']", repeat_t2_axi -> 0/1, ...

Output (wide format, one row per session) - same columns/values as the file-level parser:
    Subject Label, Session Label, Session Timestamp,
    quality_AXI_T2, ..., banding_AXI_T2, ..., zipper_SAG_T2, QC_all_T1, QC_all_T2,
    <any other question in the form, e.g. clinical_findings_description, reviewer_type, ...>
    values: good / unsure / bad   and   passed / unsure / failed / incomplete

Any question that is not a quality_*, artifacts_*, or repeat_* question (free text, categorical,
reviewer metadata, or anything added to the form later) is still exported, as its own column
named after the question, holding the raw answer verbatim. Nothing is silently dropped.

A session can be reviewed independently by more than one person (e.g. one reviewer doing
clinical reporting + visual QC, another doing visual QC only). Each distinct 'reviewer_type'
answer for a session produces its own output row (not merged together), so neither reviewer's
answers overwrite the other's. Multiple form responses that share the same reviewer_type (i.e.
the same reviewer revising their own answers) are still merged into a single row, as before.

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

# --- Question names in the session-level form --------------------------------
_QUALITY_RE = re.compile(r'^quality_(t1|t2)_(axi|cor|sag)(_repeat)?$')
_ARTIFACT_RE = re.compile(r'^artifacts_(t1|t2)_(axi|cor|sag)(_repeat)?$')
_REPEAT_FLAG_RE = re.compile(r'^repeat_(t1|t2)_(axi|cor|sag)$')
# Any question that isn't one of the three patterns above (free text, categorical,
# reviewer metadata, reported-series bookkeeping, or anything added to the form later)
# is still exported - see the catch-all branch in flatten_session_level() below.


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


def flatten_session_level(raw_df, repeat_policy='primary', artifact_encoding='label'):
    """Convert long-format session-level form responses to the wide QC table.

    Args:
        raw_df: DataFrame from the form-and-annotations exporter, session-level rows only.
        repeat_policy: how repeat scans feed QC_all_*:
            'primary' - only the primary (reported) scan counts; matches the session tagging
                        in main.py (resolve_session_qc).
            'best'    - the better of primary and repeat counts per orientation.
            The repeat's own rating is always exported as quality_<ORI>_<CON>_repeat.
        artifact_encoding: values used in the per-artifact columns (e.g. motion_AXI_T2):
            'label'  - 'bad' if that artifact was listed, else 'good' (file-level parity, default).
            'binary' - 1 if that artifact was listed, else 0.

    Returns:
        (wide_df, issues_df)
    """
    if repeat_policy not in ('primary', 'best'):
        raise ValueError("repeat_policy must be 'primary' or 'best'")
    if artifact_encoding not in ('label', 'binary'):
        raise ValueError("artifact_encoding must be 'label' or 'binary'")
    artifact_present, artifact_absent = (1, 0) if artifact_encoding == 'binary' else ('bad', 'good')

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

        # A single session can be reviewed independently by MORE THAN ONE person - e.g. one
        # reviewer doing clinical reporting + visual QC, another doing visual QC only. These are
        # two genuinely different assessments and must end up as two SEPARATE rows: silently
        # merging them would mean arbitrarily overwriting one reviewer's answers with the
        # other's. We use the 'reviewer_type' question to tell them apart. Multiple form
        # responses that share the same reviewer_type (i.e. one reviewer revising their own
        # answers over time) are still merged together, as before.
        g = g.copy()
        if 'formresponse.id' in g.columns:
            resp_to_reviewer = {}
            for resp_id, resp_rows in g.groupby('formresponse.id'):
                q_norm = resp_rows['Question'].astype(str).str.strip().str.lower().str.replace(' ', '_')
                rt_answers = resp_rows.loc[q_norm == 'reviewer_type', 'Answer']
                rt_answers = [a for a in rt_answers if not _is_blank(a)]
                resp_to_reviewer[resp_id] = str(rt_answers[0]).strip() if rt_answers else 'unspecified'
            g['_reviewer_type'] = g['formresponse.id'].map(resp_to_reviewer)
        else:
            g['_reviewer_type'] = 'unspecified'

        for reviewer_type, g_rev in g.groupby('_reviewer_type', sort=False):

            # A session+reviewer can still have several form responses (that reviewer revising
            # their QC over time). A later response does NOT necessarily repeat every question -
            # e.g. a revision that only fixes a quality score may leave a free-text field
            # blank/untouched even though an earlier response answered it. Previously we kept
            # ONLY the single latest formresponse.id and discarded every row from the other
            # responses wholesale, which silently erased any answer that happened to live in a
            # non-latest response (this is what made questions like additional_repeats_notes
            # vanish). Instead, merge responses question-by-question: process rows oldest ->
            # newest so a later NON-BLANK answer overrides an earlier one, but a blank/missing
            # answer in a newer response never erases a valid earlier answer.
            if 'formresponse.id' in g_rev.columns and g_rev['formresponse.id'].nunique() > 1:
                sort_col = 'Modified' if 'Modified' in g_rev.columns else 'Created'
                issue(subj, sess, '', f'multiple form responses for reviewer_type={reviewer_type!r}',
                      f'{g_rev["formresponse.id"].nunique()} found, merging all (newest non-blank '
                      f'answer wins per question, nothing discarded)')
                g_rev = g_rev.sort_values(sort_col)

            info = {}  # (CONTRAST, ORI) -> {'primary', 'repeat', 'arts', 'flag'}
            extras = {}  # question -> raw answer, for any catch-all question
            for _, r in g_rev.iterrows():
                q, ans = str(r['Question']).strip(), r['Answer']
                m = _QUALITY_RE.match(q)
                if m:
                    key = (CONTRASTS[m.group(1)], ORIENTATIONS[m.group(2)])
                    sub_key = 'repeat' if m.group(3) else 'primary'
                    if _is_blank(ans):
                        continue  # don't let a blank answer in a later response erase an earlier one
                    label = _to_label(ans)
                    if label is None:
                        issue(subj, sess, q, 'unknown quality answer', repr(ans))
                        continue
                    d = info.setdefault(key, {})
                    if sub_key in d and d[sub_key] != label:
                        issue(subj, sess, q, 'multiple differing answers for reviewer',
                              f'{d[sub_key]!r} -> keeping {label!r}')
                    d[sub_key] = label
                    continue
                m = _ARTIFACT_RE.match(q)
                if m:
                    if not m.group(3) and not _is_blank(ans):  # artifacts of repeat scans are not exported
                        key = (CONTRASTS[m.group(1)], ORIENTATIONS[m.group(2)])
                        info.setdefault(key, {})['arts'] = _parse_artifacts(ans)
                    continue
                m = _REPEAT_FLAG_RE.match(q)
                if m:
                    if not _is_blank(ans):
                        key = (CONTRASTS[m.group(1)], ORIENTATIONS[m.group(2)])
                        info.setdefault(key, {})['flag'] = str(ans).strip() in ('1', '1.0')  # 1 => repeat exists
                    continue
                # Catch-all: any other question (free text, categorical, reviewer metadata,
                # reported-series bookkeeping, or anything added to the form later) is exported
                # verbatim as its own column, named after the question, instead of being dropped.
                if not _is_blank(ans):
                    if q in extras and extras[q] != ans:
                        issue(subj, sess, q, 'multiple differing answers for reviewer',
                              f'{extras[q]!r} -> keeping {ans!r}')
                    extras[q] = ans

            rec = {c: first.get(c) for c in ID_COLS}
            rec.update(extras)
            rec.setdefault('reviewer_type', reviewer_type)
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
                        rec[f'{a}_{ori}_{con}'] = artifact_present if a in arts_norm else artifact_absent
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

    # Column order = file-level layout: ids, quality_*, artifact columns by type, QC_all_*,
    # repeat_*, then any other (catch-all) question, each kept as its own column, in the
    # order first encountered.
    quality_cols = sorted(c for c in out.columns if c.startswith('quality_') and not c.endswith('_repeat'))
    art_cols = [c for a in ARTIFACT_TYPES for c in sorted(out.columns) if c.startswith(f'{a}_')]
    qc_all_cols = [c for c in ('QC_all_T1', 'QC_all_T2') if c in out.columns]
    repeat_cols = sorted(c for c in out.columns if c.endswith('_repeat'))
    known_cols = set(ID_COLS) | set(quality_cols) | set(art_cols) | set(qc_all_cols) | set(repeat_cols)
    extra_cols = [c for c in out.columns if c not in known_cols]
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
    ap.add_argument('--artifact-encoding', choices=['label', 'binary'], default='label',
                     help="artifact column values: 'label' = good/bad (default), 'binary' = 0/1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    raw = pd.read_csv(args.form_csv)
    wide, iss = flatten_session_level(raw, args.repeat_policy, args.artifact_encoding)
    os.makedirs(args.out_dir, exist_ok=True)
    p, ip = write_outputs(wide, iss, args.out_dir)
    print(f'{len(wide)} sessions -> {p}')
    print(f'{len(iss)} issues' + (f' -> {ip}' if ip else ''))
