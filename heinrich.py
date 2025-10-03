# heinrich_fixed.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import sys

# -----------------------------
# Helper utilities
# -----------------------------
def find_sheet_by_keyword(xls, keyword):
    for s in xls.sheet_names:
        if keyword.lower() in s.lower():
            return s
    return None

def normalize_columns(df):
    df = df.copy()
    df.columns = [
        re.sub(r'[^0-9a-z]+', '_', str(c).strip().lower()).strip('_')
        for c in df.columns
    ]
    return df

def find_col(df, *keywords):
    for kw in keywords:
        for c in df.columns:
            if kw in c:
                return c
    return None

def contains_any(series, keywords):
    s = series.fillna('').astype(str).str.lower()
    pattern = "|".join(re.escape(k.lower()) for k in keywords)
    return s.str.contains(pattern, na=False)

def count_findings_with_unsafe(df):
    if df.empty:
        return 0
    # try a list of likely columns that contain findings/answers/responses
    for candidate in ['finding', 'finding_location', 'finding_','answer','response','recommendation','question','help_text']:
        col = find_col(df, candidate)
        if col:
            s = df[col].fillna('').astype(str).str.lower()
            mask = s.str.contains('|'.join([
                'violation','unsafe','non-compliance','non compliance','nonconform','obser','deficien','finding','fail','risk','near miss','hazard'
            ]), na=False)
            return int(mask.sum())
    # fallback: search across all text columns for keywords (slower but safer)
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) == 0:
        return 0
    combined = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
    mask = combined.str.contains('|'.join(['violation','unsafe','non-compliance','near miss','hazard','obser','finding','fail','risk']), na=False)
    return int(mask.sum())

# -----------------------------
# Load Excel (auto-detect file)
# -----------------------------
# CLI arg: optional path to excel file
if len(sys.argv) > 1:
    file_path = Path(sys.argv[1])
else:
    # prefer 'data.xlsx' then first .xlsx in cwd
    cwd = Path('.')
    if (cwd / 'EPCL_VEHS_Data_Processed.xlsx').exists():
        file_path = cwd / 'EPCL_VEHS_Data_Processed.xlsx'
    else:
        xlsx_files = list(cwd.glob('*.xlsx'))
        if xlsx_files:
            file_path = xlsx_files[0]
        else:
            raise SystemExit("No .xlsx file found in current directory. Provide filename as argument.")

print(f"Using Excel file: {file_path}")

xls = pd.ExcelFile(file_path)
inc_sheet = find_sheet_by_keyword(xls, "incident")
haz_sheet = find_sheet_by_keyword(xls, "hazard")
audit_sheet = find_sheet_by_keyword(xls, "audit")
inspect_sheet = find_sheet_by_keyword(xls, "inspection")

if not inc_sheet:
    raise SystemExit("Could not find a sheet with 'incident' in its name. Check your Excel file.")

# read sheets (if present)
incident_df = normalize_columns(pd.read_excel(xls, inc_sheet))
hazard_df   = normalize_columns(pd.read_excel(xls, haz_sheet)) if haz_sheet else pd.DataFrame()
audit_df    = normalize_columns(pd.read_excel(xls, audit_sheet)) if audit_sheet else pd.DataFrame()
inspect_df  = normalize_columns(pd.read_excel(xls, inspect_sheet)) if inspect_sheet else pd.DataFrame()

print("Loaded sheets:", xls.sheet_names)
print("Incident columns preview (first 60):", list(incident_df.columns)[:60])

# create robust incident_number column (use existing id if present, else index)
incident_id_col = find_col(incident_df, 'incident_id', 'incident_number', 'incident_no', 'incident')
if incident_id_col:
    incident_df['incident_number'] = incident_df[incident_id_col].astype(str)
else:
    incident_df['incident_number'] = incident_df.index.astype(str)

# --- identify candidate columns (many fallbacks) ---
inj_class_col = find_col(incident_df, "injury_class", "injury_classification", "injuryclassification", "injury")
actual_consequence_col = find_col(incident_df, "actual_consequence", "actual_consequence_incident", "actual_consequence")
relevant_consequence_col = find_col(incident_df, "relevant_consequence")
worst_case_col = find_col(incident_df, "worst_case_consequence", "worst_case_consequence_incident")
lost_days_col = find_col(incident_df, "lost_days", "num_lost", "lost_day", "restricted_days", "restricted")
reportable_col = find_col(incident_df, "reportable", "recordable", "reportable_recordable")
incident_type_col = find_col(incident_df, "incident_type", "incident_types", "incident_type_s", "type")
injury_potential_col = find_col(incident_df, "injury_potential", "injury_potential")
behavioral_col = find_col(incident_df, "behavioral_violation", "behavioral_violation_at", "behavioral")
cardinal_col = find_col(incident_df, "cardinal_rule_violation", "cardinal_rule")
ppes_col = find_col(incident_df, "ppes_violation", "ppes_violation", "ppe")
safe_wp_col = find_col(incident_df, "safe_work_practices_violation", "safe_work")

missing = []
for name, col in [
    ("injury_classification", inj_class_col),
    ("actual_consequence", actual_consequence_col),
    ("lost_days", lost_days_col),
    ("reportable", reportable_col),
    ("injury_potential", injury_potential_col)
]:
    if col is None:
        missing.append(name)
if missing:
    print("Warning: Could not find columns for:", missing, "\nScript will use safer defaults and keyword search where possible.")

# -----------------------------
# Classification logic (hierarchical)
# -----------------------------
inc = incident_df.copy()

# Fatal
is_fatal = pd.Series(False, index=inc.index)
if inj_class_col:
    is_fatal = is_fatal | contains_any(inc[inj_class_col], ["fatal", "death", "died"])
if actual_consequence_col:
    is_fatal = is_fatal | contains_any(inc[actual_consequence_col], ["fatal", "death", "died"])
if relevant_consequence_col:
    is_fatal = is_fatal | contains_any(inc[relevant_consequence_col], ["fatal", "death"])
if worst_case_col:
    is_fatal = is_fatal | contains_any(inc[worst_case_col], ["fatal", "death"])
inc['is_fatal'] = is_fatal

# Lost Workdays
is_losttime = pd.Series(False, index=inc.index)
if lost_days_col and lost_days_col in inc.columns:
    is_losttime = is_losttime | (pd.to_numeric(inc[lost_days_col], errors='coerce').fillna(0) > 0)
if inj_class_col:
    is_losttime = is_losttime | contains_any(inc[inj_class_col], ["lost time", "lti", "lost_time"])
if actual_consequence_col:
    is_losttime = is_losttime | contains_any(inc[actual_consequence_col], ["lost time", "lti"])
inc['is_losttime'] = is_losttime & ~inc['is_fatal']

# Recordable
is_recordable = pd.Series(False, index=inc.index)
if reportable_col and reportable_col in inc.columns:
    is_recordable = is_recordable | inc[reportable_col].fillna('').astype(str).str.lower().isin(['yes','y','true','1'])
if inj_class_col:
    is_recordable = is_recordable | contains_any(inc[inj_class_col], ["medical", "first aid", "restricted", "recordable", "treatment"])
# check treatment/details columns if exist (common names)
for c in ['description_of_treatment_provided','details_of_treatment','treatment','description']:
    if c in inc.columns:
        is_recordable = is_recordable | contains_any(inc[c], ["medical","treatment","physician","hospital"])
inc['is_recordable'] = is_recordable & ~inc['is_fatal'] & ~inc['is_losttime']

# Near miss / no-injury
is_nearmiss = pd.Series(False, index=inc.index)
if incident_type_col:
    is_nearmiss = is_nearmiss | contains_any(inc[incident_type_col], ["near miss", "near-miss", "near_miss", "near"])
if 'category' in inc.columns:
    is_nearmiss = is_nearmiss | contains_any(inc['category'], ["near miss", "near-miss"])
if injury_potential_col:
    is_nearmiss = is_nearmiss | contains_any(inc[injury_potential_col], ["no injury", "near miss", "potential"])
if actual_consequence_col:
    is_nearmiss = is_nearmiss | contains_any(inc[actual_consequence_col], ["no injury","no-injury"])
inc['is_nearmiss'] = is_nearmiss & ~(inc['is_fatal'] | inc['is_losttime'] | inc['is_recordable'])

# At-risk behaviours from incident flags
is_behaviour = pd.Series(False, index=inc.index)
for c in [behavioral_col, cardinal_col, ppes_col, safe_wp_col]:
    if c and c in inc.columns:
        v = inc[c].fillna('').astype(str).str.strip().str.lower()
        is_behaviour = is_behaviour | (~v.isin(['', 'no', 'none', 'n', 'false', '0']))
# also consider explicit violation type columns if present
if 'violation_type' in inc.columns:
    is_behaviour = is_behaviour | contains_any(inc['violation_type'], ['unsafe','violation'])
inc['is_behaviour'] = is_behaviour & ~(inc['is_fatal'] | inc['is_losttime'] | inc['is_recordable'] | inc['is_nearmiss'])

# final assignment
def assign_level(row):
    if row['is_fatal']:
        return 'Fatality'
    if row['is_losttime']:
        return 'Lost Workdays'
    if row['is_recordable']:
        return 'Recordable Injuries'
    if row['is_nearmiss']:
        return 'Near Misses (estimated)'
    if row['is_behaviour']:
        return 'At-Risk Behaviors (estimated)'
    return 'Other'

inc['pyramid_level'] = inc.apply(assign_level, axis=1)

# counts
incident_counts = inc.groupby('pyramid_level')['incident_number'].nunique().to_dict()

# -----------------------------
# Hazard sheet: count near-miss/hazard rows
# -----------------------------
haz_near = 0
if not hazard_df.empty:
    haz = hazard_df.copy()
    # create incident id if missing
    haz_id_col = find_col(haz, 'incident_id','incident_number','incident')
    if haz_id_col:
        haz['incident_number'] = haz[haz_id_col].astype(str)
    else:
        haz['incident_number'] = haz.index.astype(str)
    haz_incident_type_col = find_col(haz, "incident_type", "incident_types", "type")
    haz_injury_potential = find_col(haz, "injury_potential")
    haz_is_nearmiss = pd.Series(False, index=haz.index)
    if haz_incident_type_col:
        haz_is_nearmiss = haz_is_nearmiss | contains_any(haz[haz_incident_type_col], ["near miss", "near-miss", "hazard", "unsafe"])
    if haz_injury_potential:
        haz_is_nearmiss = haz_is_nearmiss | contains_any(haz[haz_injury_potential], ["no injury", "near miss", "potential"])
    haz_near = int(haz_is_nearmiss.sum())

# -----------------------------
# Audits & Inspections → estimate at-risk base
# -----------------------------
audit_unsafe_count = count_findings_with_unsafe(audit_df) if not audit_df.empty else 0
inspect_unsafe_count = count_findings_with_unsafe(inspect_df) if not inspect_df.empty else 0

at_risk_from_incidents = int(inc['is_behaviour'].sum())
at_risk_from_audits = int(audit_unsafe_count + inspect_unsafe_count)
at_risk_total = at_risk_from_incidents + at_risk_from_audits + haz_near

near_miss_total = int(inc[inc['pyramid_level']=='Near Misses (estimated)']['incident_number'].nunique()) + haz_near

# ensure we get top-3 counts from incidents only (hierarchical)
fatal_count = int(incident_counts.get('Fatality', 0))
lost_count  = int(incident_counts.get('Lost Workdays', 0))
recordable_count = int(incident_counts.get('Recordable Injuries', 0))
near_count = int(near_miss_total)
at_risk_count = int(at_risk_total)

pyramid = {
    'Fatality': fatal_count,
    'Lost Workdays': lost_count,
    'Recordable Injuries': recordable_count,
    'Near Misses (estimated)': near_count,
    'At-Risk Behaviors (estimated)': at_risk_count
}

print("\nComputed pyramid counts:")
for k,v in pyramid.items():
    print(f"  {k}: {v}")

# -----------------------------
# Plot centered pyramid
# -----------------------------
levels = list(pyramid.keys())
values = np.array([pyramid[l] for l in levels], dtype=float)
maxv = max(values.max(), 1)

fig, ax = plt.subplots(figsize=(8, 6))
ypos = np.arange(len(levels))

for i, val in enumerate(values):
    left = (maxv - val) / 2.0
    ax.barh(i, val, left=left, height=0.6, align='center', color=plt.cm.Blues((i+1)/len(levels)))
    ax.text(maxv/2, i, f"{int(val):,}", va='center', ha='center', color='white' if val>maxv*0.15 else 'black', fontsize=10, fontweight='bold')

ax.set_yticks(ypos)
ax.set_yticklabels(levels)
ax.invert_yaxis()
ax.set_xlim(0, maxv * 1.02)
ax.set_xlabel("Count")
ax.set_title("Heinrich Safety Pyramid (computed from dataset)")
plt.tight_layout()
plt.show()
