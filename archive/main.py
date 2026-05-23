import streamlit as st
import pandas as pd
import json
import ast
from typing import List, Dict, Any

# -------------------------------------------------------------------
# 1. Attempt to import st_annotator
# -------------------------------------------------------------------
try:
    from st_annotator import text_annotator
    ST_ANNOTATOR_AVAILABLE = True
except ImportError:
    ST_ANNOTATOR_AVAILABLE = False
    text_annotator = None

st.set_page_config(layout="wide")
st.title("📝 NER Annotation Tool – Edit Your Model's Predictions")

# -------------------------------------------------------------------
# 2. Helper functions for conversion & overlap cleaning
# -------------------------------------------------------------------
def clean_overlaps(annotations: List[Dict]) -> List[Dict]:
    """
    Remove overlapping spans, keeping the longer one when conflict occurs.
    Assumes each annotation has 'start', 'end', 'label'.
    """
    if not annotations:
        return []
    # Sort by start, then by longer span first
    annotations.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    filtered = []
    last_end = -1
    for ann in annotations:
        if ann["start"] >= last_end:
            filtered.append(ann)
            last_end = ann["end"]
    return filtered

def dict_of_lists_to_spans(ner_dict: Dict[str, List[str]], text: str) -> List[Dict]:
    """
    Convert { "LOC": ["New York", "Paris"], "PER": ["John"] } into a list of spans.
    Finds all occurrences of each entity string in the text (case‑sensitive).
    """
    spans = []
    for label, entities in ner_dict.items():
        if not isinstance(entities, list):
            entities = [entities]       # in case a single string is stored
        for entity in entities:
            if not isinstance(entity, str) or not entity:
                continue
            start = 0
            while True:
                pos = text.find(entity, start)
                if pos == -1:
                    break
                spans.append({
                    "start": pos,
                    "end": pos + len(entity),
                    "label": label
                })
                start = pos + 1   # allow overlapping if needed
    return spans

def parse_ner_cell(cell_value: Any, text_content: str) -> List[Dict]:
    """
    Parse the content of the NER column (could be dict, list, or string)
    and return a list of spans suitable for st_annotator.
    """
    # 1. If it's a string, try to evaluate it as a Python literal
    if isinstance(cell_value, str):
        try:
            cell_value = ast.literal_eval(cell_value)
        except (ValueError, SyntaxError):
            # If it's plain text, return empty list
            return []

    # 2. Case: already a list of spans (each with start/end/label)
    if isinstance(cell_value, list):
        # Validate that it looks like a list of spans
        if all(isinstance(item, dict) and "start" in item and "end" in item for item in cell_value):
            return cell_value
        else:
            # Unknown list format – return empty
            return []

    # 3. Case: dictionary { label: list_of_strings }
    if isinstance(cell_value, dict):
        # Check if it's already a span‑based dict (values contain start/end)
        first_val = next(iter(cell_value.values())) if cell_value else None
        if isinstance(first_val, list) and len(first_val) > 0 and isinstance(first_val[0], dict):
            # Convert span dict to flat list
            spans = []
            for label, items in cell_value.items():
                for item in items:
                    if isinstance(item, dict) and "start" in item and "end" in item:
                        spans.append({
                            "start": item["start"],
                            "end": item["end"],
                            "label": label
                        })
            return spans
        else:
            # Standard case: { "LOC": ["entity1", "entity2"], ... }
            return dict_of_lists_to_spans(cell_value, text_content)

    # 4. Fallback: empty list
    return []

# -------------------------------------------------------------------
# 3. Streamlit UI
# -------------------------------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

uploaded_file = st.file_uploader("Upload CSV or Parquet file", type=['csv', 'parquet'])

if uploaded_file is not None and st.session_state.df is None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)
    st.session_state.df = df

if st.session_state.df is not None:
    df = st.session_state.df
    columns = df.columns.tolist()

    # --- Sidebar configuration ---
    st.sidebar.header("⚙️ Configuration")
    default_text = "content_clean" if "content_clean" in columns else columns[0]
    default_ner = "ner_consensus" if "ner_consensus" in columns else (
        "ner" if "ner" in columns else columns[0]
    )

    text_col = st.sidebar.selectbox("Text column", columns, index=columns.index(default_text))
    ner_col = st.sidebar.selectbox("NER column (to edit)", columns, index=columns.index(default_ner))

    # Optional: colour map for entity types
    st.sidebar.subheader("🎨 Entity colours")
    default_colors = {"LOC": "#ff9999", "ORG": "#99ccff", "PER": "#99ff99"}
    color_map = st.sidebar.text_area(
        "Colour mapping (JSON dict)",
        value=json.dumps(default_colors, indent=2),
        help='Example: {"LOC": "#ff9999", "ORG": "#99ccff", "PER": "#99ff99"}'
    )
    try:
        color_map = json.loads(color_map)
    except:
        color_map = default_colors

    total_rows = len(df)
    idx = st.session_state.current_index

    # --- Navigation ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("◀ Previous") and idx > 0:
            st.session_state.current_index -= 1
            st.rerun()
    with col2:
        st.markdown(f"<h4 style='text-align: center;'>Row {idx+1} of {total_rows}</h4>", unsafe_allow_html=True)
    with col3:
        if st.button("Next ▶") and idx < total_rows - 1:
            st.session_state.current_index += 1
            st.rerun()

    # --- Get current row data ---
    row = df.iloc[idx]
    text_content = str(row[text_col])

    # --- Convert existing NER annotations to span format ---
    existing_spans = parse_ner_cell(row[ner_col], text_content)
    existing_spans = clean_overlaps(existing_spans)

    # --- Display annotation area ---
    st.subheader("✏️ Annotate / Edit Entities")
    st.write("Highlight text below to add or edit labels.")

    if not ST_ANNOTATOR_AVAILABLE:
        st.error(
            "`st-annotator` is not installed. Please run:\n\n"
            "```bash\npip install st-annotator\n```"
        )
    else:
        # Use a unique key per row to avoid conflicts
        updated_spans = text_annotator(
            text=text_content,
            entities=existing_spans,
            colors=color_map,
            show_label_input=True,
            key=f"annotator_{idx}"
        )

        # --- Save button ---
        if st.button("💾 Save annotations for this row"):
            # Convert updated spans back to your preferred storage format.
            # Here we store as a JSON string (easy CSV export).
            st.session_state.df.at[idx, ner_col] = json.dumps(updated_spans)
            st.success(f"Saved! Row {idx+1} annotations updated.")

    # --- Export section ---
    st.markdown("---")
    st.subheader("📥 Export annotated dataset")
    csv_data = st.session_state.df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name="annotated_ner_dataset.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Please upload a CSV or Parquet file to begin.")