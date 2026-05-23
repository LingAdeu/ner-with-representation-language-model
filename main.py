import streamlit as st
import pandas as pd
import json
import ast
import re
import os
import numpy as np
import string
import plotly.graph_objects as go
from wordcloud import WordCloud
from st_annotator import text_annotator

st.set_page_config(layout="wide", page_title="NER & KPE Annotator")

# CUSTOM CSS FOR BUTTONS AND LAYOUT
st.markdown("""
<style>
    /* Force standard buttons to use the custom blue color */
    div.stButton > button {
        background-color: #1c83e1 !important;
        color: white !important;
        border-color: #1c83e1 !important;
    }
    /* Make the button a slightly darker blue when you hover over it */
    div.stButton > button:hover {
        background-color: #156ab5 !important;
        border-color: #156ab5 !important;
        color: white !important;
    }
    
    /* Target only 'primary' buttons to make the Reset Memory button red */
    div.stButton > button[kind="primary"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #ff3333 !important;
        border-color: #ff3333 !important;
    }

    /* Force the Resume and Reset buttons in the sidebar to be the same height */
    [data-testid="stSidebar"] [data-testid="column"] div.stButton > button {
        height: 65px !important;
    }

    /* Aggressively center the label and tooltip of the number input */
    [data-testid="stNumberInput"] label {
        width: 100% !important;
    }
    [data-testid="stNumberInput"] label > div {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# HELPER FUNCTIONS
def parse_tags(data):
    if data is None: return []
    if isinstance(data, (list, dict)): return data
    if hasattr(data, "tolist"): return data.tolist()
    if pd.api.types.is_scalar(data) and pd.isna(data): return []
    if str(data).strip() == "": return []
    if isinstance(data, str):
        try: return ast.literal_eval(data)
        except (ValueError, SyntaxError):
            try: return json.loads(data.replace("'", '"'))
            except: return []
    return []

def to_js_index(text_content, python_index):
    if python_index <= 0: return 0
    return len(text_content[:python_index].encode('utf-16-le')) // 2

def convert_to_spans(text_content, tags_data, task_mode):
    if not tags_data or not isinstance(text_content, str): return []
    if isinstance(tags_data, list) and len(tags_data) > 0 and isinstance(tags_data[0], dict) and 'start' in tags_data[0]:
        return tags_data
        
    converted = []
    if isinstance(tags_data, dict):
        for label, entities in tags_data.items():
            if isinstance(entities, str): entities = [entities]
            elif hasattr(entities, "tolist"): entities = entities.tolist()
            elif not isinstance(entities, list):
                try: entities = list(entities)
                except TypeError: continue
            
            for entity_text in set(entities):
                if not isinstance(entity_text, str) or not entity_text.strip(): continue
                safe_text = re.escape(entity_text.strip())
                
                # Strict word boundaries using Negative Lookarounds
                pattern = r'(?<!\w)' + safe_text + r'(?!\w)'
                matches = list(re.finditer(pattern, text_content, re.IGNORECASE))
                
                for match in matches:
                    converted.append({
                        "start": to_js_index(text_content, match.start()),
                        "end": to_js_index(text_content, match.end()),
                        "label": str(label).strip().upper(),
                        "text": match.group()
                    })
    elif isinstance(tags_data, list) and len(tags_data) > 0 and isinstance(tags_data[0], str):
        lbl = "Keyphrase" if task_mode != "NER" else "ORG" 
        for entity_text in set(tags_data):
            if not isinstance(entity_text, str) or not entity_text.strip(): continue
            safe_text = re.escape(entity_text.strip())
            
            # Strict word boundaries using Negative Lookarounds
            pattern = r'(?<!\w)' + safe_text + r'(?!\w)'
            matches = list(re.finditer(pattern, text_content, re.IGNORECASE))
            
            for match in matches:
                converted.append({
                    "start": to_js_index(text_content, match.start()),
                    "end": to_js_index(text_content, match.end()),
                    "label": lbl,
                    "text": match.group()
                })
    return converted

def clean_ner_overlaps(tags):
    if not tags or not isinstance(tags, list): return []
    sorted_tags = sorted(tags, key=lambda x: (int(x.get('start', 0)), -(int(x.get('end', 0)) - int(x.get('start', 0)))))
    cleaned = []
    last_end = -1
    for tag in sorted_tags:
        s, e = int(tag.get('start', 0)), int(tag.get('end', 0))
        if s >= last_end:
            cleaned.append(tag)
            last_end = e
    return cleaned

def count_ner_tags(tags_data):
    if not isinstance(tags_data, list): return {}
    stats = {}
    for tag in tags_data:
        label = tag.get('label')
        text_content = tag.get('text', '').strip().lower()
        if label:
            if label not in stats: stats[label] = {'total': 0, 'unique': set()}
            stats[label]['total'] += 1
            if text_content: stats[label]['unique'].add(text_content)
    
    display_counts = {}
    for label, data in stats.items():
        total = data['total']
        unique = len(data['unique'])
        display_counts[label] = f"{total}" if total == unique else f"{total} ({unique} unique)"
    return display_counts

# SIDEBAR: UPLOAD & CONFIGURATION
st.sidebar.title("Configuration")

AUTOSAVE_FILE = "autosave_data.parquet"
AUTOSAVE_META = "autosave_meta.json"

def save_session_meta(idx):
    """Saves the current row index and sidebar selections."""
    meta = {
        "idx": idx,
        "text_col": st.session_state.get("text_col_key"),
        "input_col": st.session_state.get("input_col_key"),
        "output_col": st.session_state.get("output_col_key"),
        "new_col_name": st.session_state.get("new_col_key"),
        "task_mode": st.session_state.get("task_mode_key")
    }
    with open(AUTOSAVE_META, "w") as f:
        json.dump(meta, f)

# --- CALLBACK: Resume Autosave Safely ---
def resume_autosave():
    st.session_state.df = pd.read_parquet(AUTOSAVE_FILE)
    st.session_state.file_name = "autosave_recovery"
    if os.path.exists(AUTOSAVE_META):
        with open(AUTOSAVE_META, "r") as f:
            meta = json.load(f)
            st.session_state.current_idx = meta.get("idx", 0)
            st.session_state.jump_row_input = st.session_state.current_idx + 1 
            
            if meta.get("text_col"): st.session_state.text_col_key = meta["text_col"]
            if meta.get("input_col"): st.session_state.input_col_key = meta["input_col"]
            if meta.get("output_col"): st.session_state.output_col_key = meta["output_col"]
            if meta.get("new_col_name"): st.session_state.new_col_key = meta["new_col_name"]
            if meta.get("task_mode"): st.session_state.task_mode_key = meta["task_mode"]
    else:
        st.session_state.current_idx = 0
        st.session_state.jump_row_input = 1
        
    st.session_state.last_idx = -1 

if os.path.exists(AUTOSAVE_FILE):
    st.sidebar.info("Interrupted session found")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Resume", use_container_width=True, on_click=resume_autosave):
            pass 
    with col2:
        if st.button("Start Over", type="primary", use_container_width=True):
            if os.path.exists(AUTOSAVE_FILE):
                os.remove(AUTOSAVE_FILE)
            if os.path.exists(AUTOSAVE_META):
                os.remove(AUTOSAVE_META)
            st.session_state.clear()
            st.rerun()

with st.sidebar.expander("**Step 1**. Upload Data", expanded=False):
    uploaded_file = st.file_uploader("Choose a CSV or Parquet file", type=["csv", "parquet"])

    if uploaded_file:
        if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
            if uploaded_file.name.endswith(".csv"): st.session_state.df = pd.read_csv(uploaded_file)
            else: st.session_state.df = pd.read_parquet(uploaded_file)
            
            st.session_state.file_name = uploaded_file.name
            st.session_state.current_idx = 0
            
            if "jump_row_input" in st.session_state:
                del st.session_state["jump_row_input"]
                
            st.session_state.last_idx = -1 
            
            st.session_state.df.to_parquet(AUTOSAVE_FILE, index=False)

if "df" in st.session_state:
    df = st.session_state.df
    
    with st.sidebar.expander("**Step 2**. Specify Column", expanded=False):
        text_col = st.selectbox("Text Column", df.columns, key="text_col_key")
        
        input_col_option = st.selectbox("Input Predictions Column (Optional)", ["None"] + list(df.columns), key="input_col_key")
        
        label_col_option = st.selectbox("Output Labels Column", ["-- Create New Column --"] + list(df.columns), key="output_col_key")
        if label_col_option == "-- Create New Column --":
            label_col = st.text_input("New Column Name", value="annotations", key="new_col_key")
            if label_col not in df.columns: df[label_col] = None
        else:
            label_col = label_col_option

    # --- CALLBACK: Reset Step 2 and clear cache when Task changes ---
    def reset_mapping_on_task_change():
        if "input_col_key" in st.session_state:
            st.session_state.input_col_key = "None"
        if "output_col_key" in st.session_state:
            st.session_state.output_col_key = "-- Create New Column --"
        if "new_col_key" in st.session_state:
            st.session_state.new_col_key = "annotations"
        st.session_state.last_idx = -1

    with st.sidebar.expander("**Step 3**. Select Task", expanded=False):
        task_mode = st.radio(
            "Task Type", 
            ["NER", "Keyphrase Extraction"], 
            key="task_mode_key",
            on_change=reset_mapping_on_task_change
        )
        
        if task_mode == "NER":
            raw_labels = st.text_input("NER Labels (comma separated)", "PER, LOC, ORG")
            labels_list = [l.strip().upper() for l in raw_labels.split(",") if l.strip()]
        else:
            labels_list = ["Keyphrase"]
            
        hex_palette = [
        "#FF0033", # Neon Red (PER)
        "#00CC44", # Neon Green (LOC)
        "#1c83e1", # Blue (ORG)
        "#FF7700", 
        "#CC00FF", 
        "#FF0099"  
        ]
        colors_map = {lbl: hex_palette[i % len(hex_palette)] for i, lbl in enumerate(labels_list)}

    # EXPORT DATA (STEP 4)
    with st.sidebar.expander("**Step 4**. Export Data", expanded=False):
        output_style = st.radio("Label Format", ["Dictionary (Original)", "Spans (Exact Coordinates)"])
        export_format = st.radio("File Type", ["CSV", "Parquet"])
        
        st.markdown("---")
        st.markdown("**Export Range**")
        
        range_col1, range_col2 = st.columns(2)
        with range_col1:
            start_row = st.number_input("Start Row", min_value=1, max_value=len(df), value=1)
        with range_col2:
            end_row = st.number_input("End Row", min_value=1, max_value=len(df), value=len(df))

        if start_row > end_row:
            st.error("Start Row cannot be greater than End Row.")
        else:
            export_df = df.copy()
            
            if output_style == "Dictionary (Original)":
                def revert_to_dict(tags_data):
                    base_dict = {lbl: [] for lbl in labels_list}
                    try:
                        if isinstance(tags_data, str): tags = json.loads(tags_data)
                        else: tags = tags_data
                            
                        if not isinstance(tags, list): return str(base_dict) if export_format == "CSV" else base_dict
                            
                        result_dict = base_dict.copy()
                        for tag in tags:
                            lbl = tag.get("label")
                            txt = tag.get("text", "").lower() 
                            
                            if lbl and txt:
                                if lbl not in result_dict: result_dict[lbl] = []
                                if txt not in result_dict[lbl]: result_dict[lbl].append(txt)
                                    
                        return str(result_dict) if export_format == "CSV" else result_dict
                    except Exception:
                        return str(base_dict) if export_format == "CSV" else base_dict
                        
                export_df[label_col] = export_df[label_col].apply(revert_to_dict)
            
            sliced_df = export_df.iloc[start_row - 1 : end_row]
            
            if export_format == "CSV":
                csv_data = sliced_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Final CSV",
                    data=csv_data,
                    file_name="annotated_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                parquet_data = sliced_df.to_parquet(index=False)
                st.download_button(
                    label="Download Final Parquet",
                    data=parquet_data,
                    file_name="annotated_data.parquet",
                    mime="application/octet-stream",
                    use_container_width=True
                )

    # MAIN WORKSPACE WITH PROGRAMMATIC TABS
    st.title(f"{task_mode} Tool")
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Data Annotation"

    st.radio(
        "Navigation",
        ["Data Annotation", "Data Audit"],
        horizontal=True,
        key="active_tab",
        label_visibility="collapsed"
    )
    
    # --- TAB 1: THE ANNOTATOR ---
    if st.session_state.active_tab == "Data Annotation":
        total_rows = len(df)
        idx = st.session_state.current_idx
        
        st.markdown(f"<div style='text-align: center; font-size: 20px;'><b>Page:</b> <code>{idx + 1} / {total_rows}</code></div>", unsafe_allow_html=True)
        
        current_text = str(df.iloc[idx][text_col])
        
        if "last_idx" not in st.session_state or st.session_state.last_idx != idx:
            out_data = parse_tags(df.iloc[idx][label_col])
            
            if not out_data and input_col_option != "None":
                in_data = parse_tags(df.iloc[idx][input_col_option])
                existing_tags = convert_to_spans(current_text, in_data, task_mode)
            else:
                existing_tags = convert_to_spans(current_text, out_data, task_mode)

            existing_tags = clean_ner_overlaps(existing_tags)
            
            labels_dict = {cat: [] for cat in labels_list}
            for tag in existing_tags:
                cat = tag.get("label")
                if cat not in labels_dict: continue
                
                body = tag.get("text", "")
                if not body and "start" in tag and "end" in tag:
                    try: body = current_text[int(tag["start"]):int(tag["end"])]
                    except: pass
                    
                labels_dict[cat].append({
                    "start": int(tag["start"]),
                    "end": int(tag["end"]),
                    "label": body
                })

            st.session_state.frozen_labels_dict = labels_dict
            st.session_state.last_idx = idx

        st.info("Guide: Highlight text to apply a label. Double-click a highlighted span to remove it.")
        
        updated_dict = text_annotator(
            current_text,
            st.session_state.frozen_labels_dict,
            colors=colors_map,
            show_label_input=False,
            key=f"annotator_{idx}"
        )

        new_tags = []
        if isinstance(updated_dict, dict):
            for cat, items in updated_dict.items():
                if isinstance(items, list):
                    for item in items:
                        try:
                            new_tags.append({
                                "start": int(item["start"]),
                                "end": int(item["end"]),
                                "label": cat,
                                "text": item.get("label", "")
                            })
                        except (ValueError, TypeError): continue

        new_tags = clean_ner_overlaps(new_tags)
        
        counts = count_ner_tags(new_tags)
        if counts:
            c_html = " ".join([f"<span style='background:#e0e0e0;padding:4px 8px;border-radius:4px;margin-right:5px;font-weight:bold;color:#333'>{k}: {v}</span>" for k, v in counts.items()])
            st.markdown(f"<div style='margin-top:20px; margin-bottom:20px'>{c_html}</div>", unsafe_allow_html=True)
        
        current_saved = df.iloc[idx][label_col]
        new_saved_json = json.dumps(new_tags)
        
        if str(current_saved) != new_saved_json:
            df.at[idx, label_col] = new_saved_json
            st.session_state.df = df
            df.to_parquet(AUTOSAVE_FILE, index=False)

        # NAVIGATION WITH JUMP TO ROW
        if "jump_row_input" not in st.session_state:
            st.session_state.jump_row_input = idx + 1
            
        def go_previous():
            st.session_state.current_idx -= 1
            st.session_state.jump_row_input = st.session_state.current_idx + 1
            save_session_meta(st.session_state.current_idx)

        def go_next():
            st.session_state.current_idx += 1
            st.session_state.jump_row_input = st.session_state.current_idx + 1
            save_session_meta(st.session_state.current_idx)
            
        col1, col2, col3 = st.columns([1, 2, 1], vertical_alignment="bottom")
        
        with col1:
            st.button("Previous", disabled=(idx == 0), use_container_width=True, on_click=go_previous)
                
        with col2:
            def jump_to_input_row():
                target = st.session_state.jump_row_input - 1
                st.session_state.current_idx = target
                save_session_meta(target)
                
            st.number_input(
                f"Jump to Page (1 - {total_rows})",
                min_value=1,
                max_value=total_rows,
                key="jump_row_input",
                on_change=jump_to_input_row,
                help="Type a page number and press Enter to jump to it."
            )
            
        with col3:
            st.button("Next", disabled=(idx == total_rows - 1), use_container_width=True, on_click=go_next)

    # --- TAB 2: DATASET AUDIT ---
    elif st.session_state.active_tab == "Data Audit":
        st.info("Info: Scan your entire dataset to track your annotation progress, find modified tags, and easily spot conflicting labels.")
        
        if st.button("Run Full Audit Scan", type="primary"):
            with st.spinner("Analyzing all rows... this might take a moment."):
                total_base = 0
                total_curr = 0
                added = 0
                removed = 0
                edited = 0
                
                word_to_labels = {}
                adjacent_pairs = {}
                
                kpe_boundaries = {}
                indonesian_stopwords = {"dan", "yang", "di", "ke", "dari", "pada", "dalam", "untuk", "dengan", "adalah", "sebagai", "atau"}
                
                all_span_lengths = []
                all_spans_meta = [] 
                ner_stats = {} 
                
                for r_idx, r_data in df.iterrows():
                    txt_val = str(r_data[text_col])
                    
                    base_spans = []
                    if input_col_option != "None":
                        base_raw = parse_tags(r_data[input_col_option])
                        base_spans = convert_to_spans(txt_val, base_raw, task_mode) if base_raw else []
                        base_spans = clean_ner_overlaps(base_spans)
                    
                    curr_raw = parse_tags(r_data[label_col])
                    if not curr_raw and input_col_option != "None":
                        curr_spans = base_spans
                    else:
                        curr_spans = convert_to_spans(txt_val, curr_raw, task_mode) if curr_raw else []
                        curr_spans = clean_ner_overlaps(curr_spans)
                    
                    # CHECKER 1: Fragmented Tags Logic (NER Only)
                    if task_mode == "NER":
                        sorted_spans = sorted(curr_spans, key=lambda x: int(x.get('start', 0)))
                        for i in range(len(sorted_spans) - 1):
                            s1, s2 = sorted_spans[i], sorted_spans[i+1]
                            end1, start2 = int(s1.get('end', 0)), int(s2.get('start', 0))
                            
                            if 0 <= (start2 - end1) <= 12:
                                gap_text = txt_val[end1:start2]
                                clean_gap = gap_text.strip().lower()
                                
                                if clean_gap == "" or clean_gap in ["kota", "kab.", "kabupaten", "prov.", "provinsi"]:
                                    lbl1 = str(s1.get('label', '')).strip().upper()
                                    lbl2 = str(s2.get('label', '')).strip().upper()
                                    
                                    if {lbl1, lbl2} == {"ORG", "LOC"}:
                                        t1 = s1.get('text') or txt_val[int(s1.get('start',0)):int(s1.get('end',0))]
                                        t2 = s2.get('text') or txt_val[int(s2.get('start',0)):int(s2.get('end',0))]
                                        
                                        combo_text = f"{t1} + {t2}"
                                        combo_labels = f"[{lbl1}] + [{lbl2}]"
                                        
                                        if combo_text not in adjacent_pairs:
                                            adjacent_pairs[combo_text] = {}
                                        if combo_labels not in adjacent_pairs[combo_text]:
                                            adjacent_pairs[combo_text][combo_labels] = []
                                        
                                        if r_idx not in adjacent_pairs[combo_text][combo_labels]:
                                            adjacent_pairs[combo_text][combo_labels].append(r_idx)
                    
                    # CHECKERS & STATS
                    for t in curr_spans:
                        word = t.get("text", "").strip().lower()
                        raw_word = t.get("text", "").strip() 
                        lbl = t.get("label", "")
                        
                        if raw_word and lbl:
                            # 1. Length outlier collection
                            length = len(raw_word)
                            all_span_lengths.append(length)
                            all_spans_meta.append({
                                "text": raw_word,
                                "label": lbl,
                                "row": r_idx,
                                "len": length
                            })
                            
                            # 2. Stats & Tracking
                            if lbl not in ner_stats:
                                ner_stats[lbl] = {'total': 0, 'entities': {}}
                            ner_stats[lbl]['total'] += 1
                            ner_stats[lbl]['entities'][raw_word] = ner_stats[lbl]['entities'].get(raw_word, 0) + 1
                            
                            if word not in word_to_labels: word_to_labels[word] = {}
                            if lbl not in word_to_labels[word]: word_to_labels[word][lbl] = []
                            if r_idx not in word_to_labels[word][lbl]: word_to_labels[word][lbl].append(r_idx)
                            
                            # 3. Keyphrase Boundary Checker
                            if task_mode == "Keyphrase Extraction":
                                words = word.split()
                                if words:
                                    has_flaw = False
                                    if words[0] in indonesian_stopwords or words[-1] in indonesian_stopwords:
                                        has_flaw = True
                                    if word[0] in string.punctuation or word[-1] in string.punctuation:
                                        has_flaw = True
                                        
                                    if has_flaw:
                                        if raw_word not in kpe_boundaries: kpe_boundaries[raw_word] = []
                                        if r_idx not in kpe_boundaries[raw_word]: kpe_boundaries[raw_word].append(r_idx)
                    
                    # --- DELTA METRICS ---
                    total_base += len(base_spans)
                    total_curr += len(curr_spans)
                    
                    b_set = {(t.get('start'), t.get('end'), t.get('label')) for t in base_spans}
                    c_set = {(t.get('start'), t.get('end'), t.get('label')) for t in curr_spans}
                    
                    b_bounds = {(t.get('start'), t.get('end')): t.get('label') for t in base_spans}
                    c_bounds = {(t.get('start'), t.get('end')): t.get('label') for t in curr_spans}
                    
                    row_removed = len(b_set - c_set)
                    row_added = len(c_set - b_set)
                    
                    row_edits = 0
                    for bounds, clbl in c_bounds.items():
                        if bounds in b_bounds and b_bounds[bounds] != clbl:
                            row_edits += 1
                            row_added -= 1
                            row_removed -= 1
                    
                    added += max(0, row_added)
                    removed += max(0, row_removed)
                    edited += row_edits
                
                # --- Post-Loop Outlier Logic ---
                outliers_dict = {}
                if len(all_span_lengths) > 3:
                    q1 = np.percentile(all_span_lengths, 25)
                    q3 = np.percentile(all_span_lengths, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - (1.5 * iqr)
                    upper_bound = q3 + (1.5 * iqr)
                    
                    for meta in all_spans_meta:
                        is_outlier = False
                        if meta["len"] < lower_bound or meta["len"] > upper_bound: 
                            is_outlier = True
                            
                        if is_outlier:
                            outlier_text = f"{meta['text']} [{meta['label']}]"
                            if outlier_text not in outliers_dict:
                                outliers_dict[outlier_text] = {"len": meta["len"], "rows": []}
                            if meta['row'] not in outliers_dict[outlier_text]["rows"]:
                                outliers_dict[outlier_text]["rows"].append(meta['row'])

                inconsistencies = {w: data for w, data in word_to_labels.items() if len(data) > 1}
                
                st.session_state.audit_stats = {
                    "base": total_base, "curr": total_curr, 
                    "add": added, "rem": removed, "edit": edited
                }
                st.session_state.inconsistencies = inconsistencies
                st.session_state.adjacencies = adjacent_pairs
                st.session_state.outliers = outliers_dict
                st.session_state.ner_stats = ner_stats
                
                if task_mode == "Keyphrase Extraction":
                    st.session_state.kpe_boundaries = kpe_boundaries

        if "audit_stats" in st.session_state:
            stats = st.session_state.audit_stats
            
            st.markdown("---")
            st.markdown("#### Word Cloud")
            ner_stats = st.session_state.get("ner_stats", {})
            
            if not ner_stats:
                st.info("No tags found in the dataset.")
            else:
                if task_mode == "Keyphrase Extraction":
                    # --- PLOTLY DYNAMIC WORD CLOUD FOR KPE ---
                    lbl = list(ner_stats.keys())[0] 
                    freq_dict = ner_stats[lbl]['entities']
                    
                    if freq_dict:
                        # 1. Use wordcloud library to calculate math coordinates
                        wc_width = 1500
                        wc_height = 700
                        wc = WordCloud(
                            width=wc_width, height=wc_height, 
                            max_words=100, 
                            prefer_horizontal=1.0, 
                            margin=5
                        ).generate_from_frequencies(freq_dict)
                        
                        # FIX: Switch from Scatter trace to Annotations for perfect bounding-box hovers
                        annotations = []
                        
                        for (word, norm_float), font_size, position, orientation, color in wc.layout_:
                            y, x = position
                            actual_count = freq_dict[word]
                            
                            annotations.append(dict(
                                x=x,
                                y=-y,
                                text=word,
                                font=dict(size=font_size, color='#1c83e1'),
                                showarrow=False,
                                xanchor='left', # Matches WordCloud's default math
                                yanchor='top',
                                hovertext=f"<b>{word}</b><br>Count: {actual_count:,}",
                                hoverlabel=dict(bgcolor="white", font_size=14, font_family="sans-serif")
                            ))
                            
                        # 2. Feed annotations into Plotly
                        fig = go.Figure()
                        
                        # Add an invisible scatter trace just to set the coordinate system bounds
                        fig.add_trace(go.Scatter(
                            x=[0, wc_width], 
                            y=[0, -wc_height],
                            mode='markers',
                            marker=dict(color='rgba(0,0,0,0)'),
                            hoverinfo='none',
                            showlegend=False
                        ))
                        
                        # Lock the Plotly axes strictly to the WordCloud math dimensions
                        fig.update_layout(
                            annotations=annotations,
                            xaxis=dict(range=[0, wc_width], showgrid=False, showticklabels=False, zeroline=False),
                            yaxis=dict(range=[-wc_height, 0], showgrid=False, showticklabels=False, zeroline=False),
                            margin=dict(t=10, b=10, l=10, r=10),
                            plot_bgcolor='white',
                            width=wc_width,   
                            height=wc_height
                        )
                        
                        st.plotly_chart(fig, use_container_width=False)
                        st.caption(f"**Total Keyphrases:** {ner_stats[lbl]['total']:,}")
                else:
                    # --- KEEP STANDARD TEXT COLUMNS FOR NER ---
                    sorted_labels = sorted(ner_stats.keys())
                    cols = st.columns(len(sorted_labels) if len(sorted_labels) > 0 else 1)
                    for i, lbl in enumerate(sorted_labels):
                        data = ner_stats[lbl]
                        with cols[i % len(cols)]:
                            st.markdown(f"**{lbl}** (Total: {data['total']:,})")
                            top_5 = sorted(data['entities'].items(), key=lambda x: x[1], reverse=True)[:5]
                            for text, count in top_5:
                                st.markdown(f"- {text}: `{count:,}`")

            st.markdown("---")
            st.markdown("#### Before vs After Changes")
            
            if input_col_option == "None":
                st.info("No baseline input column selected. Delta tracking is disabled.")
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Baseline Tags", f"{stats['base']:,}")
                m2.metric("Label Swaps", f"{stats['edit']:,}")
                m3.metric("Tags Added", f"{stats['add']:,}")
                m4.metric("Tags Removed", f"{stats['rem']:,}")
            
            def jump_to_row(target_row):
                st.session_state.active_tab = "Data Annotation"
                st.session_state.current_idx = target_row
                st.session_state.jump_row_input = target_row + 1
                save_session_meta(target_row)
            
            if task_mode == "NER":
                st.markdown("---")
                st.markdown("#### Fragmented Entities Spotting")
                adj = st.session_state.get("adjacencies", {})
                
                if not adj:
                    st.success("Great! No consecutive ORG and LOC tags found.")
                else:
                    st.warning(f"**Result**: Found **{len(adj)}** pairs of adjacent ORG and LOC tags that might need to be merged.")
                    
                    for combo_txt, label_data in adj.items():
                        with st.container():
                            st.markdown(f"**Adjacent Position:** `{combo_txt}`")
                            for lbl, rows in label_data.items():
                                st.markdown(f"Tagged as **{lbl}** in:")
                                btn_cols = st.columns(10)
                                for j, r in enumerate(rows):
                                    with btn_cols[j % 10]:
                                        st.button(
                                            f"Row {r + 1}", 
                                            key=f"jump_adj_{combo_txt}_{lbl}_{r}",
                                            on_click=jump_to_row,
                                            args=(r,)
                                        )
                            st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)
                            
            elif task_mode == "Keyphrase Extraction":
                st.markdown("---")
                st.markdown("#### Boundary Flaw Spotting")
                bnd = st.session_state.get("kpe_boundaries", {})
                
                if not bnd:
                    st.success("Great! No boundary flaws (stopwords or trailing punctuation) found.")
                else:
                    st.warning(f"**Result**: Found **{len(bnd)}** keyphrases starting or ending with punctuation/stopwords.")
                    
                    for flaw_txt, rows in bnd.items():
                        with st.container():
                            st.markdown(f"**Flawed Boundary:** `{flaw_txt}`")
                            btn_cols = st.columns(10)
                            for j, r in enumerate(rows):
                                with btn_cols[j % 10]:
                                    st.button(
                                        f"Row {r + 1}", 
                                        key=f"jump_bnd_{flaw_txt}_{r}",
                                        on_click=jump_to_row,
                                        args=(r,)
                                    )
                            st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)
            
            # st.markdown("---")
            st.markdown("#### Length Outlier Spotting")
            outls = st.session_state.get("outliers", {})
            
            if not outls:
                st.success("Great! No unusually long or short tags detected based on your dataset's IQR.")
            else:
                st.warning(f"**Result**: Found **{len(outls)}** tags that fall outside the normal length distribution (Q1 - 1.5*IQR or Q3 + 1.5*IQR).")
                
                for outlier_txt, data in outls.items():
                    with st.container():
                        st.markdown(f"**Outlier:** `{outlier_txt}` (Length: {data['len']} chars)")
                        btn_cols = st.columns(10)
                        for j, r in enumerate(data["rows"]):
                            with btn_cols[j % 10]:
                                st.button(
                                    f"Row {r + 1}", 
                                    key=f"jump_outl_{outlier_txt}_{r}",
                                    on_click=jump_to_row,
                                    args=(r,)
                                )
                        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Inconsistency Spotting")
            inc = st.session_state.inconsistencies
            
            if not inc:
                st.success("Excellent! No conflicting labels found in your annotated dataset.")
            else:
                st.warning(f"**Result**: Found **{len(inc)}** words with conflicting labels.")

                for word, label_data in inc.items():
                    with st.container():
                        st.markdown(f"**Word:** `{word}`")
                        for lbl, rows in label_data.items():
                            st.markdown(f"Tagged as **{lbl}** in:")
                            btn_cols = st.columns(10)
                            for j, r in enumerate(rows):
                                with btn_cols[j % 10]:
                                    st.button(
                                        f"Row {r + 1}", 
                                        key=f"jump_inc_{word}_{lbl}_{r}",
                                        on_click=jump_to_row,
                                        args=(r,)
                                    )
                        st.markdown("<hr style='margin:10px 0'>", unsafe_allow_html=True)

else:
    st.info("Please upload a CSV or Parquet file in the sidebar, or resume a previous session.")

# --- READ GUIDE MODAL ---
@st.dialog("CMS User Guide", width="medium")
def guide_modal():
    st.markdown("<h4 style='text-align: center; color: #1c83e1;'>NER & KPE Annotation Tools</h4>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin-bottom: 20px;'>Here is how to use the CMS:</p>", unsafe_allow_html=True)
    
    st.markdown("#### Step-1. Dataset Upload")
    st.markdown("You can upload **.csv** or **.parquet** files. Ensure your dataset has at least one column containing the raw text you want to annotate.")

    st.markdown("#### Step-2. Specify Column")
    st.markdown("""
    * **Text Column:** Select the column containing the sentences you want to read.
    * **Input Predictions (Optional):** If your machine learning model already generated baseline tags, select that column here. The app will pre-highlight them on the text for you to review.
    * **Output Labels Column:** Tell the app where to save your final human-reviewed tags. If you select `-- Create New Column --`, type a name (like `annotations`) and the app will generate it for you.
    """)

    st.markdown("#### Step-3. Choose Task")
    st.markdown("Choose between **NER** or **Keyphrase Extraction**. Type your desired labels separated by commas (e.g., `PER, LOC, ORG`). The app will automatically assign them distinct, neon colors for easy scanning.")

    st.markdown("#### Step-4. Exporting Data")
    st.markdown("When you are finished annotating, choose how you want your data structured:")
    st.success("""
    * **Dictionary (Original):** Strips away the UI coordinates and groups words cleanly by category (e.g., `{'ORG': ['ASEAN', 'DPR']}`). Best if your legacy scripts expect this format.
    * **Spans (Exact Coordinates):** Exports precise character mapping (e.g., `[{"start": 10, "end": 15, "label": "ORG"}]`). This is the industry standard format required for training modern NLP models.
    """)
    st.markdown("**Finally, select CSV or Parquet and hit download!**")

    st.warning("""
    **:red[HEADS UP: This app does not use an external database. It runs purely in your computer's active memory.]**
    
    However, to keep your data safe, it utilizes a background **Auto-Save**. Every time you edit a tag or navigate to a new row, it silently saves your progress locally (`autosave_data.parquet`).
    
    *If you accidentally close the tab or refresh your browser, simply click **"Resume"** in the sidebar to pick up exactly where you left off!*
    
    *If you want to completely clear the cache and start fresh with a new dataset, click the red **"Start Over"** button.*
    """)

if st.sidebar.button("Confused? Read Me", use_container_width=True):
    guide_modal()