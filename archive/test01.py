import streamlit as st
import json

st.set_page_config(page_title="NER Editor", layout="wide")
st.title("✨ Token‑Based NER Annotation")

# ---------- CONFIG ----------
text = "Apple is looking at buying a startup in San Francisco."
LABELS = ["ORG", "GPE", "PERSON", "DATE"]
LABEL_COLORS = {
    "ORG": "#F1948A",
    "GPE": "#85C1E9",
    "PERSON": "#82E0AA",
    "DATE": "#F7DC6F",
}

# ---------- SESSION STATE ----------
if "annotations" not in st.session_state:
    st.session_state.annotations = [
        {"start": 0, "end": 5, "label": "ORG", "text": "Apple"},
        {"start": 40, "end": 53, "label": "GPE", "text": "San Francisco"},
    ]

# Convert Python objects to JSON for embedding in JavaScript
labels_json = json.dumps(LABELS)
label_colors_json = json.dumps(LABEL_COLORS)

# ---------- THE COMPLETE HTML / JS ----------
html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 10px; }}
  #text-container {{ line-height: 2.2; font-size: 18px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; user-select: none; }}
  .token {{ display: inline-block; margin-right: 4px; padding: 2px 4px; border-radius: 3px; cursor: pointer; border: 1px solid transparent; }}
  .token.selected {{ background: #d3e0ff !important; border-color: #7aa3ff; }}
  .token.annotated {{ border: 1px solid rgba(0,0,0,0.2); }}
  .controls {{ margin-top: 15px; display: flex; gap: 10px; align-items: center; }}
  button {{ padding: 6px 14px; cursor: pointer; }}
  .annotation-item {{ background: #f8f9fa; padding: 8px; margin: 4px 0; border-radius: 4px; display: flex; gap: 12px; align-items: center; }}
</style></head>
<body>
<div id="text-container"></div>
<div class="controls">
  <select id="label-select">{''.join(f'<option value="{l}">{l}</option>' for l in LABELS)}</select>
  <button id="add-btn">➕ Add</button>
  <button id="clear-btn">🧹 Clear</button>
</div>
<h3>Annotations</h3>
<div id="annotation-list"></div>
<script>
const TEXT = {json.dumps(text)};
const INITIAL = {json.dumps(st.session_state.annotations)};
const COLORS = {label_colors_json};
const LABELS = {labels_json};

function tokenise(t) {{
  const tokens = [];
  const re = /\w+(?:'\w+)?|[^\w\s]/g;
  let m;
  while((m=re.exec(t))!==null) tokens.push({{text:m[0],start:m.index,end:m.index+m[0].length}});
  return tokens;
}}
const tokens = tokenise(TEXT);
let annotations = JSON.parse(JSON.stringify(INITIAL));
let selected = new Set();

function renderText() {{
  const c = document.getElementById('text-container');
  c.innerHTML = '';
  tokens.forEach((tok,i)=>{{
    const s = document.createElement('span');
    s.className = 'token';
    s.textContent = tok.text;
    s.dataset.index = i;
    if(selected.has(i)) s.classList.add('selected');
    const ann = annotations.find(a=>a.start<=tok.start && a.end>tok.start);
    if(ann) {{ s.style.backgroundColor = COLORS[ann.label]||'#ddd'; s.classList.add('annotated'); }}
    s.onclick = ()=>{{
      const idx = parseInt(s.dataset.index);
      selected.has(idx) ? selected.delete(idx) : selected.add(idx);
      renderText(); renderList(); send();
    }};
    c.appendChild(s);
  }});
}}

function renderList() {{
  const l = document.getElementById('annotation-list');
  l.innerHTML = '';
  annotations.forEach((ann,i)=>{{
    const d = document.createElement('div');
    d.className = 'annotation-item';
    d.innerHTML = `<span><b>"${{ann.text}}"</b> [${{ann.start}}:${{ann.end}}]</span>`;
    const sel = document.createElement('select');
    LABELS.forEach(lb=>{{ const o = document.createElement('option'); o.value=lb; o.textContent=lb; if(lb===ann.label) o.selected=true; sel.appendChild(o); }});
    sel.onchange = ()=>{{ ann.label = sel.value; renderText(); renderList(); send(); }};
    const del = document.createElement('button'); del.textContent='🗑';
    del.onclick = ()=>{{ annotations.splice(i,1); renderText(); renderList(); send(); }};
    d.appendChild(sel); d.appendChild(del);
    l.appendChild(d);
  }});
}}

function send() {{
  window.parent.postMessage({{type:"streamlit:setComponentValue", data:annotations}}, "*");
}}

document.getElementById('add-btn').onclick = ()=>{{
  if(selected.size===0) return alert('Select words first.');
  const sorted = Array.from(selected).sort((a,b)=>a-b);
  for(let i=1;i<sorted.length;i++) if(sorted[i]!==sorted[i-1]+1) return alert('Must be contiguous.');
  const s = tokens[sorted[0]], e = tokens[sorted[sorted.length-1]];
  annotations = annotations.filter(a=>!(a.start<e.end && a.end>s.start));
  annotations.push({{start:s.start, end:e.end, text:TEXT.substring(s.start,e.end), label:document.getElementById('label-select').value}});
  selected.clear();
  renderText(); renderList(); send();
}};

document.getElementById('clear-btn').onclick = ()=>{{ selected.clear(); renderText(); }};
renderText(); renderList();
</script>
</body></html>"""

# ---------- RENDER THE COMPONENT ----------
result = st.components.v1.html(html, height=600, scrolling=True)

# ---------- HANDLE RETURNED DATA ----------
if result is not None:
    st.session_state.annotations = result

st.subheader("Output (Python)")
st.json(st.session_state.annotations, expanded=False)