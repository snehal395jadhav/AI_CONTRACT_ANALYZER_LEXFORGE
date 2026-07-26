"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              N A V N E E T   C O N T R A C T   A I                            ║
║       Enterprise Contract Intelligence Platform                              ║
║  Company   : Navneet Education Limited                                      ║
║  Engine    : LexForge hybrid workflow (LangGraph + RAG + OpenRouter)        ║
║  Year      : 2026                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import os, json, time, io, sys, hashlib, html
import importlib.util
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import base64

sys.path.insert(0, os.path.dirname(__file__))
from core.agent import (
    ContractAnalysisAgent,
    ContractWriterAgent,
    WRITER_TEMPLATES,
    CLAUSE_LIBRARY,
    strip_markdown_stars,
)
from core.config import (
    OPENROUTER_MODEL,
    OPENROUTER_MODEL_LABEL,
    OPENROUTER_MODELS,
    RETRIEVAL_ENGINE_LABEL,
    openrouter_model_label,
)
from core.intelligence import enrich_analysis
from core.offline_analyzer import analyze_contract_offline
from core.pdf_parser import extract_text_from_pdf, extract_contract_file, chunk_text
from core.rag_engine import RAGEngine
from database.db import (
    save_contract_analysis, get_all_contracts, get_contract_by_id,
    update_contract, delete_contract, search_contracts,
    save_chat_message, get_chat_history, clear_chat_history,
    save_generated_contract, get_all_generated_contracts,
    update_generated_contract, delete_generated_contract,
    get_dashboard_stats, get_recent_activity, verify_user, create_user, get_user
)

# ── Brand / Config ──────────────────────────────────────────────────────────────
BRAND_NAME = "Navneet ContractAI"
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

OPENROUTER_SITE_URL = "http://localhost:8501"
OPENROUTER_SITE_NAME = "Navneet ContractAI"

# Pages that anyone can view without signing in. Everything else requires login.
PUBLIC_PAGES = {"home", "features", "architecture", "about", "contact"}


def _logo_data_uri() -> str:
    """Return the Navneet logo as a data URI. Prefers an official PNG/JPG if present."""
    for name in ("navneet_logo.png", "navneet_logo.jpg", "navneet_logo.jpeg"):
        p = os.path.join(ASSETS_DIR, name)
        if os.path.exists(p):
            mime = "jpeg" if name.endswith(("jpg", "jpeg")) else "png"
            with open(p, "rb") as f:
                return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()
    svg = os.path.join(ASSETS_DIR, "navneet_logo.svg")
    if os.path.exists(svg):
        with open(svg, "rb") as f:
            return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode()
    return ""


LOGO_URI = _logo_data_uri()


def _configured_openrouter_key() -> str:
    """Read the single workspace OpenRouter key from Streamlit secrets."""
    try:
        return str(st.secrets.get("OPENROUTER_API_KEY", ""))
    except Exception:
        return ""


def _mask_secret(value: str) -> str:
    if not value:
        return "Not configured"
    return f"{value[:7]}...{value[-4:]}" if len(value) > 14 else "Configured"

# ── Three.js 3D hero (self-contained iframe) ────────────────────────────────────
THREEJS_HERO_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  html,body{height:100%;overflow:hidden;font-family:'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif}
  #wrap{position:relative;width:100%;height:620px;
        background:linear-gradient(165deg,#ffffff 0%,#eef4ff 48%,#d9e7ff 100%);}
  #c{position:absolute;inset:0;z-index:0;display:block}
  #overlay{position:absolute;inset:0;z-index:2;display:flex;flex-direction:column;
           align-items:center;justify-content:center;text-align:center;padding:0 24px 120px;
           pointer-events:none;
           background:radial-gradient(ellipse 58% 50% at 50% 30%,rgba(255,255,255,0.85),rgba(255,255,255,0) 70%)}
  .badge{display:inline-flex;align-items:center;gap:8px;background:rgba(37,99,235,0.10);
         color:#2563eb;border:1px solid rgba(37,99,235,0.28);padding:7px 16px;border-radius:100px;
         font-size:12px;font-weight:700;letter-spacing:.05em;margin-bottom:24px}
  .badge .dot{width:7px;height:7px;border-radius:50%;background:#2563eb;box-shadow:0 0 10px #2563eb}
  h1{font-family:'Sora','Plus Jakarta Sans',sans-serif;font-size:clamp(34px,5.4vw,68px);font-weight:800;
     color:#0f2247;line-height:1.05;letter-spacing:-2px;margin-bottom:18px}
  h1 .grad{background:linear-gradient(120deg,#2563eb,#7c3aed);-webkit-background-clip:text;
           background-clip:text;-webkit-text-fill-color:transparent}
  p.sub{font-size:17px;color:#475569;max-width:620px;line-height:1.7;margin:0 auto 34px;font-weight:500}
  .stats{display:flex;gap:46px;flex-wrap:wrap;justify-content:center;margin-top:8px}
  .stat .v{font-family:'Sora',sans-serif;font-size:30px;font-weight:800;color:#2563eb}
  .stat .l{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-top:2px;font-weight:600}
</style></head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="overlay">
    <div class="badge"><span class="dot"></span>Navneet ContractAI · Evidence-grounded LangGraph Engine</div>
    <h1>AI-Powered Legal<br><span class="grad">Contract Intelligence</span></h1>
    <p class="sub">Analyze contracts with reasoning-enabled GPT-OSS and NVIDIA Nemotron models,
       configured through OpenRouter for Navneet Education Limited.</p>
    <div class="stats">
      <div class="stat"><div class="v">8</div><div class="l">Agent Nodes</div></div>
      <div class="stat"><div class="v">12+</div><div class="l">Contract Types</div></div>
      <div class="stat"><div class="v">100+</div><div class="l">Clause Types</div></div>
      <div class="stat"><div class="v">GDPR</div><div class="l">Compliant</div></div>
    </div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){
  if(typeof THREE==='undefined'){return;}
  const wrap=document.getElementById('wrap');
  const canvas=document.getElementById('c');
  const W=()=>wrap.clientWidth, H=()=>620;
  const scene=new THREE.Scene();
  scene.fog=new THREE.Fog(0xe7f0ff,24,78);
  const camera=new THREE.PerspectiveCamera(62,W()/H(),0.1,200);
  camera.position.set(0,5.4,15);
  const renderer=new THREE.WebGLRenderer({canvas:canvas,antialias:true,alpha:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.setSize(W(),H(),false);

  scene.add(new THREE.AmbientLight(0xffffff,0.95));
  const dir=new THREE.DirectionalLight(0xffffff,0.85); dir.position.set(6,14,10); scene.add(dir);
  const sky=new THREE.HemisphereLight(0xffffff,0x9bbbff,0.6); scene.add(sky);
  const pb=new THREE.PointLight(0x3b82f6,0.8,120); pb.position.set(-12,8,12); scene.add(pb);

  // ── Animated field of glowing glass blocks (white, receding to horizon) ──
  const COLS=24, ROWS=56, GAP=1.7, SPEED=0.07, WRAP_Z=13;
  const group=new THREE.Group(); group.rotation.y=-0.30; group.position.y=2.2; scene.add(group);
  const colA=new THREE.Color(0x2f7df0), colB=new THREE.Color(0xbcd6ff);
  const blockGeo=new THREE.BoxGeometry(1.18,1,1.18);
  const edgeGeo=new THREE.EdgesGeometry(blockGeo);
  const blocks=[];
  for(let zi=0; zi<ROWS; zi++){
    for(let xi=0; xi<COLS; xi++){
      const h=0.4+Math.random()*Math.random()*6.0;
      const c=colA.clone().lerp(colB,Math.random());
      const mat=new THREE.MeshStandardMaterial({color:c,transparent:true,opacity:0.42,
        metalness:0.25,roughness:0.12});
      const m=new THREE.Mesh(blockGeo,mat);
      m.scale.y=h; m.position.set((xi-COLS/2)*GAP+GAP/2, h/2, -zi*GAP);
      const edges=new THREE.LineSegments(edgeGeo,
        new THREE.LineBasicMaterial({color:0x2563eb,transparent:true,opacity:0.55}));
      m.add(edges);
      group.add(m); blocks.push(m);
    }
  }
  // bright ground haze plane for the white horizon
  const ground=new THREE.Mesh(new THREE.PlaneGeometry(400,400),
    new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0.0}));
  ground.rotation.x=-Math.PI/2; ground.position.y=-0.02; scene.add(ground);

  let mx=0,my=0;
  wrap.addEventListener('mousemove',e=>{const rc=wrap.getBoundingClientRect();
    mx=((e.clientX-rc.left)/rc.width-0.5)*2; my=((e.clientY-rc.top)/rc.height-0.5)*2;});

  const clock=new THREE.Clock();
  function animate(){
    const dt=Math.min(clock.getDelta(),0.05);
    for(let i=0;i<blocks.length;i++){
      const m=blocks[i];
      m.position.z+=SPEED*dt*60;
      if(m.position.z>WRAP_Z){
        m.position.z-=ROWS*GAP;
        const h=0.4+Math.random()*Math.random()*6.0;
        m.scale.y=h; m.position.y=h/2;
      }
    }
    camera.position.x+=((mx*3.0)-camera.position.x)*0.04;
    camera.position.y+=((5.4-my*1.6)-camera.position.y)*0.04;
    camera.lookAt(-1,0.4,-28);
    renderer.render(scene,camera);
    requestAnimationFrame(animate);
  }
  animate();
  window.addEventListener('resize',()=>{camera.aspect=W()/H();camera.updateProjectionMatrix();
    renderer.setSize(W(),H(),false);});
})();
</script>
</body></html>
"""

# ── Capability cards (custom line-icon SVGs, no emojis) ─────────────────────────
FEATURE_CARDS = [
    ('<svg viewBox="0 0 24 24"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7"/>'
     '<path d="M14 3v5h5"/><circle cx="17" cy="16" r="3"/><line x1="19.2" y1="18.2" x2="22" y2="21"/></svg>',
     "Contract Analysis",
     "Four-stage LangGraph workflow across eight domains: extraction, model review, evidence grounding and validation."),
    ('<svg viewBox="0 0 24 24"><path d="M12 20h9"/>'
     '<path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>',
     "AI Contract Writer",
     "Generate 12+ contract types with proper legal structure. GDPR-ready, jurisdiction-aware, fully formatted."),
    ('<svg viewBox="0 0 24 24"><path d="M21 11.5a8.5 8.5 0 0 1-12.2 7.6L3 21l1.9-5.8A8.5 8.5 0 1 1 21 11.5Z"/>'
     '<line x1="8" y1="10" x2="16" y2="10"/><line x1="8" y1="13" x2="13" y2="13"/></svg>',
     "RAG-Powered Chat",
     "Ask contract questions with document-scoped ChromaDB or Pinecone retrieval and visible relevance evidence."),
    ('<svg viewBox="0 0 24 24"><path d="M12 3l7 3v5c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6Z"/>'
     '<polyline points="9 12 11 14 15 9.5"/></svg>',
     "Compliance Center",
     "GDPR, CCPA, HIPAA, SOC2 auto-detection with a scored compliance report and missing-clause recommendations."),
    ('<svg viewBox="0 0 24 24"><line x1="3" y1="20" x2="21" y2="20"/><line x1="6" y1="20" x2="6" y2="12"/>'
     '<line x1="12" y1="20" x2="12" y2="6"/><line x1="18" y1="20" x2="18" y2="14"/></svg>',
     "Visual Dashboard",
     "Risk gauge, obligation breakdown, deadline timeline, clause heatmap, anomaly scores — all with Plotly charts."),
    ('<svg viewBox="0 0 24 24"><line x1="12" y1="4" x2="12" y2="20"/><line x1="8" y1="20" x2="16" y2="20"/>'
     '<line x1="6" y1="8" x2="18" y2="8"/><path d="M6 8l-3 6h6Z"/><path d="M18 8l-3 6h6Z"/></svg>',
     "Contract Comparison",
     "Upload two contracts and get a side-by-side dimension analysis with a recommendation on the more favorable one."),
    ('<svg viewBox="0 0 24 24"><path d="M5 5a2 2 0 0 1 2-2h12v15H7a2 2 0 0 0-2 2Z"/>'
     '<line x1="9" y1="7" x2="15" y2="7"/><line x1="9" y1="10.5" x2="13" y2="10.5"/></svg>',
     "Clause Library",
     "20+ pre-built legal clauses — indemnification, LOL, GDPR DPA, force majeure, non-compete. One-click generation."),
    ('<svg viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="7" ry="3"/>'
     '<path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></svg>',
     "Full CRUD History",
     "SQLite-backed contract history. Add, edit, delete, search, tag, and export all your analyses and drafts."),
    ('<svg viewBox="0 0 24 24"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/>'
     '<line x1="6.7" y1="7" x2="10.5" y2="16.3"/><line x1="17.3" y1="7" x2="13.5" y2="16.3"/><line x1="7" y1="6" x2="17" y2="6"/></svg>',
     "Multi-Model Support",
     "Choose GPT-OSS 20B or NVIDIA Nemotron 3 Super 120B for AI-enhanced analysis, with reasoning context preserved."),
    ('<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/></svg>',
     "Vector Infrastructure",
     "Switch between local persistent ChromaDB and a configured Pinecone index with namespaces and metadata filters."),
    ('<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="8.5" y="14" width="7" height="7" rx="2"/><path d="M6.5 10v2h11v-2M12 12v2"/></svg>',
     "MCP Integration Registry",
     "Register Streamable HTTP MCP endpoints, keep credentials masked, and test protocol initialization from the workspace."),
    ('<svg viewBox="0 0 24 24"><path d="M4 18V8m5 10V4m5 14v-7m5 7V6"/><path d="M2 21h20"/></svg>',
     "Retrieval Diagnostics",
     "Inspect provider readiness, embedding configuration, namespace, indexed chunks and scored retrieval evidence."),
    ('<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v3m0 14v3M2 12h3m14 0h3M4.9 4.9 7 7m10 10 2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1"/></svg>',
     "Agent Operations",
     "Goal-driven contract workflows with context controls, structured plans, outputs, caveats and exportable results."),
]

UI_ICONS = {
    "document": '<svg viewBox="0 0 24 24"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M14 3v6h6"/><path d="M8 13h8M8 17h6"/></svg>',
    "risk": '<svg viewBox="0 0 24 24"><path d="M12 3 2.8 20h18.4L12 3Z"/><path d="M12 9v5m0 3v.01"/></svg>',
    "draft": '<svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>',
    "vector": '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/></svg>',
    "graph": '<svg viewBox="0 0 24 24"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M7 6h10M6 8l5 8m7-8-5 8"/></svg>',
    "server": '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="6" rx="2"/><rect x="3" y="14" width="18" height="6" rx="2"/><path d="M7 7h.01M7 17h.01M11 7h6M11 17h6"/></svg>',
    "search": '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>',
    "shield": '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 5-3.4 8.2-8 9.5C7.4 20.2 4 17 4 12V6l8-3Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></svg>',
}

SIDEBAR_ICONS = {
    "dashboard": '<svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>',
    "analyze": '<svg viewBox="0 0 24 24"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7"/><path d="M14 3v6h6"/><circle cx="17" cy="16" r="3"/><path d="m19.3 18.3 2.2 2.2"/></svg>',
    "writer": '<svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>',
    "chatbot": '<svg viewBox="0 0 24 24"><path d="M21 12a8 8 0 0 1-8 8H6l-4 2 1.5-5A9 9 0 1 1 21 12Z"/><path d="M8 11h8M8 15h5"/></svg>',
    "agent": '<svg viewBox="0 0 24 24"><rect x="4" y="6" width="16" height="13" rx="3"/><path d="M9 2h6M12 2v4M8 11h.01M16 11h.01M8 15h8"/></svg>',
    "chat": '<svg viewBox="0 0 24 24"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M14 3v6h6M8 13h8M8 17h5"/></svg>',
    "knowledge": '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7"/></svg>',
    "compare": '<svg viewBox="0 0 24 24"><path d="M8 3 4 7l4 4M4 7h14M16 13l4 4-4 4M20 17H6"/></svg>',
    "history": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2M3 4v5h5"/></svg>',
    "integrations": '<svg viewBox="0 0 24 24"><path d="M8 12h8M12 8v8"/><rect x="3" y="3" width="6" height="6" rx="2"/><rect x="15" y="3" width="6" height="6" rx="2"/><rect x="3" y="15" width="6" height="6" rx="2"/><rect x="15" y="15" width="6" height="6" rx="2"/></svg>',
    "about": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></svg>',
    "contact": '<svg viewBox="0 0 24 24"><path d="M4 13v-2a8 8 0 0 1 16 0v2"/><path d="M4 13a2 2 0 0 1 2-2h1v6H6a2 2 0 0 1-2-2ZM20 13a2 2 0 0 0-2-2h-1v6h1a2 2 0 0 0 2-2ZM17 19c-1 2-3 2-5 2"/></svg>',
    "theme": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    "settings": '<svg viewBox="0 0 24 24"><path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5"/><circle cx="16" cy="6" r="2"/><circle cx="8" cy="12" r="2"/><circle cx="13" cy="18" r="2"/></svg>',
    "logout": '<svg viewBox="0 0 24 24"><path d="M10 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h5M14 8l4 4-4 4M18 12H8"/></svg>',
}


def _sidebar_icon_uri(svg: str) -> str:
    svg = svg.replace(
        "<svg ",
        '<svg xmlns="http://www.w3.org/2000/svg" fill="none" stroke="black" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" ',
        1,
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


SIDEBAR_ICON_URIS = {name: _sidebar_icon_uri(svg) for name, svg in SIDEBAR_ICONS.items()}

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=BRAND_NAME,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded" if st.session_state.get("auth_user") else "collapsed",
)

# ── Session State ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "dark_mode":          False,
    "page":               "home",
    "api_key":            _configured_openrouter_key(),
    "api_key_fingerprint": "",
    "rag_config":         None,
    "vector_backend":     "pinecone" if os.getenv("PINECONE_API_KEY") and os.getenv("PINECONE_INDEX_HOST") else "chroma",
    "pinecone_api_key":   os.getenv("PINECONE_API_KEY", ""),
    "pinecone_host":      os.getenv("PINECONE_INDEX_HOST", ""),
    "pinecone_namespace": os.getenv("PINECONE_NAMESPACE", "contracts"),
    "knowledge_doc_id":   "",
    "knowledge_doc_name": "",
    "mcp_servers":        [],
    "analysis_results":   None,
    "analysis_model":     OPENROUTER_MODEL,
    "doc_info":           None,
    "current_doc_id":     None,
    "current_doc_chunks": [],
    "current_doc_indexed": False,
    "chat_history":       [],
    "rag_engine":         None,
    "analysis_agent":     None,
    "writer_agent":       None,
    "generated_contract": "",
    "contract_review":    None,
    "compare_a":          None,
    "compare_b":          None,
    "compare_result":     None,
    "auth_user":          None,
    "auth_role":          None,
    "username":           None,
    "show_auth":          False,
    "edit_contract_id":   None,
    "toast_msg":          "",
    "chatbot_history":    [],
    "agent_result":       None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Never retain a manually entered or stale OpenRouter key in a long-running
# Streamlit session. The workspace secret is the only authorized key source.
_workspace_api_key = _configured_openrouter_key()
_workspace_key_fingerprint = hashlib.sha256(_workspace_api_key.encode()).hexdigest() if _workspace_api_key else ""
if st.session_state.api_key_fingerprint != _workspace_key_fingerprint:
    st.session_state.api_key = _workspace_api_key
    st.session_state.api_key_fingerprint = _workspace_key_fingerprint
    st.session_state.analysis_agent = None
    st.session_state.writer_agent = None
    st.session_state.rag_engine = None
    st.session_state.rag_config = None


def nav(page: str):
    st.session_state.page = page
    st.rerun()


def get_agent(model: str = OPENROUTER_MODEL) -> ContractAnalysisAgent:
    if (
        not st.session_state.analysis_agent
        or st.session_state.analysis_agent.model != model
    ):
        st.session_state.analysis_agent = ContractAnalysisAgent(
            st.session_state.api_key, model=model
        )
    return st.session_state.analysis_agent


def get_writer() -> ContractWriterAgent:
    if not st.session_state.writer_agent:
        st.session_state.writer_agent = ContractWriterAgent(
            st.session_state.api_key
        )
    return st.session_state.writer_agent


def get_rag() -> RAGEngine:
    rag_config = (
        st.session_state.api_key,
        RETRIEVAL_ENGINE_LABEL,
        st.session_state.vector_backend,
        st.session_state.pinecone_api_key,
        st.session_state.pinecone_host,
        st.session_state.pinecone_namespace,
    )
    if not st.session_state.rag_engine or st.session_state.rag_config != rag_config:
        st.session_state.rag_engine = RAGEngine(
            api_key=st.session_state.api_key,
            backend=st.session_state.vector_backend,
            pinecone_api_key=st.session_state.pinecone_api_key,
            pinecone_host=st.session_state.pinecone_host,
            pinecone_namespace=st.session_state.pinecone_namespace,
        )
        st.session_state.rag_config = rag_config
    return st.session_state.rag_engine


def build_contract_context(question: str, doc_id: str, top_k: int = 6) -> str:
    """Index the current document only when chat retrieval is actually requested."""
    rag = get_rag()
    if (
        doc_id == st.session_state.current_doc_id
        and st.session_state.current_doc_chunks
        and not st.session_state.current_doc_indexed
    ):
        rag.add_document(doc_id, st.session_state.current_doc_chunks)
        st.session_state.current_doc_indexed = True
    return rag.build_context(question, doc_id=doc_id, top_k=top_k)


# ══════════════════════════════════════════════════════════════════════════════
#  THEME & CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css(dark: bool, show_sidebar: bool = False):
    if dark:
        bg, surf, surf2 = "#08090f", "#111827", "#1a2235"
        text, text2     = "#f1f5f9", "#94a3b8"
        primary         = "#3b82f6"
        accent          = "#8b5cf6"
        border          = "#1e293b"
        card_bg         = "#111827"
        inp_bg          = "#0d1117"
        success, warn, danger = "#10b981", "#f59e0b", "#ef4444"
        grad_hero       = "linear-gradient(135deg, #08090f 0%, #0f172a 40%, #111827 100%)"
        nav_bg          = "rgba(8,9,15,0.95)"
        risk_low_bg     = "#0d2137"; risk_low_t = "#60a5fa"
        risk_med_bg     = "#1c1a07"; risk_med_t = "#fbbf24"
        risk_hi_bg      = "#1f0d0d"; risk_hi_t  = "#f87171"
        badge_new_bg    = "#1e3a5f"; badge_new_t = "#93c5fd"
    else:
        bg, surf, surf2 = "#f0f4ff", "#ffffff", "#f1f5f9"
        text, text2     = "#0f172a", "#64748b"
        primary         = "#2563eb"
        accent          = "#7c3aed"
        border          = "#e2e8f0"
        card_bg         = "#ffffff"
        inp_bg          = "#f8fafc"
        success, warn, danger = "#059669", "#d97706", "#dc2626"
        grad_hero       = "linear-gradient(135deg, #dbeafe 0%, #eff6ff 50%, #e0e7ff 100%)"
        nav_bg          = "rgba(255,255,255,0.96)"
        risk_low_bg     = "#dbeafe"; risk_low_t = "#1d4ed8"
        risk_med_bg     = "#fef3c7"; risk_med_t = "#92400e"
        risk_hi_bg      = "#fee2e2"; risk_hi_t  = "#991b1b"
        badge_new_bg    = "#dbeafe"; badge_new_t = "#1d4ed8"

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Sora:wght@400;500;600;700;800&display=swap');

*,*::before,*::after {{ box-sizing:border-box; }}
html,body,.stApp {{ font-family:'Plus Jakarta Sans',sans-serif !important; letter-spacing:-0.1px; }}

/* ── Lift the hero CTA (label + buttons together) up onto the 3D design.
   negative margin keeps them visible without leaving an empty gap below. ────── */
.st-key-hero_cta {{
    position:relative; z-index:6;
    margin-top:-122px !important;
    padding:0 clamp(16px,4vw,60px) 36px;
    background:linear-gradient(180deg,transparent,{'rgba(240,244,255,0.96)' if not dark else 'rgba(8,9,15,0.96)'} 88%);
}}
.st-key-hero_cta .hero-cta-label {{
    text-align:center; color:{primary}; font-size:12px; font-weight:700;
    text-transform:uppercase; letter-spacing:0.1em; padding-top:10px; margin-bottom:14px;
}}
.st-key-hero_cta [data-testid="stHorizontalBlock"] {{
    align-items:center; gap:14px;
}}
.st-key-hero_cta .stButton > button {{ min-height:48px; }}

/* ── Advanced buttons ──────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    position:relative; overflow:hidden;
    border-radius:14px !important; font-weight:700 !important; font-size:14.5px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    padding:0.62rem 1.25rem !important; letter-spacing:0.2px;
    transition:transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
}}
.stButton > button p, .stDownloadButton > button p, .stFormSubmitButton > button p {{
    margin:0 !important;
    padding:0 !important;
    line-height:1.25 !important;
}}
/* moving sheen */
.stButton > button::after, .stDownloadButton > button::after, .stFormSubmitButton > button::after {{
    content:''; position:absolute; top:0; left:-130%; width:60%; height:100%;
    background:linear-gradient(120deg,transparent,rgba(255,255,255,0.45),transparent);
    transform:skewX(-20deg); transition:left 0.6s ease; pointer-events:none;
}}
.stButton > button:hover::after, .stDownloadButton > button:hover::after,
.stFormSubmitButton > button:hover::after {{ left:140%; }}

.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {{
    background:linear-gradient(135deg,{primary} 0%,{accent} 100%) !important;
    border:none !important; color:#fff !important;
    box-shadow:0 10px 26px {primary}45, inset 0 1px 0 rgba(255,255,255,0.35) !important;
}}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {{
    transform:translateY(-3px) scale(1.015) !important; filter:brightness(1.07) !important;
    box-shadow:0 16px 34px {primary}60, inset 0 1px 0 rgba(255,255,255,0.45) !important;
}}
.stButton > button[kind="primary"]:active, .stFormSubmitButton > button[kind="primary"]:active {{
    transform:translateY(-1px) scale(0.99) !important;
}}
.stButton > button[kind="secondary"], .stDownloadButton > button[kind="secondary"] {{
    background:{card_bg} !important; border:1.5px solid {border} !important;
    color:{text} !important;
    box-shadow:0 4px 14px {'rgba(0,0,0,0.28)' if dark else 'rgba(15,23,42,0.06)'} !important;
}}
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button[kind="secondary"]:hover {{
    border-color:{primary} !important; color:{primary} !important;
    transform:translateY(-3px) scale(1.015) !important;
    box-shadow:0 12px 26px {primary}30 !important;
}}

/* ── Shell ─────────────────────────────────────────────────────────────────── */
.stApp {{ background:{bg} !important; color:{text} !important; min-height:100vh; }}
.stApp > header,[data-testid="stHeader"] {{
    display:block !important;
    position:fixed !important;
    inset:0 0 auto 0 !important;
    height:0 !important;
    min-height:0 !important;
    background:transparent !important;
    box-shadow:none !important;
    pointer-events:none !important;
    overflow:visible !important;
}}
{"" if show_sidebar else '[data-testid="stSidebar"] { display:none !important; }'}
[data-testid="stToolbar"] {{
    display:block !important;
    height:0 !important;
    min-height:0 !important;
    pointer-events:none !important;
    background:transparent !important;
}}
[data-testid="stToolbar"] button {{ display:none !important; }}
[data-testid="stToolbar"] [data-testid="stExpandSidebarButton"] {{
    display:{'flex' if show_sidebar else 'none'} !important;
}}
#MainMenu, footer {{ display:none !important; }}
.block-container {{ padding:0 !important; max-width:100% !important; }}
section.main > div {{ padding:0 !important; }}

/* Keep both sidebar directions available. The collapsed control normally lives
   in Streamlit's header, so the zero-height header must allow this one control. */
[data-testid="stSidebarCollapsedControl"] {{
    display:{'flex' if show_sidebar else 'none'} !important;
    position:fixed !important;
    top:14px !important;
    left:14px !important;
    z-index:100000 !important;
    pointer-events:auto !important;
}}
[data-testid="stExpandSidebarButton"] {{
    display:{'flex' if show_sidebar else 'none'} !important;
    align-items:center !important;
    justify-content:center !important;
    position:fixed !important;
    top:14px !important;
    left:14px !important;
    z-index:100001 !important;
    width:38px !important;
    height:38px !important;
    min-height:38px !important;
    padding:0 !important;
    border:1px solid {border} !important;
    border-radius:11px !important;
    color:{text} !important;
    background:{card_bg} !important;
    box-shadow:0 8px 24px {'rgba(0,0,0,.24)' if dark else 'rgba(15,23,42,.10)'} !important;
    pointer-events:auto !important;
}}
[data-testid="stExpandSidebarButton"]:hover {{
    color:{primary} !important;
    border-color:{primary}55 !important;
    background:{card_bg} !important;
}}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {{
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    width:38px !important;
    height:38px !important;
    min-height:38px !important;
    padding:0 !important;
    border:1px solid {border} !important;
    border-radius:11px !important;
    color:{text} !important;
    background:{card_bg} !important;
    box-shadow:0 8px 24px {'rgba(0,0,0,.24)' if dark else 'rgba(15,23,42,.10)'} !important;
}}
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover {{
    color:{primary} !important;
    border-color:{primary}55 !important;
    transform:none !important;
}}
[data-testid="stSidebarCollapseButton"] {{
    pointer-events:auto !important;
    margin:4px 2px 8px auto !important;
}}

/* ── Navbar ─────────────────────────────────────────────────────────────────── */
.lf-nav {{
    position:fixed; top:0; left:0; right:0; z-index:9999;
    background:{nav_bg};
    backdrop-filter:blur(20px);
    border-bottom:1px solid {border};
    padding:0 40px;
    height:68px;
    display:flex; align-items:center; justify-content:space-between;
}}
.lf-logo {{
    display:flex; align-items:center; gap:12px; cursor:pointer;
    text-decoration:none;
}}
.lf-logo-mark {{
    width:40px; height:40px; border-radius:10px;
    background:linear-gradient(135deg,{primary},{accent});
    position:relative; overflow:hidden;
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 4px 16px rgba(37,99,235,0.35);
}}
.lf-logo-inner {{
    width:20px; height:20px; border:2.5px solid white; border-radius:4px;
    position:relative;
}}
.lf-logo-inner::before {{
    content:''; position:absolute; top:3px; left:3px;
    width:10px; height:2px; background:white; border-radius:1px;
    box-shadow:0 4px 0 white, 0 8px 0 white;
}}
.lf-logo-title {{
    font-family:'Sora',sans-serif;
    font-size:20px; font-weight:700; color:{text};
    letter-spacing:-0.5px;
}}
.lf-logo-title span {{ color:{primary}; }}
.lf-nav-links {{
    display:flex; align-items:center; gap:4px;
}}
.lf-nav-link {{
    padding:8px 16px; border-radius:8px; font-size:14px; font-weight:500;
    color:{text2}; cursor:pointer; border:none; background:transparent;
    transition:all 0.15s; text-decoration:none;
    font-family:'Plus Jakarta Sans',sans-serif;
}}
.lf-nav-link:hover {{ color:{primary}; background:{'rgba(37,99,235,0.08)'}; }}
.lf-nav-link.active {{ color:{primary}; background:{'rgba(37,99,235,0.12)'}; }}
.lf-nav-right {{ display:flex; align-items:center; gap:10px; }}
.lf-btn-primary {{
    padding:9px 20px; border-radius:9px; font-size:14px; font-weight:600;
    color:white; background:{primary}; border:none; cursor:pointer;
    font-family:'Plus Jakarta Sans',sans-serif; transition:all 0.15s;
    box-shadow:0 2px 8px rgba(37,99,235,0.3);
}}
.lf-btn-primary:hover {{ filter:brightness(1.08); transform:translateY(-1px); }}
.lf-btn-ghost {{
    padding:9px 16px; border-radius:9px; font-size:14px; font-weight:500;
    color:{text2}; background:transparent; border:1px solid {border}; cursor:pointer;
    font-family:'Plus Jakarta Sans',sans-serif; transition:all 0.15s;
}}
.lf-btn-ghost:hover {{ border-color:{primary}; color:{primary}; }}
.theme-pill {{
    width:52px; height:28px; border-radius:14px; cursor:pointer;
    background:{'#1e293b' if dark else '#e2e8f0'};
    border:none; position:relative; transition:background 0.2s;
    display:flex; align-items:center; padding:3px;
}}
.theme-dot {{
    width:22px; height:22px; border-radius:50%;
    background:{'#3b82f6' if dark else '#94a3b8'};
    transition:transform 0.2s;
    transform:{'translateX(24px)' if dark else 'translateX(0)'};
}}

.lf-logo-img {{ height:38px; width:auto; border-radius:8px; display:block; }}
.lf-brand-name {{
    font-family:'Sora',sans-serif; font-size:19px; font-weight:700;
    color:{text}; letter-spacing:-0.5px;
}}
.lf-brand-name span {{ color:{primary}; }}

/* Public navigation. Streamlit buttons are restyled as compact navigation
   links; the flexible columns keep the main links visually centered. */
.st-key-public_navbar {{
    position:sticky; top:0; z-index:9999;
    min-height:72px;
    padding:10px clamp(16px,3vw,48px);
    background:{nav_bg};
    border-bottom:1px solid {border};
    backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px);
    box-shadow:0 8px 24px {'rgba(0,0,0,0.16)' if dark else 'rgba(15,23,42,0.04)'};
}}
.st-key-public_navbar [data-testid="stHorizontalBlock"] {{
    align-items:center;
    gap:clamp(5px,0.8vw,12px);
}}
.st-key-public_navbar [data-testid="stColumn"] {{
    flex:0 0 auto !important;
    width:auto !important;
    min-width:0 !important;
}}
.st-key-public_navbar [data-testid="stColumn"]:first-child,
.st-key-public_navbar [data-testid="stColumn"]:nth-child(8) {{
    flex:1 1 0 !important;
}}
.st-key-public_navbar .lf-brand-name {{ white-space:nowrap; }}
.st-key-public_navbar .stButton > button {{
    width:auto !important;
    min-height:40px;
    padding:0.5rem 0.9rem !important;
    border:1px solid transparent !important;
    border-radius:999px !important;
    background:transparent !important;
    color:{text2} !important;
    box-shadow:none !important;
    font-size:13.5px !important;
    font-weight:650 !important;
}}
.st-key-public_navbar .stButton > button:hover {{
    transform:none !important;
    color:{primary} !important;
    border-color:{primary}30 !important;
    background:{primary}0d !important;
    box-shadow:none !important;
}}
.st-key-public_navbar .st-key-nav_home button[kind="primary"],
.st-key-public_navbar .st-key-nav_features button[kind="primary"],
.st-key-public_navbar .st-key-nav_architecture button[kind="primary"],
.st-key-public_navbar .st-key-nav_about button[kind="primary"],
.st-key-public_navbar .st-key-nav_contact button[kind="primary"] {{
    color:{primary} !important;
    border-color:{primary}24 !important;
    background:{primary}12 !important;
}}
.st-key-public_navbar .st-key-theme_btn button {{
    width:72px !important;
    min-width:72px !important;
    padding:0 10px !important;
    border-color:{border} !important;
    background:{card_bg} !important;
}}
.st-key-public_navbar .st-key-theme_btn button::before {{
    content:''; width:13px; height:13px; flex:0 0 13px;
    border:1.8px solid currentColor; border-radius:50%;
    box-shadow:5px -3px 0 -2px {card_bg};
}}
.st-key-public_navbar .st-key-signin_btn button {{
    min-width:94px !important;
    padding:0.52rem 1.15rem !important;
    color:#fff !important;
    border-color:transparent !important;
    background:linear-gradient(135deg,{primary},{accent}) !important;
    box-shadow:0 8px 20px {primary}35 !important;
}}
.st-key-public_navbar .st-key-signin_btn button:hover {{
    color:#fff !important;
    background:linear-gradient(135deg,{primary},{accent}) !important;
    box-shadow:0 10px 24px {primary}45 !important;
}}

/* ── Sidebar (logged-in app shell) ─────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background:
        linear-gradient(180deg,{'rgba(37,99,235,0.10)' if not dark else 'rgba(37,99,235,0.08)'} 0%,transparent 34%),
        {surf} !important;
    border-right:1px solid {border};
    box-shadow:8px 0 34px {'rgba(0,0,0,0.22)' if dark else 'rgba(15,23,42,0.06)'};
}}
[data-testid="stSidebar"] > div {{ padding:18px 14px 18px; }}
.sb-brand {{
    display:flex; align-items:center; gap:12px;
    padding:4px 4px 18px;
    border-bottom:1px solid {border};
    margin-bottom:14px;
}}
.sb-brand img {{ height:38px; border-radius:8px; }}
.sb-brand .sb-title {{ font-family:'Sora',sans-serif; font-weight:800; font-size:16px; color:{text}; line-height:1.12; letter-spacing:0; }}
.sb-brand .sb-title span {{ color:{primary}; }}
.sb-user {{
    background:
        linear-gradient(135deg,{'rgba(37,99,235,0.14)' if not dark else 'rgba(59,130,246,0.16)'},transparent),
        {card_bg};
    border:1px solid {border};
    border-radius:14px;
    padding:13px;
    margin:0 0 14px;
    min-height:66px;
}}
.sb-user .sb-avatar {{
    width:38px; height:38px; border-radius:12px; float:left; margin-right:10px;
    background:linear-gradient(135deg,{primary},{accent}); color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700; font-size:15px;
    box-shadow:0 8px 20px {primary}40;
}}
.sb-user .sb-name {{ font-weight:800; font-size:13px; color:{text}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.sb-user .sb-role {{ font-size:10px; color:{primary}; text-transform:uppercase; letter-spacing:0.08em; font-weight:800; margin-top:2px; }}
.sb-api {{
    border:1px solid {'rgba(16,185,129,0.22)' if dark else 'rgba(5,150,105,0.18)'};
    background:{'rgba(16,185,129,0.08)' if dark else 'rgba(236,253,245,0.88)'};
    border-radius:12px;
    padding:10px 12px;
    margin:0 0 16px;
}}
.sb-api-top {{ display:flex; justify-content:space-between; gap:8px; align-items:center; }}
.sb-api-title {{ color:{text}; font-size:11px; font-weight:800; }}
.sb-api-pill {{
    color:{success};
    font-size:9px;
    font-weight:900;
    letter-spacing:0.08em;
    text-transform:uppercase;
}}
.sb-api-model {{ color:{text2}; font-size:10.5px; line-height:1.45; margin-top:5px; word-break:break-word; }}
.sb-section {{
    font-size:9.5px; font-weight:850; color:{text2}; text-transform:uppercase;
    letter-spacing:0.14em; margin:20px 10px 8px;
}}
[data-testid="stSidebar"] .stButton > button {{
    width:100%;
    min-height:44px;
    text-align:left;
    justify-content:flex-start;
    gap:11px !important;
    border-radius:10px !important;
    font-weight:700 !important;
    font-size:12.5px !important;
    border:1px solid transparent !important;
    background:transparent !important;
    color:{text2} !important;
    box-shadow:none !important;
    margin-bottom:3px;
    padding:0.62rem 0.72rem !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background:{primary}12 !important;
    color:{primary} !important;
    border-color:{primary}24 !important;
    box-shadow:inset 3px 0 0 {primary} !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    color:{primary} !important;
    border-color:{primary}20 !important;
    background:{primary}0b !important;
    box-shadow:none !important;
    transform:none !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    box-shadow:inset 3px 0 0 {primary} !important;
}}
.sb-footer {{
    color:{text2};
    font-size:10.5px;
    line-height:1.6;
    padding:16px 6px 4px;
    border-top:1px solid {border};
    margin-top:12px;
}}
.app-topbar {{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:18px;
    padding:14px 28px;
    min-height:68px;
    background:{nav_bg};
    border-bottom:1px solid {border};
    backdrop-filter:blur(20px);
}}
.app-topbar-title {{
    font-family:'Sora',sans-serif;
    font-size:15px;
    font-weight:800;
    color:{text};
}}
.app-topbar-sub {{
    color:{text2};
    font-size:11.5px;
    margin-top:2px;
}}
.app-topbar-right {{
    display:flex;
    align-items:center;
    justify-content:flex-end;
    gap:10px;
    color:{text};
    font-size:12px;
    font-weight:700;
}}
.api-status-pill {{
    border:1px solid {'rgba(16,185,129,0.24)' if dark else 'rgba(5,150,105,0.18)'};
    background:{'rgba(16,185,129,0.10)' if dark else 'rgba(236,253,245,0.92)'};
    color:{success};
    border-radius:999px;
    padding:7px 11px;
    font-size:11px;
    font-weight:900;
    letter-spacing:0.04em;
}}

/* ── 3D Hero background ─────────────────────────────────────────────────────── */
.hero3d-wrap {{ position:relative; }}
.hero3d-canvas {{ position:absolute; inset:0; z-index:0; }}
.hero3d-content {{ position:relative; z-index:2; }}

/* ── Signature blocks ──────────────────────────────────────────────────────── */
.sig-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:24px; margin-top:18px; }}
.sig-box {{ border:1px solid {border}; border-radius:12px; padding:18px 20px 14px; background:{card_bg}; }}
.sig-line {{ border-bottom:1.5px solid {text2}; height:34px; margin:14px 0 6px; }}
.sig-label {{ font-size:12px; color:{text2}; }}
.sig-role {{ font-size:12px; font-weight:700; color:{primary}; text-transform:uppercase; letter-spacing:0.05em; }}

/* ── Page Wrapper ─────────────────────────────────────────────────────────── */
.page-wrap {{ padding-top:68px; }}

/* ── Hero ─────────────────────────────────────────────────────────────────── */
.hero-section {{
    background:{grad_hero};
    padding:100px 60px 80px;
    text-align:center;
    position:relative; overflow:hidden;
    min-height:90vh;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
}}
.hero-badge {{
    display:inline-flex; align-items:center; gap:8px;
    background:{badge_new_bg}; color:{badge_new_t};
    padding:6px 16px; border-radius:100px; font-size:12px; font-weight:600;
    border:1px solid {badge_new_t}33;
    margin-bottom:28px; letter-spacing:0.05em;
}}
.hero-title {{
    font-family:'Sora',sans-serif;
    font-size:clamp(42px,6vw,80px); font-weight:700;
    color:{text}; line-height:1.08; letter-spacing:-2px;
    margin-bottom:24px;
}}
.hero-title .grad-text {{
    background:linear-gradient(135deg,{primary},{accent});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}}
.hero-sub {{
    font-size:18px; color:{text2}; max-width:640px;
    margin:0 auto 40px; line-height:1.7; font-weight:400;
}}
.hero-cta-row {{
    display:flex; align-items:center; justify-content:center; gap:14px;
    flex-wrap:wrap;
}}
.cta-primary {{
    display:inline-flex; align-items:center; gap:8px;
    padding:14px 32px; border-radius:12px;
    background:{primary}; color:white;
    font-size:16px; font-weight:700;
    border:none; cursor:pointer;
    box-shadow:0 4px 20px rgba(37,99,235,0.4);
    font-family:'Plus Jakarta Sans',sans-serif; transition:all 0.2s;
}}
.cta-primary:hover {{ transform:translateY(-2px); filter:brightness(1.08); }}
.cta-secondary {{
    display:inline-flex; align-items:center; gap:8px;
    padding:14px 28px; border-radius:12px;
    background:transparent; color:{text};
    font-size:16px; font-weight:600;
    border:1.5px solid {border}; cursor:pointer;
    font-family:'Plus Jakarta Sans',sans-serif; transition:all 0.2s;
}}
.cta-secondary:hover {{ border-color:{primary}; color:{primary}; }}
.hero-stats {{
    display:flex; justify-content:center; gap:48px;
    margin-top:60px; flex-wrap:wrap;
}}
.hero-stat {{ text-align:center; }}
.hero-stat-val {{
    font-family:'Sora',sans-serif;
    font-size:32px; font-weight:700; color:{primary};
}}
.hero-stat-label {{
    font-size:12px; color:{text2}; font-weight:500;
    text-transform:uppercase; letter-spacing:0.08em; margin-top:4px;
}}
.hero-float {{
    position:absolute; border-radius:50%; opacity:0.06;
    background:{primary}; animation:float 8s ease-in-out infinite;
}}
@keyframes float {{
    0%,100% {{ transform:translateY(0) scale(1); }}
    50% {{ transform:translateY(-20px) scale(1.05); }}
}}

/* ── Section ─────────────────────────────────────────────────────────────── */
.section {{ padding:80px 60px; }}
.section-sm {{ padding:48px 60px; }}
.section-label {{
    font-size:12px; font-weight:700; color:{primary};
    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;
}}
.section-title {{
    font-family:'Sora',sans-serif;
    font-size:clamp(28px,3vw,42px); font-weight:700;
    color:{text}; letter-spacing:-1px; margin-bottom:14px;
}}
.section-sub {{ font-size:16px; color:{text2}; max-width:560px; line-height:1.7; }}

/* ── Feature Cards ────────────────────────────────────────────────────────── */
.feat-grid {{
    display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:20px;
    margin-top:40px;
}}
.feat-card {{
    background:{card_bg}; border:1px solid {border};
    border-radius:16px; padding:28px;
    transition:all 0.2s;
    position:relative; overflow:hidden;
}}
.feat-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,{primary},{accent});
    opacity:0; transition:opacity 0.2s;
}}
.feat-card:hover {{ transform:translateY(-4px); box-shadow:0 12px 40px {'rgba(0,0,0,0.3)' if dark else 'rgba(0,0,0,0.1)'}; }}
.feat-card:hover::before {{ opacity:1; }}
.feat-icon {{
    width:52px; height:52px; border-radius:14px;
    background:linear-gradient(135deg,{'rgba(37,99,235,0.22)' if dark else '#e6efff'},
                                       {'rgba(139,92,246,0.18)' if dark else '#efe8ff'});
    border:1px solid {'rgba(99,140,255,0.25)' if dark else 'rgba(37,99,235,0.18)'};
    display:flex; align-items:center; justify-content:center;
    margin-bottom:18px; box-shadow:0 6px 18px {'rgba(0,0,0,0.35)' if dark else 'rgba(37,99,235,0.12)'};
}}
.feat-icon svg {{ width:26px; height:26px; display:block; }}
.feat-icon svg path, .feat-icon svg line, .feat-icon svg polyline,
.feat-icon svg circle, .feat-icon svg rect, .feat-icon svg polygon {{
    stroke:{primary}; stroke-width:1.8; fill:none;
    stroke-linecap:round; stroke-linejoin:round;
}}
.feat-title {{ font-size:16px; font-weight:700; color:{text}; margin-bottom:8px; }}
.feat-desc {{ font-size:13px; color:{text2}; line-height:1.6; }}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.card {{
    background:{card_bg}; border:1px solid {border};
    border-radius:16px; padding:24px;
    box-shadow:0 2px 12px {'rgba(0,0,0,0.2)' if dark else 'rgba(0,0,0,0.06)'};
    transition:all 0.2s;
}}
.card:hover {{ box-shadow:0 6px 24px {'rgba(0,0,0,0.3)' if dark else 'rgba(0,0,0,0.1)'}; }}
.card-sm {{ padding:16px; border-radius:12px; }}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
.metric-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }}
.metric-card {{
    background:{card_bg}; border:1px solid {border}; border-radius:14px;
    padding:20px; border-top:3px solid {primary};
    box-shadow:0 2px 8px {'rgba(0,0,0,0.2)' if dark else 'rgba(0,0,0,0.05)'};
}}
.metric-val {{ font-family:'Sora',sans-serif; font-size:28px; font-weight:700; color:{primary}; }}
.metric-label {{ font-size:11px; color:{text2}; text-transform:uppercase; letter-spacing:0.07em; margin-top:4px; font-weight:600; }}
.metric-sub {{ font-size:12px; color:{success}; margin-top:2px; }}

/* Professional workspace surfaces */
.command-shell {{ padding:34px clamp(22px,3vw,46px) 60px; }}
.command-hero {{
    position:relative; overflow:hidden;
    display:flex; align-items:flex-end; justify-content:space-between; gap:28px;
    min-height:190px; padding:34px 38px; margin-bottom:22px;
    border:1px solid {border}; border-radius:22px;
    background:
      radial-gradient(circle at 88% 18%,{accent}24,transparent 28%),
      linear-gradient(135deg,{primary}18,transparent 52%),{card_bg};
    box-shadow:0 20px 55px {'rgba(0,0,0,0.28)' if dark else 'rgba(15,23,42,0.08)'};
}}
.command-kicker {{ color:{primary}; font-size:11px; font-weight:850; letter-spacing:.12em; text-transform:uppercase; margin-bottom:10px; }}
.command-title {{ font-family:'Sora',sans-serif; color:{text}; font-size:clamp(29px,3.4vw,48px); line-height:1.06; font-weight:800; letter-spacing:-1.4px; }}
.command-copy {{ color:{text2}; font-size:14px; line-height:1.65; max-width:720px; margin-top:13px; }}
.command-runtime {{ display:flex; gap:9px; flex-wrap:wrap; justify-content:flex-end; max-width:360px; }}
.runtime-chip {{ display:inline-flex; align-items:center; gap:7px; padding:8px 11px; border:1px solid {border}; border-radius:999px; background:{card_bg}; color:{text}; font-size:10.5px; font-weight:800; white-space:nowrap; }}
.runtime-chip::before {{ content:''; width:7px; height:7px; border-radius:50%; background:{success}; box-shadow:0 0 0 4px {success}18; }}
.runtime-chip.warn::before {{ background:{warn}; box-shadow:0 0 0 4px {warn}18; }}
.pro-metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:18px 0 28px; }}
.pro-metric {{ position:relative; overflow:hidden; min-height:132px; border:1px solid {border}; border-radius:17px; padding:20px; background:{card_bg}; box-shadow:0 10px 30px {'rgba(0,0,0,.18)' if dark else 'rgba(15,23,42,.055)'}; }}
.pro-metric-icon,.action-icon,.infra-icon {{ width:36px; height:36px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:{primary}12; border:1px solid {primary}22; }}
.pro-metric-icon svg,.action-icon svg,.infra-icon svg {{ width:19px; height:19px; fill:none; stroke:{primary}; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }}
.pro-metric-value {{ color:{text}; font-family:'Sora',sans-serif; font-size:30px; line-height:1; font-weight:800; margin-top:15px; }}
.pro-metric-label {{ color:{text2}; font-size:11px; text-transform:uppercase; letter-spacing:.08em; font-weight:800; margin-top:7px; }}
.workspace-grid {{ display:grid; grid-template-columns:minmax(0,1.35fr) minmax(300px,.65fr); gap:18px; margin-top:22px; }}
.workspace-panel {{ border:1px solid {border}; border-radius:18px; padding:22px; background:{card_bg}; box-shadow:0 12px 34px {'rgba(0,0,0,.18)' if dark else 'rgba(15,23,42,.05)'}; }}
.panel-head {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:18px; }}
.panel-title {{ font-family:'Sora',sans-serif; color:{text}; font-size:17px; font-weight:800; }}
.panel-sub {{ color:{text2}; font-size:11.5px; line-height:1.55; margin-top:4px; }}
.pipeline-row,.contract-row {{ display:grid; grid-template-columns:34px minmax(0,1fr) auto; align-items:center; gap:12px; padding:11px 0; border-bottom:1px solid {border}; }}
.pipeline-row:last-child,.contract-row:last-child {{ border-bottom:0; }}
.pipeline-index {{ width:28px; height:28px; display:flex; align-items:center; justify-content:center; border-radius:9px; background:{primary}12; color:{primary}; font-size:10px; font-weight:900; }}
.row-title {{ color:{text}; font-size:12.5px; font-weight:750; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.row-sub {{ color:{text2}; font-size:10.5px; margin-top:2px; }}
.row-status {{ color:{success}; font-size:9.5px; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }}
.section-heading {{ color:{text}; font-family:'Sora',sans-serif; font-size:22px; font-weight:800; margin:28px 0 12px; }}
.action-labels {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
.action-label {{ border:1px solid {border}; border-radius:16px 16px 8px 8px; background:{card_bg}; padding:16px 17px 10px; }}
.action-label-title {{ color:{text}; font-size:12.5px; font-weight:800; margin-top:11px; }}
.action-label-copy {{ color:{text2}; font-size:10.5px; line-height:1.45; margin-top:3px; min-height:30px; }}
.st-key-dashboard_actions {{ margin:-9px clamp(22px,3vw,46px) 0; }}
.st-key-dashboard_actions .stButton > button {{ border-radius:8px 8px 14px 14px !important; min-height:42px; }}
.infra-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:18px 0; }}
.infra-card {{ border:1px solid {border}; border-radius:16px; padding:18px; background:{card_bg}; }}
.infra-name {{ color:{text}; font-size:13px; font-weight:850; margin-top:12px; }}
.infra-detail {{ color:{text2}; font-size:10.5px; line-height:1.5; margin-top:4px; min-height:31px; }}
.infra-state {{ display:inline-flex; margin-top:12px; padding:5px 8px; border-radius:999px; background:{success}12; color:{success}; font-size:9px; font-weight:900; letter-spacing:.08em; text-transform:uppercase; }}
.infra-state.ready {{ background:{primary}12; color:{primary}; }}
.evidence-card {{ border:1px solid {border}; border-left:3px solid {primary}; border-radius:12px; padding:14px 16px; margin:9px 0; background:{card_bg}; }}
.evidence-score {{ color:{primary}; font-size:10px; font-weight:900; text-transform:uppercase; letter-spacing:.08em; }}
.evidence-text {{ color:{text2}; font-size:12px; line-height:1.6; margin-top:5px; }}

/* AI decision room */
.ai-runtime {{
    position:relative; overflow:hidden; margin:4px 0 18px; padding:22px;
    border:1px solid {primary}32; border-radius:18px;
    background:linear-gradient(135deg,{primary}12,{accent}0d 55%,{card_bg});
}}
.ai-runtime::after {{
    content:''; position:absolute; width:190px; height:190px; right:-85px; top:-110px;
    border-radius:50%; background:{accent}18; filter:blur(2px);
}}
.ai-runtime-top {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; position:relative; z-index:1; }}
.ai-runtime-kicker {{ color:{primary}; font-size:9.5px; font-weight:900; letter-spacing:.13em; text-transform:uppercase; }}
.ai-runtime-title {{ color:{text}; font-family:'Sora',sans-serif; font-size:19px; font-weight:850; margin-top:6px; }}
.ai-runtime-copy {{ color:{text2}; font-size:11.5px; line-height:1.55; margin-top:5px; max-width:680px; }}
.ai-live {{ display:inline-flex; align-items:center; gap:7px; color:{success}; font-size:9.5px; font-weight:900; letter-spacing:.09em; text-transform:uppercase; white-space:nowrap; }}
.ai-live::before {{ content:''; width:8px; height:8px; border-radius:50%; background:{success}; box-shadow:0 0 0 5px {success}16; }}
.ai-trace {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:9px; margin-top:18px; position:relative; z-index:1; }}
.ai-node {{ min-height:82px; padding:12px; border:1px solid {border}; border-radius:12px; background:{card_bg}; }}
.ai-node-id {{ color:{primary}; font-size:9px; font-weight:900; letter-spacing:.09em; text-transform:uppercase; }}
.ai-node-name {{ color:{text}; font-size:11.5px; font-weight:800; margin-top:6px; }}
.ai-node-detail {{ color:{text2}; font-size:9.5px; line-height:1.35; margin-top:3px; }}
.ai-node.fallback {{ border-color:{warn}45; }}
.ai-node.fallback .ai-node-id {{ color:{warn}; }}
.decision-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:14px 0 20px; }}
.decision-metric {{ border:1px solid {border}; border-radius:14px; padding:16px; background:{card_bg}; }}
.decision-value {{ color:{text}; font-family:'Sora',sans-serif; font-size:25px; font-weight:850; }}
.decision-label {{ color:{text2}; font-size:9.5px; font-weight:850; letter-spacing:.08em; text-transform:uppercase; margin-top:5px; }}
.deal-row {{ display:grid; grid-template-columns:38px minmax(0,1fr) auto; gap:12px; align-items:start; padding:13px 0; border-bottom:1px solid {border}; }}
.deal-row:last-child {{ border-bottom:0; }}
.deal-rank {{ width:32px; height:32px; display:flex; align-items:center; justify-content:center; border-radius:10px; color:#fff; background:linear-gradient(135deg,{primary},{accent}); font-size:11px; font-weight:900; }}
.deal-title {{ color:{text}; font-size:12.5px; font-weight:850; }}
.deal-copy {{ color:{text2}; font-size:10.5px; line-height:1.5; margin-top:4px; }}
.deal-score {{ color:{primary}; font-family:'Sora',sans-serif; font-size:16px; font-weight:850; white-space:nowrap; }}
.guardrail {{ display:flex; align-items:flex-start; gap:10px; padding:11px 0; border-bottom:1px solid {border}; }}
.guardrail:last-child {{ border-bottom:0; }}
.guardrail-dot {{ width:9px; height:9px; margin-top:4px; border-radius:50%; background:{success}; box-shadow:0 0 0 4px {success}14; flex:0 0 auto; }}
.guardrail-dot.review {{ background:{warn}; box-shadow:0 0 0 4px {warn}14; }}
.guardrail-dot.fail {{ background:{danger}; box-shadow:0 0 0 4px {danger}14; }}
.guardrail-title {{ color:{text}; font-size:11.5px; font-weight:800; }}
.guardrail-detail {{ color:{text2}; font-size:10px; margin-top:2px; }}

/* Distinct vector icons for every workspace destination. */
[data-testid="stSidebar"] .stButton > button::before {{
    content:'';
    width:18px; height:18px; flex:0 0 18px;
    border:0; border-radius:0; box-shadow:none;
    background-color:currentColor;
    -webkit-mask-image:var(--sb-icon);
    mask-image:var(--sb-icon);
    -webkit-mask-position:center; mask-position:center;
    -webkit-mask-repeat:no-repeat; mask-repeat:no-repeat;
    -webkit-mask-size:contain; mask-size:contain;
}}
.st-key-sb_dashboard {{ --sb-icon:url("{SIDEBAR_ICON_URIS['dashboard']}"); }}
.st-key-sb_analyze {{ --sb-icon:url("{SIDEBAR_ICON_URIS['analyze']}"); }}
.st-key-sb_writer {{ --sb-icon:url("{SIDEBAR_ICON_URIS['writer']}"); }}
.st-key-sb_chatbot {{ --sb-icon:url("{SIDEBAR_ICON_URIS['chatbot']}"); }}
.st-key-sb_agent {{ --sb-icon:url("{SIDEBAR_ICON_URIS['agent']}"); }}
.st-key-sb_chat {{ --sb-icon:url("{SIDEBAR_ICON_URIS['chat']}"); }}
.st-key-sb_knowledge {{ --sb-icon:url("{SIDEBAR_ICON_URIS['knowledge']}"); }}
.st-key-sb_compare {{ --sb-icon:url("{SIDEBAR_ICON_URIS['compare']}"); }}
.st-key-sb_history {{ --sb-icon:url("{SIDEBAR_ICON_URIS['history']}"); }}
.st-key-sb_integrations {{ --sb-icon:url("{SIDEBAR_ICON_URIS['integrations']}"); }}
.st-key-sb_about {{ --sb-icon:url("{SIDEBAR_ICON_URIS['about']}"); }}
.st-key-sb_contact {{ --sb-icon:url("{SIDEBAR_ICON_URIS['contact']}"); }}
.st-key-sb_theme {{ --sb-icon:url("{SIDEBAR_ICON_URIS['theme']}"); }}
.st-key-sb_settings {{ --sb-icon:url("{SIDEBAR_ICON_URIS['settings']}"); }}
.st-key-sb_logout {{ --sb-icon:url("{SIDEBAR_ICON_URIS['logout']}"); }}

/* ── Badges ───────────────────────────────────────────────────────────────── */
.badge {{
    display:inline-flex; align-items:center; gap:4px;
    padding:4px 12px; border-radius:100px;
    font-size:11px; font-weight:700; letter-spacing:0.04em;
}}
.b-critical {{ background:#fee2e2; color:#991b1b; }}
.b-high     {{ background:#fef3c7; color:#92400e; }}
.b-medium   {{ background:#dcfce7; color:#166534; }}
.b-low      {{ background:#dbeafe; color:#1d4ed8; }}
.b-warn     {{ background:#fef3c7; color:#92400e; }}
.b-pass     {{ background:#dcfce7; color:#166534; }}
.b-fail     {{ background:#fee2e2; color:#991b1b; }}
.b-na       {{ background:{surf2};  color:{text2};  }}
.b-favorable  {{ background:#dcfce7; color:#166534; }}
.b-neutral    {{ background:{surf2};  color:{text2};  }}
.b-concerning {{ background:#fee2e2; color:#991b1b; }}

/* ── Risk Items ───────────────────────────────────────────────────────────── */
.risk-item {{
    background:{surf2}; border-left:4px solid; border-radius:10px;
    padding:14px 16px; margin:8px 0; transition:0.2s;
}}
.risk-critical {{ border-color:{danger}; background:{risk_hi_bg}; }}
.risk-high     {{ border-color:{warn};   background:{risk_med_bg}; }}
.risk-medium   {{ border-color:#3b82f6;  background:{risk_low_bg}; }}
.risk-low      {{ border-color:{success}; }}
.risk-title    {{ font-size:14px; font-weight:700; color:{text}; margin-bottom:4px; }}
.risk-desc     {{ font-size:12px; color:{text2}; line-height:1.5; }}

/* ── Obligations ──────────────────────────────────────────────────────────── */
.obl-item {{
    background:{surf2}; border:1px solid {border}; border-radius:10px;
    padding:14px 16px; margin:6px 0;
    display:flex; align-items:flex-start; gap:12px;
}}
.obl-num {{
    min-width:28px; height:28px; border-radius:50%;
    background:{primary}22; color:{primary};
    display:flex; align-items:center; justify-content:center;
    font-size:11px; font-weight:700; flex-shrink:0;
}}
.obl-body {{ flex:1; }}
.obl-title {{ font-size:13px; font-weight:600; color:{text}; margin-bottom:3px; }}
.obl-meta  {{ font-size:11px; color:{text2}; }}

/* ── Timeline ─────────────────────────────────────────────────────────────── */
.timeline-item {{
    display:flex; gap:16px; padding:12px 0;
    border-bottom:1px solid {border};
}}
.timeline-dot {{
    width:10px; height:10px; border-radius:50%;
    background:{primary}; margin-top:4px; flex-shrink:0;
}}
.timeline-body {{ flex:1; }}
.timeline-title {{ font-size:13px; font-weight:600; color:{text}; }}
.timeline-meta  {{ font-size:11px; color:{text2}; margin-top:2px; }}

/* ── Compliance ───────────────────────────────────────────────────────────── */
.compliance-row {{
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 0; border-bottom:1px solid {border};
}}
.compliance-label {{ font-size:13px; font-weight:600; color:{text}; }}
.score-bar-wrap {{ width:120px; height:6px; background:{border}; border-radius:3px; overflow:hidden; }}
.score-bar-fill {{ height:100%; border-radius:3px; }}

/* ── Chat ────────────────────────────────────────────────────────────────── */
.chat-msg {{
    display:flex; gap:12px; margin:10px 0;
    animation:fadein 0.3s ease;
}}
.chat-avatar {{
    width:32px; height:32px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:13px; font-weight:700; flex-shrink:0;
}}
.chat-user .chat-avatar {{ background:{primary}; color:white; }}
.chat-ai   .chat-avatar {{ background:linear-gradient(135deg,{primary},{accent}); color:white; }}
.chat-bubble {{
    max-width:75%; padding:12px 16px; border-radius:14px;
    font-size:14px; line-height:1.6; color:{text};
}}
.chat-user .chat-bubble {{ background:{primary}22; border-bottom-right-radius:4px; margin-left:auto; }}
.chat-ai   .chat-bubble {{ background:{card_bg}; border:1px solid {border}; border-bottom-left-radius:4px; }}
@keyframes fadein {{ from{{opacity:0;transform:translateY(6px)}} to{{opacity:1;transform:none}} }}

/* ── Table ────────────────────────────────────────────────────────────────── */
.lf-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.lf-table th {{
    background:{surf2}; color:{text2}; font-weight:600; font-size:11px;
    text-transform:uppercase; letter-spacing:0.06em;
    padding:10px 14px; border-bottom:1px solid {border}; text-align:left;
}}
.lf-table td {{ padding:12px 14px; border-bottom:1px solid {border}; color:{text}; vertical-align:top; }}
.lf-table tr:hover td {{ background:{surf2}; }}

/* ── Form Elements ────────────────────────────────────────────────────────── */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea,
.stSelectbox>div>div {{
    background:{inp_bg} !important;
    border:1px solid {border} !important;
    border-radius:10px !important;
    color:{text} !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
}}
.stButton>button {{
    border-radius:10px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:600 !important;
    transition:all 0.15s !important;
}}
.stButton>button[kind="primary"] {{
    background:{primary} !important;
    color:white !important;
    border:none !important;
    box-shadow:0 2px 8px rgba(37,99,235,0.3) !important;
}}
.stButton>button[kind="primary"]:hover {{
    filter:brightness(1.08) !important;
    transform:translateY(-1px) !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    display:flex !important;
    align-items:center !important;
    background:{surf2} !important;
    border:1px solid {border} !important;
    border-radius:13px !important;
    padding:6px !important;
    gap:8px !important;
}}
.stTabs [data-baseweb="tab"] {{
    min-height:40px !important;
    padding:8px 16px !important;
    margin:0 !important;
    border-radius:9px !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:650 !important; font-size:13px !important;
    color:{text2} !important;
    white-space:nowrap !important;
    justify-content:center !important;
}}
.stTabs [data-baseweb="tab"] p {{
    margin:0 !important;
    padding:0 !important;
    line-height:1.25 !important;
}}
.stTabs [aria-selected="true"] {{
    background:{card_bg} !important; color:{primary} !important;
    box-shadow:0 1px 4px {'rgba(0,0,0,0.3)' if dark else 'rgba(0,0,0,0.1)'} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top:20px !important;
}}
[data-testid="stDialog"] .stTabs [data-baseweb="tab"] {{
    flex:1 1 0 !important;
}}
[data-testid="stDialog"] [data-testid="stVerticalBlock"] {{
    gap:14px !important;
}}
[data-testid="stDialog"] [role="dialog"] {{
    border-radius:20px !important;
    border:1px solid {border} !important;
    background:{card_bg} !important;
    box-shadow:0 28px 80px {'rgba(0,0,0,.45)' if dark else 'rgba(15,23,42,.20)'} !important;
}}
[data-testid="stDialog"] [data-testid="stForm"] {{
    border:0 !important;
    padding:0 !important;
}}
[data-testid="stFileUploader"] {{
    border:2px dashed {border} !important;
    border-radius:14px !important;
    background:{surf2} !important;
    transition:border-color 0.2s;
}}
[data-testid="stFileUploader"]:hover {{ border-color:{primary} !important; }}
[data-testid="stExpander"] {{
    border:1px solid {border} !important;
    border-radius:12px !important;
    background:{card_bg} !important;
}}
.stProgress > div > div {{ background:{primary} !important; border-radius:4px !important; }}

/* ── About/Dev Page ───────────────────────────────────────────────────────── */
.about-shell,.contact-shell {{
    padding:34px 60px 56px;
}}
.about-hero,.contact-hero {{
    position:relative; overflow:hidden;
    border:1px solid {border};
    border-radius:24px;
    background:
        linear-gradient(135deg,{'rgba(37,99,235,0.14)' if dark else 'rgba(37,99,235,0.10)'} 0%,transparent 42%),
        linear-gradient(160deg,{'rgba(16,185,129,0.10)' if dark else 'rgba(16,185,129,0.08)'} 28%,transparent 66%),
        {card_bg};
    box-shadow:0 24px 70px {'rgba(0,0,0,0.34)' if dark else 'rgba(15,23,42,0.10)'};
}}
.about-hero {{
    display:grid;
    grid-template-columns:minmax(0,1.15fr) minmax(320px,0.85fr);
    gap:34px;
    padding:42px;
}}
.contact-hero {{ padding:42px; }}
.eyebrow {{
    display:inline-flex; align-items:center; gap:8px;
    color:{primary};
    background:{'rgba(37,99,235,0.16)' if dark else 'rgba(37,99,235,0.09)'};
    border:1px solid {'rgba(96,165,250,0.26)' if dark else 'rgba(37,99,235,0.16)'};
    border-radius:999px;
    padding:7px 12px;
    font-size:11px;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:0.08em;
    width:max-content;
    max-width:100%;
}}
.eyebrow-dot {{
    width:7px; height:7px; border-radius:999px; background:{success};
    box-shadow:0 0 0 5px {'rgba(16,185,129,0.12)' if dark else 'rgba(5,150,105,0.12)'};
    flex:0 0 auto;
}}
.about-title,.contact-title {{
    font-family:'Sora',sans-serif;
    color:{text};
    font-size:clamp(32px,4.5vw,58px);
    line-height:1.03;
    letter-spacing:0;
    font-weight:800;
    margin:18px 0;
    max-width:840px;
}}
.about-copy,.contact-copy {{
    color:{text2};
    font-size:16px;
    line-height:1.8;
    max-width:760px;
}}
.about-actions {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:28px; }}
.about-chip,.contact-chip {{
    display:inline-flex; align-items:center; gap:8px;
    border:1px solid {border};
    background:{'rgba(255,255,255,0.05)' if dark else 'rgba(255,255,255,0.74)'};
    color:{text};
    border-radius:999px;
    padding:9px 13px;
    font-size:12px;
    font-weight:700;
}}
.chip-swatch {{ width:8px; height:8px; border-radius:50%; background:{warn}; }}
.about-system-panel {{
    border:1px solid {border};
    background:{'rgba(13,17,23,0.76)' if dark else 'rgba(248,250,252,0.82)'};
    border-radius:18px;
    padding:18px;
    align-self:stretch;
}}
.system-head {{
    display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding-bottom:14px;
    border-bottom:1px solid {border};
    margin-bottom:16px;
}}
.system-title {{ color:{text}; font-size:13px; font-weight:800; }}
.system-live {{
    color:{success};
    background:{'rgba(16,185,129,0.12)' if dark else 'rgba(5,150,105,0.10)'};
    border:1px solid {'rgba(16,185,129,0.24)' if dark else 'rgba(5,150,105,0.18)'};
    border-radius:999px;
    padding:5px 9px;
    font-size:10px;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:0.08em;
}}
.system-row {{
    display:grid;
    grid-template-columns:34px minmax(0,1fr) auto;
    gap:12px;
    align-items:center;
    padding:11px 0;
    border-bottom:1px solid {'rgba(148,163,184,0.14)' if dark else 'rgba(226,232,240,0.88)'};
}}
.system-row:last-child {{ border-bottom:none; }}
.system-icon {{
    width:34px; height:34px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    color:white; background:linear-gradient(135deg,{primary},{accent});
}}
.system-icon svg,.contact-method-icon svg {{ width:18px; height:18px; }}
.system-icon svg path,.system-icon svg rect,.system-icon svg polyline,.system-icon svg circle,
.contact-method-icon svg path,.contact-method-icon svg rect,.contact-method-icon svg polyline,.contact-method-icon svg circle {{
    stroke:currentColor; stroke-width:1.9; fill:none; stroke-linecap:round; stroke-linejoin:round;
}}
.system-name {{ color:{text}; font-size:13px; font-weight:800; }}
.system-desc {{ color:{text2}; font-size:11px; line-height:1.45; margin-top:2px; }}
.system-score {{
    color:{primary};
    background:{'rgba(96,165,250,0.13)' if dark else 'rgba(37,99,235,0.08)'};
    border:1px solid {'rgba(96,165,250,0.20)' if dark else 'rgba(37,99,235,0.14)'};
    border-radius:999px;
    padding:5px 8px;
    font-size:11px;
    font-weight:800;
    white-space:nowrap;
}}
.about-section-head,.contact-section-head {{
    display:flex; justify-content:space-between; align-items:end; gap:20px;
    margin:34px 0 16px;
}}
.about-section-title,.contact-section-title {{
    font-family:'Sora',sans-serif;
    color:{text};
    font-size:22px;
    font-weight:800;
    letter-spacing:0;
}}
.about-section-sub,.contact-section-sub {{
    color:{text2};
    font-size:13px;
    line-height:1.65;
    max-width:560px;
}}
.about-kpi-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
.about-kpi,.contact-stat {{
    border:1px solid {border};
    background:{card_bg};
    border-radius:16px;
    padding:20px;
    box-shadow:0 14px 34px {'rgba(0,0,0,0.22)' if dark else 'rgba(15,23,42,0.06)'};
}}
.about-kpi-value,.contact-stat-value {{
    font-family:'Sora',sans-serif;
    color:{text};
    font-size:30px;
    font-weight:800;
    letter-spacing:0;
}}
.about-kpi-label,.contact-stat-label {{
    color:{text2};
    font-size:12px;
    font-weight:700;
    margin-top:6px;
    line-height:1.4;
}}
.about-feature-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
.about-feature {{
    border:1px solid {border};
    background:{card_bg};
    border-radius:16px;
    padding:20px;
    min-height:170px;
    transition:transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}}
.about-feature:hover {{
    transform:translateY(-3px);
    border-color:{primary};
    box-shadow:0 18px 42px {'rgba(0,0,0,0.28)' if dark else 'rgba(37,99,235,0.10)'};
}}
.about-feature-top {{ display:flex; align-items:center; gap:12px; margin-bottom:12px; }}
.about-feature-index {{
    width:32px; height:32px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    color:white;
    font-size:12px; font-weight:900;
    background:linear-gradient(135deg,{primary},{accent});
}}
.about-feature-title {{ color:{text}; font-size:14px; font-weight:800; line-height:1.35; }}
.about-feature-body {{ color:{text2}; font-size:12.5px; line-height:1.7; }}
.tech-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
.tech-pill {{
    background:{card_bg};
    border:1px solid {border};
    border-radius:12px;
    padding:14px;
    min-height:96px;
    color:{text};
    box-shadow:0 10px 28px {'rgba(0,0,0,0.18)' if dark else 'rgba(15,23,42,0.045)'};
}}
.tech-pill strong {{ display:block; color:{text}; font-size:12px; font-weight:800; margin-bottom:7px; }}
.tech-pill span {{ display:block; color:{text2}; font-size:11.5px; line-height:1.55; }}
.about-footer-note {{
    margin-top:28px;
    display:flex; align-items:center; gap:18px;
    border:1px solid {border};
    background:{card_bg};
    border-radius:16px;
    padding:20px;
}}
.about-footer-note img {{ height:46px; width:auto; display:block; flex:0 0 auto; }}
.about-footer-title {{ color:{text}; font-weight:800; font-size:15px; margin-bottom:4px; }}
.about-footer-copy {{ color:{text2}; font-size:12px; line-height:1.65; }}

/* ── Steps ────────────────────────────────────────────────────────────────── */
.steps-row {{
    display:grid; grid-template-columns:repeat(4,1fr); gap:0;
    position:relative;
}}
.step-item {{ text-align:center; padding:0 16px; position:relative; }}
.step-num {{
    width:48px; height:48px; border-radius:50%;
    background:{primary}; color:white;
    font-size:18px; font-weight:800;
    display:flex; align-items:center; justify-content:center;
    margin:0 auto 14px;
    box-shadow:0 4px 16px rgba(37,99,235,0.35);
}}
.step-title {{ font-size:15px; font-weight:700; color:{text}; margin-bottom:6px; }}
.step-desc  {{ font-size:13px; color:{text2}; line-height:1.5; }}
.step-connector {{
    position:absolute; top:24px; left:50%; right:-50%;
    height:2px; background:{border};
    z-index:0;
}}

/* ── Page Header ──────────────────────────────────────────────────────────── */
.page-header {{
    position:relative; isolation:isolate; overflow:hidden;
    min-height:176px; padding:64px 36px 30px;
    border:1px solid {border}; border-radius:22px;
    background:
        radial-gradient(circle at 90% 18%,{'rgba(124,58,237,0.16)' if not dark else 'rgba(124,58,237,0.20)'} 0%,transparent 30%),
        linear-gradient(135deg,{'rgba(37,99,235,0.13)' if not dark else 'rgba(37,99,235,0.12)'} 0%,transparent 52%),
        {card_bg};
    box-shadow:0 22px 58px {'rgba(0,0,0,0.30)' if dark else 'rgba(15,23,42,0.085)'};
}}
.page-header::before {{
    content:'NAVNEET CONTRACTAI  /  INTELLIGENCE WORKSPACE';
    position:absolute; top:28px; left:36px; z-index:0;
    color:{primary}; font-size:10px; font-weight:850; letter-spacing:.13em;
}}
.page-header::after {{
    content:''; position:absolute; z-index:0; right:-30px; bottom:-76px;
    width:310px; height:210px; opacity:.52; transform:rotate(-8deg);
    background-image:
      linear-gradient({primary}1c 1px,transparent 1px),
      linear-gradient(90deg,{primary}1c 1px,transparent 1px);
    background-size:26px 26px;
    -webkit-mask-image:linear-gradient(135deg,transparent 3%,#000 48%);
    mask-image:linear-gradient(135deg,transparent 3%,#000 48%);
}}
.page-header-title {{
    font-family:'Sora',sans-serif; position:relative; z-index:1;
    font-size:clamp(30px,3vw,44px); line-height:1.08; font-weight:800;
    color:{text}; margin-bottom:10px; letter-spacing:-1.1px;
}}
.page-header-sub {{ position:relative; z-index:1; font-size:14px; color:{text2}; line-height:1.65; max-width:760px; }}

/* ── Inner content padding ────────────────────────────────────────────────── */
.inner {{ padding:32px 60px 56px; }}

/* Real authenticated workspace container. The old HTML-only .inner markers do
   not wrap later Streamlit elements, so they are hidden inside this shell. */
[class*="st-key-workspace_page_"] {{
    padding:28px clamp(20px,3.1vw,48px) 64px;
}}
[class*="st-key-workspace_page_"] .inner {{ display:none !important; }}
[class*="st-key-workspace_page_"] .page-header {{ margin:0 0 24px; }}
[class*="st-key-workspace_page_"] [data-testid="stVerticalBlock"] {{ gap:14px; }}
[class*="st-key-workspace_page_"] [data-testid="stHorizontalBlock"] {{ gap:18px; }}
[class*="st-key-workspace_page_"] h2,
[class*="st-key-workspace_page_"] h3,
[class*="st-key-workspace_page_"] h4 {{
    font-family:'Sora',sans-serif; color:{text}; letter-spacing:-.35px;
}}
[class*="st-key-workspace_page_"] h3 {{ font-size:19px; margin-top:8px; }}
[class*="st-key-workspace_page_"] [data-testid="stWidgetLabel"] p {{
    color:{text2} !important; font-size:11px !important; font-weight:800 !important;
    letter-spacing:.055em !important; text-transform:uppercase;
}}
[class*="st-key-workspace_page_"] [data-baseweb="input"],
[class*="st-key-workspace_page_"] [data-baseweb="textarea"],
[class*="st-key-workspace_page_"] [data-baseweb="select"] > div {{
    min-height:48px; border-radius:12px !important;
    background:{inp_bg} !important; border-color:{border} !important;
    box-shadow:0 1px 0 {'rgba(255,255,255,.04)' if dark else 'rgba(255,255,255,.8)'} inset;
}}
[class*="st-key-workspace_page_"] [data-baseweb="input"]:focus-within,
[class*="st-key-workspace_page_"] [data-baseweb="textarea"]:focus-within,
[class*="st-key-workspace_page_"] [data-baseweb="select"] > div:focus-within {{
    border-color:{primary} !important; box-shadow:0 0 0 3px {primary}16 !important;
}}
[class*="st-key-workspace_page_"] [data-testid="stFileUploader"] {{
    padding:10px; border-radius:18px !important; background:{card_bg} !important;
    box-shadow:0 12px 34px {'rgba(0,0,0,.16)' if dark else 'rgba(15,23,42,.05)'};
}}
[class*="st-key-workspace_page_"] [data-testid="stFileUploaderDropzone"] {{
    min-height:104px; padding:20px; border:1px dashed {primary}55 !important;
    border-radius:13px !important; background:{primary}08 !important;
}}
[class*="st-key-workspace_page_"] [data-testid="stExpander"] {{
    overflow:hidden; border-radius:15px !important;
    box-shadow:0 8px 24px {'rgba(0,0,0,.14)' if dark else 'rgba(15,23,42,.04)'};
}}
[class*="st-key-workspace_page_"] [data-testid="stAlert"] {{
    border:1px solid {border}; border-radius:14px;
    box-shadow:0 8px 24px {'rgba(0,0,0,.13)' if dark else 'rgba(15,23,42,.04)'};
}}
[class*="st-key-workspace_page_"] [data-testid="stDataFrame"],
[class*="st-key-workspace_page_"] .stPlotlyChart {{
    overflow:hidden; padding:8px; border:1px solid {border}; border-radius:17px;
    background:{card_bg}; box-shadow:0 12px 34px {'rgba(0,0,0,.17)' if dark else 'rgba(15,23,42,.055)'};
}}
[class*="st-key-workspace_page_"] [data-testid="stChatMessage"] {{
    margin:8px 0; padding:16px 18px; border:1px solid {border}; border-radius:16px;
    background:{card_bg}; box-shadow:0 9px 28px {'rgba(0,0,0,.15)' if dark else 'rgba(15,23,42,.045)'};
}}
[class*="st-key-workspace_page_"] [data-testid="stForm"] {{
    padding:22px; border:1px solid {border}; border-radius:18px;
    background:{card_bg}; box-shadow:0 14px 38px {'rgba(0,0,0,.18)' if dark else 'rgba(15,23,42,.06)'};
}}
[class*="st-key-workspace_page_"] [data-testid="stVerticalBlockBorderWrapper"] {{
    border-color:{border} !important; border-radius:18px !important;
    background:{card_bg}; box-shadow:0 14px 38px {'rgba(0,0,0,.17)' if dark else 'rgba(15,23,42,.055)'};
}}
[class*="st-key-workspace_page_"] .stTabs [data-baseweb="tab-list"] {{
    width:max-content; max-width:100%; overflow-x:auto; margin-bottom:12px;
}}
[class*="st-key-workspace_page_"] .stTabs [data-baseweb="tab-panel"] {{
    min-height:260px; padding:24px !important; border:1px solid {border}; border-radius:18px;
    background:{card_bg}; box-shadow:0 14px 40px {'rgba(0,0,0,.18)' if dark else 'rgba(15,23,42,.06)'};
}}
[class*="st-key-workspace_page_"] .chat-transcript {{
    min-height:320px; max-height:520px; overflow-y:auto; padding:18px;
    border:1px solid {border}; border-radius:18px; margin-bottom:16px;
    background:{card_bg}; box-shadow:0 14px 40px {'rgba(0,0,0,.18)' if dark else 'rgba(15,23,42,.06)'};
}}
[data-testid="stMetric"] {{
    background:{card_bg};
    border:1px solid {border};
    border-radius:16px;
    padding:16px 18px;
    box-shadow:0 12px 32px {'rgba(0,0,0,0.18)' if dark else 'rgba(15,23,42,0.055)'};
}}
[data-testid="stMetricLabel"] {{ color:{text2} !important; font-weight:750 !important; }}
[data-testid="stMetricValue"] {{ color:{text} !important; font-family:'Sora',sans-serif !important; letter-spacing:0 !important; }}

/* ── Toasts ───────────────────────────────────────────────────────────────── */
.toast {{
    position:fixed; bottom:24px; right:24px; z-index:99999;
    background:{success}; color:white; padding:12px 24px;
    border-radius:10px; font-size:14px; font-weight:600;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);
    animation:slidein 0.3s ease;
}}
@keyframes slidein {{ from{{transform:translateX(100%)}} to{{transform:translateX(0)}} }}

/* ── Editor ───────────────────────────────────────────────────────────────── */
.contract-editor {{
    background:{inp_bg}; border:1px solid {border}; border-radius:12px;
    padding:20px; font-family:'Sora',monospace; font-size:13px;
    line-height:1.7; color:{text}; min-height:400px;
    white-space:pre-wrap; word-break:break-word;
    overflow-y:auto; max-height:600px;
}}

/* ── History Card ─────────────────────────────────────────────────────────── */
.hist-card {{
    background:{card_bg}; border:1px solid {border}; border-radius:12px;
    padding:16px; margin:8px 0; transition:all 0.2s;
    display:flex; align-items:center; gap:16px;
}}
.hist-card:hover {{ border-color:{primary}; box-shadow:0 4px 16px {'rgba(0,0,0,0.2)' if dark else 'rgba(0,0,0,0.08)'}; }}
.hist-icon {{
    width:44px; height:44px; border-radius:10px;
    background:{primary}15; color:{primary};
    display:flex; align-items:center; justify-content:center;
    font-size:20px; flex-shrink:0;
}}
.hist-body {{ flex:1; min-width:0; }}
.hist-title {{ font-size:14px; font-weight:600; color:{text}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.hist-meta  {{ font-size:11px; color:{text2}; margin-top:2px; }}

/* ── Floating orbs for hero ───────────────────────────────────────────────── */
.orb {{
    position:absolute; border-radius:50%; opacity:0.12;
    background:radial-gradient(circle,{primary},{accent});
    animation:float 10s ease-in-out infinite;
    pointer-events:none;
}}

/* ── Contact ──────────────────────────────────────────────────────────────── */
.contact-card {{
    border:1px solid {border};
    background:{card_bg};
    border-radius:16px;
    padding:20px;
    transition:transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}}
.contact-card:hover {{
    transform:translateY(-3px);
    border-color:{primary};
    box-shadow:0 18px 42px {'rgba(0,0,0,0.28)' if dark else 'rgba(37,99,235,0.10)'};
}}
.contact-method-grid {{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:14px;
    margin-top:28px;
}}
.contact-method-icon {{
    width:42px; height:42px; border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    color:white;
    background:linear-gradient(135deg,{primary},{accent});
    margin-bottom:16px;
}}
.contact-card-title {{ color:{text}; font-size:15px; font-weight:800; margin-bottom:7px; }}
.contact-card-copy {{ color:{text2}; font-size:12.5px; line-height:1.65; }}
.contact-layout {{
    display:grid;
    grid-template-columns:minmax(320px,0.88fr) minmax(0,1.12fr);
    gap:18px;
    align-items:start;
}}
.contact-side {{
    border:1px solid {border};
    background:{card_bg};
    border-radius:18px;
    padding:22px;
}}
.contact-side-title {{
    font-family:'Sora',sans-serif;
    color:{text};
    font-size:18px;
    font-weight:800;
    letter-spacing:0;
    margin-bottom:8px;
}}
.contact-side-copy {{ color:{text2}; font-size:13px; line-height:1.7; margin-bottom:18px; }}
.contact-timeline {{
    display:grid;
    gap:12px;
}}
.contact-step {{
    display:grid;
    grid-template-columns:28px minmax(0,1fr);
    gap:12px;
    align-items:start;
}}
.contact-step-num {{
    width:28px; height:28px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    color:white;
    background:{primary};
    font-size:11px;
    font-weight:900;
}}
.contact-step-title {{ color:{text}; font-size:13px; font-weight:800; }}
.contact-step-copy {{ color:{text2}; font-size:12px; line-height:1.55; margin-top:2px; }}
.contact-form-shell {{
    border:1px solid {border};
    background:{card_bg};
    border-radius:18px;
    padding:22px;
    box-shadow:0 16px 42px {'rgba(0,0,0,0.22)' if dark else 'rgba(15,23,42,0.07)'};
}}
.contact-form-title {{
    font-family:'Sora',sans-serif;
    color:{text};
    font-size:20px;
    font-weight:800;
    letter-spacing:0;
    margin-bottom:4px;
}}
.contact-form-sub {{ color:{text2}; font-size:12.5px; line-height:1.65; margin-bottom:18px; }}
.contact-stats {{
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:14px;
    margin-top:18px;
}}

/* ── Misc ─────────────────────────────────────────────────────────────────── */
.divider {{ height:1px; background:{border}; margin:24px 0; }}
.text-primary {{ color:{text}; }}
.text-secondary {{ color:{text2}; }}
.text-accent {{ color:{primary}; }}
.fw-700 {{ font-weight:700; }}
.fw-600 {{ font-weight:600; }}
.fw-500 {{ font-weight:500; }}
.fs-13 {{ font-size:13px; }}
.fs-12 {{ font-size:12px; }}
.mt-8  {{ margin-top:8px; }}
.mt-16 {{ margin-top:16px; }}
.mb-16 {{ margin-bottom:16px; }}
.gap-8 {{ gap:8px; }}

@media (max-width: 1080px) {{
    .pro-metric-grid,.infra-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .workspace-grid {{ grid-template-columns:1fr; }}
    .action-labels {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .about-shell,.contact-shell,.inner {{ padding-left:28px; padding-right:28px; }}
    [class*="st-key-workspace_page_"] {{ padding-left:24px; padding-right:24px; }}
    .about-hero,.contact-layout {{ grid-template-columns:1fr; }}
    .about-kpi-grid,.tech-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .about-feature-grid,.contact-method-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .ai-trace {{ grid-template-columns:repeat(3,minmax(0,1fr)); }}
    .decision-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
}}
@media (max-width: 720px) {{
    .command-shell {{ padding:20px 14px 42px; }}
    .st-key-dashboard_actions {{ margin:-18px 14px 0; }}
    .command-hero {{ display:block; padding:25px 22px; border-radius:17px; }}
    .command-runtime {{ justify-content:flex-start; margin-top:18px; }}
    .pro-metric-grid,.infra-grid,.action-labels {{ grid-template-columns:1fr; }}
    .ai-runtime-top {{ display:block; }}
    .ai-live {{ margin-top:12px; }}
    .ai-trace,.decision-grid {{ grid-template-columns:1fr; }}
    .st-key-public_navbar {{ padding:8px 12px; min-height:62px; }}
    .st-key-public_navbar .lf-brand-name {{ display:none; }}
    .st-key-public_navbar [data-testid="stColumn"]:nth-child(8) {{ display:none; }}
    .st-key-public_navbar .st-key-nav_about,
    .st-key-public_navbar .st-key-nav_contact {{ display:none; }}
    .st-key-public_navbar .stButton > button {{
        min-height:38px; padding:0.42rem 0.58rem !important; font-size:12px !important;
    }}
    .st-key-public_navbar .st-key-theme_btn button {{
        width:38px !important; min-width:38px !important; padding:0 !important; font-size:0 !important;
    }}
    .st-key-public_navbar .st-key-signin_btn button {{
        min-width:76px !important; padding:0.44rem 0.72rem !important;
    }}
    .st-key-hero_cta {{ margin-top:-104px !important; padding-bottom:26px; }}
    .st-key-hero_cta [data-testid="stHorizontalBlock"] {{ gap:8px; }}
    .about-shell,.contact-shell,.inner {{ padding:22px 16px 40px; }}
    [class*="st-key-workspace_page_"] {{ padding:16px 14px 42px; }}
    [class*="st-key-workspace_page_"] .page-header {{
        min-height:154px; padding:58px 22px 24px; margin-bottom:18px; border-radius:18px;
    }}
    [class*="st-key-workspace_page_"] .page-header::before {{ top:24px; left:22px; font-size:9px; }}
    [class*="st-key-workspace_page_"] .page-header-title {{ font-size:29px; }}
    [class*="st-key-workspace_page_"] .stTabs [data-baseweb="tab-panel"] {{ padding:17px !important; }}
    [class*="st-key-workspace_page_"] [data-testid="stHorizontalBlock"] {{ gap:12px; }}
    .about-hero,.contact-hero {{ padding:24px; border-radius:18px; }}
    .about-title,.contact-title {{ font-size:32px; line-height:1.12; }}
    .about-section-head,.contact-section-head {{ display:block; }}
    .about-section-sub,.contact-section-sub {{ margin-top:8px; }}
    .about-kpi-grid,.about-feature-grid,.tech-grid,.contact-method-grid,.contact-stats {{
        grid-template-columns:1fr;
    }}
    .about-footer-note {{ align-items:flex-start; flex-direction:column; }}
    .system-row {{ grid-template-columns:34px minmax(0,1fr); }}
    .system-score {{ grid-column:2; width:max-content; }}
}}
@media (max-width: 520px) {{
    .st-key-public_navbar [data-testid="stColumn"]:first-child {{ display:none; }}
    .st-key-public_navbar .st-key-nav_features {{ display:none; }}
}}

[data-testid="stMarkdownContainer"] p {{
    color:{text} !important;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_navbar():
    p = st.session_state.page
    dark = st.session_state.dark_mode
    logged_in = bool(st.session_state.auth_user)

    divider = "rgba(255,255,255,0.10)" if dark else "rgba(15,23,42,0.10)"
    logo_html = (f'<img src="{LOGO_URI}" alt="Navneet" '
                 f'style="height:42px;width:auto;display:block"/>' if LOGO_URI else '')
    brand_html = (
        '<div style="display:flex;align-items:center;gap:13px;height:100%">'
        f'{logo_html}<span class="lf-brand-name">Navneet <span>ContractAI</span></span></div>'
    )
    links = [
        ("home", "Home"),
        ("features", "Capabilities"),
        ("architecture", "Architecture"),
        ("about", "About"),
        ("contact", "Contact"),
    ]

    if logged_in:
        key_state = "Connected" if st.session_state.api_key else "Setup needed"
        page_title = dict(
            dashboard="Command Center", analyze="Contract Analysis", writer="Contract Studio",
            chatbot="Legal Assistant", agent="Agent Workspace", chat="RAG Contract Chat",
            knowledge="Knowledge and Retrieval", integrations="AI Infrastructure",
            compare="Contract Comparison", history="Contract History", features="Capabilities",
            architecture="Architecture", settings="Settings", about="About", contact="Support",
            home="Home",
        ).get(p, "Workspace")
        vector_name = "Pinecone" if st.session_state.vector_backend == "pinecone" else "ChromaDB"
        current_model = OPENROUTER_MODEL_LABEL
        st.markdown(
            f"""
<div class="app-topbar">
  <div>
    <div class="app-topbar-title">{page_title}</div>
    <div class="app-topbar-sub">Navneet ContractAI / {current_model} / {vector_name}</div>
  </div>
  <div class="app-topbar-right">
    <div class="api-status-pill">{key_state}</div>
    <div class="api-status-pill">{vector_name}</div>
    <div>{html.escape(st.session_state.username or 'User')}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='height:1px;background:{divider};margin:0'></div>",
                    unsafe_allow_html=True)
        return

    if logged_in:
        cols = st.columns([2.6, 0.9, 0.9, 0.95, 0.7, 2.3, 1.9], vertical_alignment="center")
        cols[0].markdown(brand_html, unsafe_allow_html=True)
        for i, (key, label) in enumerate(links):
            if cols[i + 1].button(label, key=f"nav_{key}", use_container_width=True,
                                  type="primary" if p == key else "secondary"):
                nav(key)
        if cols[4].button("Theme", key="theme_btn",
                          use_container_width=True, help="Toggle light / dark"):
            st.session_state.dark_mode = not dark
            st.rerun()
        cols[6].markdown(
            f"<div style='text-align:right;font-size:13px;font-weight:600'>"
            f"👤 {st.session_state.username or 'User'}</div>", unsafe_allow_html=True)
    else:
        # Compact link-style controls with flexible space on both sides of the
        # main links, so the sign-in action never becomes oversized.
        with st.container(key="public_navbar"):
            cols = st.columns([2.6, 0.9, 1.15, 1.15, 0.8, 0.9, 0.65, 1.8, 0.95],
                              vertical_alignment="center")
            cols[0].markdown(brand_html, unsafe_allow_html=True)
            for i, (key, label) in enumerate(links):
                if cols[i + 1].button(label, key=f"nav_{key}",
                                      type="primary" if p == key else "secondary"):
                    nav(key)
            if cols[6].button("Light" if dark else "Dark", key="theme_btn",
                              help="Toggle light / dark"):
                st.session_state.dark_mode = not dark
                st.rerun()
            if cols[8].button("Sign In", key="signin_btn", type="primary"):
                st.session_state.show_auth = True
                st.session_state.page = "home"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ── HOME ──────────────────────────────────────────────────────────────────────
def page_home():
    st.iframe(THREEJS_HERO_HTML, height=620, width="stretch", tab_index=-1)

    # ── Primary CTAs — label + buttons in one keyed container so CSS lifts the
    #    whole unit onto the hero while preserving responsive document flow. ─────
    with st.container(key="hero_cta"):
        st.markdown("<div class='hero-cta-label'>Get started in seconds</div>",
                    unsafe_allow_html=True)
        sp1, b1, b2, sp2 = st.columns([2.6, 1.5, 1.5, 2.6], vertical_alignment="center")
        with b1:
            if st.button("Analyze a Contract", type="primary", use_container_width=True):
                nav("analyze")
        with b2:
            if st.button("Draft a Contract", use_container_width=True):
                nav("writer")

    st.markdown("""
<div class="section">
  <div class="section-label">How It Works</div>
  <div class="section-title">From Upload to Insight in Minutes</div>
  <div style="height:40px"></div>
  <div class="steps-row">
    <div class="step-item">
      <div class="step-num">1</div>
      <div class="step-title">Ingest contract</div>
      <div class="step-desc">Parse PDF, Word, TXT or Markdown locally with page-aware evidence mapping.</div>
    </div>
    <div class="step-item">
      <div class="step-num">2</div>
      <div class="step-title">Hybrid AI reasoning</div>
      <div class="step-desc">LangGraph coordinates symbolic extraction, model review, grounding and validation.</div>
    </div>
    <div class="step-item">
      <div class="step-num">3</div>
      <div class="step-title">Decision intelligence</div>
      <div class="step-desc">Evidence ledger, calibrated confidence, risk matrix and negotiation playbook.</div>
    </div>
    <div class="step-item">
      <div class="step-num">4</div>
      <div class="step-title">Chat & Export</div>
      <div class="step-desc">Ask questions, compare contracts, export PDF/DOCX reports.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    sec_bg = "rgba(255,255,255,0.03)" if st.session_state.dark_mode else "#f8faff"
    cards_html = "".join(
        f'<div class="feat-card"><div class="feat-icon">{svg}</div>'
        f'<div class="feat-title">{title}</div><div class="feat-desc">{desc}</div></div>'
        for svg, title, desc in FEATURE_CARDS
    )
    st.markdown(
        f'<div class="section" style="background:{sec_bg}">'
        f'<div class="section-label">Capabilities</div>'
        f'<div class="section-title">Everything Your Legal Team Needs</div>'
        f'<div class="feat-grid">{cards_html}</div></div>',
        unsafe_allow_html=True)


# ── FEATURES ───────────────────────────────────────────────────────────────────
def page_features():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">Features</div>
  <div class="page-header-sub">Full-stack AI legal intelligence — built by SNEHAL LAXMAN JADHAV, Navneet Education Limited</div>
</div>
<div class="inner">
""", unsafe_allow_html=True)
    tabs = st.tabs(["AI Analysis", "Processing Pipeline", "AI Runtime", "RAG Engine", "Database", "Export"])
    with tabs[0]:
        st.markdown("""
**8-Dimension Contract Review:**
1. **NER Node** — Named Entity Recognition: parties, dates, amounts, jurisdictions, contract type
2. **Obligation Node** — Full obligation extraction with priority, conditions, clause references
3. **Risk Node** — Risk identification with severity, likelihood, financial exposure, mitigations
4. **Deadline Node** — All dates, terms, auto-renewals, notice periods, consequences
5. **Clause Node** — 20+ clause types classified as Favorable/Neutral/Concerning/Missing
6. **Compliance Node** — GDPR, CCPA, HIPAA, SOC2 compliance scoring with missing clause detection
7. **Anomaly Node** — ML-based detection of unusual provisions, hidden risks, contradictions
8. **Summary Node** — Executive summary with PROCEED/PROCEED WITH MODIFICATIONS/DO NOT SIGN recommendation
        """)
    with tabs[1]:
        st.markdown("""
**Fast Hybrid Processing Architecture:**
- Immediate deterministic extraction for every report section
- Document-grounded checks for risky and missing contract terms
- Local structured results suitable for history and export
- Retrieval indexing deferred until contract chat is requested

**Runtime Behavior:**
- The main Analyze button never waits for an external AI provider
- The complete local report appears before any optional model request
- A selectable GPT-OSS or NVIDIA Nemotron enhancement reviews the three most important errors on demand
- Provider failures never replace the already-visible local report
        """)
    with tabs[2]:
        st.markdown("""
**Reasoning-Enabled AI Runtime:**
- **Analysis models** — `openai/gpt-oss-20b:free` or `nvidia/nemotron-3-super-120b-a12b:free`
- **Gateway** — OpenRouter chat-completions API
- **Reasoning** — enabled per model; returned reasoning details are preserved unchanged for follow-up turns
- **Structured output** — JSON findings are validated before display or storage
- **Configuration** — one workspace API key; the Analyzer exposes only the approved model list
        """)
    with tabs[3]:
        st.markdown("""
**Local Retrieval Engine:**
- **Vectorization** — deterministic lexical hashing with no additional AI model
- **Storage** — ChromaDB cosine index or optional Pinecone index
- **Isolation** — document filters and Pinecone namespaces
- **Privacy and speed** — retrieval creates no additional OpenRouter request
        """)
    with tabs[4]:
        st.markdown("""
**Project Data Stores:**
- **ChromaDB** — local vector store for contract retrieval
- **SQLite** — contract history, chat logs, users, activity, and drafts
- **Pinecone** — optional external vector backend when configured

**Full CRUD Operations:**
- Create, Read, Update, Delete contract analyses
- Tag management, notes, status tracking
- Generated contract library with versioning
- Clause snippet storage and reuse
- Dashboard aggregate stats
        """)
    with tabs[5]:
        st.markdown("""
**Export Formats:**
- **PDF Report** — Professional PDF with risk summary, tables (ReportLab)
- **DOCX** — Word document with formatted contract or analysis (python-docx)
- **JSON** — Full structured analysis data
- **Markdown** — Human-readable analysis report
- **CSV** — Risk register and obligation tables for Excel
        """)
    st.markdown("</div>", unsafe_allow_html=True)


# ── ANALYZE ────────────────────────────────────────────────────────────────────
ANALYSIS_SECTIONS = ("ner", "obligations", "risks", "deadlines", "clauses", "compliance", "anomalies", "summary")


def _analysis_failures(results: Dict) -> List[str]:
    if not isinstance(results, dict):
        return ["The analysis service returned an invalid response."]
    failures = []
    for section in ANALYSIS_SECTIONS:
        value = results.get(section)
        if not isinstance(value, dict):
            failures.append(f"{section}: missing or invalid result")
        elif value.get("error"):
            failures.append(f"{section}: {value['error']}")
    return failures


def _friendly_analysis_error(failures: List[str]) -> str:
    joined = " ".join(failures).lower()
    if "402" in joined or "insufficient credits" in joined:
        return "The AI provider rejected the request because the OpenRouter account has insufficient credits. No risk score was produced."
    if "401" in joined or "unauthorized" in joined or "api key" in joined:
        return "The configured OpenRouter API key was rejected. No risk score was produced."
    if "429" in joined or "rate limit" in joined:
        return "The AI provider is temporarily rate-limited. No risk score was produced."
    return "The contract analysis did not complete, so the app will not display or save misleading zero scores."


def page_analyze():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">AI Contract Intelligence</div>
  <div class="page-header-sub">A LangGraph hybrid workflow combines deterministic extraction, LLM legal reasoning, source evidence grounding, confidence calibration and guardrail validation.</div>
</div>
""", unsafe_allow_html=True)

    with st.container(key="analysis_workspace"):
        st.markdown('<div class="inner">', unsafe_allow_html=True)

        # API Key check
        if not st.session_state.api_key:
            st.warning("The workspace OpenRouter key is missing from .streamlit/secrets.toml. Saved demo analyses remain available below and from Contract Repository.")
            if st.button("Go to Settings"):
                nav("settings")

        col_upload, col_cfg = st.columns([2, 1])

        with col_upload.container(border=True):
            st.markdown("### Source document")
            uploaded = st.file_uploader(
                "Drop your contract here",
                type=["pdf", "docx", "txt", "md"],
                help="PDF, Word, TXT, or Markdown. Parsing stays local; only extracted contract text is sent when AI analysis runs.",
                label_visibility="visible"
            )

            # Sample contracts
            with st.expander("Use a sample contract instead"):
                sample = st.selectbox("Select sample", [
                    "None",
                    "Sample NDA (TechCorp vs InnovateLab)",
                ])
                if sample != "None":
                    sample_texts = {
                        "Sample NDA (TechCorp vs InnovateLab)": """NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("Agreement") is entered into as of January 15, 2026,
by and between TechCorp Solutions Inc., a Delaware corporation ("Disclosing Party"),
and InnovateLab Pvt. Ltd., an Indian company ("Receiving Party").

1. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any non-public information disclosed by either party,
including technical data, trade secrets, business plans, financial data, or customer lists.

2. OBLIGATIONS
The Receiving Party shall: (a) hold Confidential Information in strict confidence;
(b) not disclose to any third party without prior written consent; (c) use only for
evaluation purposes; (d) protect with at least the same care as its own confidential info.

3. TERM
This Agreement shall remain in effect for THREE (3) years from the Effective Date.
Obligations survive termination for an additional FIVE (5) years.

4. EXCLUSIONS
Confidentiality obligations do not apply to information: (a) already public knowledge;
(b) independently developed; (c) required by law (with notice to disclosing party).

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware, USA.
Disputes resolved by binding arbitration under AAA Commercial Rules.

6. REMEDIES
Both parties acknowledge that breach may cause irreparable harm for which monetary
damages are insufficient. Injunctive relief shall be available without bond.

NOTE: This contract is missing: Limitation of Liability, GDPR clause, and arbitration seat.""",
                    }
                    if sample in sample_texts:
                        st.session_state["sample_text"] = sample_texts[sample]
                        st.session_state["sample_name"] = sample
                        st.info(f"Sample loaded: {sample}")

        with col_cfg.container(border=True):
            st.markdown("### Analysis controls")
            st.selectbox(
                "AI enhancement model",
                options=list(OPENROUTER_MODELS),
                format_func=openrouter_model_label,
                key="analysis_model",
                help="The instant local report is model-free. This model reviews its key findings when you select AI enhancement.",
            )

            analysis_depth = st.select_slider(
                "Analysis Depth",
                options=["Quick", "Standard", "Deep"],
                value="Standard",
                help="Deep requests a full eight-domain model analysis. Quick and Standard use a compact model review over the deterministic baseline.",
            )
            # Compliance and anomaly validation are mandatory guardrails in v6,
            # so users cannot accidentally produce a partial approval report.
            include_compliance = True
            include_anomaly = True
            st.caption("Mandatory guardrails: compliance · anomaly detection · evidence grounding")
            execution_label = (
                "Hybrid AI · LangGraph · evidence validation"
                if st.session_state.api_key else
                "Local pre-screen · no provider credits"
            )
            st.caption(f"Execution: {execution_label}")

        # Run Analysis
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        run_col, _ = st.columns([1, 3])
        with run_col:
            run_btn = st.button(
                "Run Hybrid AI Analysis" if st.session_state.api_key else "Run Local Pre-screen",
                type="primary", use_container_width=True,
            )

        if run_btn:
            # Determine source text
            if uploaded:
                with st.spinner("Reading contract text..."):
                    raw_bytes = uploaded.read()
                    doc_info = extract_contract_file(raw_bytes, uploaded.name)
            elif "sample_text" in st.session_state:
                text = st.session_state.sample_text
                doc_info = {
                    "text": text, "word_count": len(text.split()),
                    "page_count": 1, "char_count": len(text),
                    "doc_hash": "sample_" + st.session_state.get("sample_name","")[:8].replace(" ","_"),
                    "filename": st.session_state.get("sample_name", "Sample Contract"),
                    "method": "text", "pages": [{"page": 1, "text": text}],
                }
            else:
                st.error("Please upload a PDF or select a sample contract.")
                return

            text = doc_info["text"]
            if len(text) < 50:
                st.error("Could not extract text from this PDF. Try a text-based PDF.")
                return

            # Store chunks for lazy chat indexing. Contract analysis itself does
            # not wait for vector storage or an external provider.
            chunks = chunk_text(text)
            doc_id = doc_info["doc_hash"]
            st.session_state.current_doc_id = doc_id
            st.session_state.current_doc_chunks = chunks
            st.session_state.current_doc_indexed = False
            st.session_state.doc_info = doc_info

            # Progress
            prog_bar = st.progress(0)
            prog_text = st.empty()

            def progress_cb(step, total, label):
                pct = int(step / total * 100)
                prog_bar.progress(pct)
                prog_text.markdown(f"**{label}...**")

            with st.spinner("Executing the hybrid intelligence pipeline..."):
                if st.session_state.api_key:
                    results = get_agent(st.session_state.analysis_model).run_full_analysis(
                        text=text,
                        doc_info=doc_info,
                        progress_callback=progress_cb,
                        depth=analysis_depth,
                    )
                else:
                    results = analyze_contract_offline(
                        text=text, doc_info=doc_info, progress_callback=progress_cb
                    )
                    results["meta"]["analysis_mode"] = "instant_local"
                    results["meta"]["fallback_reason"] = ""
                    results["summary"]["executive_summary"] = results["summary"]["executive_summary"].replace(
                        "\n\nLocal contract-review engine selected.", ""
                    )
                    results = enrich_analysis(text, results, doc_info)
                results.setdefault("meta", {})["requested_checks"] = {
                    "compliance": include_compliance,
                    "anomaly_detection": include_anomaly,
                }

            prog_bar.progress(100)
            failures = _analysis_failures(results)
            prog_text.markdown("**Analysis failed**" if failures else "**Analysis complete!**")

            st.session_state.analysis_results = results

            if failures:
                st.error(_friendly_analysis_error(failures))
            else:
                summary = results.get("summary", {})
                risks = results.get("risks", {})
                ner = results.get("ner", {})
                meta = results.get("meta", {})
                save_contract_analysis(
                    filename=doc_info["filename"],
                    doc_hash=doc_id,
                    word_count=doc_info.get("word_count", 0),
                    page_count=doc_info.get("page_count", 0),
                    contract_type=ner.get("contract_type", "Unknown"),
                    risk_score=risks.get("overall_risk_score", 0),
                    assessment=summary.get("overall_assessment", "Unknown"),
                    model_used=meta.get("model_used", OPENROUTER_MODEL),
                    tokens_used=meta.get("tokens_used", 0),
                    analysis_json=results,
                    owner_id=(get_user(st.session_state.auth_user) or {}).get("id", ""),
                    owner_name=st.session_state.username or st.session_state.auth_user or "",
                )
                mode_label = "AI-enhanced" if meta.get("analysis_mode") == "api_enhanced" else "locally validated"
                st.success(f"{mode_label.capitalize()} contract review complete and saved to history.")
                st.rerun()

        # Show results
        if st.session_state.analysis_results:
            render_analysis_results(st.session_state.analysis_results)

        st.markdown('</div>', unsafe_allow_html=True)


def _render_ai_runtime(intelligence: Dict, meta: Dict) -> None:
    """Render the observable hybrid-AI execution trace."""
    if not intelligence:
        return
    pipeline_html = "".join(
        f'''<div class="ai-node {"fallback" if node.get("status") == "fallback" else ""}">
          <div class="ai-node-id">{html.escape(str(node.get("status", "complete")))}</div>
          <div class="ai-node-name">{html.escape(str(node.get("label", "Pipeline stage")))}</div>
          <div class="ai-node-detail">{html.escape(str(node.get("detail", "")))}</div>
        </div>'''
        for node in intelligence.get("pipeline", [])
    )
    architecture = intelligence.get("architecture", "hybrid")
    model = openrouter_model_label(meta.get("model_used", "")) if meta.get("analysis_mode") == "api_enhanced" else "Local symbolic engine"
    st.markdown(f'''
<div class="ai-runtime">
  <div class="ai-runtime-top">
    <div>
      <div class="ai-runtime-kicker">Observable AI execution · {html.escape(str(architecture))}</div>
      <div class="ai-runtime-title">LexForge Hybrid Intelligence v{html.escape(str(intelligence.get("engine_version", "6.0")))}</div>
      <div class="ai-runtime-copy">{html.escape(str(model))} · grounded findings · confidence calibration · deterministic guardrails · audit ID {html.escape(str(intelligence.get("analysis_fingerprint", "N/A")))}</div>
    </div>
    <div class="ai-live">Validated result</div>
  </div>
  <div class="ai-trace">{pipeline_html}</div>
</div>''', unsafe_allow_html=True)


def _render_ai_decision_room(results: Dict) -> None:
    intelligence = results.get("intelligence", {})
    if not intelligence:
        st.info("Decision intelligence is available for newly analyzed contracts.")
        return

    risks = results.get("risks", {}).get("risks", [])
    conflicts = intelligence.get("conflict_checks", [])
    playbook = intelligence.get("negotiation_playbook", [])
    guardrails = intelligence.get("guardrails", [])
    priority = intelligence.get("priority_queue", [])
    human_gate = "Required" if intelligence.get("human_review_required") else "Advisory"
    st.markdown(f'''
<div class="decision-grid">
  <div class="decision-metric"><div class="decision-value">{intelligence.get("calibrated_confidence", 0):.0%}</div><div class="decision-label">Calibrated confidence</div></div>
  <div class="decision-metric"><div class="decision-value">{intelligence.get("evidence_coverage", 0):.0%}</div><div class="decision-label">Evidence coverage</div></div>
  <div class="decision-metric"><div class="decision-value">{len(conflicts)}</div><div class="decision-label">Cross-clause alerts</div></div>
  <div class="decision-metric"><div class="decision-value">{human_gate}</div><div class="decision-label">Human approval gate</div></div>
</div>''', unsafe_allow_html=True)

    matrix_col, queue_col = st.columns([1.12, 0.88])
    with matrix_col:
        st.markdown("#### Risk decision matrix")
        if risks:
            likelihood_value = {"Low": 1, "Medium": 2, "High": 3}
            severity_value = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
            scores = [float(risk.get("priority_score", 0) or 0) for risk in risks]
            fig = go.Figure(go.Scatter(
                x=[likelihood_value.get(risk.get("likelihood", "Medium"), 2) for risk in risks],
                y=[severity_value.get(risk.get("severity", "Medium"), 2) for risk in risks],
                mode="markers",
                text=[html.escape(str(risk.get("title", "Risk"))) for risk in risks],
                customdata=[[risk.get("id", ""), risk.get("category", ""), f"{risk.get('confidence', 0):.0%}"] for risk in risks],
                hovertemplate="<b>%{text}</b><br>%{customdata[0]} · %{customdata[1]}<br>confidence %{customdata[2]}<extra></extra>",
                marker={
                    "size": [max(15, min(42, score / 2 + 12)) for score in scores],
                    "color": scores,
                    "colorscale": [[0, "#60a5fa"], [0.55, "#f59e0b"], [1, "#ef4444"]],
                    "showscale": False,
                    "line": {"color": "rgba(255,255,255,.85)", "width": 1.5},
                    "opacity": 0.88,
                },
            ))
            fig.update_xaxes(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"], title="Likelihood", range=[0.6, 3.4], gridcolor="rgba(148,163,184,.16)")
            fig.update_yaxes(tickvals=[1, 2, 3, 4], ticktext=["Low", "Medium", "High", "Critical"], title="Impact", range=[0.6, 4.4], gridcolor="rgba(148,163,184,.16)")
            fig.update_layout(height=355, margin=dict(l=25, r=15, t=15, b=25), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#64748b")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk items were available for matrix scoring.")

    with queue_col:
        st.markdown("#### Prioritized review queue")
        queue_html = "".join(
            f'''<div class="deal-row"><div class="deal-rank">{index:02d}</div><div><div class="deal-title">{html.escape(str(item.get("title", "Finding")))}</div><div class="deal-copy">{html.escape(str(item.get("severity", "")))} · {html.escape(str(item.get("category", "")))} · owner {html.escape(str(item.get("owner", "Legal")))} · {html.escape(str(item.get("source_id", "")))}</div></div><div class="deal-score">{float(item.get("priority_score", 0) or 0):.0f}</div></div>'''
            for index, item in enumerate(priority[:6], 1)
        ) or '<div class="panel-sub">No material findings are queued.</div>'
        st.markdown(f'<div class="workspace-panel">{queue_html}</div>', unsafe_allow_html=True)

    negotiation_tab, evidence_tab, validation_tab = st.tabs(["Negotiation playbook", "Evidence ledger", "Guardrails"])
    with negotiation_tab:
        if not playbook:
            st.info("No negotiation positions were generated.")
        for item in playbook:
            signal = " · WALK-AWAY SIGNAL" if item.get("walk_away_signal") else ""
            st.markdown(f'''
<div class="evidence-card">
  <div class="evidence-score">Priority {item.get("rank", "-")} · score {float(item.get("priority_score", 0) or 0):.0f}{signal}</div>
  <div class="deal-title" style="margin-top:7px">{html.escape(str(item.get("issue", "Negotiation issue")))}</div>
  <div class="evidence-text"><strong>Opening position:</strong> {html.escape(str(item.get("opening_position", "")))}<br><strong>Fallback:</strong> {html.escape(str(item.get("fallback_position", "")))}<br><strong>Evidence:</strong> {html.escape(str(item.get("source_id", "UNRESOLVED")))}</div>
</div>''', unsafe_allow_html=True)
    with evidence_tab:
        ledger = intelligence.get("grounded_findings", [])
        if ledger:
            st.dataframe(pd.DataFrame([
                {
                    "Finding": item.get("finding_id"),
                    "Type": item.get("kind"),
                    "Source": item.get("source_id"),
                    "Section": item.get("section"),
                    "Page": str(item.get("page")) if item.get("page") else "—",
                    "Grounding": f"{item.get('grounding_score', 0):.0%}",
                    "Verification": item.get("verification"),
                }
                for item in ledger
            ]), use_container_width=True, hide_index=True)
            for item in ledger[:5]:
                st.markdown(f'<div class="evidence-card"><div class="evidence-score">{html.escape(str(item.get("finding_id", "")))} · {html.escape(str(item.get("source_id", "")))} · {float(item.get("grounding_score", 0) or 0):.0%} grounded</div><div class="evidence-text">{html.escape(str(item.get("excerpt", "")))}</div></div>', unsafe_allow_html=True)
    with validation_tab:
        guardrail_html = "".join(
            f'''<div class="guardrail"><div class="guardrail-dot {html.escape(str(check.get("status", "review")))}"></div><div><div class="guardrail-title">{html.escape(str(check.get("name", "Validation")))}</div><div class="guardrail-detail">{html.escape(str(check.get("detail", "")))}</div></div></div>'''
            for check in guardrails
        )
        st.markdown(f'<div class="workspace-panel">{guardrail_html}</div>', unsafe_allow_html=True)
        if conflicts:
            st.markdown("#### Cross-clause consistency alerts")
            for alert in conflicts:
                st.warning(f"{alert.get('title')}: {alert.get('detail')} Review: {alert.get('review')}")
        else:
            st.success("No deterministic cross-clause conflicts were detected.")


def render_analysis_results(results: Dict):
    failures = _analysis_failures(results)
    if failures:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.error(_friendly_analysis_error(failures))
        st.info("Run the analysis again after resolving the provider issue. This failed result is intentionally not shown as a 0/10 assessment.")
        return

    summary     = results.get("summary", {})
    obligations = results.get("obligations", {})
    risks       = results.get("risks", {})
    deadlines   = results.get("deadlines", {})
    clauses     = results.get("clauses", {})
    compliance  = results.get("compliance", {})
    anomalies   = results.get("anomalies", {})
    ner         = results.get("ner", {})
    meta        = results.get("meta", {})
    safe = lambda value: html.escape(str(value or ""))

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    if meta.get("analysis_mode") == "instant_local":
        st.success("Instant contract analysis is complete. All report tabs are ready below.")
        analysis_model = st.session_state.get("analysis_model", OPENROUTER_MODEL)
        analysis_model_label = openrouter_model_label(analysis_model)
        current_doc = st.session_state.doc_info or {}
        can_enhance = bool(
            st.session_state.api_key
            and current_doc.get("text")
            and current_doc.get("filename") == meta.get("filename")
        )
        if can_enhance and st.button(
            f"Optional: Enhance key findings with {analysis_model_label}",
            key="enhance_current_analysis",
        ):
            with st.spinner(f"{analysis_model_label} is reviewing the three highest-priority legal errors..."):
                enhanced = get_agent(analysis_model).run_full_analysis(
                    text=current_doc["text"], doc_info=current_doc
                )
            if enhanced.get("meta", {}).get("analysis_mode") == "api_enhanced":
                st.session_state.analysis_results = enhanced
                st.success(f"{analysis_model_label} enhancement complete.")
                st.rerun()
            else:
                st.warning(
                    f"{analysis_model_label} did not return a valid structured review. "
                    "Your complete instant analysis remains displayed below."
                )
    if meta.get("analysis_mode") == "local_fallback":
        st.warning(meta.get("fallback_reason", "The AI provider was unavailable, so the local contract-review engine completed this analysis."))

    _render_ai_runtime(results.get("intelligence", {}), meta)

    # Key metrics
    risk_score = risks.get("overall_risk_score", 0) or summary.get("overall_risk_score", 0)
    risk_color = "#ef4444" if risk_score >= 7 else "#f59e0b" if risk_score >= 4 else "#10b981"
    obl_count = obligations.get("total_count", 0)
    comp_score = compliance.get("overall_score", 0)
    rec = safe(summary.get("recommendation", "N/A"))
    gdpr_data = compliance.get("gdpr", {})
    gdpr_state = "N/A" if gdpr_data.get("applicable") is False else "OK" if gdpr_data.get("compliant") else "Issues"

    st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-val" style="color:{risk_color}">{risk_score}/10</div>
    <div class="metric-label">Risk Score</div>
    <div class="metric-sub">{safe(risks.get('assessment',''))}</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{obl_count}</div>
    <div class="metric-label">Obligations</div>
    <div class="metric-sub">{obligations.get('high_priority_count',0)} high priority</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{comp_score}/100</div>
    <div class="metric-label">Compliance Score</div>
    <div class="metric-sub">GDPR {gdpr_state}</div>
  </div>
  <div class="metric-card">
    <div class="metric-val" style="font-size:14px;padding-top:8px">{rec}</div>
    <div class="metric-label">Recommendation</div>
    <div class="metric-sub">Confidence: {summary.get('confidence_score',0):.0%}</div>
  </div>
</div>
<div style="height:24px"></div>
""", unsafe_allow_html=True)
    # NER summary
    if ner:
        parties = ner.get("parties", [])
        party_str = " | ".join(f"{safe(p.get('name',''))} ({safe(p.get('role',''))})" for p in parties[:3])
        st.markdown(f"""
<div class="card card-sm" style="margin-bottom:16px">
  <div style="display:flex;gap:32px;flex-wrap:wrap">
    <div><span style="font-size:11px;color:var(--text2)">CONTRACT TYPE</span><br><strong>{safe(ner.get('contract_type','Unknown'))}</strong></div>
    <div><span style="font-size:11px;color:var(--text2)">PARTIES</span><br><strong>{party_str or 'See analysis'}</strong></div>
    <div><span style="font-size:11px;color:var(--text2)">GOVERNING LAW</span><br><strong>{safe(ner.get('governing_law','Unknown'))}</strong></div>
    <div><span style="font-size:11px;color:var(--text2)">DISPUTE RESOLUTION</span><br><strong>{safe(ner.get('dispute_resolution','Unknown'))}</strong></div>
  </div>
</div>
""", unsafe_allow_html=True)

    tabs = st.tabs(["Summary", "Obligations", "Risks", "Deadlines", "Clauses", "Compliance", "Anomalies", "Export", "AI Decision Room"])

    with tabs[0]:  # Summary
        exec_sum = summary.get("executive_summary", "No summary generated.")
        st.markdown(f'<div class="card"><p style="line-height:1.8;font-size:14px">{safe(exec_sum)}</p></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            strengths = summary.get("key_strengths", [])
            if strengths:
                st.markdown("**Key Strengths**")
                for s in strengths:
                    st.markdown(f"- {s}")
            actions = summary.get("immediate_actions", [])
            if actions:
                st.markdown("**Immediate Actions Required**")
                for a in actions:
                    st.markdown(f"- {a}")
        with c2:
            concerns = summary.get("key_concerns", [])
            if concerns:
                st.markdown("**Key Concerns**")
                for c in concerns:
                    st.markdown(f"- {c}")
            nego = summary.get("negotiation_points", [])
            if nego:
                st.markdown("**Negotiation Points**")
                for n in nego:
                    st.markdown(f"- {n}")
        disc = summary.get("disclaimer", "")
        if disc:
            st.caption(f"*{disc}*")

    with tabs[1]:  # Obligations
        obls = obligations.get("obligations", [])
        if obls:
            obligations_html = ""
            for i, ob in enumerate(obls):
                pri = ob.get("priority","Medium")
                pri_cls = "b-critical" if pri=="High" else "b-medium" if pri=="Low" else "b-warn"
                obligations_html += f"""
<div class="obl-item">
  <div class="obl-num">{safe(ob.get('id','OBL').split('-')[-1] if '-' in str(ob.get('id','')) else i+1)}</div>
  <div class="obl-body">
    <div class="obl-title">{safe(ob.get('description',''))}</div>
    <div class="obl-meta">
      Party: <strong>{safe(ob.get('party',''))}</strong> &nbsp;|&nbsp;
      Category: {safe(ob.get('category',''))} &nbsp;|&nbsp;
      Clause: {safe(ob.get('clause_ref',''))} &nbsp;|&nbsp;
      <span class="badge {pri_cls}">{safe(pri)}</span>
    </div>
    {f'<div class="obl-meta" style="margin-top:4px">Conditions: {safe(ob.get("conditions",""))}</div>' if ob.get('conditions') else ''}
  </div>
</div>"""
            st.markdown(obligations_html, unsafe_allow_html=True)
        else:
            st.info("No obligations extracted.")

    with tabs[2]:  # Risks
        risk_list = risks.get("risks", [])
        if risk_list:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risks.get("overall_risk_score", 0),
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Overall Risk Score"},
                gauge={
                    "axis":{"range":[0,10]},
                    "bar":{"color":"#3b82f6"},
                    "steps":[
                        {"range":[0,3],"color":"#dcfce7"},
                        {"range":[3,6],"color":"#fef3c7"},
                        {"range":[6,8],"color":"#fee2e2"},
                        {"range":[8,10],"color":"#dc2626"},
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(t=40,b=0,l=0,r=0),
                              paper_bgcolor="rgba(0,0,0,0)", font_color="#64748b")
            st.plotly_chart(fig, use_container_width=True)

            # Missing protections
            missing = risks.get("missing_protections", [])
            red_flags = risks.get("red_flags", [])
            if missing or red_flags:
                c1, c2 = st.columns(2)
                with c1:
                    if missing:
                        st.markdown("**Missing Protections**")
                        for m in missing:
                            st.markdown(f'<span class="badge b-fail">{safe(m)}</span> ', unsafe_allow_html=True)
                with c2:
                    if red_flags:
                        st.markdown("**Red Flags**")
                        for rf in red_flags:
                            st.markdown(f'<span class="badge b-warn">{safe(rf)}</span> ', unsafe_allow_html=True)

            st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
            for r in risk_list:
                sev = r.get("severity","Medium")
                cls = {"Critical":"risk-critical","High":"risk-high","Medium":"risk-medium","Low":"risk-low"}.get(sev,"risk-low")
                sev_badge = {"Critical":"b-critical","High":"b-high","Medium":"b-medium","Low":"b-low"}.get(sev,"b-low")
                st.markdown(f"""
<div class="risk-item {cls}">
  <div class="risk-title">
    {safe(r.get('id',''))} — {safe(r.get('title',''))}
    <span class="badge {sev_badge}" style="margin-left:8px">{safe(sev)}</span>
    <span class="badge b-na" style="margin-left:4px">{safe(r.get('likelihood',''))} likelihood</span>
  </div>
  <div class="risk-desc">{safe(r.get('description',''))}</div>
  <div class="risk-desc" style="margin-top:6px">
    Financial Exposure: <strong>{safe(r.get('financial_exposure','Unknown'))}</strong> &nbsp;|&nbsp;
    Clause: {safe(r.get('clause_ref',''))} &nbsp;|&nbsp; Confidence: {float(r.get('confidence',0) or 0):.0%}
  </div>
  <div class="risk-desc" style="margin-top:4px;color:#059669">
    Mitigation: {safe(r.get('mitigation',''))}
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No risks extracted.")

    with tabs[3]:  # Deadlines
        dl_list = deadlines.get("deadlines", [])
        info_cols = st.columns(4)
        info_cols[0].metric("Contract Term", deadlines.get("contract_term","N/A"))
        info_cols[1].metric("Effective Date", deadlines.get("effective_date","N/A"))
        info_cols[2].metric("Notice Period", deadlines.get("notice_period","N/A"))
        info_cols[3].metric("Auto-Renewal", "Yes" if deadlines.get("auto_renewal") else "No")
        if dl_list:
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            for dl in dl_list:
                pri = dl.get("priority","Medium")
                pri_cls = "b-critical" if pri=="Critical" else "b-high" if pri=="High" else "b-medium" if pri=="Low" else "b-warn"
                st.markdown(f"""
<div class="timeline-item">
  <div class="timeline-dot"></div>
  <div class="timeline-body">
    <div class="timeline-title">
      {safe(dl.get('event',''))} — <strong>{safe(dl.get('date',''))}</strong>
      <span class="badge {pri_cls}" style="margin-left:8px">{safe(pri)}</span>
    </div>
    <div class="timeline-meta">
      Party: {safe(dl.get('party',''))} | Clause: {safe(dl.get('clause_ref',''))} |
      Consequence: {safe(dl.get('consequence',''))}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    with tabs[4]:  # Clauses
        clause_list = clauses.get("clauses", [])
        balance = clauses.get("overall_balance","")
        if balance:
            st.markdown(f"**Overall Balance:** {balance}")
        if clause_list:
            rows = []
            for cl in clause_list:
                present = "Yes" if cl.get("present") else "No"
                assess = cl.get("assessment","")
                badge_cls = {"Favorable":"b-favorable","Neutral":"b-neutral",
                             "Concerning":"b-concerning","Missing":"b-fail"}.get(assess,"b-na")
                rows.append({
                    "Clause": cl.get("title", cl.get("type","")),
                    "Present": present,
                    "Assessment": assess,
                    "Clause Ref": cl.get("clause_ref",""),
                    "Notes": cl.get("notes","")[:80] + "..." if len(cl.get("notes","")) > 80 else cl.get("notes","")
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        missing_cl = clauses.get("standard_clauses_missing",[])
        if missing_cl:
            st.markdown("**Standard Clauses Missing:**")
            st.markdown(" ".join(f'<span class="badge b-fail">{safe(m)}</span>' for m in missing_cl), unsafe_allow_html=True)

    with tabs[5]:  # Compliance
        gdpr  = compliance.get("gdpr",{})
        ccpa  = compliance.get("ccpa",{})
        hipaa = compliance.get("hipaa",{})
        soc2  = compliance.get("soc2",{})
        score = compliance.get("overall_score", 0)
        score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
        st.markdown(f"""
<div class="card" style="margin-bottom:16px">
  <div style="display:flex;align-items:center;gap:24px">
    <div>
      <div style="font-size:40px;font-weight:800;color:{score_color}">{score}</div>
      <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.07em">Overall Score</div>
    </div>
    <div style="flex:1">
""", unsafe_allow_html=True)
        for reg_name, reg_data in [("GDPR",gdpr),("CCPA",ccpa),("HIPAA",hipaa),("SOC2",soc2)]:
            reg_score = reg_data.get("score",0)
            applicable = reg_data.get("applicable", True)
            reg_color = "#94a3b8" if applicable is False else "#10b981" if reg_score >= 70 else "#f59e0b" if reg_score >= 40 else "#ef4444"
            reg_label = "Not Applicable" if applicable is False else "Compliant" if reg_data.get("compliant") else "Issues"
            reg_badge = "b-na" if applicable is False else "b-pass" if reg_data.get("compliant") else "b-fail"
            reg_value = "N/A" if applicable is False else str(reg_score)
            st.markdown(f"""
<div class="compliance-row">
  <div class="compliance-label">{reg_name}</div>
  <div style="display:flex;align-items:center;gap:12px">
    <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{reg_score}%;background:{reg_color}"></div></div>
    <span style="font-size:13px;font-weight:700;color:{reg_color};min-width:32px">{reg_value}</span>
    <span class="badge {reg_badge}">{reg_label}</span>
  </div>
</div>""", unsafe_allow_html=True)
        st.markdown("</div></div></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Missing Clauses**")
            for m in compliance.get("recommendations", []):
                st.markdown(f"- {m}")
        with c2:
            for reg_name, reg_data in [("GDPR",gdpr),("CCPA",ccpa)]:
                missing = reg_data.get("missing",[])
                if missing:
                    st.markdown(f"**{reg_name} Missing:**")
                    for m in missing:
                        st.markdown(f"- {m}")

    with tabs[6]:  # Anomalies
        anom_list = anomalies.get("anomalies",[])
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Anomalies", anomalies.get("total_anomalies",len(anom_list)))
        c2.metric("Critical", anomalies.get("critical_count",0))
        c3.metric("Market Deviation", f"{anomalies.get('market_deviation_score',0)}/10")
        if anomalies.get("summary"):
            st.markdown(f"*{anomalies['summary']}*")
        for an in anom_list:
            sev = an.get("severity","Medium")
            cls = {"Critical":"risk-critical","High":"risk-high","Medium":"risk-medium","Low":"risk-low"}.get(sev,"risk-low")
            st.markdown(f"""
<div class="risk-item {cls}">
  <div class="risk-title">{safe(an.get('id',''))} — {safe(an.get('type',''))}: {safe(str(an.get('description',''))[:80])}</div>
  <div class="risk-desc">Location: {safe(an.get('location',''))} | Impact: {safe(an.get('impact',''))}</div>
  <div class="risk-desc" style="color:#059669;margin-top:4px">Action: {safe(an.get('recommendation',''))}</div>
</div>""", unsafe_allow_html=True)

    with tabs[7]:  # Export
        st.markdown("**Export Analysis Report**")
        ecols = st.columns(4)
        with ecols[0]:
            json_data = json.dumps(results, indent=2)
            st.download_button("Download JSON", json_data, "lexforge_analysis.json", "application/json", use_container_width=True)
        with ecols[1]:
            md_text = f"""# LexForge AI Analysis Report\n\n**Contract:** {meta.get('filename','')}\n**Risk Score:** {risks.get('overall_risk_score',0)}/10\n**Assessment:** {summary.get('overall_assessment','')}\n**Recommendation:** {summary.get('recommendation','')}\n\n## Executive Summary\n{summary.get('executive_summary','')}\n\n## Key Risks\n{chr(10).join(f"- {r.get('title','')}: {r.get('severity','')}" for r in risks.get('risks',[]))}\n\n## Disclaimer\n{summary.get('disclaimer','')}"""
            st.download_button("Download Markdown", md_text, "lexforge_analysis.md", "text/markdown", use_container_width=True)
        with ecols[2]:
            risk_df = pd.DataFrame(risks.get("risks",[]))
            if not risk_df.empty:
                csv = risk_df.to_csv(index=False)
                st.download_button("Risk Register CSV", csv, "risk_register.csv", "text/csv", use_container_width=True)
        with ecols[3]:
            if st.button("Chat about this", use_container_width=True):
                nav("chat")

    with tabs[8]:  # AI Decision Room
        _render_ai_decision_room(results)


# ── WRITER ─────────────────────────────────────────────────────────────────────
def build_contract_pdf(title: str, body: str, signatories: List[Dict]) -> bytes:
    """Render a contract to a professional PDF with signature blocks for managers/authorities."""
    import re
    from datetime import datetime
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as _canvas
    from reportlab.lib.utils import ImageReader
    from xml.sax.saxutils import escape

    title = strip_markdown_stars(title)
    body = strip_markdown_stars(body)
    buf = io.BytesIO()
    NAVY = colors.HexColor("#0f2247")
    BLUE = colors.HexColor("#1f6fd0")
    GREY = colors.HexColor("#475569")
    RULE = colors.HexColor("#c9d6ea")
    INK  = colors.HexColor("#1a2231")
    today  = datetime.now().strftime("%d %B %Y")
    ref_no = "NEL/" + datetime.now().strftime("%Y%m%d") + "/" + (
        "".join(ch for ch in str(title).upper() if ch.isalnum())[:6] or "CONTRACT")

    # Prefer Times New Roman (classic serif legal typeface); fall back to built-in Times.
    font_regular, font_bold, font_italic = "Times-Roman", "Times-Bold", "Times-Italic"
    try:
        _tnr = {"NEL": r"C:\Windows\Fonts\times.ttf",
                "NEL-Bold": r"C:\Windows\Fonts\timesbd.ttf",
                "NEL-Italic": r"C:\Windows\Fonts\timesi.ttf"}
        if all(os.path.exists(p) for p in _tnr.values()):
            for _n, _p in _tnr.items():
                pdfmetrics.registerFont(TTFont(_n, _p))
            font_regular, font_bold, font_italic = "NEL", "NEL-Bold", "NEL-Italic"
    except Exception:
        pass

    def ascii_safe(t):
        t = strip_markdown_stars(t)
        for s, d in {chr(0x2013): "-", chr(0x2014): "-", chr(0x2018): "'",
                     chr(0x2019): "'", chr(0x201c): '"', chr(0x201d): '"',
                     chr(0x2022): "-", chr(0x00b7): "-", chr(0x00a0): " "}.items():
            t = t.replace(s, d)
        return t

    _logo_img = None
    for _nm in ("navneet_logo.png", "navneet_logo.jpg", "navneet_logo.jpeg"):
        _pp = os.path.join(ASSETS_DIR, _nm)
        if os.path.exists(_pp):
            _logo_img = _pp
            break

    def pdf_meta_safe(t: str) -> str:
        replacements = {
            "\u2013": "-",
            "\u2014": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2022": "-",
            "\u00b7": "-",
        }
        for src, dst in replacements.items():
            t = str(t).replace(src, dst)
        return t.encode("ascii", "ignore").decode("ascii") or "Navneet ContractAI Contract"

    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2.7*cm, bottomMargin=2.0*cm,
                            leftMargin=2.2*cm, rightMargin=2.2*cm,
                            title=pdf_meta_safe(title), author="Navneet Education Limited")

    title_s = ParagraphStyle("title", fontName=font_bold, fontSize=17, leading=22,
                             alignment=TA_CENTER, textColor=NAVY, spaceAfter=2)
    meta_s  = ParagraphStyle("meta", fontName=font_regular, fontSize=9, leading=13,
                             alignment=TA_CENTER, textColor=GREY)
    head_s  = ParagraphStyle("head", fontName=font_bold, fontSize=11.5, leading=16,
                             textColor=NAVY, spaceBefore=12, spaceAfter=5, keepWithNext=1)
    body_s  = ParagraphStyle("body", fontName=font_regular, fontSize=10.5, leading=16,
                             alignment=TA_JUSTIFY, spaceAfter=7, textColor=INK)
    wit_s   = ParagraphStyle("wit", parent=body_s, fontName=font_italic, spaceBefore=6)
    s_lbl   = ParagraphStyle("slbl", fontName=font_regular, fontSize=9.5, leading=14, textColor=GREY)
    s_party = ParagraphStyle("sp", fontName=font_bold, fontSize=10.5, leading=14, textColor=NAVY)

    def draw_logo(c, x, y, w, h):
        if _logo_img:
            try:
                c.drawImage(ImageReader(_logo_img), x, y, width=w, height=h,
                            preserveAspectRatio=True, mask="auto")
                return
            except Exception:
                pass
        c.saveState()
        c.setFillColor(colors.HexColor("#0a4f8f"))
        c.ellipse(x, y, x + w, y + h, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#1f8fdc"))
        c.ellipse(x + 0.7, y + 0.7, x + w - 0.7, y + h - 0.7, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#7fc4ef"))
        c.ellipse(x + w * 0.10, y + h * 0.48, x + w * 0.90, y + h * 0.93, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont(font_bold, h * 0.34)
        c.drawCentredString(x + w / 2, y + h * 0.33, "NAVNEET")
        c.restoreState()

    def is_heading(s):
        s = s.strip()
        if not s or len(s) > 90:
            return False
        if re.match(r'^(ARTICLE|SECTION|SCHEDULE|ANNEXURE?|APPENDIX|RECITALS?|WHEREAS|DEFINITIONS)\b', s, re.I):
            return True
        if re.match(r'^\d{1,2}(\.\d{1,2})*\.?\s+[A-Z]', s):
            return True
        letters = [ch for ch in s if ch.isalpha()]
        if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.85 and len(s.split()) <= 10:
            return True
        return False

    def header_footer(c, d):
        pw, ph = A4
        c.saveState()
        draw_logo(c, d.leftMargin, ph - 1.98*cm, 2.55*cm, 1.12*cm)
        c.setFillColor(NAVY); c.setFont(font_bold, 10.5)
        c.drawRightString(pw - d.rightMargin, ph - 1.34*cm, "NAVNEET EDUCATION LIMITED")
        c.setFillColor(GREY); c.setFont(font_regular, 8)
        c.drawRightString(pw - d.rightMargin, ph - 1.70*cm, "Private & Confidential")
        c.setStrokeColor(RULE); c.setLineWidth(0.9)
        c.line(d.leftMargin, ph - 2.16*cm, pw - d.rightMargin, ph - 2.16*cm)
        c.setStrokeColor(RULE); c.setLineWidth(0.6)
        c.line(d.leftMargin, 1.58*cm, pw - d.rightMargin, 1.58*cm)
        c.setFillColor(GREY); c.setFont(font_regular, 7.5)
        c.drawString(d.leftMargin, 1.18*cm, "Ref: " + ascii_safe(ref_no))
        c.drawCentredString(pw / 2.0, 1.18*cm, "Private & Confidential")
        c.drawRightString(pw - d.rightMargin, 1.18*cm, "Initials: ______ / ______")
        c.restoreState()

    def esc(t): return escape(ascii_safe(t))

    flow = [Paragraph(esc(title), title_s),
            HRFlowable(width="40%", thickness=1.4, color=BLUE, spaceBefore=4,
                       spaceAfter=8, hAlign="CENTER"),
            Paragraph("Reference No.: " + esc(ref_no) + " &nbsp;&nbsp;&bull;&nbsp;&nbsp; Date: " + esc(today), meta_s),
            Spacer(1, 14)]
    for raw in body.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            flow.append(Spacer(1, 5))
        elif is_heading(line):
            flow.append(Paragraph(esc(line), head_s))
        else:
            flow.append(Paragraph(esc(line), body_s))

    flow.append(Spacer(1, 14))
    flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceAfter=10))
    flow.append(Paragraph("IN WITNESS WHEREOF, the parties hereto have executed this Agreement "
                          "as of the date first written above, on this " + esc(today) + ".", wit_s))
    flow.append(Spacer(1, 18))

    def sig_cell(s):
        return [Paragraph("For and on behalf of", s_lbl),
                Paragraph(esc(s.get("party", "")) or "_______________________", s_party),
                Spacer(1, 34),
                Paragraph("_______________________________", s_lbl),
                Paragraph("Authorised Signatory", s_lbl),
                Spacer(1, 5),
                Paragraph("Name: " + (esc(s.get("name", "")) or "_______________________"), s_lbl),
                Paragraph("Designation: " + (esc(s.get("title", "")) or "____________________"), s_lbl),
                Paragraph("Date: ____________________", s_lbl),
                Paragraph("Place: ___________________", s_lbl)]

    rows = []
    sigs = [s for s in signatories if s.get("party") or s.get("name")]
    for i in range(0, len(sigs), 2):
        left = sig_cell(sigs[i])
        right = sig_cell(sigs[i+1]) if i + 1 < len(sigs) else [Spacer(1, 1)]
        rows.append([left, right])
    if rows:
        t = Table(rows, colWidths=[8.1*cm, 8.1*cm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 0),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 26)]))
        flow.append(t)

    flow.append(Spacer(1, 6))
    flow.append(Paragraph("In the presence of (Witnesses):", s_party))
    flow.append(Spacer(1, 4))
    _wcell = lambda n: [Paragraph(str(n) + ".  _______________________________", s_lbl),
                        Paragraph("Name: ____________________", s_lbl),
                        Paragraph("Address: __________________", s_lbl)]
    _wt = Table([[_wcell(1), _wcell(2)]], colWidths=[8.1*cm, 8.1*cm])
    _wt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("TOPPADDING", (0, 0), (-1, -1), 8)]))
    flow.append(_wt)

    class NumberedCanvas(_canvas.Canvas):
        def __init__(self, *a, **k):
            _canvas.Canvas.__init__(self, *a, **k)
            self._saved = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved)
            for state in self._saved:
                self.__dict__.update(state)
                self.setFont(font_regular, 7.5)
                self.setFillColor(GREY)
                self.drawCentredString(A4[0] / 2.0, 0.82 * cm,
                                       "Page %d of %d" % (self._pageNumber, total))
                _canvas.Canvas.showPage(self)
            _canvas.Canvas.save(self)

    doc.build(flow, onFirstPage=header_footer, onLaterPages=header_footer,
              canvasmaker=NumberedCanvas)
    return buf.getvalue()


def page_writer():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">Contract Studio</div>
  <div class="page-header-sub">Generate, review, edit and govern structured contract drafts with reusable clauses, version-ready records and professional exports.</div>
</div>
""", unsafe_allow_html=True)

    with st.container(key="writer_workspace"):
        st.markdown('<div class="inner">', unsafe_allow_html=True)

        writer_tabs = st.tabs(["Generate New", "Clause Library", "My Contracts", "Edit Contract"])

        with writer_tabs[0]:  # Generate
            c1, c2 = st.columns([1, 2])
            with c1.container(border=True):
                contract_type = st.selectbox(
                    "Contract Type",
                    list(WRITER_TEMPLATES.keys()),
                    format_func=lambda x: WRITER_TEMPLATES[x]["name"]
                )
                template = WRITER_TEMPLATES[contract_type]
                st.caption(template.get("description",""))
                st.markdown("---")

                fields = template.get("fields",[])
                params = {}
                for f in fields:
                    label = f.replace("_", " ").title()
                    params[f] = st.text_input(label, key=f"wr_{f}")

                st.markdown("**Additional Instructions**")
                extra = st.text_area("Custom requirements", height=80,
                    placeholder="e.g. Include GDPR DPA addendum, use Singapore law...")
                if extra:
                    params["additional_instructions"] = extra

                gen_btn = st.button("Generate Contract", type="primary", use_container_width=True)

            with c2.container(border=True):
                if gen_btn:
                    if not st.session_state.api_key:
                        st.error("The workspace OpenRouter key is not configured.")
                    else:
                        with st.spinner("Drafting your contract with AI..."):
                            writer = get_writer()
                            contract_text = writer.generate_contract(contract_type, params)
                            if contract_text.strip().lower().startswith("error:"):
                                st.session_state.generated_contract = ""
                                st.error(contract_text)
                                return
                            st.session_state.generated_contract = contract_text
                        st.success("Contract generated!")

                if st.session_state.generated_contract:
                    ct = strip_markdown_stars(st.session_state.generated_contract)
                    st.session_state.generated_contract = ct
                    wc = len(ct.split())
                    st.markdown(f"**Word count:** {wc} | **Reading level:** Professional")

                    st.text_area("Generated Contract", ct, height=500, key="contract_display")

                    # Review
                    if st.button("AI Review this Draft"):
                        with st.spinner("Reviewing..."):
                            writer = get_writer()
                            review = writer.review_contract(ct)
                            st.session_state.contract_review = review

                    if st.session_state.contract_review:
                        rv = st.session_state.contract_review
                        rcols = st.columns(3)
                        rcols[0].metric("Completeness", f"{rv.get('completeness_score',0)}%")
                        rcols[1].metric("Quality", f"{rv.get('quality_score',0)}%")
                        rcols[2].metric("Risk", f"{rv.get('risk_score',0)}/10")
                        if rv.get("missing_clauses"):
                            st.warning("Missing: " + ", ".join(rv["missing_clauses"]))

                    # Save & Export
                    scols = st.columns(3)
                    with scols[0]:
                        _party = params.get('party_a') or params.get('employer') or params.get('client') or ''
                        _ttl = WRITER_TEMPLATES[contract_type]['name'] + (f" — {_party}" if _party else "")
                        title = st.text_input("Contract title", value=_ttl)
                        if st.button("Save to Library", use_container_width=True):
                            cid = save_generated_contract(
                                title, contract_type, ct,
                                owner_id=(get_user(st.session_state.auth_user) or {}).get("id", ""),
                                owner_name=st.session_state.username or st.session_state.auth_user or "",
                            )
                            st.success(f"Saved! ID: {cid[:8]}")
                    with scols[1]:
                        st.download_button("Download TXT", ct, f"{contract_type}.txt", "text/plain", use_container_width=True)
                    with scols[2]:
                        st.download_button("Download MD", f"# {WRITER_TEMPLATES[contract_type]['name']}\n\n{ct}", f"{contract_type}.md", "text/markdown", use_container_width=True)

                    # ── Signature blocks + professional PDF ──────────────────
                    st.markdown("---")
                    st.markdown("**Signature blocks — for managers / authorised signatories**")
                    st.caption("These appear as signing spaces at the end of the exported PDF.")
                    sg1, sg2 = st.columns(2)
                    with sg1:
                        st.markdown("**Party A — authority**")
                        a_party = st.text_input("Party / Entity", "Navneet Education Limited", key="sig_a_party")
                        a_name  = st.text_input("Authorised signatory name", key="sig_a_name")
                        a_title = st.text_input("Designation / Title", "Manager", key="sig_a_title")
                    with sg2:
                        st.markdown("**Party B — counterparty**")
                        b_party = st.text_input("Party / Entity", params.get("party_b", params.get("client", "")), key="sig_b_party")
                        b_name  = st.text_input("Authorised signatory name", key="sig_b_name")
                        b_title = st.text_input("Designation / Title", "Director", key="sig_b_title")

                    signatories = [
                        {"party": a_party, "name": a_name, "title": a_title},
                        {"party": b_party, "name": b_name, "title": b_title},
                    ]
                    try:
                        pdf_bytes = build_contract_pdf(title, ct, signatories)
                        st.download_button("📄 Download Signed-Ready PDF", pdf_bytes,
                                           f"{contract_type}_contract.pdf", "application/pdf",
                                           type="primary", use_container_width=True)
                    except Exception as e:
                        st.error(f"PDF export needs reportlab (pip install reportlab). {e}")

        with writer_tabs[1]:  # Clause Library
            st.markdown("**Pre-Built Legal Clause Library**")
            st.caption("Click any clause to generate a full draft. 20+ professional clauses ready.")
            search_cl = st.text_input("Search clauses", placeholder="e.g. GDPR, indemnification...")
            filtered = {k:v for k,v in CLAUSE_LIBRARY.items()
                       if not search_cl or search_cl.lower() in k.lower() or search_cl.lower() in v.lower()}
            cols = st.columns(2)
            for i,(name,desc) in enumerate(filtered.items()):
                with cols[i % 2]:
                    with st.expander(name):
                        st.caption(desc)
                        context_inp = st.text_input("Context/Parameters", key=f"cl_ctx_{name}",
                            placeholder="e.g. 12-month cap, US jurisdiction...")
                        if st.button(f"Generate {name} Clause", key=f"cl_gen_{name}"):
                            if not st.session_state.api_key:
                                st.error("The workspace OpenRouter key is not configured.")
                            else:
                                with st.spinner("Generating..."):
                                    writer = get_writer()
                                    clause_text = writer.generate_clause(name, context_inp)
                                st.text_area("Generated Clause", clause_text, height=200, key=f"cl_out_{name}")
                                st.download_button("Download", clause_text, f"{name.replace(' ','_')}.txt", key=f"cl_dl_{name}")

        with writer_tabs[2]:  # My Contracts CRUD
            st.markdown("**Saved Contracts**")
            contracts = get_all_generated_contracts()
            if not contracts:
                st.info("No saved contracts yet. Generate one above!")
            else:
                search_my = st.text_input("Search saved contracts")
                if search_my:
                    contracts = [c for c in contracts if search_my.lower() in c["title"].lower() or search_my.lower() in c["contract_type"].lower()]
                for c in contracts:
                    clean_content = strip_markdown_stars(c["content"])
                    with st.expander(f"{c['title']} — {c['contract_type']} ({c['word_count']} words) — {c['updated_at'][:10]}"):
                        c1, c2, c3 = st.columns([2,1,1])
                        with c1:
                            st.caption(f"Owner: {c.get('owner_name') or 'Workspace user'} · Status: {c.get('status', 'draft').title()}")
                            new_title = st.text_input("Title", c["title"], key=f"ct_title_{c['id']}")
                            new_notes = st.text_input("Notes", c.get("notes",""), key=f"ct_notes_{c['id']}")
                        with c2:
                            if st.button("Update", key=f"ct_upd_{c['id']}", use_container_width=True):
                                update_generated_contract(c["id"], title=new_title, notes=new_notes)
                                st.success("Updated!")
                                st.rerun()
                            st.download_button("Download", clean_content, f"{c['title']}.txt",
                                             key=f"ct_dl_{c['id']}", use_container_width=True)
                        with c3:
                            if st.button("Edit in Editor", key=f"ct_edit_{c['id']}", use_container_width=True):
                                st.session_state.edit_contract_id = c["id"]
                                st.session_state.generated_contract = clean_content
                            if st.button("Delete", key=f"ct_del_{c['id']}", use_container_width=True, type="secondary"):
                                delete_generated_contract(c["id"])
                                st.success("Deleted!")
                                st.rerun()
                        st.text_area("Preview", clean_content[:500] + "...", height=120, disabled=True, key=f"ct_prev_{c['id']}")

        with writer_tabs[3]:  # Edit Contract
            edit_id = st.session_state.get("edit_contract_id")
            if edit_id:
                ec = next((c for c in get_all_generated_contracts() if c["id"] == edit_id), None)
                if ec:
                    st.markdown(f"**Editing:** {ec['title']}")
                    edited = st.text_area(
                        "Contract Content", strip_markdown_stars(ec["content"]),
                        height=500, key="editor_main"
                    )
                    improve_inst = st.text_input("AI Improvement Instruction",
                        placeholder="e.g. Add a stronger limitation of liability clause, make payment terms stricter...")
                    ecols = st.columns(3)
                    with ecols[0]:
                        if st.button("Save Changes", type="primary"):
                            update_generated_contract(edit_id, content=strip_markdown_stars(edited))
                            st.success("Saved!")
                            st.rerun()
                    with ecols[1]:
                        if st.button("AI Improve") and improve_inst:
                            with st.spinner("AI improving contract..."):
                                writer = get_writer()
                                improved = writer.improve_contract(edited, improve_inst)
                            update_generated_contract(edit_id, content=improved)
                            st.success("Improved and saved!")
                            st.rerun()
                    with ecols[2]:
                        st.download_button("Download", strip_markdown_stars(edited), f"{ec['title']}.txt")
            else:
                st.info("Select a contract from 'My Contracts' to edit it here.")

        st.markdown('</div>', unsafe_allow_html=True)


# ── RAG CHAT ───────────────────────────────────────────────────────────────────
def page_chat():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">Grounded Contract Chat</div>
  <div class="page-header-sub">Interrogate contract evidence through retrieval-augmented generation, document-scoped vector search and source-aware answers.</div>
</div>
""", unsafe_allow_html=True)

    with st.container(key="rag_chat_workspace"):
        st.markdown('<div class="inner">', unsafe_allow_html=True)

        if not st.session_state.api_key:
            st.warning("The workspace OpenRouter key is not configured.")
            return

        # Contract selector
        contracts = get_all_contracts()
        if not contracts and not st.session_state.current_doc_id:
            st.info("Analyze a contract first to enable chat.")
            if st.button("Go to Analyze"):
                nav("analyze")
            return

        col1, col2 = st.columns([2,1])
        with col1:
            contract_options = {c["id"]: f"{c['filename']} ({c['uploaded_at'][:10]})" for c in contracts}
            if st.session_state.current_doc_id:
                contract_options = {st.session_state.current_doc_id: "Current document"} | contract_options
            selected_id = st.selectbox("Contract to chat about",
                list(contract_options.keys()),
                format_func=lambda x: contract_options.get(x, x))
        with col2:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                clear_chat_history(selected_id)
                st.rerun()

        # Quick questions
        st.markdown("**Quick Questions:**")
        quick_qs = [
            "What are the main obligations of each party?",
            "What are the key risks in this contract?",
            "When does this contract expire?",
            "What is the liability cap?",
            "Is this contract GDPR compliant?",
            "What are the termination conditions?",
        ]
        qcols = st.columns(3)
        for i, q in enumerate(quick_qs):
            with qcols[i % 3]:
                if st.button(q, key=f"qq_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":q})
                    ctx = build_contract_context(q, selected_id, top_k=6)
                    agent = get_agent()
                    answer = agent.chat_with_contract(q, ctx)
                    st.session_state.chat_history.append({"role":"assistant","content":answer})
                    save_chat_message(selected_id, "user", q)
                    save_chat_message(selected_id, "assistant", answer, OPENROUTER_MODEL)
                    st.rerun()

        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

        # Load DB history
        if selected_id and not st.session_state.chat_history:
            db_hist = get_chat_history(selected_id)
            st.session_state.chat_history = [{"role":h["role"],"content":h["content"]} for h in db_hist[-20:]]

        # Render chat
        chat_html = ""
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"].replace("<","&lt;").replace(">","&gt;")
            if role == "user":
                chat_html += f'<div class="chat-msg chat-user"><div class="chat-bubble" style="margin-left:auto">{content}</div><div class="chat-avatar">U</div></div>'
            else:
                chat_html += f'<div class="chat-msg chat-ai"><div class="chat-avatar">AI</div><div class="chat-bubble">{content}</div></div>'

        st.markdown(f'<div class="chat-transcript">{chat_html}</div>', unsafe_allow_html=True)

        # Input
        with st.form("chat_form", clear_on_submit=True):
            user_q = st.text_input("Ask a question about your contract...", label_visibility="collapsed",
                                    placeholder="e.g. What happens if I terminate early?")
            submitted = st.form_submit_button("Send", type="primary", use_container_width=False)

        if submitted and user_q.strip():
            st.session_state.chat_history.append({"role":"user","content":user_q})
            ctx = build_contract_context(user_q, selected_id, top_k=6)
            agent = get_agent()
            with st.spinner("Retrieving and generating answer..."):
                answer = agent.chat_with_contract(user_q, ctx)
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            save_chat_message(selected_id, "user", user_q)
            save_chat_message(selected_id, "assistant", answer, OPENROUTER_MODEL)
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# ── HISTORY ────────────────────────────────────────────────────────────────────
def page_history():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">Contract Repository</div>
  <div class="page-header-sub">Search, inspect, classify and maintain analyzed contracts with risk posture, tags, notes and complete lifecycle controls.</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="inner">', unsafe_allow_html=True)

    stats = get_dashboard_stats()
    st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="metric-val">{stats['total_contracts']}</div><div class="metric-label">Contracts Analyzed</div></div>
  <div class="metric-card"><div class="metric-val" style="color:#ef4444">{stats['high_risk_count']}</div><div class="metric-label">High Risk</div></div>
  <div class="metric-card"><div class="metric-val">{stats['avg_risk_score']}</div><div class="metric-label">Avg Risk Score</div></div>
  <div class="metric-card"><div class="metric-val">{stats['generated_contracts']}</div><div class="metric-label">Contracts Drafted</div></div>
</div>
<div style="height:24px"></div>
""", unsafe_allow_html=True)
    if stats.get("failed_analyses", 0):
        st.warning(f"{stats['failed_analyses']} earlier API-failed record(s) are excluded from risk statistics. Re-run them to generate a valid review.")

    # Search + filter
    c1, c2 = st.columns([3,1])
    with c1:
        search = st.text_input("Search contracts", placeholder="filename, type, assessment...")
    with c2:
        sort_by = st.selectbox("Sort", ["Newest First", "Risk High-Low", "Risk Low-High"])

    contracts = search_contracts(search) if search else get_all_contracts()
    if sort_by == "Risk High-Low":
        contracts = sorted(contracts, key=lambda x: x.get("risk_score",0), reverse=True)
    elif sort_by == "Risk Low-High":
        contracts = sorted(contracts, key=lambda x: x.get("risk_score",0))

    if not contracts:
        st.info("No contracts found. Analyze a contract to get started.")
    else:
        for c in contracts:
            risk = c.get("risk_score",0)
            risk_color = "#ef4444" if risk >= 7 else "#f59e0b" if risk >= 4 else "#10b981"
            risk_label = "Analysis failed" if c.get("status") == "failed" else f"Risk: {risk}/10"
            with st.expander(f"**{c['filename']}** | {risk_label} | {c.get('contract_type','?')} | {c['uploaded_at'][:10]}"):
                col1, col2, col3 = st.columns([2,2,1])
                with col1:
                    st.markdown(f"**Owner:** {c.get('owner_name') or 'Workspace user'}")
                    st.markdown(f"**Assessment:** {c.get('assessment','')}")
                    st.markdown(f"**Model:** {c.get('model_used','')}")
                    st.markdown(f"**Words:** {c.get('word_count',0)} | **Pages:** {c.get('page_count',0)}")
                    new_notes = st.text_area("Notes", c.get("notes",""), key=f"notes_{c['id']}", height=80)
                    new_tags  = st.text_input("Tags (comma-separated)", c.get("tags","[]").strip("[]\"'").replace('"',''), key=f"tags_{c['id']}")
                    new_type  = st.text_input("Contract Type", c.get("contract_type",""), key=f"ctype_{c['id']}")
                with col2:
                    analysis = c.get("analysis_json")
                    if analysis and isinstance(analysis, str):
                        try:
                            analysis = json.loads(analysis)
                        except Exception:
                            analysis = {}
                    if analysis:
                        risks_data = analysis.get("risks",{})
                        risk_list  = risks_data.get("risks",[])[:3]
                        if risk_list:
                            st.markdown("**Top Risks:**")
                            for r in risk_list:
                                sev = r.get("severity","")
                                st.markdown(f'<span class="badge {"b-critical" if sev=="Critical" else "b-high" if sev=="High" else "b-medium"}">{sev}</span> {r.get("title","")}', unsafe_allow_html=True)
                with col3:
                    if st.button("Update", key=f"upd_{c['id']}", use_container_width=True, type="primary"):
                        update_contract(c["id"], {"notes":new_notes, "contract_type":new_type})
                        st.success("Updated!")
                        st.rerun()
                    if st.button("Load Analysis", key=f"load_{c['id']}", use_container_width=True):
                        if analysis:
                            st.session_state.analysis_results = analysis
                            st.session_state.current_doc_id = c.get("doc_hash","")
                            st.session_state.current_doc_chunks = []
                            st.session_state.current_doc_indexed = False
                            st.session_state.doc_info = None
                            nav("analyze")
                    if analysis:
                        st.download_button("Export JSON", json.dumps(analysis,indent=2),
                            f"{c['filename']}_analysis.json", key=f"exp_{c['id']}", use_container_width=True)
                    if st.button("Delete", key=f"del_{c['id']}", use_container_width=True):
                        delete_contract(c["id"])
                        st.success("Deleted!")
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ── COMPARE ────────────────────────────────────────────────────────────────────
def page_compare():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">Contract Comparison</div>
  <div class="page-header-sub">Evaluate two agreements side by side with dimension scoring, term-level differences and an AI-assisted recommendation.</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="inner">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("### Contract A")
        file_a = st.file_uploader("Upload Contract A", type=["pdf","txt"], key="cmp_a")
        name_a = st.text_input("Label for Contract A", "Contract A")
    with c2.container(border=True):
        st.markdown("### Contract B")
        file_b = st.file_uploader("Upload Contract B", type=["pdf","txt"], key="cmp_b")
        name_b = st.text_input("Label for Contract B", "Contract B")

    if st.button("Compare Contracts", type="primary"):
        if not file_a or not file_b:
            st.error("Please upload both contracts.")
        elif not st.session_state.api_key:
            st.error("The workspace OpenRouter key is not configured.")
        else:
            info_a = extract_text_from_pdf(file_a.read())
            info_b = extract_text_from_pdf(file_b.read())
            with st.spinner("Comparing contracts with AI..."):
                agent = get_agent()
                result = agent.compare_contracts(
                    info_a["text"], info_b["text"], name_a, name_b
                )
            st.session_state.compare_result = result

    if st.session_state.compare_result:
        res = st.session_state.compare_result
        winner = res.get("winner","")
        winner_reason = res.get("winner_reason","")
        st.markdown(f"""
<div class="card" style="text-align:center;margin-bottom:24px">
  <div style="font-size:28px;font-weight:800;color:#2563eb">{winner}</div>
  <div style="font-size:14px;color:#64748b;margin-top:8px">{winner_reason}</div>
</div>
""", unsafe_allow_html=True)

        dims = res.get("dimensions",[])
        if dims:
            rows = []
            for d in dims:
                rows.append({
                    "Category": d.get("category",""),
                    f"{name_a} Score": d.get("contract_a_score",""),
                    f"{name_b} Score": d.get("contract_b_score",""),
                    "Winner": d.get("winner",""),
                    f"{name_a} Notes": d.get("contract_a_notes",""),
                    f"{name_b} Notes": d.get("contract_b_notes",""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart
        if dims:
            fig = go.Figure(data=[
                go.Bar(name=name_a, x=[d["category"] for d in dims],
                       y=[d.get("contract_a_score",0) for d in dims], marker_color="#3b82f6"),
                go.Bar(name=name_b, x=[d["category"] for d in dims],
                       y=[d.get("contract_b_score",0) for d in dims], marker_color="#8b5cf6"),
            ])
            fig.update_layout(barmode="group", height=320, title="Score Comparison by Dimension",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#64748b")
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{name_a} Strengths**")
            for s in res.get("contract_a_strengths",[]):
                st.markdown(f"- {s}")
        with c2:
            st.markdown(f"**{name_b} Strengths**")
            for s in res.get("contract_b_strengths",[]):
                st.markdown(f"- {s}")
        if res.get("recommendation"):
            st.markdown("**Recommendation**")
            st.markdown(res["recommendation"])

        st.download_button("Export Comparison JSON", json.dumps(res,indent=2),
                          "comparison.json", "application/json")

    st.markdown('</div>', unsafe_allow_html=True)


# ── ABOUT ──────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown('<div class="inner">', unsafe_allow_html=True)
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">About Navneet ContractAI</div>
  <div class="page-header-sub">Enterprise contract intelligence, engineered at Navneet Education Limited</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card" style="text-align:center;padding:40px 32px;border-radius:20px">
  <div style="font-size:12px;font-weight:700;color:#2563eb;text-transform:uppercase;letter-spacing:.12em;margin-bottom:12px">Our Mission</div>
  <div style="font-size:24px;font-weight:800;max-width:780px;margin:0 auto 16px;line-height:1.35;letter-spacing:-0.5px">
    Make every contract readable, reviewable, and defensible — in minutes, not days.
  </div>
  <div style="max-width:700px;margin:0 auto;font-size:14.5px;line-height:1.75;color:#64748b">
    Navneet ContractAI unifies large language models, stateful orchestration (LangGraph),
    retrieval-augmented generation and classical ML to analyze, draft, compare and manage
    contracts for legal and business teams — securely and at scale.
  </div>
  <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:22px">
    <span class="badge b-low">Artificial Intelligence</span>
    <span class="badge b-low">Legal Tech</span>
    <span class="badge b-low">NLP &amp; LLM</span>
    <span class="badge b-low">LangGraph</span>
    <span class="badge b-low">RAG</span>
    <span class="badge b-low">GDPR Ready</span>
  </div>
</div>
<div style="height:26px"></div>
<div class="metric-grid">
  <div class="metric-card"><div class="metric-val">8</div><div class="metric-label">Agent Nodes</div></div>
  <div class="metric-card"><div class="metric-val">12+</div><div class="metric-label">Contract Types</div></div>
  <div class="metric-card"><div class="metric-val">100+</div><div class="metric-label">Clause Types</div></div>
  <div class="metric-card"><div class="metric-val">2</div><div class="metric-label">Analysis Models</div></div>
</div>
<div style="height:30px"></div>
""", unsafe_allow_html=True)

    st.markdown("### Technology Stack")
    tech_items = [
        ("LLM Gateway", "OpenRouter API — GPT-OSS 20B and NVIDIA Nemotron with reasoning enabled"),
        ("Orchestration", "LangChain v0.3 + LangGraph v0.2 stateful workflow"),
        ("RAG Engine", "ChromaDB + Sentence-Transformers + Cross-Encoder Reranking"),
        ("Deep Learning", "PyTorch — BERT, 1D-CNN, Bi-LSTM, Transformer Encoder"),
        ("Classical ML", "Scikit-learn — Isolation Forest, Gradient Boosting, KMeans"),
        ("Database", "SQLite (local) + PostgreSQL schema + Redis cache + ChromaDB"),
        ("PDF Parsing", "pdfplumber + pypdf fallback + OCR ready"),
        ("UI Framework", "Streamlit + Plotly + custom CSS design system"),
        ("Backend API", "FastAPI + Pydantic + Docker + Kubernetes"),
        ("Export", "ReportLab PDF + python-docx + JSON + CSV"),
        ("Security", "JWT auth + bcrypt + rate limiting + AES encryption"),
        ("Compliance", "GDPR + CCPA + HIPAA + SOC2 auto-detection"),
    ]
    tech_html = '<div class="tech-grid">'
    for label, val in tech_items:
        tech_html += f'<div class="tech-pill"><strong>{label}</strong><br><span style="font-size:11px;color:#64748b">{val}</span></div>'
    tech_html += '</div>'
    st.markdown(tech_html, unsafe_allow_html=True)

    logo = f'<img src="{LOGO_URI}" style="height:46px;margin-bottom:14px"/>' if LOGO_URI else ''
    st.markdown(f"""<div style="height:30px"></div>
<div class="card" style="text-align:center;padding:30px;border-radius:18px">
  {logo}
  <div style="font-family:'Sora',sans-serif;font-weight:700;font-size:18px;margin-bottom:6px">Navneet ContractAI</div>
  <div style="font-size:13px;color:#64748b;line-height:1.7">
    Enterprise Contract Intelligence Platform · Navneet Education Limited · 2026<br>
    Engine: LexForge core — LangGraph multi-agent + RAG + OpenRouter<br>
    <span style="color:#94a3b8">AI analysis is assistive and does not constitute legal advice.</span>
  </div>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── CONTACT ────────────────────────────────────────────────────────────────────
def page_contact():
    ic_mail = ('<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/>'
               '<polyline points="3 7 12 13 21 7"/></svg>')
    ic_phone = ('<svg viewBox="0 0 24 24"><path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 '
                '19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1.9.3 1.8.6 2.6a2 2 0 0 1-.5 2.1L8.1 9.6a16 16 0 0 0 6 6l1.2-1.2a2 2 0 0 1 2.1-.5c.8.3 1.7.5 2.6.6a2 2 0 0 1 1.7 2Z"/></svg>')
    ic_loc = ('<svg viewBox="0 0 24 24"><path d="M12 21s-7-5.5-7-11a7 7 0 0 1 14 0c0 5.5-7 11-7 11Z"/>'
              '<circle cx="12" cy="10" r="2.5"/></svg>')
    st.markdown(f"""
<div class="page-header">
  <div class="page-header-title">Contact &amp; Support</div>
  <div class="page-header-sub">Questions, partnerships or legal-tech consultations — we usually reply within 24 hours</div>
</div>
<div class="inner">
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px">
  <div class="contact-card"><div class="contact-icon">{ic_mail}</div>
    <div style="font-weight:700;margin-bottom:6px">Email</div>
    <div style="font-size:13px;color:#64748b">support@navneet.com<br>Legal-tech enquiries welcome</div></div>
  <div class="contact-card"><div class="contact-icon">{ic_phone}</div>
    <div style="font-weight:700;margin-bottom:6px">Phone</div>
    <div style="font-size:13px;color:#64748b">+91 22 6662 6565<br>Mon–Fri · 10:00–18:00 IST</div></div>
  <div class="contact-card"><div class="contact-icon">{ic_loc}</div>
    <div style="font-weight:700;margin-bottom:6px">Office</div>
    <div style="font-size:13px;color:#64748b">Navneet Education Limited<br>Mumbai, India · 2026</div></div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="inner"><div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown("### Send a message")
    with st.form("contact_form"):
        fc = st.columns(2)
        name  = fc[0].text_input("Name")
        email = fc[1].text_input("Email")
        subject = st.selectbox("Subject", ["General Enquiry", "Bug Report", "Feature Request",
                                           "Partnership", "Legal Tech Consultation"])
        message = st.text_area("Message", height=130)
        if st.form_submit_button("Send Message", type="primary"):
            if name.strip() and message.strip():
                st.success("Message received! Our team will respond within 24 hours.")
            else:
                st.warning("Please add your name and a message.")
    st.markdown('</div>', unsafe_allow_html=True)


# ── SETTINGS ───────────────────────────────────────────────────────────────────
# Updated public About page. This later definition intentionally overrides the
# earlier basic version without touching unrelated legacy markup.
def page_about():
    ic_doc = '<svg viewBox="0 0 24 24"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9Z"/><path d="M14 3v6h6"/></svg>'
    ic_graph = '<svg viewBox="0 0 24 24"><circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/><path d="M7 6h10"/><path d="m6 8 5 8"/><path d="m18 8-5 8"/></svg>'
    ic_search = '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>'
    ic_lock = '<svg viewBox="0 0 24 24"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>'

    system_rows = [
        (ic_graph, "Stateful orchestration", "LangGraph pipeline for extraction, model review, grounding and validation.", "4 stages"),
        (ic_search, "Grounded retrieval", "ChromaDB search with reranking for evidence-aware answers.", "RAG"),
        (ic_doc, "Drafting workspace", "Structured templates, clause library and export-ready contract text.", "12+ types"),
        (ic_lock, "Governance layer", "Local SQLite history with auth, audit-minded data flow and review warnings.", "Secure"),
    ]
    system_html = ""
    for icon, name, desc, score in system_rows:
        system_html += f"""
  <div class="system-row">
    <div class="system-icon">{icon}</div>
    <div><div class="system-name">{name}</div><div class="system-desc">{desc}</div></div>
    <div class="system-score">{score}</div>
  </div>"""

    st.markdown(f"""
<div class="about-shell">
  <div class="about-hero">
    <div>
      <div class="eyebrow"><span class="eyebrow-dot"></span>Enterprise legal AI platform</div>
      <div class="about-title">Contract intelligence for faster, clearer legal review.</div>
      <div class="about-copy">
        Navneet ContractAI brings analysis, drafting, comparison, compliance checks and contract chat into a single
        review workspace. It combines large language models, LangGraph agents, retrieval, classical ML and practical
        legal workflows so teams can move from document upload to decision with more confidence.
      </div>
      <div class="about-actions">
        <span class="about-chip"><span class="chip-swatch"></span>AI-assisted review</span>
        <span class="about-chip"><span class="chip-swatch" style="background:#10b981"></span>RAG grounded answers</span>
        <span class="about-chip"><span class="chip-swatch" style="background:#7c3aed"></span>Compliance ready</span>
        <span class="about-chip"><span class="chip-swatch" style="background:#f59e0b"></span>Human-in-the-loop</span>
      </div>
    </div>
    <div class="about-system-panel">
      <div class="system-head">
        <div class="system-title">Platform operating model</div>
        <div class="system-live">Online</div>
      </div>
      {system_html}
    </div>
  </div>

  <div class="about-section-head">
    <div>
      <div class="about-section-title">Built for contract operations</div>
      <div class="about-section-sub">High-signal review tools, reusable drafting assets and structured history for repeated legal workflows.</div>
    </div>
  </div>

  <div class="about-kpi-grid">
    <div class="about-kpi"><div class="about-kpi-value">8</div><div class="about-kpi-label">Agent nodes for end-to-end analysis</div></div>
    <div class="about-kpi"><div class="about-kpi-value">12+</div><div class="about-kpi-label">Contract templates for drafting</div></div>
    <div class="about-kpi"><div class="about-kpi-value">100+</div><div class="about-kpi-label">Clause and risk signals tracked</div></div>
    <div class="about-kpi"><div class="about-kpi-value">2</div><div class="about-kpi-label">Approved analysis models through OpenRouter</div></div>
  </div>

  <div class="about-section-head">
    <div>
      <div class="about-section-title">What the platform does</div>
      <div class="about-section-sub">Each module is designed around legal review tasks, not generic chat output.</div>
    </div>
  </div>

  <div class="about-feature-grid">
    <div class="about-feature">
      <div class="about-feature-top"><div class="about-feature-index">01</div><div class="about-feature-title">Analyze and explain</div></div>
      <div class="about-feature-body">Extracts parties, obligations, deadlines, risky language, anomalies and executive summaries from uploaded contracts.</div>
    </div>
    <div class="about-feature">
      <div class="about-feature-top"><div class="about-feature-index">02</div><div class="about-feature-title">Draft with structure</div></div>
      <div class="about-feature-body">Generates contract drafts from guided templates with editable sections, clause options and export-friendly formatting.</div>
    </div>
    <div class="about-feature">
      <div class="about-feature-top"><div class="about-feature-index">03</div><div class="about-feature-title">Compare alternatives</div></div>
      <div class="about-feature-body">Reviews two contracts side by side and highlights stronger terms, gaps, unfavorable positions and negotiation focus areas.</div>
    </div>
    <div class="about-feature">
      <div class="about-feature-top"><div class="about-feature-index">04</div><div class="about-feature-title">Ask grounded questions</div></div>
      <div class="about-feature-body">Uses retrieval over contract chunks so answers stay connected to source material instead of floating as generic advice.</div>
    </div>
    <div class="about-feature">
      <div class="about-feature-top"><div class="about-feature-index">05</div><div class="about-feature-title">Monitor compliance</div></div>
      <div class="about-feature-body">Checks for GDPR, CCPA, HIPAA and SOC2 related indicators, with missing-clause recommendations and risk scoring.</div>
    </div>
    <div class="about-feature">
      <div class="about-feature-top"><div class="about-feature-index">06</div><div class="about-feature-title">Retain work history</div></div>
      <div class="about-feature-body">Keeps local analysis and generated-contract history searchable, editable and ready for export or follow-up review.</div>
    </div>
  </div>
""", unsafe_allow_html=True)

    tech_items = [
        ("LLM Gateway", "OpenRouter API - GPT-OSS 20B and NVIDIA Nemotron with reasoning enabled"),
        ("Orchestration", "LangChain v0.3 + LangGraph v0.2 stateful workflow"),
        ("RAG Engine", "ChromaDB or Pinecone + OpenRouter/local embeddings + scored evidence retrieval"),
        ("Orchestration", "LangGraph specialist nodes with structured state and evidence context"),
        ("Vector Layer", "Persistent local collections or managed namespaced Pinecone indexes"),
        ("Database", "SQLite operational store + ChromaDB vectors + optional Pinecone connector"),
        ("PDF Parsing", "pdfplumber + pypdf fallback + OCR ready"),
        ("UI Framework", "Streamlit + Plotly + custom CSS design system"),
        ("Application", "Streamlit workspace + typed Python service and persistence layers"),
        ("Export", "ReportLab PDF + python-docx + JSON + CSV"),
        ("Security", "Session authentication + masked secrets + scoped connector configuration"),
        ("Compliance", "GDPR + CCPA + HIPAA + SOC2 auto-detection"),
    ]
    tech_html = """
  <div class="about-section-head">
    <div>
      <div class="about-section-title">Technology stack</div>
      <div class="about-section-sub">A practical AI stack for document parsing, retrieval, generation, scoring, storage and export.</div>
    </div>
  </div>
  <div class="tech-grid">"""
    for label, val in tech_items:
        tech_html += f'<div class="tech-pill"><strong>{label}</strong><span>{val}</span></div>'
    tech_html += '</div>'
    st.markdown(tech_html, unsafe_allow_html=True)

    logo = f'<img src="{LOGO_URI}" alt="Navneet"/>' if LOGO_URI else ''
    st.markdown(f"""
  <div class="about-footer-note">
    {logo}
    <div>
      <div class="about-footer-title">Navneet ContractAI</div>
      <div class="about-footer-copy">
        Enterprise Contract Intelligence Platform - Navneet Education Limited - 2026<br>
        Engine: LexForge core with a LangGraph hybrid workflow, RAG and OpenRouter model access.<br>
        AI analysis is assistive and does not constitute legal advice.
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


def page_contact():
    ic_mail = '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></svg>'
    ic_phone = '<svg viewBox="0 0 24 24"><path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1.9.3 1.8.6 2.6a2 2 0 0 1-.5 2.1L8.1 9.6a16 16 0 0 0 6 6l1.2-1.2a2 2 0 0 1 2.1-.5c.8.3 1.7.5 2.6.6a2 2 0 0 1 1.7 2Z"/></svg>'
    ic_loc = '<svg viewBox="0 0 24 24"><path d="M12 21s-7-5.5-7-11a7 7 0 0 1 14 0c0 5.5-7 11-7 11Z"/><circle cx="12" cy="10" r="2.5"/></svg>'

    st.markdown(f"""
<div class="contact-shell">
  <div class="contact-hero">
    <div class="eyebrow"><span class="eyebrow-dot"></span>Support and collaboration</div>
    <div class="contact-title">Get the right legal-tech help without the runaround.</div>
    <div class="contact-copy">
      Reach the Navneet ContractAI team for platform support, bug reports, feature requests,
      partnership conversations and contract-intelligence consultation.
    </div>
    <div class="contact-method-grid">
      <div class="contact-card">
        <div class="contact-method-icon">{ic_mail}</div>
        <div class="contact-card-title">Email support</div>
        <div class="contact-card-copy">support@navneet.com<br>Best for product, account and workflow questions.</div>
      </div>
      <div class="contact-card">
        <div class="contact-method-icon">{ic_phone}</div>
        <div class="contact-card-title">Phone desk</div>
        <div class="contact-card-copy">+91 22 6662 6565<br>Monday to Friday, 10:00 to 18:00 IST.</div>
      </div>
      <div class="contact-card">
        <div class="contact-method-icon">{ic_loc}</div>
        <div class="contact-card-title">Office</div>
        <div class="contact-card-copy">Navneet Education Limited<br>Mumbai, India.</div>
      </div>
    </div>
    <div class="contact-stats">
      <div class="contact-stat"><div class="contact-stat-value">24h</div><div class="contact-stat-label">Typical first response</div></div>
      <div class="contact-stat"><div class="contact-stat-value">5</div><div class="contact-stat-label">Inquiry categories routed</div></div>
      <div class="contact-stat"><div class="contact-stat-value">IST</div><div class="contact-stat-label">Business-hour support window</div></div>
    </div>
  </div>

  <div class="contact-section-head">
    <div>
      <div class="contact-section-title">Send a message</div>
      <div class="contact-section-sub">Share the context once. The form routes your note by topic and keeps the response path clear.</div>
    </div>
  </div>

  <div class="contact-layout">
    <div class="contact-side">
      <div class="contact-side-title">How requests are handled</div>
      <div class="contact-side-copy">Use a focused subject and include contract type, model behavior or workflow details where relevant.</div>
      <div class="contact-timeline">
        <div class="contact-step"><div class="contact-step-num">1</div><div><div class="contact-step-title">Triage</div><div class="contact-step-copy">Support identifies whether the request is product, technical, partnership or legal-tech consulting.</div></div></div>
        <div class="contact-step"><div class="contact-step-num">2</div><div><div class="contact-step-title">Review</div><div class="contact-step-copy">The right owner checks reproduction details, business context and any relevant contract workflow.</div></div></div>
        <div class="contact-step"><div class="contact-step-num">3</div><div><div class="contact-step-title">Response</div><div class="contact-step-copy">You receive a practical next step, clarification request or meeting path within the normal support window.</div></div></div>
      </div>
    </div>
    <div class="contact-form-shell">
      <div class="contact-form-title">Message details</div>
      <div class="contact-form-sub">Fields marked by context are enough here; no sensitive contract text is required for the first contact.</div>
""", unsafe_allow_html=True)

    with st.form("contact_form"):
        fc = st.columns(2)
        name = fc[0].text_input("Name")
        email = fc[1].text_input("Email")
        subject = st.selectbox("Subject", [
            "General Enquiry",
            "Bug Report",
            "Feature Request",
            "Partnership",
            "Legal Tech Consultation",
        ])
        message = st.text_area("Message", height=150, placeholder="Tell us what you need help with...")
        submitted = st.form_submit_button("Send Message", type="primary", use_container_width=True)
        if submitted:
            if name.strip() and email.strip() and message.strip():
                st.success("Message received. Our team will respond within 24 hours.")
            elif name.strip() and message.strip():
                st.warning("Please add your email so the team can respond.")
            else:
                st.warning("Please add your name, email and message.")

    st.markdown("""
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


def page_settings():
    st.markdown('<div class="inner">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)

    tabs = st.tabs(["AI Connection", "Theme", "Data Management"])
    with tabs[0]:
        st.info(
            f"OpenRouter key: {_mask_secret(st.session_state.api_key)}\n\n"
            f"Default generation model: {OPENROUTER_MODEL_LABEL} (`{OPENROUTER_MODEL}`)\n\n"
            f"Analysis models: {', '.join(openrouter_model_label(model) for model in OPENROUTER_MODELS)}\n\n"
            f"Retrieval engine: `{RETRIEVAL_ENGINE_LABEL}` (no second AI model)\n\n"
            "The API key is fixed; analysis models are selected from the approved list in the Analyzer."
        )

    with tabs[1]:
        dark = st.toggle("Dark Mode", st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

    with tabs[2]:
        stats = get_dashboard_stats()
        st.metric("Total Contracts in DB", stats["total_contracts"])
        st.metric("Total Generated Contracts", stats["generated_contracts"])
        if st.button("Clear All Chat History", type="secondary"):
            st.warning("This will delete all chat history. (Implement full clear in production)")
        if st.button("Export All Data as JSON"):
            all_c = get_all_contracts()
            st.download_button("Download All Data", json.dumps(all_c,indent=2,default=str),
                              "lexforge_all_data.json","application/json")

    st.markdown('</div>', unsafe_allow_html=True)


# ── DASHBOARD ──────────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown(f"""
<div class="page-header">
  <div class="page-header-title">Welcome back, {st.session_state.username or 'User'}</div>
  <div class="page-header-sub">Your Navneet ContractAI command center — analyze, draft, and manage contracts.</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="inner">', unsafe_allow_html=True)
    stats = get_dashboard_stats()
    m = st.columns(4)
    m[0].metric("Contracts Analyzed", stats.get("total_contracts", 0))
    m[1].metric("High-Risk Contracts", stats.get("high_risk_count", 0))
    m[2].metric("Avg Risk Score", f"{stats.get('avg_risk_score', 0)}/10")
    m[3].metric("Drafts Generated", stats.get("generated_contracts", 0))

    st.markdown("### Quick actions")
    q = st.columns(4)
    if q[0].button("Analyze a Contract", use_container_width=True, type="primary"):
        nav("analyze")
    if q[1].button("Draft a Contract", use_container_width=True):
        nav("writer")
    if q[2].button("Ask the Chatbot", use_container_width=True):
        nav("chatbot")
    if q[3].button("Run the AI Agent", use_container_width=True):
        nav("agent")

    if not st.session_state.api_key:
        st.warning("The workspace OpenRouter key is missing from `.streamlit/secrets.toml`.")

    st.markdown("### Recent contracts")
    recent = get_all_contracts(limit=6)
    if not recent:
        st.info("No contracts analyzed yet. Head to **AI Analyzer** to upload your first PDF.")
    else:
        for c in recent:
            cc = st.columns([3, 1, 1, 1])
            cc[0].markdown(f"**{c['filename']}**  \n<span style='color:#64748b;font-size:12px'>{c.get('contract_type','—')} · {c.get('uploaded_at','')[:10]}</span>", unsafe_allow_html=True)
            cc[1].metric("Risk", f"{c.get('risk_score',0)}/10")
            cc[2].markdown(f"<div style='padding-top:8px'>{c.get('assessment','—')}</div>", unsafe_allow_html=True)
            if cc[3].button("Open", key=f"dash_open_{c['id']}", use_container_width=True):
                nav("history")
    st.markdown('</div>', unsafe_allow_html=True)


# ── AI CHATBOT (general legal assistant) ───────────────────────────────────────
def page_chatbot():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">Legal Copilot</div>
  <div class="page-header-sub">Work through contract clauses, negotiation positions and compliance questions in a focused legal intelligence conversation.</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="inner">', unsafe_allow_html=True)

    if not st.session_state.api_key:
        st.warning("The workspace OpenRouter key is missing from `.streamlit/secrets.toml`.")

    for msg in st.session_state.chatbot_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a legal/contract question…")
    if prompt:
        st.session_state.chatbot_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            reasoning_details = None
            if not st.session_state.api_key:
                reply = "The workspace OpenRouter key is not configured."
            else:
                with st.spinner("Thinking…"):
                    agent = get_agent()
                    reply = agent.legal_chatbot(prompt, st.session_state.chatbot_history)
                    reasoning_details = agent.last_reasoning_details
            st.markdown(reply)
        assistant_message = {"role": "assistant", "content": reply}
        if reasoning_details is not None:
            assistant_message["reasoning_details"] = reasoning_details
        st.session_state.chatbot_history.append(assistant_message)

    if st.session_state.chatbot_history:
        if st.button("Clear conversation"):
            st.session_state.chatbot_history = []
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── AI AGENT (autonomous plan-and-execute) ─────────────────────────────────────
def page_agent():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">Agent Studio</div>
  <div class="page-header-sub">Define an outcome and execute a structured plan-and-reason workflow with optional contract context, caveats and next actions.</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="inner">', unsafe_allow_html=True)

    examples = [
        "Draft a mutual NDA between Navneet Education Ltd and a software vendor, 3-year term, India law.",
        "Build a pre-signing risk checklist for a SaaS subscription agreement.",
        "Generate a strong limitation-of-liability clause capped at 12 months of fees.",
    ]
    task = st.text_area("Agent task / goal", height=110,
                        placeholder="Describe what you want the agent to accomplish…")
    st.caption("Examples: " + "  •  ".join(examples))
    use_ctx = st.checkbox("Use the currently loaded contract as context", value=False)
    context = ""
    if use_ctx and st.session_state.get("doc_info"):
        context = st.session_state.doc_info.get("text", "") if isinstance(st.session_state.doc_info, dict) else ""

    if st.button("Run Agent", type="primary"):
        if not st.session_state.api_key:
            st.error("The workspace OpenRouter key is missing from `.streamlit/secrets.toml`.")
        elif not task.strip():
            st.warning("Describe a task for the agent first.")
        else:
            with st.spinner("Agent planning and executing…"):
                st.session_state.agent_result = get_agent().run_agent_task(task, context)

    res = st.session_state.agent_result
    if res:
        if res.get("parse_error"):
            st.markdown(res.get("raw", "No result."))
        else:
            if res.get("plan"):
                st.markdown("#### Execution plan")
                for i, step in enumerate(res["plan"], 1):
                    st.markdown(f"**{i}.** {step}")
            if res.get("reasoning"):
                with st.expander("Reasoning"):
                    st.markdown(res["reasoning"])
            if res.get("result"):
                st.markdown("#### Result")
                st.markdown(res["result"])
                st.download_button("Download result", res["result"], "agent_result.md", "text/markdown")
            if res.get("risks_or_caveats"):
                st.warning("**Caveats:** " + " · ".join(res["risks_or_caveats"]))
            if res.get("next_steps"):
                st.info("**Next steps:** " + " · ".join(res["next_steps"]))
    st.markdown('</div>', unsafe_allow_html=True)


# ── AUTH MODAL ─────────────────────────────────────────────────────────────────
def _do_login(user: Dict):
    st.session_state.auth_user = user["username"]
    st.session_state.username  = user.get("full_name") or user["username"]
    st.session_state.auth_role = user.get("role", "user")
    st.session_state.show_auth = False
    st.session_state.page      = "dashboard"
    st.rerun()


@st.dialog("Welcome to Navneet ContractAI")
def _auth_dialog():
    logo = f'<img src="{LOGO_URI}" style="height:46px"/>' if LOGO_URI else ''
    st.markdown(
        f"<div style='text-align:center;padding:4px 0 14px'>{logo}"
        "<div style='font-family:Sora,sans-serif;font-weight:700;font-size:17px;margin-top:8px'>"
        "Sign in to your workspace</div>"
        "<div style='font-size:13px;color:#64748b'>Analyze, draft and manage contracts with AI</div></div>",
        unsafe_allow_html=True)
    tab_in, tab_up = st.tabs(["Sign In", "Create Account"])
    with tab_in:
        u  = st.text_input("Username", key="signin_user", placeholder="snehal1")
        pw = st.text_input("Password", type="password", key="signin_pwd",
                           placeholder="Enter your password")
        if st.button("Sign In", type="primary", use_container_width=True, key="do_signin"):
            user = verify_user(u, pw)
            if user:
                _do_login(user)
            else:
                st.error("Invalid username or password.")
        st.markdown(
            "<div style='background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.2);"
            "border-radius:10px;padding:10px 12px;font-size:12.5px;color:#475569;margin-top:6px'>"
            "<b>Demo admin</b> — <code>snehal1</code> / <code>snehal123</code><br>"
            "<b>Demo team</b> — <code>priya.shah</code>, <code>arjun.mehta</code>, "
            "<code>neha.kulkarni</code> or <code>rohan.patel</code> / <code>demo123</code></div>",
            unsafe_allow_html=True)
    with tab_up:
        fn  = st.text_input("Full Name", key="reg_name")
        cc  = st.columns(2)
        u2  = cc[0].text_input("Username", key="reg_user")
        em  = cc[1].text_input("Email", key="reg_email")
        pw2 = st.text_input("Password", type="password", key="reg_pwd",
                            placeholder="Choose a strong password")
        if st.button("Create Account", type="primary", use_container_width=True, key="do_reg"):
            res = create_user(u2, pw2, fn, em)
            if res.get("ok"):
                _do_login(res["user"])
            else:
                st.error(res.get("error", "Could not create account."))


def render_auth_modal():
    if st.session_state.show_auth and not st.session_state.auth_user:
        _auth_dialog()


# ── SIDEBAR (logged-in app shell) ──────────────────────────────────────────────
SIDEBAR_ITEMS = [
    ("dashboard", "Dashboard"),
    ("analyze",   "AI Analyzer"),
    ("writer",    "AI Contract Writer"),
    ("chatbot",   "AI Chatbot"),
    ("agent",     "AI Agent"),
    ("chat",      "RAG Chat"),
    ("compare",   "Compare"),
    ("history",   "History"),
    ("features",  "Capabilities"),
]


def render_sidebar():
    p = st.session_state.page
    with st.sidebar:
        logo = f'<img src="{LOGO_URI}"/>' if LOGO_URI else ''
        st.markdown(
            f'<div class="sb-brand">{logo}<div class="sb-title">Navneet<br>ContractAI</div></div>',
            unsafe_allow_html=True)
        name = st.session_state.username or "User"
        role = (st.session_state.auth_role or "user").upper()
        initial = name.strip()[0].upper() if name.strip() else "U"
        st.markdown(
            f'<div class="sb-user"><div class="sb-avatar">{initial}</div>'
            f'<div class="sb-name">{name}</div><div class="sb-role">{role}</div></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Workspace</div>', unsafe_allow_html=True)
        for key, label in SIDEBAR_ITEMS:
            if st.button(label, key=f"sb_{key}", use_container_width=True,
                         type="primary" if p == key else "secondary"):
                nav(key)

        st.markdown('<div class="sb-section">Account</div>', unsafe_allow_html=True)
        if st.button("Settings", key="sb_settings", use_container_width=True,
                     type="primary" if p == "settings" else "secondary"):
            nav("settings")
        if st.button("Logout", key="sb_logout", use_container_width=True):
            st.session_state.auth_user = None
            st.session_state.username  = None
            st.session_state.auth_role = None
            st.session_state.show_auth = False
            st.session_state.page = "home"
            st.rerun()
        st.markdown(
            "<div style='font-size:11px;color:#94a3b8;padding:16px 8px;line-height:1.6'>"
            "Navneet Education Limited · 2026<br>AI is not legal advice.</div>",
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
SIDEBAR_GROUPS = [
    ("Overview", [
        ("dashboard", "Dashboard"),
    ]),
    ("Intelligence", [
        ("analyze", "Analyze Contracts"),
        ("writer", "Contract Studio"),
        ("chatbot", "Legal Copilot"),
        ("agent", "Agent Studio"),
        ("chat", "Grounded Contract Chat"),
        ("knowledge", "Knowledge Base"),
    ]),
    ("Operations", [
        ("compare", "Compare Contracts"),
        ("history", "Contract Repository"),
    ]),
    ("Platform", [
        ("integrations", "Integrations and Vectors"),
    ]),
    ("Support", [
        ("about", "About Platform"),
        ("contact", "Support Center"),
    ]),
]


def render_sidebar():
    p = st.session_state.page
    with st.sidebar:
        logo = f'<img src="{LOGO_URI}" alt="Navneet"/>' if LOGO_URI else ''
        st.markdown(
            f'<div class="sb-brand">{logo}<div class="sb-title">Navneet<br><span>ContractAI</span></div></div>',
            unsafe_allow_html=True,
        )

        name = st.session_state.username or "User"
        role = (st.session_state.auth_role or "user").upper()
        initial = name.strip()[0].upper() if name.strip() else "U"
        st.markdown(
            f'<div class="sb-user"><div class="sb-avatar">{initial}</div>'
            f'<div class="sb-name">{name}</div><div class="sb-role">{role}</div></div>',
            unsafe_allow_html=True,
        )

        api_label = "Connected" if st.session_state.api_key else "Missing"
        api_model = OPENROUTER_MODEL_LABEL
        vector_label = "Pinecone" if st.session_state.vector_backend == "pinecone" else "ChromaDB"
        mcp_count = len(st.session_state.mcp_servers)
        st.markdown(
            f"""
<div class="sb-api">
  <div class="sb-api-top">
    <div class="sb-api-title">OpenRouter</div>
    <div class="sb-api-pill">{api_label}</div>
  </div>
  <div class="sb-api-model">Embeddings: {api_model}<br>Vectors: {vector_label}<br>MCP servers: {mcp_count}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        for group, items in SIDEBAR_GROUPS:
            st.markdown(f'<div class="sb-section">{group}</div>', unsafe_allow_html=True)
            for key, label in items:
                if st.button(label, key=f"sb_{key}", use_container_width=True,
                             type="primary" if p == key else "secondary"):
                    nav(key)

        st.markdown('<div class="sb-section">Account</div>', unsafe_allow_html=True)
        if st.button("Light mode" if st.session_state.dark_mode else "Dark mode",
                     key="sb_theme", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        if st.button("Settings", key="sb_settings", use_container_width=True,
                     type="primary" if p == "settings" else "secondary"):
            nav("settings")
        if st.button("Logout", key="sb_logout", use_container_width=True):
            st.session_state.auth_user = None
            st.session_state.username = None
            st.session_state.auth_role = None
            st.session_state.show_auth = False
            st.session_state.page = "home"
            st.rerun()

        st.markdown(
            "<div class='sb-footer'>Navneet Education Limited - 2026<br>"
            "AI analysis is assistive and not legal advice.</div>",
            unsafe_allow_html=True,
        )


def page_settings():
    st.markdown("""
<div class="page-header">
  <div class="page-header-title">Settings</div>
  <div class="page-header-sub">Review the fixed AI connection, appearance preferences and workspace data operations.</div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="inner">', unsafe_allow_html=True)

    tabs = st.tabs(["AI Connection", "Appearance", "Data"])
    with tabs[0]:
        st.markdown(
            f"""
<div class="contact-form-shell">
  <div class="contact-form-title">OpenRouter connection</div>
  <div class="contact-form-sub">
    Status: <strong>{'Connected' if st.session_state.api_key else 'Not configured'}</strong><br>
    Key: <strong>{_mask_secret(st.session_state.api_key)}</strong><br>
    Default generation model: <strong>{OPENROUTER_MODEL_LABEL} ({OPENROUTER_MODEL})</strong><br>
    Analysis models: <strong>{', '.join(openrouter_model_label(model) for model in OPENROUTER_MODELS)}</strong><br>
    Retrieval engine: <strong>{RETRIEVAL_ENGINE_LABEL} (no second AI model)</strong><br>
    Configuration: <strong>Fixed in workspace secrets and code</strong>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with tabs[1]:
        dark = st.toggle("Dark mode", st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()

    with tabs[2]:
        stats = get_dashboard_stats()
        c1, c2 = st.columns(2)
        c1.metric("Contracts in database", stats["total_contracts"])
        c2.metric("Generated drafts", stats["generated_contracts"])
        if st.button("Export all data"):
            all_c = get_all_contracts()
            st.download_button(
                "Download JSON",
                json.dumps(all_c, indent=2, default=str),
                "lexforge_all_data.json",
                "application/json",
            )

    st.markdown('</div>', unsafe_allow_html=True)


def _runtime_flags() -> Dict[str, bool]:
    return {
        "LangGraph": importlib.util.find_spec("langgraph") is not None,
        "ChromaDB": importlib.util.find_spec("chromadb") is not None,
        "OpenRouter": bool(st.session_state.api_key),
        "Pinecone": bool(st.session_state.pinecone_api_key and st.session_state.pinecone_host),
        "MCP": bool(st.session_state.mcp_servers),
    }


def page_dashboard():
    stats = get_dashboard_stats()
    recent = get_all_contracts(limit=6)
    activity = get_recent_activity(limit=6)
    name = html.escape(st.session_state.username or "User")
    provider = "Pinecone" if st.session_state.vector_backend == "pinecone" else "ChromaDB"
    model_name = html.escape(OPENROUTER_MODEL_LABEL)
    api_state = "Gateway online" if st.session_state.api_key else "Gateway setup required"
    api_chip_class = "" if st.session_state.api_key else "warn"
    st.markdown(f"""
<div class="command-shell">
  <div class="command-hero">
    <div>
      <div class="command-kicker">Contract intelligence command center</div>
      <div class="command-title">Welcome back, {name}</div>
      <div class="command-copy">Operate analysis, drafting, retrieval and agent workflows from one evidence-aware workspace.</div>
    </div>
    <div class="command-runtime">
      <span class="runtime-chip {api_chip_class}">{api_state}</span><span class="runtime-chip">{provider} vectors</span>
      <span class="runtime-chip">{stats.get('total_users', 0)} workspace members</span><span class="runtime-chip">{stats.get('chat_messages', 0)} grounded Q&amp;A messages</span>
      <span class="runtime-chip">LangGraph · 4 stages / 8 domains</span><span class="runtime-chip">{model_name}</span>
    </div>
  </div>
  <div class="pro-metric-grid">
    <div class="pro-metric"><div class="pro-metric-icon">{UI_ICONS['document']}</div><div class="pro-metric-value">{stats.get('total_contracts', 0)}</div><div class="pro-metric-label">Contracts analyzed</div></div>
    <div class="pro-metric"><div class="pro-metric-icon">{UI_ICONS['risk']}</div><div class="pro-metric-value">{stats.get('high_risk_count', 0)}</div><div class="pro-metric-label">High-risk contracts</div></div>
    <div class="pro-metric"><div class="pro-metric-icon">{UI_ICONS['shield']}</div><div class="pro-metric-value">{stats.get('avg_risk_score', 0)}/10</div><div class="pro-metric-label">Average risk score</div></div>
    <div class="pro-metric"><div class="pro-metric-icon">{UI_ICONS['draft']}</div><div class="pro-metric-value">{stats.get('generated_contracts', 0)}</div><div class="pro-metric-label">Drafts generated</div></div>
  </div>
  <div class="section-heading">Launch a workflow</div>
  <div class="action-labels">
    <div class="action-label"><div class="action-icon">{UI_ICONS['search']}</div><div class="action-label-title">Contract analysis</div><div class="action-label-copy">Run extraction, risk and compliance agents.</div></div>
    <div class="action-label"><div class="action-icon">{UI_ICONS['draft']}</div><div class="action-label-title">Contract drafting</div><div class="action-label-copy">Generate structured, editable legal drafts.</div></div>
    <div class="action-label"><div class="action-icon">{UI_ICONS['vector']}</div><div class="action-label-title">Knowledge workspace</div><div class="action-label-copy">Index and inspect retrieval evidence.</div></div>
    <div class="action-label"><div class="action-icon">{UI_ICONS['graph']}</div><div class="action-label-title">Agent execution</div><div class="action-label-copy">Run goal-driven contract workflows.</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    with st.container(key="dashboard_actions"):
        q = st.columns(4)
        if q[0].button("Analyze contract", key="dash_action_analyze", use_container_width=True, type="primary"):
            nav("analyze")
        if q[1].button("Create draft", key="dash_action_writer", use_container_width=True):
            nav("writer")
        if q[2].button("Open knowledge", key="dash_action_knowledge", use_container_width=True):
            nav("knowledge")
        if q[3].button("Run agent", key="dash_action_agent", use_container_width=True):
            nav("agent")

    activity_html = "".join(
        f'<div class="pipeline-row"><div class="pipeline-index">{html.escape(str(a.get("user_name", "U"))[:1].upper())}</div><div><div class="row-title">{html.escape(str(a.get("user_name", "User")))} {html.escape(str(a.get("action", "updated")))} {html.escape(str(a.get("entity_type", "record")))}</div><div class="row-sub">{html.escape(str(a.get("entity_title", "")))} · {html.escape(str(a.get("details", "")))}</div></div><div class="row-status">{html.escape(str(a.get("timestamp", ""))[5:16])}</div></div>'
        for a in activity
    ) or '<div class="panel-sub">Workspace activity will appear as the team analyzes and drafts contracts.</div>'
    recent_html = "".join(
        f'<div class="contract-row"><div class="pipeline-index">{i:02d}</div><div><div class="row-title">{html.escape(str(c.get("filename", "Contract")))}</div><div class="row-sub">{html.escape(str(c.get("contract_type", "Unknown")))} · {html.escape(str(c.get("owner_name") or "Workspace user"))} · {html.escape(str(c.get("uploaded_at", ""))[:10])}</div></div><div class="row-status">{"Analysis failed" if c.get("status") == "failed" else "Risk " + str(c.get("risk_score", 0)) + "/10"}</div></div>'
        for i, c in enumerate(recent, 1)
    ) or '<div class="panel-sub">No contracts yet. Start with Contract Analysis to create the first evidence-backed record.</div>'
    st.markdown(f"""
<div class="command-shell" style="padding-top:0">
  <div class="workspace-grid">
    <div class="workspace-panel"><div class="panel-head"><div><div class="panel-title">Recent team activity</div><div class="panel-sub">Analysis, drafting and review work across the demo workspace.</div></div></div>{activity_html}</div>
    <div class="workspace-panel"><div class="panel-head"><div><div class="panel-title">Recent contracts</div><div class="panel-sub">Latest analyzed records and risk posture.</div></div></div>{recent_html}</div>
  </div>
</div>""", unsafe_allow_html=True)
    if not st.session_state.api_key:
        st.warning("Configure `OPENROUTER_API_KEY` in `.streamlit/secrets.toml` to enable model and embedding workflows.")


def page_features():
    cards_html = "".join(
        f'<div class="feat-card"><div class="feat-icon">{svg}</div><div class="feat-title">{title}</div><div class="feat-desc">{copy}</div></div>'
        for svg, title, copy in FEATURE_CARDS
    )
    st.markdown(f"""
<div class="page-header"><div class="page-header-title">Platform Capabilities</div><div class="page-header-sub">Production-minded contract AI across analysis, generation, retrieval, orchestration and operations.</div></div>
<div class="inner"><div class="feat-grid">{cards_html}</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Engineering depth</div>', unsafe_allow_html=True)
    graph_tab, rag_tab, data_tab, ops_tab = st.tabs(["Analysis domains", "RAG and vectors", "Data layer", "Agent operations"])
    with graph_tab:
        nodes = [
            ("01", "Entity extraction", "Parties, dates, money, jurisdiction and contract type"),
            ("02", "Obligation mapping", "Party-specific duties, priority, conditions and references"),
            ("03", "Risk analysis", "Severity, likelihood, exposure and mitigation"),
            ("04", "Deadline engine", "Terms, notice periods, renewals and consequences"),
            ("05", "Clause intelligence", "Classification across more than twenty clause families"),
            ("06", "Compliance", "GDPR, CCPA, HIPAA and SOC2 control review"),
            ("07", "Anomaly review", "Unusual, conflicting or hidden provisions"),
            ("08", "Executive synthesis", "Structured decision summary and recommendation"),
        ]
        st.markdown("".join(f'<div class="pipeline-row"><div class="pipeline-index">{idx}</div><div><div class="row-title">{title}</div><div class="row-sub">{copy}</div></div><div class="row-status">Domain</div></div>' for idx, title, copy in nodes), unsafe_allow_html=True)
    with rag_tab:
        st.markdown("""
        **Implemented retrieval path**

        - Overlap-aware semantic chunks with source and document metadata
        - OpenRouter embedding API or local SentenceTransformer fallback
        - Persistent ChromaDB cosine index for local deployments
        - Pinecone index data-plane integration with namespaces and metadata filters
        - Configurable top-k retrieval with visible relevance scores
        - Evidence-scoped context assembly for contract chat and agents
        """)
    with data_tab:
        st.markdown("""
        **Persistent operational data**

        - SQLite contracts, analyses, users, drafts, chat history and reusable clauses
        - ChromaDB collections isolated by embedding strategy
        - Pinecone namespaces for managed vector tenancy
        - Structured JSON analysis records and export-ready data frames
        - Update, search, delete, tag and history workflows
        """)
    with ops_tab:
        st.markdown("""
        **Agent and integration controls**

        - Selectable GPT-OSS 20B or NVIDIA Nemotron analysis with preserved reasoning details
        - Structured plan, reasoning, result, caveat and next-step outputs
        - Optional current-contract context for agent execution
        - MCP Streamable HTTP server registry and initialization checks
        - Explicit readiness labels so configured and active services are never confused
        """)
    st.markdown('</div>', unsafe_allow_html=True)


def page_architecture():
    flags = _runtime_flags()
    cards = [
        ("graph", "LangGraph orchestration", "Four stateful stages coordinate analysis across eight legal-intelligence domains.", flags["LangGraph"], "Runtime"),
        ("vector", "Vector retrieval", "Persistent ChromaDB locally or Pinecone through its index data-plane API with namespace and document filters.", flags["ChromaDB"], "Storage"),
        ("server", "MCP connectivity", "Workspace registry for Streamable HTTP MCP servers with JSON-RPC initialization health checks.", flags["MCP"], "Protocol"),
        ("shield", "Evidence controls", "Chunk metadata, scoped retrieval, relevance scores and explicit provider diagnostics.", True, "Grounding"),
    ]
    cards_html = "".join(
        f'<div class="infra-card"><div class="infra-icon">{UI_ICONS[icon]}</div><div class="infra-name">{title}</div><div class="infra-detail">{copy}</div><span class="infra-state {"" if active else "ready"}">{"Available" if active else "Configurable"}</span></div>'
        for icon, title, copy, active, _ in cards
    )
    st.markdown(f"""
<div class="about-shell">
  <div class="about-hero">
    <div><div class="eyebrow"><span class="eyebrow-dot"></span>AI systems architecture</div><div class="about-title">Built as an engineering workspace, not a prompt wrapper.</div><div class="about-copy">A modular contract-intelligence stack joins document parsing, LangGraph orchestration, model routing, vector retrieval, MCP connectivity and persistent operations.</div></div>
    <div class="system-card"><div class="system-card-title">Execution path</div><div class="system-card-sub">Ingest → chunk → embed → retrieve → reason → validate → persist</div></div>
  </div>
  <div class="about-section-head"><div><div class="about-section-title">Platform layers</div><div class="about-section-sub">Status labels distinguish locally available capabilities from integrations that need configuration.</div></div></div>
  <div class="infra-grid">{cards_html}</div>
  <div class="workspace-panel"><div class="panel-title">Reference pipeline</div><div class="panel-sub">PDF / DOCX / TXT → deterministic parsing → overlap-aware chunks → local hybrid embeddings → ChromaDB / Pinecone → scoped evidence → LangGraph workflow → structured JSON, PDF, DOCX and CSV outputs.</div></div>
</div>""", unsafe_allow_html=True)


def page_knowledge():
    provider = "Pinecone" if st.session_state.vector_backend == "pinecone" else "ChromaDB"
    st.markdown(f"""
<div class="page-header"><div class="page-header-title">Knowledge Base</div><div class="page-header-sub">Ingest sources, manage vector grounding and inspect document-scoped retrieval evidence through {provider}.</div></div>
""", unsafe_allow_html=True)
    st.markdown('<div class="inner">', unsafe_allow_html=True)
    tabs = st.tabs(["Ingest", "Retrieval lab", "Diagnostics"])
    with tabs[0]:
        source = st.file_uploader("Knowledge source", type=["pdf", "txt", "md"], key="knowledge_upload")
        cc = st.columns(2)
        chunk_size = cc[0].slider("Chunk size", 400, 1600, 800, 100)
        overlap = cc[1].slider("Chunk overlap", 50, 350, 150, 25)
        if st.button("Index knowledge source", type="primary", key="knowledge_index"):
            if not source:
                st.warning("Choose a PDF, TXT or Markdown source first.")
            else:
                raw = source.getvalue()
                if source.name.lower().endswith(".pdf"):
                    parsed = extract_text_from_pdf(raw)
                    source_text = parsed.get("text", "")
                else:
                    source_text = raw.decode("utf-8", errors="ignore")
                chunks = chunk_text(source_text, chunk_size=chunk_size, overlap=overlap)
                doc_id = hashlib.sha256(raw).hexdigest()[:16]
                try:
                    with st.spinner(f"Embedding and indexing in {provider}..."):
                        indexed = get_rag().add_document(doc_id, chunks, {"source": source.name})
                    st.session_state.knowledge_doc_id = doc_id
                    st.session_state.knowledge_doc_name = source.name
                    st.success(f"Indexed {indexed} chunks from {source.name}.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
    with tabs[1]:
        active_name = st.session_state.knowledge_doc_name or "No knowledge source indexed in this session"
        st.caption(f"Active source: {active_name}")
        question = st.text_input("Retrieval query", placeholder="Which clauses create uncapped liability?", key="knowledge_query")
        top_k = st.slider("Evidence results", 2, 10, 5, key="knowledge_top_k")
        if st.button("Retrieve evidence", type="primary", key="knowledge_retrieve"):
            if not st.session_state.knowledge_doc_id:
                st.warning("Index a source first.")
            elif not question.strip():
                st.warning("Enter a retrieval query.")
            else:
                try:
                    hits = get_rag().query(question, st.session_state.knowledge_doc_id, top_k)
                    if not hits:
                        st.info("No matching evidence was returned.")
                    for i, hit in enumerate(hits, 1):
                        source_label = hit.get("source") or active_name
                        score_line = (
                            f"Result {i} · hybrid {hit.get('score', 0):.3f} · "
                            f"semantic {hit.get('semantic_score', 0):.3f} · lexical {hit.get('lexical_score', 0):.3f} · "
                            f"{source_label} / chunk {hit.get('chunk_idx', '—')}"
                        )
                        st.markdown(f'<div class="evidence-card"><div class="evidence-score">{html.escape(score_line)}</div><div class="evidence-text">{html.escape(hit.get("text", ""))}</div></div>', unsafe_allow_html=True)
                except Exception as exc:
                    st.error(f"Retrieval failed: {exc}")
    with tabs[2]:
        configured = st.session_state.vector_backend == "chroma" or bool(st.session_state.pinecone_api_key and st.session_state.pinecone_host)
        vector_count = "Session scoped" if provider == "Pinecone" else "Persistent local collection"
        st.markdown(f"""
<div class="infra-grid">
  <div class="infra-card"><div class="infra-icon">{UI_ICONS['vector']}</div><div class="infra-name">{provider}</div><div class="infra-detail">{vector_count}</div><span class="infra-state {'' if configured else 'ready'}">{'Configured' if configured else 'Setup required'}</span></div>
  <div class="infra-card"><div class="infra-icon">{UI_ICONS['server']}</div><div class="infra-name">Retrieval engine</div><div class="infra-detail">{html.escape(RETRIEVAL_ENGINE_LABEL)}</div><span class="infra-state">Ready</span></div>
  <div class="infra-card"><div class="infra-icon">{UI_ICONS['document']}</div><div class="infra-name">Active source</div><div class="infra-detail">{html.escape(active_name)}</div><span class="infra-state ready">Session</span></div>
  <div class="infra-card"><div class="infra-icon">{UI_ICONS['shield']}</div><div class="infra-name">Retrieval scope</div><div class="infra-detail">Document filter and namespace isolation enabled.</div><span class="infra-state">Ready</span></div>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _probe_mcp(endpoint: str, token: str = "") -> Dict:
    if not endpoint.startswith(("http://", "https://")):
        return {"ok": False, "detail": "Use an http:// or https:// endpoint."}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "Navneet ContractAI", "version": "5.1"}}}
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10)
        return {"ok": response.status_code < 400, "detail": f"HTTP {response.status_code}", "session": response.headers.get("mcp-session-id", "")}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


def page_integrations():
    st.markdown("""
<div class="page-header"><div class="page-header-title">Integrations and Vectors</div><div class="page-header-sub">Operate ChromaDB or Pinecone vector storage, embedding routing and Model Context Protocol server connections.</div></div>
""", unsafe_allow_html=True)
    st.markdown('<div class="inner">', unsafe_allow_html=True)
    vector_tab, mcp_tab = st.tabs(["Vector database", "MCP registry"])
    with vector_tab:
        backend = st.selectbox("Vector backend", ["chroma", "pinecone"], index=0 if st.session_state.vector_backend == "chroma" else 1, format_func=lambda value: "ChromaDB (local)" if value == "chroma" else "Pinecone (managed)")
        if backend == "pinecone":
            pc_key = st.text_input("Pinecone API key", value=st.session_state.pinecone_api_key, type="password")
            pc_host = st.text_input("Pinecone index host", value=st.session_state.pinecone_host, placeholder="https://your-index.svc.region.pinecone.io")
            pc_namespace = st.text_input("Namespace", value=st.session_state.pinecone_namespace)
        else:
            pc_key, pc_host, pc_namespace = st.session_state.pinecone_api_key, st.session_state.pinecone_host, st.session_state.pinecone_namespace
        if st.button("Save vector configuration", type="primary"):
            if backend == "pinecone" and (not pc_key.strip() or not pc_host.strip()):
                st.warning("Pinecone requires an API key and index host.")
            else:
                st.session_state.vector_backend = backend
                st.session_state.pinecone_api_key = pc_key.strip()
                st.session_state.pinecone_host = pc_host.strip()
                st.session_state.pinecone_namespace = pc_namespace.strip() or "contracts"
                st.session_state.rag_engine = None
                st.session_state.rag_config = None
                st.success(f"{backend.title()} is now the selected vector backend.")
    with mcp_tab:
        mc = st.columns([1, 2])
        server_name = mc[0].text_input("Server name", placeholder="Contract policy tools")
        endpoint = mc[1].text_input("Streamable HTTP endpoint", placeholder="https://mcp.example.com/mcp")
        token = st.text_input("Bearer token (optional)", type="password")
        add_col, test_col = st.columns(2)
        if add_col.button("Add to registry", use_container_width=True):
            if not server_name.strip() or not endpoint.strip():
                st.warning("Server name and endpoint are required.")
            elif not endpoint.startswith(("http://", "https://")):
                st.warning("Use a valid HTTP or HTTPS endpoint.")
            else:
                st.session_state.mcp_servers.append({"name": server_name.strip(), "endpoint": endpoint.strip(), "token": token, "status": "Configured"})
                st.success("MCP server added to this session.")
        if test_col.button("Test initialization", type="primary", use_container_width=True):
            if not endpoint.strip():
                st.warning("Enter an endpoint first.")
            else:
                result = _probe_mcp(endpoint.strip(), token)
                (st.success if result["ok"] else st.error)(f"{result['detail']} — {'MCP initialization responded' if result['ok'] else 'connection failed'}")
        for i, server in enumerate(st.session_state.mcp_servers):
            rc = st.columns([2, 4, 1])
            rc[0].markdown(f"**{html.escape(server['name'])}**")
            rc[1].code(server["endpoint"], language=None)
            if rc[2].button("Remove", key=f"remove_mcp_{i}", use_container_width=True):
                st.session_state.mcp_servers.pop(i)
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


logged_in = bool(st.session_state.auth_user)

# Gate protected pages behind login
if st.session_state.page not in PUBLIC_PAGES and not logged_in:
    st.session_state.page = "home"
    st.session_state.show_auth = True

inject_css(st.session_state.dark_mode, show_sidebar=logged_in)
render_navbar()
if logged_in:
    render_sidebar()
render_auth_modal()

page = st.session_state.page
PAGES = {
    "home":      page_home,
    "dashboard": page_dashboard,
    "features":  page_features,
    "architecture": page_architecture,
    "analyze":   page_analyze,
    "writer":    page_writer,
    "chatbot":   page_chatbot,
    "agent":     page_agent,
    "chat":      page_chat,
    "knowledge": page_knowledge,
    "integrations": page_integrations,
    "history":   page_history,
    "compare":   page_compare,
    "about":     page_about,
    "contact":   page_contact,
    "settings":  page_settings,
}
WORKSPACE_PAGES = {
    "analyze", "writer", "chatbot", "agent", "chat", "knowledge",
    "integrations", "history", "compare", "settings",
}
if logged_in and page in WORKSPACE_PAGES:
    with st.container(key=f"workspace_page_{page}"):
        PAGES.get(page, page_home)()
else:
    PAGES.get(page, page_home)()
