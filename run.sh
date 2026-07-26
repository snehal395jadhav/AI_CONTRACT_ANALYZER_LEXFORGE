#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  LexForge AI v5.0 — Run Script                              ║
# ║  Developer: SNEHAL LAXMAN JADHAV                            ║
# ║  Navneet Education Limited | 2026                           ║
# ╚══════════════════════════════════════════════════════════════╝

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting LexForge AI v5.0..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
