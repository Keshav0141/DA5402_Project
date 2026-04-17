# frontend/app.py

import streamlit as st
import requests
from PIL import Image
import io
import json
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="CerebroNet",
    page_icon="🧠",
    layout="wide"
)
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")
