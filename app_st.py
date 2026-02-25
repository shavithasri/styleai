"""
Style AI — app_st.py
Flask runs ALL pages (login.html, index.html) + API (/analyze, /chat)
Streamlit is just the launcher — hides itself and shows Flask full-screen
"""

import streamlit as st
import streamlit.components.v1 as components
import threading, os, cv2, numpy as np, base64, time
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS

# Load .env from same folder as this script — always override so fresh key is picked up
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=_env_path, override=True)

# ══════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════
flask_app = Flask(__name__)
CORS(flask_app)
groq_ref = {"client": None}


def find_file(name):
    """Find an HTML file in common locations."""
    candidates = [
        name,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        f"/mnt/user-data/uploads/{name}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ─── Skin Tone Detection ──────────────────────────────────────
def detect_skin_tone(img_bytes):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return "Medium", 180, 140, 110
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cx, cy = x + w // 2, y + h // 2
                sample = img_rgb[cy - h // 6: cy + h // 6, cx - w // 6: cx + w // 6]
            else:
                h, w = img_rgb.shape[:2]
                sample = img_rgb[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        except Exception:
            h, w = img_rgb.shape[:2]
            sample = img_rgb[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        avg = sample.mean(axis=(0, 1))
        r, g, b = int(avg[0]), int(avg[1]), int(avg[2])
        br = (r + g + b) / 3
        tone = "Fair" if br > 200 else "Medium" if br > 160 else "Olive" if br > 120 else "Deep"
        return tone, r, g, b
    except Exception:
        return "Medium", 180, 140, 110


# ─── Parse AI Output ──────────────────────────────────────────
def parse_recs(raw):
    sections, current, items = {}, None, []
    keys = ["DRESS_CODE", "SUGGESTED_OUTFIT", "SHIRT_DETAILS", "PANT_DETAILS",
            "SHOES_DETAILS", "HAIRSTYLE", "ACCESSORIES", "COLOR_PALETTE", "WHY_IT_WORKS"]
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        matched = next(
            (k for k in keys if k in line.upper().replace(" ", "_").replace("-", "_")), None
        )
        if matched:
            if current:
                sections[current] = items
            current, items = matched, []
        elif current and line.startswith("→"):
            item = line.lstrip("→").strip()
            if item:
                items.append(item)
        elif current and line and not line.startswith("#"):
            if line not in keys:
                items.append(line)
    if current:
        sections[current] = items
    return sections


# ─── Shopping Data ────────────────────────────────────────────
SHOPPING = {
    ("Fair", "Male"): [
        {"name": "Light Blue Oxford Shirt", "platform": "Myntra", "url": "https://www.myntra.com/shirts", "icon": "👕"},
        {"name": "Navy Chinos", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=navy+chinos+men", "icon": "👖"},
        {"name": "White Sneakers", "platform": "Zara", "url": "https://www.zara.com/in/en/man-shoes-l769.html", "icon": "👟"},
        {"name": "Silver Watch", "platform": "Myntra", "url": "https://www.myntra.com/watches", "icon": "⌚"},
    ],
    ("Fair", "Female"): [
        {"name": "Pastel Floral Kurti", "platform": "Myntra", "url": "https://www.myntra.com/kurtis", "icon": "👗"},
        {"name": "Ivory Palazzo Pants", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=ivory+palazzo+women", "icon": "👖"},
        {"name": "Rose Gold Heels", "platform": "Zara", "url": "https://www.zara.com/in/en/woman-shoes-l1251.html", "icon": "👠"},
        {"name": "Pearl Earrings", "platform": "Myntra", "url": "https://www.myntra.com/earrings", "icon": "💎"},
    ],
    ("Medium", "Male"): [
        {"name": "Olive Green Shirt", "platform": "Myntra", "url": "https://www.myntra.com/shirts", "icon": "👕"},
        {"name": "Dark Brown Chinos", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=dark+brown+chinos", "icon": "👖"},
        {"name": "Tan Loafers", "platform": "Zara", "url": "https://www.zara.com/in/en/man-shoes-l769.html", "icon": "🥿"},
        {"name": "Leather Belt", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=leather+belt+men", "icon": "🔗"},
    ],
    ("Medium", "Female"): [
        {"name": "Terracotta Kurta", "platform": "Myntra", "url": "https://www.myntra.com/kurtis", "icon": "👗"},
        {"name": "Beige Straight Pants", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=beige+straight+pants+women", "icon": "👖"},
        {"name": "Block Heels", "platform": "Myntra", "url": "https://www.myntra.com/heels", "icon": "👠"},
        {"name": "Gold Jhumkas", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=gold+jhumka+earrings", "icon": "💛"},
    ],
    ("Olive", "Male"): [
        {"name": "Mustard Yellow Polo", "platform": "Myntra", "url": "https://www.myntra.com/polo-t-shirts", "icon": "👕"},
        {"name": "Dark Jeans", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=dark+jeans+men", "icon": "👖"},
        {"name": "Brown Derby Shoes", "platform": "Zara", "url": "https://www.zara.com/in/en/man-shoes-l769.html", "icon": "👞"},
        {"name": "Copper Bracelet", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=copper+bracelet+men", "icon": "⚡"},
    ],
    ("Olive", "Female"): [
        {"name": "Emerald Green Anarkali", "platform": "Myntra", "url": "https://www.myntra.com/anarkali-suits", "icon": "👗"},
        {"name": "Copper Leggings", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=copper+leggings+women", "icon": "👖"},
        {"name": "Wedge Sandals", "platform": "Zara", "url": "https://www.zara.com/in/en/woman-shoes-l1251.html", "icon": "👡"},
        {"name": "Oxidised Silver Necklace", "platform": "Myntra", "url": "https://www.myntra.com/necklaces", "icon": "💍"},
    ],
    ("Deep", "Male"): [
        {"name": "Royal Blue Kurta", "platform": "Myntra", "url": "https://www.myntra.com/kurtas", "icon": "👘"},
        {"name": "Deep Purple Trousers", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=purple+formal+trousers", "icon": "👖"},
        {"name": "Black Chelsea Boots", "platform": "Zara", "url": "https://www.zara.com/in/en/man-shoes-l769.html", "icon": "🥾"},
        {"name": "Gold Chain", "platform": "Myntra", "url": "https://www.myntra.com/chains", "icon": "🏆"},
    ],
    ("Deep", "Female"): [
        {"name": "Burgundy Saree", "platform": "Myntra", "url": "https://www.myntra.com/sarees", "icon": "👘"},
        {"name": "Fuchsia Blouse", "platform": "Amazon.in", "url": "https://www.amazon.in/s?k=fuchsia+blouse+women", "icon": "👗"},
        {"name": "Gold Stilettos", "platform": "Zara", "url": "https://www.zara.com/in/en/woman-shoes-l1251.html", "icon": "👠"},
        {"name": "Statement Bangles", "platform": "Myntra", "url": "https://www.myntra.com/bangles", "icon": "🔮"},
    ],
}


# ─── Flask Routes: Serve HTML pages ──────────────────────────
@flask_app.route("/")
def root():
    return redirect("/login")


@flask_app.route("/login")
@flask_app.route("/login.html")
def login_page():
    path = find_file("login.html")
    if path:
        return send_file(path, mimetype="text/html")
    return "<h1 style='font-family:sans-serif;padding:2rem'>login.html not found — put it in the same folder as app_st.py</h1>", 404


@flask_app.route("/index")
@flask_app.route("/index.html")
def index_page():
    path = find_file("index.html")
    if path:
        return send_file(path, mimetype="text/html")
    return "<h1 style='font-family:sans-serif;padding:2rem'>index.html not found — put it in the same folder as app_st.py</h1>", 404


# ─── Flask Route: /analyze ────────────────────────────────────
@flask_app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json or {}
        image_b64 = data.get("image", "")
        gender = data.get("gender", "Male")
        occasion = data.get("occasion", "Casual")
        prompt = data.get("prompt", "")

        # Decode image
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)

        # Detect skin tone
        tone, r, g, b = detect_skin_tone(img_bytes)

        # Try to init client fresh if not set (handles cache issue)
        if not groq_ref["client"]:
            load_dotenv(dotenv_path=_env_path, override=True)
            key = os.getenv("GROQ_API_KEY", "").strip()
            if key:
                groq_ref["client"] = Groq(api_key=key)
        client = groq_ref["client"]
        if not client:
            return jsonify({"success": False, "error": "GROQ_API_KEY not found in .env — add it and restart"})

        style_ctx = f"Occasion: {occasion}. Extra style notes: {prompt}" if prompt else f"Occasion: {occasion}."
        ai_prompt = f"""You are an expert fashion stylist. A {gender} client has a {tone} skin tone (RGB: {r},{g},{b}).
{style_ctx}

Give personalised recommendations in this EXACT format — use → prefix for every item:

DRESS_CODE
→ Formal
→ Business Casual
→ Casual

SUGGESTED_OUTFIT
→ [Complete head-to-toe outfit in one vivid sentence]

SHIRT_DETAILS
→ Color: [specific color]
→ Type: [shirt type]
→ Brand: [Indian or global brand]
→ Fabric: [fabric type]

PANT_DETAILS
→ Color: [specific color]
→ Type: [pant type]
→ Brand: [brand]
→ Fabric: [fabric]

SHOES_DETAILS
→ Color: [color]
→ Type: [shoe type]
→ Brand: [brand]

HAIRSTYLE
→ Style: [specific hairstyle name]
→ How-to: [step-by-step maintenance tip]
→ Products: [recommended hair product]
→ Tip: [pro styling tip]

ACCESSORIES
→ 1. [accessory name] — [why it works for this skin tone and occasion]
→ 2. [accessory name] — [why it works for this skin tone and occasion]
→ 3. [accessory name] — [why it works for this skin tone and occasion]
→ 4. [accessory name] — [why it works for this skin tone and occasion]

COLOR_PALETTE
→ Primary: [one specific color name e.g. "Navy Blue", "Emerald Green", "Dusty Rose"]
→ Secondary: [one specific color name e.g. "Ivory", "Charcoal Grey", "Copper"]
→ Accent: [one specific color name e.g. "Gold", "Coral", "Sage Green"]

WHY_IT_WORKS
→ [2 sentences: why these choices complement this skin tone and style]

Be specific, stylish, and consider Indian fashion. Output ONLY the format above — nothing else."""

        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": ai_prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=1200,
            temperature=0.7,
        )
        raw = resp.choices[0].message.content
        parsed = parse_recs(raw)
        products = SHOPPING.get((tone, gender), SHOPPING.get(("Medium", "Male"), []))

        return jsonify({
            "success": True,
            "skin_tone": tone,
            "rgb": {"r": r, "g": g, "b": b},
            "recommendations": parsed,
            "products": products,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ─── Flask Route: /chat ───────────────────────────────────────
@flask_app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        message = data.get("message", "")
        history = data.get("history", [])

        if not groq_ref["client"]:
            load_dotenv(dotenv_path=_env_path, override=True)
            key = os.getenv("GROQ_API_KEY", "").strip()
            if key:
                groq_ref["client"] = Groq(api_key=key)
        client = groq_ref["client"]
        if not client:
            return jsonify({"reply": "GROQ_API_KEY not found in .env — add it and restart"})

        msgs = [{"role": "system", "content": (
            "You are a witty, knowledgeable Indian fashion assistant for Style AI. "
            "You specialise in outfit pairing, colour theory, occasion dressing, "
            "and Indian/global fashion trends. Keep replies concise (under 4 sentences) "
            "unless the user asks for a detailed breakdown."
        )}]
        for m in history[-6:]:
            msgs.append(m)
        msgs.append({"role": "user", "content": message})

        resp = client.chat.completions.create(
            messages=msgs,
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            temperature=0.8,
        )
        return jsonify({"reply": resp.choices[0].message.content})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})


# ─── Start Flask in background thread ────────────────────────
def _run_flask():
    flask_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)


_flask_started = False

def start_flask():
    global _flask_started
    if _flask_started:
        return True

    # Re-load .env fresh every time
    load_dotenv(dotenv_path=_env_path, override=True)

    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "").strip()
        except Exception:
            pass

    if key:
        groq_ref["client"] = Groq(api_key=key)
    else:
        print("WARNING: GROQ_API_KEY not found in .env!")

    t = threading.Thread(target=_run_flask, daemon=True)
    t.start()
    time.sleep(1.5)
    _flask_started = True
    return True


# ══════════════════════════════════════════════════════════════
#  STREAMLIT SHELL
#  — hides all Streamlit UI
#  — renders Flask app in a full-viewport iframe
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Style AI",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Completely hide Streamlit chrome
st.markdown("""
<style>
  #MainMenu, footer, header,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stSidebarNav"],
  section[data-testid="stSidebar"],
  [data-testid="stStatusWidget"],
  [data-testid="stBottom"] { display: none !important; }

  .block-container,
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewBlockContainer"],
  .main, .appview-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    height: 100vh !important;
  }
</style>
""", unsafe_allow_html=True)

# Start Flask
start_flask()

# Embed as full-screen iframe — starts at /login
# login.html redirects to /index after Firebase login
components.html(
    """<!DOCTYPE html>
<html style="margin:0;padding:0;height:100%">
<body style="margin:0;padding:0;height:100%;overflow:hidden">
  <iframe
    src="http://localhost:5001/login"
    style="width:100%;height:100vh;border:none;display:block"
    allow="microphone; camera"
    allowfullscreen
  ></iframe>
</body>
</html>""",
    height=900,
    scrolling=False,
)
