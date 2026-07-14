import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ============================================================
# INISIALISASI FLASK & PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates") if os.path.exists(os.path.join(BASE_DIR, "templates")) else "templates",
    static_folder=os.path.join(BASE_DIR, "static") if os.path.exists(os.path.join(BASE_DIR, "static")) else "static"
)
CORS(app)


# ============================================================
# KONFIGURASI THRESHOLD & MARGIN
# ============================================================
# PENTING: Angka di bawah ini adalah titik awal, BUKAN angka final.
# Wajib dikalibrasi pakai script `calibrate_threshold.py` yang saya sertakan
# terpisah, dengan query yang benar-benar di luar dataset kamu.
#
# - SCORE_THRESHOLD  : skor minimum (E5/IP) atau maksimum (L2) supaya top-1
#                       dianggap "layak" jadi kandidat jawaban.
# - MIN_MARGIN        : selisih minimum antara skor top-1 dan top-2. Kalau
#                       selisihnya kecil, artinya model "ragu" -> banyak
#                       chunk lain yang skornya mirip -> kemungkinan besar
#                       pertanyaan user memang tidak match spesifik ke satu
#                       chunk di dataset -> dianggap not_found.
CONFIG = {
    "e5": {
        # Dikalibrasi dari data log asli:
        #   - Di luar dataset ("berapa", "nayla")        -> top1 ~0.75
        #   - Ada di dataset (match ke pertanyaan asli)   -> top1 ~0.88-0.91
        # 0.82 dipilih sebagai titik tengah dengan buffer aman.
        "score_threshold": 0.82,
        # Margin top1-top2 di dataset ini secara alami KECIL (0.001-0.004)
        # untuk kasus yang match sekalipun -- karena banyak chunk dalam
        # satu dokumen isinya context yang saling nyambung/mirip. Jadi
        # margin BUKAN sinyal yang reliable di sini, diset sangat kecil
        # supaya tidak ikut menolak jawaban yang sebenarnya valid.
        "min_margin": 0.0005,
    },
    "indobert": {
        # Dikalibrasi dari data log asli (index IndoBERT kamu = IP/cosine):
        #   - Di luar dataset ("berapa harga ukt pcr?")      -> top1 ~0.48
        #   - Ada di dataset (match ke pertanyaan asli)       -> top1 ~0.69-0.71
        # 0.58 dipilih sebagai titik tengah dengan buffer aman.
        "score_threshold": 0.58,
        # Sama seperti E5, margin top1-top2 tidak konsisten (bisa 0.005
        # bahkan untuk match yang valid), jadi dibuat sangat kecil supaya
        # tidak ikut menolak jawaban yang sebenarnya benar.
        "min_margin": 0.001,
    },
}


# ============================================================
# LOAD RESOURCE - MODEL E5
# ============================================================
print("=" * 60)
print("Loading E5 Resources...")
df_e5 = pd.read_csv(os.path.join(BASE_DIR, "chunk_metadata.csv"))
index_e5 = faiss.read_index(os.path.join(BASE_DIR, "faiss_index.bin"))

MODEL_E5 = "intfloat/e5-base-v2"
tokenizer_e5 = AutoTokenizer.from_pretrained(MODEL_E5)
model_e5 = AutoModel.from_pretrained(MODEL_E5)
model_e5.eval()
print(f"E5 index metric_type = {index_e5.metric_type} "
      f"({'INNER_PRODUCT/cosine' if index_e5.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'})")
print("E5 Resources Loaded Successfully!")


# ============================================================
# LOAD RESOURCE - MODEL INDOBERT
# ============================================================
print("=" * 60)
print("Loading IndoBERT Resources...")
INDOBERT_DIR = os.path.join(BASE_DIR, "chatbotIndoBert")

df_indobert = pd.read_csv(os.path.join(INDOBERT_DIR, "chunk_metadata_indobert.csv"))
index_indobert = faiss.read_index(os.path.join(INDOBERT_DIR, "faiss_index_indobert.bin"))

MODEL_INDOBERT = "indobenchmark/indobert-base-p1"
tokenizer_indobert = AutoTokenizer.from_pretrained(MODEL_INDOBERT)
model_indobert = AutoModel.from_pretrained(MODEL_INDOBERT)
model_indobert.eval()

# Deteksi metric_type index IndoBERT secara otomatis, supaya logika
# perbandingan (>= threshold vs <= threshold) tidak salah arah kalau
# suatu saat index dibangun ulang dengan metric berbeda.
INDOBERT_IS_IP = (index_indobert.metric_type == faiss.METRIC_INNER_PRODUCT)
print(f"IndoBERT index metric_type = {index_indobert.metric_type} "
      f"({'INNER_PRODUCT' if INDOBERT_IS_IP else 'L2'})")
print("IndoBERT Resources Loaded Successfully!")
print("=" * 60)


# ============================================================
# HELPER FUNCTIONS (EMBEDDING)
# ============================================================
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embedding_e5(text):
    text = "query: " + text
    encoded_input = tokenizer_e5(
        text, max_length=512, truncation=True, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model_e5(**encoded_input)
        embedding = average_pool(outputs.last_hidden_state, encoded_input["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy().astype("float32")

def get_embedding_indobert(text):
    encoded_input = tokenizer_indobert(
        text, max_length=512, truncation=True, padding=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model_indobert(**encoded_input)
        embedding = outputs.last_hidden_state.mean(dim=1)
        if INDOBERT_IS_IP:
            # kalau index-nya IP, embedding query juga wajib dinormalisasi
            # supaya skalanya konsisten dengan cosine similarity
            embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.cpu().numpy().astype("float32")

GREETINGS = ["halo", "hai", "hi", "hello", "assalamualaikum", "selamat pagi", "selamat siang", "selamat sore", "selamat malam"]


# ============================================================
# ROUTES FOR UI / TEMPLATES
# ============================================================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/chatbot")
def chatbot_e5_ui():
    return render_template("chatbot.html")

@app.route("/chatbot-indobert")
def chatbot_indobert_ui():
    return render_template("chatbotIndoBert.html")


# ============================================================
# CORE LOGIC: apakah top-1 hasil FAISS layak dijadikan jawaban?
# ============================================================
def is_confident_match(distances_row, model_type):
    """
    distances_row: array skor hasil FAISS search untuk 1 query, urutan
                   sudah terbaik->terburuk sesuai konvensi FAISS (index 0 = top-1).
    model_type   : "e5" atau "indobert"

    Return True kalau hasil top-1 dianggap cukup relevan DAN cukup
    menonjol dibanding kandidat lain (margin cukup besar).
    """
    cfg = CONFIG[model_type]
    top1 = float(distances_row[0])
    top2 = float(distances_row[1]) if len(distances_row) > 1 else None

    if model_type == "e5" or (model_type == "indobert" and INDOBERT_IS_IP):
        # metric IP / cosine: makin BESAR makin mirip
        score_ok = top1 >= cfg["score_threshold"]
        margin_ok = True if top2 is None else (top1 - top2) >= cfg["min_margin"]
    else:
        # metric L2: makin KECIL makin mirip
        score_ok = top1 <= cfg["score_threshold"]
        margin_ok = True if top2 is None else (top2 - top1) >= cfg["min_margin"]

    return score_ok and margin_ok


# ============================================================
# CORE LOGIC FOR SEARCH (REUSABLE FUNCTION)
# ============================================================
def process_chat_logic(user_question, model_type):
    # 1. Pilihan Resource
    if model_type == "e5":
        embedding_fn = get_embedding_e5
        faiss_index = index_e5
        dataframe = df_e5
        bot_name = "Finance AI Chatbot (E5)"
    else:
        embedding_fn = get_embedding_indobert
        faiss_index = index_indobert
        dataframe = df_indobert
        bot_name = "Finance AI Chatbot (IndoBERT)"

    # 2. Cek Salam
    if user_question.lower() in GREETINGS:
        return jsonify({
            "type": "greeting",
            "message": f"Halo! Saya adalah {bot_name}. Saya siap membantu menjawab pertanyaan mengenai laporan keuangan, aset, liabilitas, biaya operasional, investasi, dan informasi finansial lainnya."
        })

    # 3. Embedding & FAISS Search (ambil top-3 supaya bisa hitung margin)
    query_embedding = embedding_fn(user_question)
    k = 3
    distances, indices = faiss_index.search(query_embedding, k=k)

    best_idx = int(indices[0][0])
    top_scores = distances[0]

    print(f"\n[MODEL: {model_type.upper()}] | Query: {user_question} | "
          f"Top idx: {best_idx} | Top-{k} scores: {top_scores}")

    # 4. Validasi Dasar Index Utama FAISS
    if best_idx == -1 or best_idx >= len(dataframe):
        return jsonify({
            "type": "not_found",
            "message": "Maaf, pertanyaan tidak ada dalam dataset."
        })

    # 5. Cek confidence: skor cukup tinggi/rendah DAN cukup menonjol dari kandidat lain
    if not is_confident_match(top_scores, model_type):
        return jsonify({
            "type": "not_found",
            "message": "Maaf, pertanyaan tidak ada dalam dataset."
        })

    # 6. Ambil Data Aman Menggunakan .iloc langsung
    retrieved_question = str(dataframe.iloc[best_idx]["question"])
    retrieved_answer = str(dataframe.iloc[best_idx]["answer"])
    retrieved_context = str(dataframe.iloc[best_idx]["text"])

    return jsonify({
        "type": "answer",
        "question": retrieved_question,
        "answer": retrieved_answer,
        "context": retrieved_context
    })


# ============================================================
# API ENDPOINTS
# ============================================================
@app.route("/chat", methods=["POST"])
def chat_e5():
    try:
        data = request.get_json()
        if not data or "message" not in data or data["message"].strip() == "":
            return jsonify({"type": "not_found", "message": "Input atau pertanyaan tidak valid."})
        return process_chat_logic(data["message"].strip(), model_type="e5")
    except Exception as e:
        print("ERROR E5 ROUTE:", e)
        return jsonify({"type": "error", "message": "Terjadi kesalahan pada server E5."}), 500


@app.route("/chat-indobert", methods=["POST"])
def chat_indobert():
    try:
        data = request.get_json()
        if not data or "message" not in data or data["message"].strip() == "":
            return jsonify({"type": "not_found", "message": "Input atau pertanyaan tidak valid."})
        return process_chat_logic(data["message"].strip(), model_type="indobert")
    except Exception as e:
        print("ERROR INDOBERT ROUTE:", e)
        return jsonify({"type": "error", "message": "Terjadi kesalahan pada server IndoBERT."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)