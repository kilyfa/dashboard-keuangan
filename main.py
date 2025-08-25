import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import numpy as np
import requests, io, os, json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.linear_model import LinearRegression

# ========= Load master prompt =========
PROMPT_PATH = "prompt.txt"
MASTER_PROMPT = (
    open(PROMPT_PATH, "r", encoding="utf-8").read()
    if os.path.exists(PROMPT_PATH)
    else "Anda adalah penasihat keuangan pribadi."
)

# ========= App config (dark only) =====
st.set_page_config(page_title="AI Finance Studio", page_icon="💸", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html,body,[class*='css']{font-family:'Inter',sans-serif;background:#121212;color:#e1e1e1}
    .sidebar-content{background:#1e1e1e} .metric span{font-size:0.9rem!important}
    </style>
    """,
    unsafe_allow_html=True,
)

# ========= OpenRouter setup ===========
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
MODEL = "meta-llama/llama-3.3-70b-instruct:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://ai-finance-studio",
    "X-Title": "Finance Studio",
}
SESSION = requests.Session()
SESSION.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=3, backoff_factor=2, status_forcelist=[502, 503, 504])),
)

# ========= AI helpers =================
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def ai_cached(msg: str) -> str:
    return ai_call(msg)

def ai_call(msg: str, temperature: float = 0.7) -> str:
    """Wrapper around OpenRouter chat completion."""
    if not OPENROUTER_API_KEY:
        return "⚠️ API-key belum disetel (Settings → Secrets)."
    payload = {
        "model": MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": MASTER_PROMPT},
            {"role": "user", "content": msg},
        ],
    }
    try:
        r = SESSION.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ AI error: {e}"

# ========= Taxonomy ===================
TAXONOMY = {
    "Essential": [
        "Food", "Bills", "Kontrakan", "Health", "Transportation",
        "Telephone", "Tax", "Insurance", "Baby", "Education"
    ],
    "Financial": [
        "Hutang", "Savings", "Investment", "Trading"
    ],
    "Lifestyle": [
        "Beauty", "Clothing", "Electronics", "Entertainment",
        "Shopping", "Social", "Sport", "Car", "Gadgets", "Travel"
    ],
    "Other": [
        "Business", "Home", "Kerugian", "Penyesuaian", "Misc"
    ],
    "Income": [
        "Salary", "Rental", "Sale", "Coupons", "Grants", "Lottery",
        "Orang Tua", "Refunds", "Business", "Trading", "Biaya Sekunder"
    ],
}
DISCRETIONARY = set(TAXONOMY["Lifestyle"])

# ========= Forecast util ==============
@st.cache_data
def forecast_runway(df_daily: pd.DataFrame, balance: float):
    """Estimate runway menggunakan regresi linear & keluarkan statistik."""
    if df_daily.empty or balance <= 0:
        return None, None, None, None, None

    d = df_daily.copy()
    d["cum"] = d["AMOUNT"].cumsum()
    d["idx"] = (d["DATE"] - d["DATE"].min()).dt.days

    model = LinearRegression().fit(d[["idx"]], d["cum"])
    slope = model.coef_[0]          # ≈ burn-rate harian (negatif = net outflow)
    intercept = model.intercept_
    r2 = model.score(d[["idx"]], d["cum"])

    days_float = (balance - intercept) / slope if slope != 0 else np.inf
    if days_float < 0:
        return 0, date.today(), slope, intercept, r2

    days_int = int(np.ceil(days_float))
    out_date = d["DATE"].min() + timedelta(days=days_int)
    return days_int, out_date, slope, intercept, r2

# ========= Sidebar ====================
st.sidebar.header("📂 Data Keuangan")
file = st.sidebar.file_uploader("Unggah CSV atau Excel", type=["csv", "xlsx"])
current_balance_input = st.sidebar.number_input(
    "💵 Saldo Saat Ini (Rp)", min_value=0.0, step=100000.0, value=0.0,
    help="Masukan saldo rekening terbaru Anda."
)
range_opt = st.sidebar.radio("Rentang",
                             ["Bulan Ini", "Bulan Lalu", "Semua", "Kustom"], index=0)
goal_amt = st.sidebar.number_input(
    "🎯 Target Tabungan (Rp)", min_value=0.0, step=100000.0, value=10000000.0
)

# ========= Main logic ================
if file:
    # ---- Read & clean ----
    df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df["TIME"] = pd.to_datetime(df["TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df.dropna(subset=["TIME"], inplace=True)
    df["DATE"] = df["TIME"].dt.date
    df["CATEGORY"] = df["CATEGORY"].str.title()
    df.loc[~df["CATEGORY"].isin(sum(TAXONOMY.values(), [])), "CATEGORY"] = "Misc"

    # ---- Date range ----
    today = date.today()
    first_month = today.replace(day=1)
    if range_opt == "Bulan Ini":
        start_d, end_d = first_month, today
    elif range_opt == "Bulan Lalu":
        last = first_month - timedelta(days=1)
        start_d, end_d = last.replace(day=1), last
    elif range_opt == "Semua":
        start_d, end_d = df["DATE"].min(), df["DATE"].max()
    else:  # Kustom
        start_d, end_d = st.sidebar.date_input(
            "Rentang", (df["DATE"].min(), df["DATE"].max())
        )

    # ---- Filter akun ----
    accounts = st.sidebar.multiselect("Akun",
                                      df["ACCOUNT"].unique(),
                                      df["ACCOUNT"].unique())
    dff = df[df["DATE"].between(start_d, end_d) &
             df["ACCOUNT"].isin(accounts)].copy()

    # ---- Metrics ----
    exp = dff[dff.TYPE == "(-) Expense"]["AMOUNT"].sum()
    inc = dff[dff.TYPE != "(-) Expense"]["AMOUNT"].sum()
    bal_calculated = inc - exp
    bal = current_balance_input if current_balance_input > 0 else bal_calculated

    top3 = (dff[dff.TYPE == "(-) Expense"]
            .groupby("CATEGORY")["AMOUNT"].sum()
            .sort_values(ascending=False).head(3).to_dict())

    ai_summary = ai_cached(
        f"Pengeluaran={exp}, Pemasukan={inc}, Saldo={bal}, Top={top3}. "
        "Beri insight dan saran ringkas."
    )

    dash, insight, advisor, chatbot, report = st.tabs(
        ["🏠 Dashboard", "💡 Insight", "🤝 Advisor", "💬 Chatbot", "📝 Report"]
    )

    # ---------- DASHBOARD ----------
    with dash:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pengeluaran", f"Rp {exp:,.0f}")
        m2.metric("Pemasukan", f"Rp {inc:,.0f}")
        m3.metric("Saldo", f"Rp {bal:,.0f}")

        progress = max(0.0, min(bal / goal_amt if goal_amt else 0, 1.0))
        m4.markdown("**Progress Target Tabungan**")
        m4.progress(progress, text=f"{progress*100:.1f}% dari Rp {goal_amt:,.0f}")

        cat_sum = (dff[dff.TYPE == "(-) Expense"]
                   .groupby("CATEGORY")["AMOUNT"].sum().reset_index())

        col1, col2 = st.columns([2, 1])
        col1.plotly_chart(
            px.bar(cat_sum, y="CATEGORY", x="AMOUNT", orientation="h",
                   height=450,
                   color=cat_sum["CATEGORY"].apply(
                       lambda x: "Lifestyle"
                       if x in DISCRETIONARY else "Essential"),
                   color_discrete_map={"Lifestyle": "#ff6b6b",
                                       "Essential": "#4dabf7"})
        )

        daily = (dff[dff.TYPE == "(-) Expense"]
                 .groupby("DATE")["AMOUNT"].sum().reset_index())
        daily["DATE"] = pd.to_datetime(daily["DATE"])
        col2.plotly_chart(
            px.area(daily, x="DATE", y="AMOUNT",
                    height=450, title="Harian")
        )

        st.expander("Detail Transaksi").dataframe(
            dff.sort_values("TIME", ascending=False), use_container_width=True)

    # ---------- INSIGHT ----------
    with insight:
        st.subheader("Insight AI")
        st.write(ai_summary)
        if st.button("🔄 Refresh Insight"):
            st.cache_data.clear(); st.rerun()

    # ---------- ADVISOR ----------
    with advisor:
        i1, i2 = st.columns([3, 1])
        item = i1.text_input("Barang/Layanan")
        price = i2.number_input("Harga", 0.0, step=5000.0)
        note = st.text_area("Alasan", height=80)
        cat_buy = st.selectbox("Kategori", sorted(df["CATEGORY"].unique()))

        if st.button("Apakah Layak Beli?"):
            st.write(
                ai_call(
                    f"Saya mau beli {item} harga {price}, saldo {bal}, "
                    f"alasan {note}, kategori {cat_buy}. Apakah layak?"
                )
            )

        st.divider()
        st.subheader("Runway Saldo")
        ess = st.multiselect("Kategori rutin",
                             TAXONOMY["Essential"], TAXONOMY["Essential"])
        reg = (dff[(dff.TYPE == "(-) Expense") & dff.CATEGORY.isin(ess)]
               .groupby("DATE")["AMOUNT"].sum().reset_index())
        reg["DATE"] = pd.to_datetime(reg["DATE"])

        if st.button("Hitung Runway"):
            days, out, slope, intercept, r2 = forecast_runway(reg, bal)
            if out:
                st.success(
                    f"Saldo bertahan {days} hari (sampai {out:%d %b %Y})\n\n"
                    f"ℹ️ Statistik: burn-rate ≈ Rp {abs(slope):,.0f}/hari, "
                    f"intersep Rp {intercept:,.0f}, R² = {r2:.2f}"
                )
            else:
                st.warning("Data tidak cukup untuk memproyeksikan runway.")

    # ---------- CHATBOT ----------
    with chatbot:
        st.subheader("Chat dengan Finance AI")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, msg in st.session_state.chat_history:
            st.chat_message(role).write(msg)

        user_q = st.chat_input("Tanya sesuatu…")
        if user_q:
            st.session_state.chat_history.append(("user", user_q))
            st.chat_message("user").write(user_q)

            csv_preview = dff.head(200).to_json(orient="records", force_ascii=False)
            context = json.dumps({
                "saldo": bal,
                "top_kategori": list(top3.keys()),
                "target_tabungan": goal_amt,
                "preview_transaksi": csv_preview
            }, ensure_ascii=False)

            ai_ans = ai_call(f"Context: {context}\nPertanyaan: {user_q}")
            st.session_state.chat_history.append(("assistant", ai_ans))
            st.chat_message("assistant").write(ai_ans)

    # ---------- REPORT ----------
    with report:
        if st.button("Generate Ringkasan"):
            rep = ai_call(
                "Buat ringkasan + 5 tips. "
                f"Data: pengeluaran {exp}, pemasukan {inc}, saldo {bal}, "
                f"progress {(bal/goal_amt*100 if goal_amt else 0):.1f}%."
            )
            st.text(rep)
            st.download_button("⬇️ TXT",
                               io.BytesIO(rep.encode()),
                               file_name="finance_report.txt")

else:
    # ---------- Landing ----------
    st.title("AI Finance Studio")
    st.markdown(
        """
        **AI Finance Studio** adalah dashboard interaktif berbasis _Streamlit_ yang membantu Anda:
        1. **Mengimpor** data transaksi (_CSV/Excel_) dari aplikasi **MyMoney**.  
        2. **Menganalisis** arus kas dengan grafik kategori, tren harian, dan deteksi pengeluaran “boncos”.  
        3. **Mendapat insight AI** untuk ringkasan, tips budgeting, dan prediksi “saldo habis”.  
        4. **Bertanya langsung** lewat _Chatbot_ tentang transaksi & strategi keuangan pribadi.  
        5. **Melacak target tabungan** dengan progress bar, serta mengekspor laporan TXT.
        ---
        | Langkah | Deskripsi |
        |---------|-----------|
        | **1. Ekspor data** | Di MyMoney → *Export → CSV* (atau Excel). Pastikan kolom utama: `TIME`, `TYPE`, `AMOUNT`, `CATEGORY`, `ACCOUNT`. |
        | **2. Unggah file** | Tekan **📂 Unggah CSV/Excel** di sidebar. |
        | **3. Input saldo** | Isi **💵 Saldo Saat Ini** agar analisis runway akurat. |
        | **4. Jelajahi tab** | `🏠 Dashboard • 💡 Insight • 🤝 Advisor • 💬 Chatbot • 📝 Report` |
        ---
        ### Format CSV
        ```csv
        TIME,TYPE,AMOUNT,CATEGORY,ACCOUNT
        2025-06-01 08:00,(-) Expense,50000,Food,Cash
        2025-06-02 09:30,(+) Income,12000000,Salary,Payroll
        """
    )
    st.info("Unggah file + isi saldo untuk mulai.")
    st.markdown("""Program ini dibuat untuk keperluan pribadi, di mana digunakan untuk memantau dan terintegrasi oleh **AI** dari data `.csv` yang tercatat oleh aplikasi **MyMoney**.""")
