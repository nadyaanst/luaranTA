import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# ===================== CONFIG DASHBOARD =====================

st.set_page_config(
    page_title="Dashboard Financial Distress",
    page_icon="📊",
    layout="wide",
)


# ===================== MODEL ANN ============================

def build_pytorch_model(hidden_sizes, input_size=6):
    """
    Arsitektur ANN:
    input 6 rasio -> hidden layer [12] -> 1 output (sigmoid)
    """
    layers = []
    in_size = input_size
    for h in hidden_sizes:
        layers.append(nn.Linear(in_size, h))
        layers.append(nn.ReLU())
        in_size = h
    layers.append(nn.Linear(in_size, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


@st.cache_resource
def load_model():
    # Layer terbaik: [12]
    hidden_layers = [12]

    # Asumsi utama: yang disimpan adalah state_dict
    try:
        model = build_pytorch_model(hidden_layers, input_size=6)
        state_dict = torch.load("model/model_terbaik_ann.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        # Jika ternyata yang disimpan full model (torch.save(best_model, ...))
        model = torch.load("model/model_terbaik_ann.pt", map_location="cpu")
        model.eval()
        return model


model = load_model()


# ===================== HALAMAN: HOME ========================

def page_home():
    st.markdown("## 📊 Dashboard Financial Distress")
    st.markdown(
        """
    ### Dashboard ini digunakan untuk mengestimasi kemungkinan perusahaan mengalami financial distress

    Aplikasi ini dikembangkan sebagai luaran **Tugas Akhir** untuk membantu
    analis, manajer keuangan, dan investor dalam mengidentifikasi **risiko financial distress**
    pada perusahaan sektor *consumer cyclicals* di Indonesia.

    Model yang digunakan berbasis **Artificial Neural Network (ANN)**  
    dengan input beberapa rasio keuangan yang mewakili aspek **likuiditas** dan **profitabilitas**.
    """
    )

    # --- Hero section: gambar + teks ---
    col1, col2 = st.columns([1.3, 1])

    with col1:
        # Gambar dari Unsplash (langsung link image)
        st.image(
            "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?auto=format&fit=crop&w=1200&q=80",
            use_column_width=True,
        )

    with col2:
        st.markdown("#### 🎯 Tujuan Sistem")
        st.markdown(
            """
        - Mengklasifikasikan perusahaan ke dalam kategori  
          **Non-Financial Distress** atau **Financial Distress**.
        - Menyediakan alat bantu cepat untuk **screening awal** kondisi keuangan.
        - Mendukung pengambilan keputusan bagi **investor, kreditor, dan manajemen**.
        """
        )

        st.markdown("#### 🧠 Ringkasan Model")
        st.markdown(
            """
        - Algoritma: **Artificial Neural Network (Multilayer Perceptron)**  
        - Arsitektur terbaik: **6–12–1 neuron**  
        - Input: Current Ratio, Quick Ratio, Cash Ratio, ROA, ROE, NPM  
        - Output: probabilitas perusahaan mengalami *financial distress*
        """
        )

    st.markdown("---")

    # --- Validasi Performa Model ---
    st.markdown("### 🏆 Validasi Performa Model")

    # Nilai sesuai laporanmu
    acc = 0.8918   # 89,18%
    auc = 0.923    # 0,923
    f1 = 0.90      # 0,90

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Akurasi", f"{acc*100:.2f} %", help="Accuracy pada data uji")
        st.caption("✅ Kemampuan model mengklasifikasikan perusahaan dengan benar")

    with c2:
        st.metric("AUC", f"{auc:.3f}", help="Area Under Curve (ROC)")
        st.caption("📈 Kemampuan model membedakan distress vs non-distress")

    with c3:
        st.metric("F1-Score", f"{f1:.2f}", help="Harmonic mean precision & recall")
        st.caption("⚖️ Keseimbangan antara kesalahan tipe I dan tipe II")

    st.info(
        "Prediksi yang dihasilkan dashboard ini bergantung pada kualitas dan ketepatan "
        "data laporan keuangan yang diinput. Semakin akurat data (total aset, ekuitas, "
        "laba bersih, penjualan, dan rasio keuangan), semakin representatif hasil prediksi "
        "terhadap kondisi perusahaan sebenarnya."
    )


# ===================== HALAMAN: KETENTUAN ===================

def page_terms():
    st.markdown("## 📑 Ketentuan & Cara Menggunakan Dashboard")
    st.write(
        """
Ikuti panduan berikut untuk menggunakan dashboard prediksi financial distress ini secara efektif:
"""
    )

    st.markdown(
        """
1. **Memasukkan Data**  
   - Jika sudah memiliki nilai rasio keuangan (CR, QR, Cash Ratio, ROA, ROE, NPM), langsung gunakan halaman **Prediksi**.  
   - Jika hanya memiliki data laporan keuangan mentah, gunakan terlebih dahulu halaman **Kalkulator Rasio**.

2. **Validitas Data**  
   - Pastikan data bersumber dari laporan keuangan resmi (annual report atau laporan kuartalan).  
   - Gunakan pemisah desimal berupa koma (contoh: `1.5` dalam Excel ➜ tulis `1,5` di sini dengan titik sebagai pemisah ribuan bila perlu).

3. **Hasil Prediksi**  
   Setelah menekan tombol **Test Prediction**, dashboard akan menampilkan:
   - Probabilitas perusahaan mengalami **financial distress**  
   - Kategori: **Non-Financial Distress** (aman) atau **Financial Distress** (berisiko)  
   - Rekomendasi singkat berdasarkan pola rasio keuangan yang diinput.

4. **Interpretasi**  
   - Model ini bersifat **alat bantu**, bukan satu-satunya dasar keputusan.  
   - Kombinasikan dengan analisis lain: tren industri, strategi bisnis, tata kelola, dan kondisi makroekonomi.

5. **Batasan Model**  
   - Model dilatih menggunakan data historis perusahaan sektor *consumer cyclicals* di Indonesia.  
   - Penggunaan pada sektor lain atau negara lain memerlukan kajian tambahan.
"""
    )


# ===================== HALAMAN: KALKULATOR RASIO ============

def page_rasio():
    st.markdown("## 📌 Kalkulator Rasio Keuangan")
    st.write(
        "Masukkan data laporan keuangan mentah. "
        "Format angka Indonesia diperbolehkan, misalnya: `1.234.567,00`. "
        "Nilai boleh positif maupun negatif."
    )

    # Konversi format Indonesia ("1.234.567,89") -> float Python
    def to_float(value: str) -> float:
        if value is None or value == "":
            return 0.0
        value = str(value).replace(".", "").replace(",", ".")
        try:
            return float(value)
        except ValueError:
            return 0.0

    col1, col2 = st.columns(2)

    with col1:
        aset_lancar = st.text_input("Aset Lancar")
        utang_lancar = st.text_input("Utang Lancar")
        inventaris = st.text_input("Inventaris / Persediaan")
        kas = st.text_input("Kas & Setara Kas")

    with col2:
        laba_bersih = st.text_input("Laba Bersih (Net Income)")
        total_aset = st.text_input("Total Aset")
        ekuitas = st.text_input("Ekuitas")
        penjualan = st.text_input("Penjualan / Pendapatan (Sales)")

    if st.button("Hitung Rasio Keuangan"):
        # Ubah ke float
        aset_lancar_f = to_float(aset_lancar)
        utang_lancar_f = to_float(utang_lancar)
        inventaris_f = to_float(inventaris)
        kas_f = to_float(kas)
        laba_bersih_f = to_float(laba_bersih)
        total_aset_f = to_float(total_aset)
        ekuitas_f = to_float(ekuitas)
        penjualan_f = to_float(penjualan)

        # Kewajiban lancar = utang lancar (sesuai permintaanmu)
        kewajiban_lancar_f = utang_lancar_f

        # Hitung rasio (jaga-jaga kalau penyebut = 0)
        current_ratio = (
            aset_lancar_f / utang_lancar_f if utang_lancar_f != 0 else 0.0
        )
        quick_ratio = (
            (aset_lancar_f - inventaris_f) / kewajiban_lancar_f
            if kewajiban_lancar_f != 0
            else 0.0
        )
        cash_ratio = (
            kas_f / kewajiban_lancar_f if kewajiban_lancar_f != 0 else 0.0
        )
        roa = (
            laba_bersih_f / total_aset_f if total_aset_f != 0 else 0.0
        )
        roe = (
            laba_bersih_f / ekuitas_f if ekuitas_f != 0 else 0.0
        )
        npm = (
            laba_bersih_f / penjualan_f if penjualan_f != 0 else 0.0
        )

        st.markdown("---")
        st.subheader("📍 Hasil Perhitungan Rasio (3 angka di belakang koma)")

        # helper: format Indonesia (koma sebagai desimal)
        def fmt_id(x: float) -> str:
            return f"{x:.3f}".replace(".", ",")

        st.write(f"**Current Ratio (CR)** = Aset Lancar / Utang Lancar = `{fmt_id(current_ratio)}`")
        st.write(
            f"**Quick Ratio (QR)** = (Aset Lancar − Inventaris) / Kewajiban Lancar "
            f"(= Utang Lancar) = `{fmt_id(quick_ratio)}`"
        )
        st.write(
            f"**Cash Ratio** = Kas & Setara Kas / Kewajiban Lancar "
            f"(= Utang Lancar) = `{fmt_id(cash_ratio)}`"
        )
        st.write(f"**ROA** = Laba Bersih / Total Aset = `{fmt_id(roa)}`")
        st.write(f"**ROE** = Laba Bersih / Ekuitas = `{fmt_id(roe)}`")
        st.write(f"**NPM** = Laba Bersih / Penjualan = `{fmt_id(npm)}`")

        # Simpan ke session_state untuk halaman Prediksi (pakai float biasa)
        st.session_state["rasio_values"] = {
            "cr": float(round(current_ratio, 3)),
            "qr": float(round(quick_ratio, 3)),
            "cash_ratio": float(round(cash_ratio, 3)),
            "roa": float(round(roa, 3)),
            "roe": float(round(roe, 3)),
            "npm": float(round(npm, 3)),
        }

        st.success(
            "Rasio berhasil dihitung ✔ Nilai sudah disimpan dan akan otomatis muncul di halaman **Prediksi**."
        )


# ===================== FUNGSI BANTU REKOMENDASI =============

def analisis_rasio(cr, qr, cash_ratio, roa, roe, npm):
    """
    Analisis kasar rasio terhadap benchmark sederhana untuk memberi rekomendasi.
    Tidak mengubah prediksi model, hanya sebagai interpretasi tambahan.
    """
    masalah = []

    # Benchmark sangat umum (boleh disesuaikan lagi di TA)
    if cr < 1.5:
        masalah.append(("Current Ratio (CR)", "Likuiditas jangka pendek relatif lemah, aset lancar kurang memadai untuk menutup kewajiban lancar."))
    if qr < 1.0:
        masalah.append(("Quick Ratio (QR)", "Ketergantungan pada persediaan cukup tinggi, kemampuan membayar kewajiban tanpa menjual persediaan masih terbatas."))
    if cash_ratio < 0.2:
        masalah.append(("Cash Ratio", "Cadangan kas relatif tipis dibanding kewajiban lancar. Perusahaan sensitif terhadap gangguan arus kas."))
    if roa < 0.05:
        masalah.append(("ROA", "Efisiensi pemanfaatan aset untuk menghasilkan laba masih rendah."))
    if roe < 0.10:
        masalah.append(("ROE", "Pengembalian kepada pemegang saham belum optimal, struktur modal dan profitabilitas perlu dievaluasi."))
    if npm < 0.05:
        masalah.append(("NPM", "Margin laba bersih tipis, perusahaan rentan terhadap kenaikan biaya atau penurunan penjualan."))

    if masalah:
        utama = [m[0] for m in masalah[:2]]
    else:
        utama = []

    return masalah, utama


# ===================== HALAMAN: PREDIKSI ====================

def page_prediction():
    st.markdown("## 🔍 Prediksi Financial Distress")
    st.write(
        "Masukkan rasio keuangan perusahaan, lalu tekan tombol **Test Prediction**. "
        "Jika sebelumnya sudah menggunakan **Kalkulator Rasio**, nilai akan terisi otomatis."
    )

    # Ambil default dari session_state kalau ada
    rasio_default = st.session_state.get("rasio_values", {})
    def_cr = rasio_default.get("cr", 0.0)
    def_qr = rasio_default.get("qr", 0.0)
    def_cash = rasio_default.get("cash_ratio", 0.0)
    def_roa = rasio_default.get("roa", 0.0)
    def_roe = rasio_default.get("roe", 0.0)
    def_npm = rasio_default.get("npm", 0.0)

    col1, col2 = st.columns(2)

    with col1:
        cr = st.number_input(
            "Current Ratio (CR)",
            step=0.01,
            value=float(def_cr),
        )
        qr = st.number_input(
            "Quick Ratio (QR)",
            step=0.01,
            value=float(def_qr),
        )
        cash_ratio = st.number_input(
            "Cash Ratio",
            step=0.01,
            value=float(def_cash),
        )

    with col2:
        roa = st.number_input(
            "Return on Assets (ROA)",
            step=0.0001,
            format="%.4f",
            value=float(def_roa),
        )
        roe = st.number_input(
            "Return on Equity (ROE)",
            step=0.0001,
            format="%.4f",
            value=float(def_roe),
        )
        npm = st.number_input(
            "Net Profit Margin (NPM)",
            step=0.0001,
            format="%.4f",
            value=float(def_npm),
        )

    st.markdown("")
    if st.button("Test Prediction"):
        features = np.array([[cr, qr, cash_ratio, roa, roe, npm]], dtype=np.float32)
        tensor = torch.from_numpy(features)

        with torch.no_grad():
            prob = model(tensor).item()

        threshold = 0.5  # sesuaikan kalau di TA pakai threshold lain
        label = "Financial Distress" if prob >= threshold else "Non-Financial Distress"

        st.markdown("---")
        st.subheader("Hasil Prediksi")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Probabilitas Financial Distress", f"{prob:.4f}")
        with c2:
            st.metric("Klasifikasi", label)

        # Analisis rasio untuk rekomendasi
        masalah, utama = analisis_rasio(cr, qr, cash_ratio, roa, roe, npm)

        st.markdown("### 📌 Interpretasi Rasio Keuangan")
        if masalah:
            st.write("Beberapa rasio yang perlu mendapat perhatian khusus:")
            for nama, penjelasan in masalah:
                st.write(f"- **{nama}** – {penjelasan}")
        else:
            st.write(
                "Secara umum, kombinasi rasio yang diinput berada pada tingkat yang relatif sehat "
                "berdasarkan benchmark sederhana yang digunakan."
            )

        st.markdown("### 🏢 Rekomendasi untuk Perusahaan")
        if label == "Financial Distress":
            st.write(
                "Perusahaan berada pada kategori **Financial Distress**. "
                "Beberapa langkah yang dapat dipertimbangkan:"
            )
            st.write("- Menyusun kembali struktur hutang untuk mengurangi tekanan likuiditas jangka pendek.")
            st.write("- Meninjau efisiensi operasional guna meningkatkan margin laba bersih.")
            st.write("- Menguatkan posisi kas dan aset lancar, misalnya dengan mempercepat penagihan piutang.")
        else:
            st.write(
                "Perusahaan berada pada kategori **Non-Financial Distress**, "
                "namun tetap perlu menjaga kualitas kinerja keuangan:"
            )
            st.write("- Mempertahankan likuiditas pada tingkat yang nyaman agar tahan terhadap shock jangka pendek.")
            st.write("- Terus memonitor profitabilitas dan margin laba agar tidak tergerus biaya yang meningkat.")
            st.write("- Menggunakan kelebihan kas secara bijak, misalnya untuk investasi produktif atau pelunasan hutang mahal.")

        if utama:
            st.write("")
            st.write(
                f"Rasio yang **paling berpengaruh dalam penilaian risiko saat ini** "
                f"(berdasarkan deviasi dari kisaran sehat) adalah: {', '.join(utama)}."
            )

        st.markdown("### 💡 Catatan untuk Investor")
        if label == "Financial Distress":
            st.write(
                "- Pertimbangkan tingkat risiko yang lebih tinggi sebelum menambah eksposur pada emiten ini.\n"
                "- Lakukan analisis lanjutan terhadap rencana restrukturisasi, dukungan pemegang saham mayoritas, "
                "serta prospek industri.\n"
                "- Diversifikasikan portofolio untuk menghindari konsentrasi pada perusahaan dengan profil risiko serupa."
            )
        else:
            st.write(
                "- Perusahaan saat ini tampak relatif sehat dari sisi likuiditas dan profitabilitas, "
                "namun tetap perlu dipantau secara berkala.\n"
                "- Tinjau juga faktor non-keuangan seperti tata kelola, kualitas manajemen, dan strategi jangka panjang.\n"
                "- Gunakan hasil prediksi ini sebagai salah satu komponen dalam analisis fundamental yang lebih menyeluruh."
            )


# ===================== SIDE BAR NAVIGATION ==================

st.sidebar.title("📊 Dashboard Financial Distress")
page = st.sidebar.selectbox(
    "Navigasi",
    ("Home", "Ketentuan", "Kalkulator Rasio", "Prediksi"),
)

if page == "Home":
    page_home()
elif page == "Ketentuan":
    page_terms()
elif page == "Kalkulator Rasio":
    page_rasio()
elif page == "Prediksi":
    page_prediction()