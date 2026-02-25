import os
import time
import io
import wave
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import feedparser
import requests
from openai import OpenAI

# --------------------------------
# Secrets / Env
# --------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --------------------------------
# UI
# --------------------------------
st.set_page_config(page_title="BIST100 Banka Takip Agent", layout="wide")
st.title("🏦 BIST100 Banka Takip Agent (Alarm + Telegram + Akıllı Haber)")

# --------------------------------
# Hisseler
# --------------------------------
BANKA_TICKERS = {
    "Akbank": "AKBNK.IS",
    "Garanti BBVA": "GARAN.IS",
    "İş Bankası (C)": "ISCTR.IS",
    "Yapı Kredi": "YKBNK.IS",
    "Halkbank": "HALKB.IS",
    "Vakıfbank": "VAKBN.IS",
    "TSKB": "TSKB.IS",
    "Şekerbank": "SKBNK.IS",
}
NEWS_QUERY = {k: f"{v.replace('.IS','')} hisse" for k, v in BANKA_TICKERS.items()}

# --------------------------------
# İndikatörler
# --------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def pct_change(prices: pd.Series, bars: int) -> float:
    s = prices.dropna()
    if len(s) < bars + 1:
        return np.nan
    return (s.iloc[-1] / s.iloc[-(bars + 1)] - 1) * 100

# --------------------------------
# Beep
# --------------------------------
def make_beep_wav(duration_sec=0.25, freq=880, rate=44100):
    n_samples = int(duration_sec * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        for i in range(n_samples):
            t = i / rate
            val = int(32767 * 0.3 * math.sin(2 * math.pi * freq * t))
            wf.writeframesraw(val.to_bytes(2, byteorder="little", signed=True))
    return buf.getvalue()

BEEP_WAV = make_beep_wav()

# --------------------------------
# Veri
# --------------------------------
@st.cache_data(ttl=60)
def fetch_history(tickers, period="3mo", interval="1d"):
    return yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True, threads=True)

def get_close(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        return df[(ticker, "Close")].dropna()
    return df["Close"].dropna()

# --------------------------------
# Haber (Google News RSS)
# --------------------------------
def google_news_rss_url(q: str) -> str:
    return "https://news.google.com/rss/search?q=" + requests.utils.quote(q) + "&hl=tr&gl=TR&ceid=TR:tr"

@st.cache_data(ttl=300)
def fetch_news_items(query: str, max_items=10):
    feed = feedparser.parse(google_news_rss_url(query))
    items = []
    for e in feed.entries[:max_items]:
        items.append({
            "title": getattr(e, "title", "").strip(),
            "link": getattr(e, "link", "").strip(),
            "published": getattr(e, "published", "").strip(),
            "source": getattr(getattr(e, "source", None), "title", "") if hasattr(e, "source") else "",
        })
    return items

@st.cache_data(ttl=300)
def summarize_with_openai(company_name: str, items: list, max_bullets=5):
    if not client:
        return {"summary": "OPENAI_API_KEY yok: Akıllı özet kapalı.", "impact": "Bilinmiyor"}

    headlines = [it["title"] for it in items if it.get("title")]
    prompt = f"""
Şirket: {company_name}
Aşağıdaki haber başlıklarına göre Türkçe çok kısa bir özet çıkar:
- 3-5 madde
- Tekrar etme
- Yatırım tavsiyesi verme
Ayrıca tek satır 'Etki: Pozitif/Negatif/Nötr' yaz.

Başlıklar:
{chr(10).join([f"- {h}" for h in headlines[:12]])}
"""
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        text={"format": {"type": "text"}}
    )
    text = (resp.output_text or "").strip()

    impact = "Bilinmiyor"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = []
    for ln in lines:
        if ln.lower().startswith("etki"):
            impact = ln.split(":", 1)[-1].strip() or impact
        elif ln.startswith("-") or ln.startswith("•"):
            bullets.append(ln.lstrip("-• ").strip())
        else:
            if len(bullets) < max_bullets:
                bullets.append(ln)

    bullets = bullets[:max_bullets]
    summary = "\n".join([f"• {b}" for b in bullets]) if bullets else text[:400]
    return {"summary": summary, "impact": impact}

# --------------------------------
# Telegram
# --------------------------------
def telegram_send(message: str) -> bool:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, timeout=10, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        return r.status_code == 200
    except Exception:
        return False

# --------------------------------
# State + Alarm motoru (cooldown)
# --------------------------------
def ensure_state():
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "last_hist_sign" not in st.session_state:
        st.session_state.last_hist_sign = {}
    if "last_sent" not in st.session_state:
        # cooldown için: {(hisse, alarm_type): datetime}
        st.session_state.last_sent = {}

def can_send(name: str, alarm_type: str, cooldown_min: int) -> bool:
    key = (name, alarm_type)
    last = st.session_state.last_sent.get(key)
    if last is None:
        return True
    return datetime.now() - last >= timedelta(minutes=cooldown_min)

def mark_sent(name: str, alarm_type: str):
    st.session_state.last_sent[(name, alarm_type)] = datetime.now()

def add_alert(level, name, alarm_type, message, details: dict, send_telegram=False, cooldown_min=30):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # UI listesine her seferinde ekleyebiliriz; ama telegram spamini cooldown ile kesiyoruz
    st.session_state.alerts.insert(0, {"zaman": ts, "seviye": level, "hisse": name, "mesaj": message})
    st.session_state.alerts = st.session_state.alerts[:300]

    if send_telegram and can_send(name, alarm_type, cooldown_min):
        # zengin telegram mesajı
        price = details.get("price")
        ch1 = details.get("ch1")
        ch5 = details.get("ch5")
        rsi14 = details.get("rsi14")
        hist = details.get("hist")

        msg = (
            f"🚨 {level} - {name}\n"
            f"{message}\n\n"
            f"Son: {price:.2f}\n"
            f"1 bar %: {ch1:+.2f}\n"
            f"~5 bar %: {ch5:+.2f}\n"
            f"RSI(14): {rsi14:.1f}\n"
            f"MACD Hist: {hist:+.4f}\n"
            f"Zaman: {ts}"
        )
        telegram_send(msg)
        mark_sent(name, alarm_type)

ensure_state()

# --------------------------------
# Kontroller
# --------------------------------
c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
with c1:
    selected = st.multiselect("Takip edilecek banka hisseleri", list(BANKA_TICKERS.keys()), default=list(BANKA_TICKERS.keys()))
with c2:
    interval = st.selectbox("Zaman aralığı", ["1d", "1h", "15m"], index=0)
with c3:
    period = st.selectbox("Veri dönemi", ["1mo", "3mo", "6mo", "1y"], index=1)
with c4:
    auto_refresh = st.toggle("Otomatik yenile", value=True)

st.divider()

st.subheader("🔔 Alarm Ayarları")
a1, a2, a3, a4, a5, a6 = st.columns([1, 1, 1, 1, 1, 1])
with a1:
    alarm_pct_1 = st.number_input("1 bar % eşiği", value=3.0, step=0.5)
with a2:
    alarm_pct_5 = st.number_input("~5 bar % eşiği", value=7.0, step=0.5)
with a3:
    alarm_rsi_low = st.number_input("RSI alt", value=30.0, step=1.0)
with a4:
    alarm_rsi_high = st.number_input("RSI üst", value=70.0, step=1.0)
with a5:
    alarm_macd_flip = st.toggle("MACD hist yön değişimi", value=True)
with a6:
    cooldown_min = st.number_input("Telegram cooldown (dk)", value=30, step=5, min_value=5)

play_sound = st.toggle("Alarm sesi", value=True)
send_tg = st.toggle("Telegram bildirimi", value=True)

st.caption("Telegram için: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID ortam değişkenleri gerekli.")
st.divider()

if not selected:
    st.warning("En az 1 hisse seç.")
    st.stop()

tickers = [BANKA_TICKERS[n] for n in selected]
data = fetch_history(tickers, period=period, interval=interval)

rows = []
series_map = {}
sound_needed = False

# --------------------------------
# Hesapla + Alarm kontrol
# --------------------------------
for name in selected:
    t = BANKA_TICKERS[name]
    close = get_close(data, t)
    series_map[name] = close

    last = close.iloc[-1] if len(close) else np.nan
    ch1 = pct_change(close, 1)
    ch5 = pct_change(close, 5)

    rsi14 = rsi(close, 14).iloc[-1] if len(close) >= 20 else np.nan
    _, _, hist = macd(close)
    hist_last = hist.iloc[-1] if len(close) >= 30 else np.nan

    rows.append({
        "Hisse": name,
        "Ticker": t,
        "Son": float(last) if pd.notna(last) else np.nan,
        "1 bar %": float(ch1) if pd.notna(ch1) else np.nan,
        "~5 bar %": float(ch5) if pd.notna(ch5) else np.nan,
        "RSI(14)": float(rsi14) if pd.notna(rsi14) else np.nan,
        "MACD Hist": float(hist_last) if pd.notna(hist_last) else np.nan,
    })

    details = {
        "price": float(last) if pd.notna(last) else np.nan,
        "ch1": float(ch1) if pd.notna(ch1) else np.nan,
        "ch5": float(ch5) if pd.notna(ch5) else np.nan,
        "rsi14": float(rsi14) if pd.notna(rsi14) else np.nan,
        "hist": float(hist_last) if pd.notna(hist_last) else np.nan,
    }

    fired = False

    if pd.notna(ch1) and abs(ch1) >= alarm_pct_1:
        add_alert("YÜKSEK", name, "pct_1", f"1 bar değişim %{ch1:.2f} (eşik %{alarm_pct_1:.2f})",
                  details, send_telegram=send_tg, cooldown_min=cooldown_min)
        fired = True

    if pd.notna(ch5) and abs(ch5) >= alarm_pct_5:
        add_alert("ORTA", name, "pct_5", f"~5 bar değişim %{ch5:.2f} (eşik %{alarm_pct_5:.2f})",
                  details, send_telegram=send_tg, cooldown_min=cooldown_min)
        fired = True

    if pd.notna(rsi14) and rsi14 <= alarm_rsi_low:
        add_alert("ORTA", name, "rsi_low", f"RSI(14) {rsi14:.1f} ≤ {alarm_rsi_low:.1f}",
                  details, send_telegram=send_tg, cooldown_min=cooldown_min)
        fired = True

    if pd.notna(rsi14) and rsi14 >= alarm_rsi_high:
        add_alert("ORTA", name, "rsi_high", f"RSI(14) {rsi14:.1f} ≥ {alarm_rsi_high:.1f}",
                  details, send_telegram=send_tg, cooldown_min=cooldown_min)
        fired = True

    if alarm_macd_flip and pd.notna(hist_last) and len(hist.dropna()) >= 2:
        sign = 1 if hist_last > 0 else (-1 if hist_last < 0 else 0)
        prev = st.session_state.last_hist_sign.get(name, None)
        if prev is not None and sign != 0 and prev != 0 and sign != prev:
            add_alert("DÜŞÜK", name, "macd_flip", "MACD histogram yön değiştirdi (pozitif↔negatif)",
                      details, send_telegram=send_tg, cooldown_min=cooldown_min)
            fired = True
        st.session_state.last_hist_sign[name] = sign

    if fired:
        sound_needed = True

# --------------------------------
# Tablo + alarm merkezi
# --------------------------------
df = pd.DataFrame(rows)

st.subheader("📋 Özet Tablo")
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("🚨 Alarm Merkezi")
x1, x2 = st.columns([1, 1])
with x1:
    if st.button("Alarmları temizle"):
        st.session_state.alerts = []
        st.session_state.last_sent = {}
with x2:
    st.caption("Telegram cooldown açık: aynı alarm tipi aynı hisse için belirlediğin dakika içinde tekrar gönderilmez.")

if st.session_state.alerts:
    st.dataframe(pd.DataFrame(st.session_state.alerts), use_container_width=True, hide_index=True)
else:
    st.info("Henüz alarm yok.")

if play_sound and sound_needed:
    st.audio(BEEP_WAV, format="audio/wav")

st.divider()

# --------------------------------
# Grafikler
# --------------------------------
st.subheader("📈 Grafikler")
g1, g2 = st.columns([1, 1])

with g1:
    for name in selected:
        s = series_map[name].dropna()
        if len(s) < 5:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.iloc[-120:], mode="lines", name=name))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), title=name)
        st.plotly_chart(fig, use_container_width=True)

with g2:
    pick = st.selectbox("Detay grafik", selected, index=0)
    s = series_map[pick].dropna()
    if len(s) >= 5:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name="Fiyat"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), title=f"{pick} - Fiyat")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------
# Haberler (Akıllı Özet)
# --------------------------------
st.subheader("📰 Haberler (Akıllı Özet)")
tab1, tab2 = st.tabs(["Özet", "Detay Linkler"])

with tab1:
    for name in selected:
        q = NEWS_QUERY.get(name, name + " hisse")
        items = fetch_news_items(q, max_items=10)
        st.markdown(f"**{name}**")
        if not items:
            st.caption("Haber bulunamadı.")
            st.write("")
            continue
        smart = summarize_with_openai(name, items)
        st.caption(f"Etki: {smart['impact']}")
        st.write(smart["summary"])
        st.write("")

with tab2:
    name = st.selectbox("Haber detayı", selected, index=0, key="news_pick")
    q = NEWS_QUERY.get(name, name + " hisse")
    items = fetch_news_items(q, max_items=12)
    if not items:
        st.info("Haber bulunamadı.")
    else:
        for it in items:
            title = it["title"]
            link = it["link"]
            published = it.get("published", "")
            src = it.get("source", "")
            label = f"{src} — {published}".strip(" —")
            st.markdown(f"- [{title}]({link})")
            if label:
                st.caption(label)

# --------------------------------
# Auto refresh
# --------------------------------
if auto_refresh:
    time.sleep(10)
    st.rerun()
