"""Máy chủ nhỏ: nhận link video/ảnh -> trả về danh sách từ vựng."""
import os, re, json, base64, glob, subprocess, tempfile
import asyncio, hashlib, pathlib
import httpx, yt_dlp, edge_tts, lameenc
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

APP_TOKEN = os.getenv("APP_TOKEN", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
TEXT_MODEL = os.getenv("TEXT_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
MAX_SECONDS = int(os.getenv("MAX_SECONDS", "900"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- Giọng đọc tự nhiên (Microsoft Edge) để tạo file âm thanh cả bài ----------
VOICES = {
    "cs": "cs-CZ-VlastaNeural", "vi": "vi-VN-HoaiMyNeural", "en": "en-US-AriaNeural",
    "ja": "ja-JP-NanamiNeural", "de": "de-DE-KatjaNeural", "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural", "ko": "ko-KR-SunHiNeural", "zh": "zh-CN-XiaoxiaoNeural",
    "ru": "ru-RU-SvetlanaNeural", "pl": "pl-PL-ZofiaNeural", "sk": "sk-SK-ViktoriaNeural",
}
BYTES_PER_SEC = 6000  # mp3 48 kbps, 24 kHz, mono (định dạng giọng Edge)
FRAME = 144           # 1 khung mp3 = 24 ms
CACHE = pathlib.Path("/tmp/tts")
CACHE.mkdir(parents=True, exist_ok=True)


def _silence_frames():
    e = lameenc.Encoder()
    e.set_bit_rate(48); e.set_in_sample_rate(24000); e.set_channels(1); e.set_quality(2)
    b = e.encode(b"\x00\x00" * 24000 * 5) + e.flush()
    return b[FRAME * 4: FRAME * 4 + FRAME * 150]  # lấy các khung im lặng ở giữa


SIL = _silence_frames()


def silence(ms):
    n = max(1, round(ms / 24))
    out = b""
    while n > 0:
        k = min(n, len(SIL) // FRAME)
        out += SIL[: k * FRAME]
        n -= k
    return out


sem = asyncio.Semaphore(6)


async def synth(text, voice):
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()
    p = CACHE / f"{key}.mp3"
    if p.exists():
        return p.read_bytes()
    async with sem:
        for attempt in range(3):
            try:
                await edge_tts.Communicate(text, voice).save(str(p))
                return p.read_bytes()
            except Exception:
                await asyncio.sleep(1 + attempt)
    return b""


class Item(BaseModel):
    w: str
    m: str
    ex: str = ""


class LessonReq(BaseModel):
    items: list[Item]
    a: str
    b: str = "vi"
    rep: int = 1
    say: str = "both"   # both | a | b
    gap: float = 1.0
    ex: bool = False


@app.post("/api/tts-lesson")
async def tts_lesson(req: LessonReq, x_token: str = Header(default="")):
    if APP_TOKEN and x_token != APP_TOKEN:
        raise HTTPException(401, "Sai mật khẩu máy chủ")
    if len(req.items) > 400:
        raise HTTPException(400, "Bài quá dài (tối đa 400 từ)")
    va, vb = VOICES.get(req.a, VOICES["en"]), VOICES.get(req.b, VOICES["vi"])
    want = set()
    for it in req.items:
        if req.say != "b": want.add((it.w, va))
        if req.say != "a": want.add((it.m, vb))
        if req.ex and it.ex and req.say != "b": want.add((it.ex, va))
    texts = list(want)
    audio = dict(zip(texts, await asyncio.gather(*(synth(t, v) for t, v in texts))))
    out, marks, t = bytearray(), [], 0.0

    def add(b, secs=None):
        nonlocal t
        out.extend(b)
        t += secs if secs is not None else len(b) / BYTES_PER_SEC

    for i, it in enumerate(req.items):
        marks.append({"i": i, "t": round(t, 2)})
        for r in range(max(1, min(req.rep, 3))):
            if req.say != "b": add(audio[(it.w, va)])
            if req.say == "both": add(silence(400), 0.4)
            if req.say != "a": add(audio[(it.m, vb)])
            if req.ex and it.ex and req.say != "b" and r == req.rep - 1:
                add(silence(400), 0.4); add(audio[(it.ex, va)])
            g = max(0.05, req.gap)
            add(silence(g * 1000), g)
    return {"audio": base64.b64encode(bytes(out)).decode(), "marks": marks, "dur": round(t, 2)}


class Req(BaseModel):
    url: str
    lang_name: str  # ví dụ "tiếng Nhật"


def sh(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def transcribe(audio_path):
    if not GROQ_KEY:
        return ""
    with open(audio_path, "rb") as f:
        r = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": ("a.mp3", f, "audio/mpeg")},
            data={"model": "whisper-large-v3", "response_format": "text"},
            timeout=180,
        )
    r.raise_for_status()
    return r.text.strip()


def img_block(data: bytes, mime="image/jpeg"):
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def extract_vocab(lang_name, transcript, images):
    prompt = (
        f"Đây là nội dung (lời nói và/hoặc hình ảnh) của một video hoặc ảnh dạy {lang_name}. "
        f"Hãy liệt kê TẤT CẢ từ vựng/câu mẫu được dạy. Với mỗi mục trả về: "
        f'"w" = từ/câu bằng {lang_name} (đúng chính tả), "m" = nghĩa tiếng Việt, '
        f'"ex" = câu ví dụ bằng {lang_name} nếu nội dung có, không có thì để chuỗi rỗng. '
        "Nếu nội dung không nói rõ nghĩa tiếng Việt thì tự dịch. Bỏ qua lời chào hỏi, quảng cáo. "
        "Chỉ trả về một mảng JSON, không giải thích."
    )
    images = images[:5]  # Groq nhận tối đa 5 ảnh mỗi lần
    text = prompt + (f"\n\nLỜI NÓI (phiên âm tự động, có thể sai):\n{transcript}" if transcript else "")
    content = [{"type": "text", "text": text}] + images
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={"model": VISION_MODEL if images else TEXT_MODEL, "temperature": 0.2, "max_tokens": 4000,
              "messages": [{"role": "user", "content": content if images else text}]},
        timeout=120,
    )
    if r.status_code >= 400:
        raise HTTPException(502, f"Lỗi AI: {r.text[:200]}")
    out = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return []
    items = json.loads(m.group(0))
    return [{"w": i["w"].strip(), "m": i["m"].strip(), "ex": (i.get("ex") or "").strip()}
            for i in items if i.get("w") and i.get("m")]


def looks_like_image(url):
    if re.search(r"\.(jpe?g|png|webp)(\?|$)", url, re.I):
        return True
    try:
        h = httpx.head(url, follow_redirects=True, timeout=10)
        return h.headers.get("content-type", "").startswith("image/")
    except Exception:
        return False


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/extract")
@app.post("/api/extract")
def extract(req: Req, x_token: str = Header(default="")):
    if not GROQ_KEY:
        raise HTTPException(500, "Máy chủ chưa có GROQ_API_KEY")
    if APP_TOKEN and x_token != APP_TOKEN:
        raise HTTPException(401, "Sai mật khẩu máy chủ")
    url = req.url.strip()
    with tempfile.TemporaryDirectory() as d:
        if looks_like_image(url):
            r = httpx.get(url, follow_redirects=True, timeout=30)
            r.raise_for_status()
            mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
            return {"items": extract_vocab(req.lang_name, "", [img_block(r.content, mime)]), "transcript": ""}
        opts = {
            "outtmpl": f"{d}/v.%(ext)s",
            "format": "worst[height>=360]/worst/best",
            "quiet": True, "noplaylist": True,
            "match_filter": yt_dlp.utils.match_filter_func(f"duration<{MAX_SECONDS}"),
            "max_filesize": 120_000_000,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as y:
                y.download([url])
        except Exception as e:
            raise HTTPException(422, f"Không tải được video (nền tảng có thể đang chặn): {str(e)[:200]}")
        files = glob.glob(f"{d}/v.*")
        if not files:
            raise HTTPException(422, "Video quá dài hoặc không tải được")
        video = files[0]
        transcript = ""
        try:
            sh("ffmpeg", "-y", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", f"{d}/a.mp3")
            transcript = transcribe(f"{d}/a.mp3")
        except Exception:
            pass
        images = []
        try:
            sh("ffmpeg", "-y", "-i", video, "-vf", "fps=1/4,scale=640:-1", "-frames:v", "5", f"{d}/f%02d.jpg")
            for p in sorted(glob.glob(f"{d}/f*.jpg")):
                images.append(img_block(open(p, "rb").read()))
        except Exception:
            pass
        items = extract_vocab(req.lang_name, transcript, images)
        return {"items": items, "transcript": transcript}


class Seg(BaseModel):
    text: str
    lang: str          # cs | vi | en | ja ...
    pause: int = 300   # ms nghỉ sau đoạn
    i: int = 0         # số thứ tự đoạn (để app tô sáng đoạn đang đọc)


class SegReq(BaseModel):
    segs: list[Seg]


@app.post("/api/tts-segments")
async def tts_segments(req: SegReq, x_token: str = Header(default="")):
    """Đọc ngữ pháp / câu chuyện: mỗi đoạn một giọng (Việt xen Séc...)."""
    if APP_TOKEN and x_token != APP_TOKEN:
        raise HTTPException(401, "Sai mật khẩu máy chủ")
    if len(req.segs) > 800 or sum(len(s.text) for s in req.segs) > 12000:
        raise HTTPException(400, "Nội dung quá dài")
    voice = lambda s: VOICES.get(s.lang, VOICES["en"])
    todo = list({(s.text, voice(s)) for s in req.segs})
    audio = dict(zip(todo, await asyncio.gather(*(synth(t, v) for t, v in todo))))
    out, marks, t = bytearray(), [], 0.0
    for s in req.segs:
        marks.append({"i": s.i, "t": round(t, 2)})
        b = audio[(s.text, voice(s))]
        out.extend(b)
        t += len(b) / BYTES_PER_SEC
        p = max(50, min(s.pause, 3000))
        out.extend(silence(p))
        t += p / 1000
    return {"audio": base64.b64encode(bytes(out)).decode(), "marks": marks, "dur": round(t, 2)}


# ---------- Tự sắp xếp ngữ pháp / câu chuyện từ văn bản dán vào ----------
class OrgReq(BaseModel):
    kind: str                       # grammar | story
    text: str
    lang_name: str = "tiếng Séc"
    words: list[str] = []           # từ vựng của bài (để đặt vào ngoặc vuông trong truyện)


def groq_items(system: str, user: str) -> list:
    if not GROQ_KEY:
        raise HTTPException(500, "Máy chủ chưa có GROQ_API_KEY")
    for _ in range(2):
        r = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            json={"model": TEXT_MODEL, "temperature": 0.2, "max_tokens": 6000,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            timeout=90,
        )
        if r.status_code == 429:
            raise HTTPException(429, "Groq đang giới hạn lượt dùng, đợi 1 phút rồi thử lại")
        if r.status_code >= 400:
            raise HTTPException(502, f"Lỗi AI: {r.text[:200]}")
        try:
            data = json.loads(r.json()["choices"][0]["message"]["content"])
            items = data.get("items") if isinstance(data, dict) else data
            if isinstance(items, list):
                return items
        except Exception:
            pass
    raise HTTPException(502, "AI trả kết quả không đúng định dạng, hãy thử lại")


@app.post("/api/organize")
def organize(req: OrgReq, x_token: str = Header(default="")):
    if APP_TOKEN and x_token != APP_TOKEN:
        raise HTTPException(401, "Sai mật khẩu máy chủ")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Chưa có nội dung")
    if len(text) > 9000:
        raise HTTPException(400, "Văn bản dài quá (tối đa 9000 ký tự), hãy chia nhỏ ra dán từng đợt")
    lang = req.lang_name
    if req.kind == "grammar":
        system = (
            f"Bạn là giáo viên dạy {lang} cho người Việt. Người dùng dán ghi chú ngữ pháp còn lộn xộn. "
            "Hãy sắp xếp thành các điểm ngữ pháp riêng biệt, mỗi điểm một ý chính. Với mỗi điểm trả về: "
            '"title" (tên điểm, ngắn gọn, tiếng Việt), '
            '"body" (giải thích bằng tiếng Việt, rõ ràng, giữ đúng ý và quy tắc trong ghi chú của người dùng, '
            "không bịa thêm quy tắc ngoài ghi chú; có thể nhiều đoạn), "
            f'"ex" (đúng 2 ví dụ, mỗi ví dụ gồm "cs" = câu bằng {lang} đúng chính tả và ngữ pháp, '
            '"vi" = nghĩa tiếng Việt). Nếu ghi chú đã có ví dụ thì dùng ví dụ của người dùng '
            "(chỉ sửa lỗi chính tả); nếu thiếu thì tự viết ví dụ đơn giản, đúng, minh hoạ đúng quy tắc đó. "
            'Chỉ trả về JSON dạng {"items":[{"title":"","body":"","ex":[{"cs":"","vi":""},{"cs":"","vi":""}]}]}'
        )
        items = groq_items(system, text)
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            title, body = str(it.get("title", "")).strip(), str(it.get("body", "")).strip()
            ex = [{"cs": str(e.get("cs", "")).strip(), "vi": str(e.get("vi", "")).strip()}
                  for e in (it.get("ex") or []) if isinstance(e, dict) and str(e.get("cs", "")).strip()][:2]
            if title and body:
                out.append({"title": title, "body": body, "ex": ex})
        return {"items": out}
    if req.kind == "story":
        words = ", ".join(w for w in req.words[:80] if w.strip())
        system = (
            f"Người dùng gửi các phần của câu chuyện viết bằng tiếng Việt có chêm {lang} "
            "(có thể ghi 'phần 1', 'phần 2'... hoặc để lẫn lộn). Hãy gom lại thành các câu chuyện hoàn chỉnh, "
            "sắp xếp các phần theo đúng thứ tự, giữ nguyên nội dung và giọng văn của người dùng (chỉ sửa lỗi chính tả). "
            f"Mọi từ hoặc cụm từ bằng {lang} phải được đặt trong ngoặc vuông [ ], phần tiếng Việt thì không. "
            "Giữ lại các ngoặc vuông người dùng đã đặt. "
            + (f"Các từ vựng sau cũng là {lang}, hãy đặt vào ngoặc vuông nếu xuất hiện: {words}. " if words else "")
            + 'Mỗi truyện có "title" ngắn bằng tiếng Việt và "text" là toàn bộ nội dung (giữ xuống dòng). '
            'Chỉ trả về JSON dạng {"items":[{"title":"","text":""}]}'
        )
        items = groq_items(system, text)
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            title, body = str(it.get("title", "")).strip(), str(it.get("text", "")).strip()
            if body:
                out.append({"title": title or "Câu chuyện", "text": body})
        return {"items": out}
    raise HTTPException(400, "kind phải là grammar hoặc story")


# Trang app (bản web cho iPhone / trình duyệt) phục vụ ngay tại địa chỉ máy chủ
if pathlib.Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
