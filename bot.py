"""NOVA UNIVERSE — Telegram-бот (порт веб-проекта).

Мульти-провайдерный ИИ-чат: OpenRouter по умолчанию, свои ключи и свои
OpenAI-совместимые провайдеры, режимы, память, навыки, фото и текстовые файлы.
"""
import asyncio
import base64
import io
import ipaddress
import json
import logging
import math
import os
import re
import socket
import time
from urllib.parse import urlparse

import aiohttp
import aiosqlite
from aiohttp import web
from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.filters import Command, CommandObject, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, BufferedInputFile, CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from PIL import Image

from persist import Backup

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ───────────────────────── Конфиг ─────────────────────────
BOT_TOKEN = os.environ["BOT_TOKEN"]
DEFAULT_OR_KEY = os.getenv("DEFAULT_OPENROUTER_KEY", "").strip()  # общий ключ (необязателен)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/auto")
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USERS", "").split(",") if x.strip()}
ALLOW_PRIVATE = os.getenv("ALLOW_PRIVATE_URLS", "0") == "1"  # http/localhost для своих провайдеров
HISTORY_CHARS = int(os.getenv("HISTORY_CHARS", "60000"))  # сколько символов истории сессии уходит модели
HISTORY_MSGS = 80      # максимум последних сообщений в запросе
SESSION_CAP = 400      # максимум сообщений, хранимых в одной сессии
Image.MAX_IMAGE_PIXELS = 50_000_000
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "nova.db")

BUILTIN = {
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1"),
    "google": ("Google AI Studio", "https://generativelanguage.googleapis.com/v1beta/openai"),
    "mistral": ("Mistral AI", "https://api.mistral.ai/v1"),
    "groq": ("Groq", "https://api.groq.com/openai/v1"),
    "deepseek": ("DeepSeek", "https://api.deepseek.com/v1"),
    "openai": ("OpenAI", "https://api.openai.com/v1"),
    "anthropic": ("Claude (Anthropic)", "https://api.anthropic.com/v1"),
    "xai": ("xAI (Grok)", "https://api.x.ai/v1"),
    "together": ("Together AI", "https://api.together.xyz/v1"),
    "perplexity": ("Perplexity", "https://api.perplexity.ai"),
    "cerebras": ("Cerebras", "https://api.cerebras.ai/v1"),
    "fireworks": ("Fireworks AI", "https://api.fireworks.ai/inference/v1"),
    "deepinfra": ("DeepInfra", "https://api.deepinfra.com/v1/openai"),
    "sambanova": ("SambaNova", "https://api.sambanova.ai/v1"),
    "novita": ("Novita AI", "https://api.novita.ai/v3/openai"),
    "nvidia": ("NVIDIA NIM", "https://integrate.api.nvidia.com/v1"),
    "huggingface": ("Hugging Face", "https://router.huggingface.co/v1"),
}
PROVIDER_DEFAULT_MODEL = {
    "openrouter": DEFAULT_MODEL, "google": "gemini-2.0-flash", "mistral": "mistral-small-latest",
    "groq": "llama-3.3-70b-versatile", "deepseek": "deepseek-chat", "openai": "gpt-4o-mini",
}

BASE_RULES = "Отвечай на языке пользователя. Будь точным; не выдумывай факты."
MODES = {
    "normal": ("💬 Обычный", "Ты — NOVA, дружелюбный и полезный ИИ-ассистент. " + BASE_RULES),
    "expert": ("🎓 Эксперт", "Ты — NOVA в режиме эксперта. Давай глубокие, структурированные ответы: "
               "сначала вывод, затем аргументы, нюансы и ограничения. " + BASE_RULES),
    "code": ("💻 NOVA Code", "Ты — NOVA Code, старший инженер. Сначала уточни структуру и контекст, "
             "предлагай минимальные патчи, показывай код блоками, объясняй риски. Финальный файл "
             "собирай только после согласования. " + BASE_RULES),
    "tutor": ("📚 Репетитор", "Ты — NOVA Репетитор. Перед началом задай 2–3 вопроса об уровне ученика. "
              "Разбирай тему по шагам, давай практику и домашнее задание, проверяй знания, не выдавай "
              "готовый ответ сразу, а веди к нему. " + BASE_RULES),
    "klyro": ("🧠 Klyro", "Ты — NOVA в режиме Klyro Engine. На сложные вопросы отвечай структурой: "
              "Решение; Уверенность (%); Причины; Риски и противоречия; Альтернативы; Следующий шаг. "
              "На простые вопросы отвечай коротко. " + BASE_RULES),
}


# ───────────────────────── Шифрование ключей ─────────────────────────
def load_fernet() -> Fernet:
    k = os.getenv("ENCRYPTION_KEY", "").strip()
    if not k:
        p = os.path.join(DATA_DIR, "secret.key")
        if os.path.exists(p):
            k = open(p).read().strip()
        else:
            k = Fernet.generate_key().decode()
            with open(p, "w") as f:
                f.write(k)
            os.chmod(p, 0o600)
    return Fernet(k.encode())


fernet = load_fernet()

# ───────────────────────── БД ─────────────────────────
db: aiosqlite.Connection


async def init_db():
    global db
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.executescript("""
    CREATE TABLE IF NOT EXISTS users(user_id INTEGER PRIMARY KEY, provider TEXT, model TEXT,
        mode TEXT, temperature REAL, sys TEXT);
    CREATE TABLE IF NOT EXISTS keys(user_id INTEGER, provider TEXT, key_enc TEXT,
        PRIMARY KEY(user_id, provider));
    CREATE TABLE IF NOT EXISTS custom_providers(user_id INTEGER, name TEXT, base_url TEXT, model TEXT,
        PRIMARY KEY(user_id, name));
    CREATE TABLE IF NOT EXISTS memory(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, text TEXT);
    CREATE TABLE IF NOT EXISTS skills(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        name TEXT, text TEXT);
    CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        role TEXT, content TEXT);
    CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
        title TEXT, created REAL, updated REAL);
    """)
    await ensure_column("users", "session_id", "INTEGER")
    await ensure_column("history", "session_id", "INTEGER")
    await ensure_column("history", "img", "TEXT")
    await ensure_column("custom_providers", "model", "TEXT")
    await db.commit()


async def ensure_column(table, col, decl):
    cols = [r["name"] for r in await fetch(f"PRAGMA table_info({table})")]
    if col not in cols:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


async def fetch(sql, args=()):
    async with db.execute(sql, args) as c:
        return await c.fetchall()


async def run(sql, args=()):
    await db.execute(sql, args)
    await db.commit()


async def get_user(uid):
    rows = await fetch("SELECT * FROM users WHERE user_id=?", (uid,))
    if not rows:
        await run("INSERT INTO users(user_id,provider,model,mode,temperature,sys) VALUES(?,?,?,?,?,?)",
                  (uid, "openrouter", DEFAULT_MODEL, "normal", 0.7, ""))
        rows = await fetch("SELECT * FROM users WHERE user_id=?", (uid,))
    return rows[0]


async def upd(uid, **kw):
    await get_user(uid)
    cols = ", ".join(f"{k}=?" for k in kw)
    await run(f"UPDATE users SET {cols} WHERE user_id=?", (*kw.values(), uid))


async def set_key(uid, pid, key):
    await run("INSERT OR REPLACE INTO keys VALUES(?,?,?)", (uid, pid, fernet.encrypt(key.encode()).decode()))


async def get_key(uid, pid):
    rows = await fetch("SELECT key_enc FROM keys WHERE user_id=? AND provider=?", (uid, pid))
    if not rows:
        return None
    try:
        return fernet.decrypt(rows[0]["key_enc"].encode()).decode()
    except Exception:
        return None


# ───────────────────────── Провайдеры ─────────────────────────
async def check_url(url: str) -> str:
    """Защита от SSRF: только https и только публичные адреса."""
    p = urlparse(url.strip())
    if p.scheme not in ("https", "http") or not p.hostname:
        raise ValueError("Нужен адрес вида https://api.example.com/v1")
    if not ALLOW_PRIVATE:
        if p.scheme != "https":
            raise ValueError("Разрешён только https.")
        try:
            infos = await asyncio.to_thread(socket.getaddrinfo, p.hostname, p.port or 443)
        except socket.gaierror:
            raise ValueError("Не удалось найти такой домен.")
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise ValueError("Адреса внутренней сети запрещены.")
    return url.strip().rstrip("/")


async def all_providers(uid):
    res = [(pid, name, base) for pid, (name, base) in BUILTIN.items()]
    for r in await fetch("SELECT name, base_url FROM custom_providers WHERE user_id=?", (uid,)):
        res.append((f"c:{r['name']}", f"{r['name']} (свой)", r["base_url"]))
    return res


async def resolve(uid, pid):
    if pid in BUILTIN:
        name, base = BUILTIN[pid]
        key = await get_key(uid, pid)
        if not key and pid == "openrouter":
            key = DEFAULT_OR_KEY or None
        return dict(id=pid, name=name, base=base, key=key, custom=False)
    if pid.startswith("c:"):
        rows = await fetch("SELECT base_url FROM custom_providers WHERE user_id=? AND name=?", (uid, pid[2:]))
        if rows:
            base = await check_url(rows[0]["base_url"])
            return dict(id=pid, name=pid[2:], base=base, key=await get_key(uid, pid), custom=True)
    return None


class ProviderError(Exception):
    def __init__(self, msg, status=0):
        super().__init__(msg)
        self.status = status


session: aiohttp.ClientSession


def auth_headers(base, key):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    if "openrouter.ai" in base:
        h["X-Title"] = "NOVA UNIVERSE Bot"
    return h


VMETA: dict[str, dict] = {}  # base_url -> {"all": set, "vis": set, "ts": float}
VISION_RE = re.compile(
    r"(gpt-4o|gpt-4\.1|gpt-4-turbo|gpt-5|o4-mini|gemini|claude|llava|vision|-vl\b|-vl-|/vl|pixtral|"
    r"llama-?3\.2-(11|90)b|llama-4|grok-(4|vision|2-vision)|qwen.*vl|internvl|minicpm-v|gemma-?3|"
    r"mistral-(small|medium)|molmo)", re.I)


def model_has_vision(m):
    """True/False, если провайдер сообщает модальности модели; None — если данных нет."""
    if not isinstance(m, dict):
        return None
    arch = m.get("architecture") or {}
    mods = arch.get("input_modalities")
    if isinstance(mods, list):
        return "image" in mods
    mod = arch.get("modality")
    if isinstance(mod, str):
        return "image" in mod.split("->")[0]
    caps = m.get("capabilities")
    if isinstance(caps, dict) and "vision" in caps:
        return bool(caps["vision"])
    return None


async def fetch_models(base, key):
    """→ (список id, множество id с vision, есть ли метаданные о vision)"""
    async with session.get(base + "/models", headers=auth_headers(base, key), allow_redirects=False,
                           timeout=aiohttp.ClientTimeout(total=30)) as r:
        if r.status != 200:
            raise ProviderError(f"HTTP {r.status}", r.status)
        d = await r.json(content_type=None)
    items = d.get("data", []) if isinstance(d, dict) else d
    ids, vis, meta = [], set(), False
    for m in items:
        if not m:
            continue
        mid = m["id"] if isinstance(m, dict) else str(m)
        ids.append(mid)
        v = model_has_vision(m)
        if v is not None:
            meta = True
            if v:
                vis.add(mid)
    ids = sorted(set(ids))
    if meta:
        VMETA[base] = {"all": set(ids), "vis": vis, "ts": time.time()}
    return ids, vis, meta


async def vision_status(prov, model):
    """True — модель видит картинки, False — точно нет, None — неизвестно."""
    base = prov["base"]
    stale = base not in VMETA or time.time() - VMETA[base]["ts"] > 3600
    if "openrouter.ai" in base and stale:
        try:
            await fetch_models(base, prov["key"])
        except Exception:
            pass
    meta = VMETA.get(base)
    if meta and model in meta["all"]:
        return model in meta["vis"]
    if VISION_RE.search(model or ""):
        return True
    return None


async def stream_chat(base, key, model, messages, temperature):
    payload = {"model": model, "messages": messages, "stream": True, "temperature": temperature}
    async with session.post(base + "/chat/completions", json=payload, headers=auth_headers(base, key),
                            allow_redirects=False) as r:
        if r.status != 200:
            text = (await r.text())[:600]
            if key:
                text = text.replace(key, "***")
            raise ProviderError(f"{r.status}: {text}", r.status)
        async for raw in r.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("error"):
                raise ProviderError(str(obj["error"])[:600], 0)
            ch = obj.get("choices") or []
            if ch:
                piece = (ch[0].get("delta") or {}).get("content")
                if piece:
                    yield piece


# ───────────────────────── UI ─────────────────────────
def mk(rows, width=1, back="m:main"):
    b = InlineKeyboardBuilder()
    for t, d in rows:
        b.button(text=t, callback_data=d)
    b.adjust(width)
    if back:
        b.row(InlineKeyboardButton(text="◀️ Назад", callback_data=back))
    return b.as_markup()


async def render(t, text, markup=None):
    if isinstance(t, CallbackQuery):
        try:
            await t.message.edit_text(text, reply_markup=markup)
        except TelegramBadRequest:
            pass
        try:
            await t.answer()
        except Exception:
            pass
    else:
        await t.answer(text, reply_markup=markup)


async def screen_main(uid):
    u = await get_user(uid)
    pname = (await resolve_name(uid, u["provider"]))
    sid = await active_session(uid)
    title = (await fetch("SELECT title FROM sessions WHERE id=?", (sid,)))[0]["title"]
    text = (f"🌌 NOVA UNIVERSE\n\nПровайдер: {pname}\nМодель: {u['model'] or '— не выбрана'}\n"
            f"Режим: {MODES[u['mode']][0]}\nСессия: {title}\n\n"
            "Просто напишите сообщение. Можно прислать фото, стикер или текстовый файл.")
    rows = [("🤖 Модель", "m:model"), ("🔑 Ключи и провайдеры", "m:prov"), ("🗂 Сессии", "m:sess"),
            ("🎭 Режим", "m:mode"), ("🧠 Память", "m:mem"), ("🧩 Навыки", "m:skills"),
            ("⚙️ Настройки", "m:set"), ("🆕 Новая сессия", "m:new")]
    return text, mk(rows, 2, back=None)


async def resolve_name(uid, pid):
    if pid in BUILTIN:
        return BUILTIN[pid][0]
    return pid[2:] + " (свой)"


async def screen_prov(uid):
    u = await get_user(uid)
    rows = []
    for pid, name, _ in await all_providers(uid):
        has = bool(await get_key(uid, pid)) or (pid == "openrouter" and DEFAULT_OR_KEY)
        mark = "▶️" if pid == u["provider"] else ("🔑" if has else "▫️")
        rows.append((f"{mark} {name}", f"p:{pid}"))
    rows.append(("➕ Добавить свой провайдер", "padd"))
    text = "🔑 Провайдеры\n▶️ активный · 🔑 есть ключ · ▫️ нет ключа\n\nВыберите провайдера, чтобы вставить ключ или сделать его активным."
    return text, mk(rows, 2)


async def screen_provider(uid, pid):
    u = await get_user(uid)
    provs = {p[0]: p for p in await all_providers(uid)}
    if pid not in provs:
        return await screen_prov(uid)
    _, name, base = provs[pid]
    own = await get_key(uid, pid)
    if own:
        ks = "ваш ключ сохранён"
    elif pid == "openrouter" and DEFAULT_OR_KEY:
        ks = "используется общий ключ бота (можно вставить свой)"
    else:
        ks = "нет — вставьте ключ"
    text = f"{name}\n{base}\n\nКлюч: {ks}\nАктивный: {'да' if u['provider'] == pid else 'нет'}"
    rows = [("🔑 Вставить / заменить ключ", f"k:{pid}")]
    if own:
        rows.append(("🗑 Удалить ключ", f"kd:{pid}"))
    rows.append(("✅ Использовать", f"use:{pid}"))
    if pid.startswith("c:"):
        rows.append(("❌ Удалить провайдера", f"pdel:{pid[2:]}"))
    return text, mk(rows, 1, back="m:prov")


async def screen_mode(uid):
    u = await get_user(uid)
    rows = [(("✅ " if k == u["mode"] else "") + v[0], f"mode:{k}") for k, v in MODES.items()]
    return "🎭 Режим ассистента", mk(rows, 2)


async def screen_mem(uid):
    rows = [(f"🗑 {r['text'][:40]}", f"md:{r['id']}") for r in
            await fetch("SELECT id, text FROM memory WHERE user_id=? ORDER BY id", (uid,))]
    rows.append(("➕ Добавить факт", "ma"))
    text = "🧠 Память о вас. Эти факты передаются модели в каждом чате. Нажмите на факт, чтобы удалить."
    return text, mk(rows, 1)


async def screen_skills(uid):
    rows = [(f"🗑 {r['name'][:40]}", f"skd:{r['id']}") for r in
            await fetch("SELECT id, name FROM skills WHERE user_id=? ORDER BY id", (uid,))]
    rows.append(("➕ Добавить навык", "ska"))
    text = ("🧩 Навыки — сохранённые инструкции, которые ИИ применяет, когда запрос подходит.\n"
            "Нажмите на навык, чтобы удалить.")
    return text, mk(rows, 1)


async def screen_settings(uid):
    u = await get_user(uid)
    text = (f"⚙️ Настройки\n\nTemperature: {u['temperature']}\n"
            f"Свой системный промпт: {'задан' if u['sys'] else 'нет'}")
    rows = [(("✅ " if abs(u["temperature"] - t) < 0.01 else "") + str(t), f"t:{t}") for t in (0.2, 0.7, 1.0, 1.3)]
    b = InlineKeyboardBuilder()
    for t, d in rows:
        b.button(text=t, callback_data=d)
    b.adjust(4)
    b.row(InlineKeyboardButton(text="✍️ Свой системный промпт", callback_data="sp"))
    if u["sys"]:
        b.row(InlineKeyboardButton(text="🗑 Сбросить промпт", callback_data="spc"))
    b.row(InlineKeyboardButton(text="◀️ Назад", callback_data="m:main"))
    return text, b.as_markup()


# Выбор модели
MC: dict[int, dict] = {}
PS = 8


def models_screen(uid, u):
    c = MC[uid]
    flt, page = c["flt"], c["page"]
    pages = max(1, math.ceil(len(flt) / PS))
    b = InlineKeyboardBuilder()
    for i, mid in enumerate(flt[page * PS:(page + 1) * PS]):
        mark = ("✅ " if mid == u["model"] else "") + ("👁 " if mid in c.get("vis", ()) else "")
        b.button(text=mark + mid[:52], callback_data=f"mm:{page * PS + i}")
    b.adjust(1)
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"mp:{page - 1}"))
    nav.append(InlineKeyboardButton(text=f"{page + 1}/{pages}", callback_data="noop"))
    if (page + 1) * PS < len(flt):
        nav.append(InlineKeyboardButton(text="➡️", callback_data=f"mp:{page + 1}"))
    b.row(*nav)
    tools = [InlineKeyboardButton(text="🔎 Поиск", callback_data="msearch"),
             InlineKeyboardButton(text="✍️ Ввести ID", callback_data="mmanual")]
    if c.get("meta"):
        tools.append(InlineKeyboardButton(text="👁 Vision" + (" ✅" if c.get("vf") else ""), callback_data="mvis"))
    b.row(*tools)
    b.row(InlineKeyboardButton(text="◀️ Назад", callback_data="m:main"))
    q = (f"\nФильтр: {c['q']}" if c["q"] else "") + ("\nПоказаны только vision-модели" if c.get("vf") else "")
    legend = "\n👁 — модель понимает изображения" if c.get("meta") else ""
    return f"🤖 Выбор модели ({len(flt)} шт.)\nТекущая: {u['model'] or '—'}{q}{legend}", b.as_markup()


def apply_filter(c):
    q = c.get("q", "")
    c["flt"] = [x for x in c["all"] if q in x.lower() and (not c.get("vf") or x in c["vis"])]
    c["page"] = 0


async def open_models(t, uid):
    u = await get_user(uid)
    try:
        prov = await resolve(uid, u["provider"])
        ids, vis, meta = await fetch_models(prov["base"], prov["key"])
        if not ids:
            raise ProviderError("пустой список")
    except Exception as e:
        MC[uid] = dict(all=[], flt=[], q="", page=0, vis=set(), meta=False, vf=False)
        msg = (f"Не удалось получить список моделей ({e}).\nМожно ввести ID модели вручную "
               "или сначала проверить ключ.")
        await render(t, msg, mk([("✍️ Ввести ID модели", "mmanual"), ("🔑 Ключи", "m:prov")], 1))
        return
    MC[uid] = dict(all=ids, flt=ids, q="", page=0, vis=vis, meta=meta, vf=False)
    await render(t, *models_screen(uid, u))


# ───────────────────────── Бот и FSM ─────────────────────────
class S(StatesGroup):
    key = State()
    p_name = State()
    p_url = State()
    p_key = State()
    p_model = State()
    mem = State()
    sk_name = State()
    sk_text = State()
    msearch = State()
    mmanual = State()
    sysprompt = State()
    rename = State()


class Gate(BaseMiddleware):
    async def __call__(self, handler, event, data):
        u = data.get("event_from_user")
        if ALLOWED and (not u or u.id not in ALLOWED):
            return None
        return await handler(event, data)


dp = Dispatcher(storage=MemoryStorage())
dp.message.outer_middleware(Gate())
dp.callback_query.outer_middleware(Gate())
dp.message.filter(F.chat.type == "private")

# ── команды
HELP = ("Основное:\n/menu — главное меню\n/models — выбрать модель (👁 = видит изображения)\n"
        "/providers — ключи и свои провайдеры\n/mode — режим ассистента\n/status — текущее состояние\n\n"
        "Сессии (у каждой своя история):\n/sessions — список сессий\n/newsession [название] — новая сессия\n"
        "/session — текущая сессия\n/rename [название] — переименовать\n/history — последние сообщения\n"
        "/export — сохранить сессию в файл\n/clear — очистить текущую сессию\n\n"
        "Память и настройки:\n/memory, /remember <факт> — память о вас\n/skills — навыки\n"
        "/settings — настройки\n/temp 0.7 — temperature\n/system <текст> — свой системный промпт ('-' сбросить)\n"
        "/vision — видит ли текущая модель изображения\n/cancel — отменить ввод\n\n"
        "Картинки и стикеры: просто отправьте их боту (нужна vision-модель).")


@dp.message(CommandStart())
@dp.message(Command("menu"))
async def cmd_start(m: Message, state: FSMContext):
    await state.clear()
    await get_user(m.from_user.id)
    await render(m, *await screen_main(m.from_user.id))


@dp.message(Command("help"))
async def cmd_help(m: Message, state: FSMContext):
    await state.clear()
    await m.answer(HELP)


@dp.message(Command("cancel"))
async def cmd_cancel(m: Message, state: FSMContext):
    await state.clear()
    await m.answer("Отменено.")


@dp.message(Command("model", "models"))
async def cmd_model(m: Message, state: FSMContext):
    await state.clear()
    await open_models(m, m.from_user.id)


@dp.message(Command("providers", "provider", "key"))
async def cmd_prov(m: Message, state: FSMContext):
    await state.clear()
    await render(m, *await screen_prov(m.from_user.id))


@dp.message(Command("mode"))
async def cmd_mode(m: Message, state: FSMContext):
    await state.clear()
    await render(m, *await screen_mode(m.from_user.id))


@dp.message(Command("memory"))
async def cmd_mem(m: Message, state: FSMContext):
    await state.clear()
    await render(m, *await screen_mem(m.from_user.id))


@dp.message(Command("skills"))
async def cmd_skills(m: Message, state: FSMContext):
    await state.clear()
    await render(m, *await screen_skills(m.from_user.id))


@dp.message(Command("remember"))
async def cmd_remember(m: Message, command: CommandObject, state: FSMContext):
    await state.clear()
    if not command.args:
        return await m.answer("Использование: /remember Я живу в Алматы")
    await run("INSERT INTO memory(user_id,text) VALUES(?,?)", (m.from_user.id, command.args.strip()[:500]))
    await m.answer("🧠 Запомнил.")


# ── callbacks: меню
@dp.callback_query(F.data == "noop")
async def cb_noop(cb: CallbackQuery):
    await cb.answer()


@dp.callback_query(F.data == "m:main")
async def cb_main(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await render(cb, *await screen_main(cb.from_user.id))


@dp.callback_query(F.data == "m:prov")
async def cb_prov(cb: CallbackQuery):
    await render(cb, *await screen_prov(cb.from_user.id))


@dp.callback_query(F.data == "m:mode")
async def cb_mode(cb: CallbackQuery):
    await render(cb, *await screen_mode(cb.from_user.id))


@dp.callback_query(F.data == "m:mem")
async def cb_mem(cb: CallbackQuery):
    await render(cb, *await screen_mem(cb.from_user.id))


@dp.callback_query(F.data == "m:skills")
async def cb_skills(cb: CallbackQuery):
    await render(cb, *await screen_skills(cb.from_user.id))


@dp.callback_query(F.data == "m:set")
async def cb_set(cb: CallbackQuery):
    await render(cb, *await screen_settings(cb.from_user.id))


@dp.callback_query(F.data == "m:model")
async def cb_model(cb: CallbackQuery):
    try:
        await cb.message.edit_text("⏳ Загружаю список моделей…")
    except TelegramBadRequest:
        pass
    await open_models(cb, cb.from_user.id)


@dp.callback_query(F.data.startswith("mode:"))
async def cb_setmode(cb: CallbackQuery):
    k = cb.data[5:]
    if k in MODES:
        await upd(cb.from_user.id, mode=k)
    await render(cb, *await screen_mode(cb.from_user.id))


@dp.callback_query(F.data.startswith("t:"))
async def cb_temp(cb: CallbackQuery):
    await upd(cb.from_user.id, temperature=float(cb.data[2:]))
    await render(cb, *await screen_settings(cb.from_user.id))


@dp.callback_query(F.data == "sp")
async def cb_sp(cb: CallbackQuery, state: FSMContext):
    await state.set_state(S.sysprompt)
    await render(cb, "Пришлите свой системный промпт одним сообщением (добавится к режиму). /cancel — отмена.")


@dp.callback_query(F.data == "spc")
async def cb_spc(cb: CallbackQuery):
    await upd(cb.from_user.id, sys="")
    await render(cb, *await screen_settings(cb.from_user.id))


# ── провайдеры и ключи
@dp.callback_query(F.data.startswith("p:"))
async def cb_provider(cb: CallbackQuery):
    await render(cb, *await screen_provider(cb.from_user.id, cb.data[2:]))


@dp.callback_query(F.data.startswith("k:"))
async def cb_key(cb: CallbackQuery, state: FSMContext):
    pid = cb.data[2:]
    await state.set_state(S.key)
    await state.update_data(pid=pid)
    await render(cb, "Пришлите API-ключ одним сообщением. Я сохраню его в зашифрованном виде "
                     "и сразу удалю ваше сообщение. /cancel — отмена.")


@dp.callback_query(F.data.startswith("kd:"))
async def cb_keydel(cb: CallbackQuery):
    pid = cb.data[3:]
    await run("DELETE FROM keys WHERE user_id=? AND provider=?", (cb.from_user.id, pid))
    await render(cb, *await screen_provider(cb.from_user.id, pid))


@dp.callback_query(F.data.startswith("use:"))
async def cb_use(cb: CallbackQuery):
    uid, pid = cb.from_user.id, cb.data[4:]
    prov = await resolve(uid, pid)
    if not prov:
        return await cb.answer("Провайдер не найден", show_alert=True)
    u = await get_user(uid)
    if u["provider"] != pid:
        model = PROVIDER_DEFAULT_MODEL.get(pid, "")
        if pid.startswith("c:"):
            model = await get_custom_model(uid, pid)
        await upd(uid, provider=pid, model=model)
    if not prov["key"] and not prov["custom"]:
        await cb.answer("Провайдер выбран, но нужен ключ", show_alert=True)
        return await render(cb, *await screen_provider(uid, pid))
    await open_models(cb, uid)


async def get_custom_model(uid, pid):
    rows = await fetch("SELECT model FROM custom_providers WHERE user_id=? AND name=?", (uid, pid[2:]))
    return (rows[0]["model"] or "") if rows else ""


@dp.callback_query(F.data.startswith("pdel:"))
async def cb_pdel(cb: CallbackQuery):
    uid, name = cb.from_user.id, cb.data[5:]
    await run("DELETE FROM custom_providers WHERE user_id=? AND name=?", (uid, name))
    await run("DELETE FROM keys WHERE user_id=? AND provider=?", (uid, f"c:{name}"))
    u = await get_user(uid)
    if u["provider"] == f"c:{name}":
        await upd(uid, provider="openrouter", model=DEFAULT_MODEL)
    await render(cb, *await screen_prov(uid))


@dp.callback_query(F.data == "padd")
async def cb_padd(cb: CallbackQuery, state: FSMContext):
    await state.set_state(S.p_name)
    await render(cb, "➕ Свой провайдер (любой OpenAI-совместимый API)\n\nШаг 1/4. Придумайте короткое имя "
                     "(латиница, цифры, - и _, до 20 символов). /cancel — отмена.")


@dp.message(StateFilter(S.p_name), F.text)
async def p_name(m: Message, state: FSMContext):
    name = m.text.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", name):
        return await m.answer("Имя: латиница, цифры, - и _, до 20 символов. Попробуйте ещё раз.")
    await state.update_data(name=name)
    await state.set_state(S.p_url)
    await m.answer("Шаг 2/4. Базовый адрес API без /chat/completions на конце, например:\n"
                   "https://api.example.com/v1")


@dp.message(StateFilter(S.p_url), F.text)
async def p_url(m: Message, state: FSMContext):
    try:
        url = await check_url(m.text)
    except ValueError as e:
        return await m.answer(f"⚠️ {e}\nПопробуйте ещё раз.")
    await state.update_data(url=url)
    await state.set_state(S.p_key)
    await m.answer("Шаг 3/4. Пришлите API-ключ (сообщение будет удалено). Если ключ не нужен — отправьте «-».")


@dp.message(StateFilter(S.p_key), F.text)
async def p_key(m: Message, state: FSMContext):
    key = m.text.strip()
    try:
        await m.delete()
    except Exception:
        pass
    await state.update_data(key=None if key == "-" else key)
    await state.set_state(S.p_model)
    await m.answer("Шаг 4/4. ID модели по умолчанию (например, gpt-4o-mini). "
                   "Отправьте «-», чтобы выбрать из списка позже.")


@dp.message(StateFilter(S.p_model), F.text)
async def p_model(m: Message, state: FSMContext):
    uid, d = m.from_user.id, await state.get_data()
    model = "" if m.text.strip() == "-" else m.text.strip()[:200]
    pid = f"c:{d['name']}"
    await run("INSERT OR REPLACE INTO custom_providers VALUES(?,?,?,?)", (uid, d["name"], d["url"], model))
    if d.get("key"):
        await set_key(uid, pid, d["key"])
    await upd(uid, provider=pid, model=model)
    await state.clear()
    await m.answer(f"✅ Провайдер «{d['name']}» добавлен и выбран. Можно писать сообщения"
                   f"{'' if model else ' (сначала выберите модель: /model)'}.",
                   reply_markup=mk([("🤖 Выбрать модель", "m:model")], back="m:main"))


@dp.message(StateFilter(S.key), F.text)
async def got_key(m: Message, state: FSMContext):
    pid = (await state.get_data())["pid"]
    await set_key(m.from_user.id, pid, m.text.strip())
    try:
        await m.delete()
    except Exception:
        pass
    await state.clear()
    await m.answer("✅ Ключ сохранён, ваше сообщение с ключом удалено.",
                   reply_markup=mk([("✅ Использовать этого провайдера", f"use:{pid}"),
                                    ("◀️ К провайдерам", "m:prov")], back=None))


# ── выбор модели
@dp.callback_query(F.data.startswith("mp:"))
async def cb_mpage(cb: CallbackQuery):
    uid = cb.from_user.id
    if uid not in MC:
        return await cb.answer("Список устарел — откройте /model", show_alert=True)
    MC[uid]["page"] = int(cb.data[3:])
    await render(cb, *models_screen(uid, await get_user(uid)))


@dp.callback_query(F.data.startswith("mm:"))
async def cb_mpick(cb: CallbackQuery):
    uid = cb.from_user.id
    try:
        model = MC[uid]["flt"][int(cb.data[3:])]
    except (KeyError, IndexError):
        return await cb.answer("Список устарел — откройте /model", show_alert=True)
    await upd(uid, model=model)
    await cb.answer(f"Модель: {model}")
    await render(cb, *models_screen(uid, await get_user(uid)))


@dp.callback_query(F.data == "msearch")
async def cb_msearch(cb: CallbackQuery, state: FSMContext):
    await state.set_state(S.msearch)
    await render(cb, "Введите часть названия модели для поиска (например, llama или free). "
                     "«-» — сбросить фильтр.")


@dp.message(StateFilter(S.msearch), F.text)
async def got_msearch(m: Message, state: FSMContext):
    uid = m.from_user.id
    await state.clear()
    c = MC.get(uid)
    if not c:
        return await open_models(m, uid)
    q = "" if m.text.strip() == "-" else m.text.strip().lower()
    c["q"] = q
    apply_filter(c)
    await render(m, *models_screen(uid, await get_user(uid)))


@dp.callback_query(F.data == "mvis")
async def cb_mvis(cb: CallbackQuery):
    uid = cb.from_user.id
    if uid not in MC:
        return await cb.answer("Список устарел — откройте /models", show_alert=True)
    MC[uid]["vf"] = not MC[uid].get("vf")
    apply_filter(MC[uid])
    await render(cb, *models_screen(uid, await get_user(uid)))


@dp.callback_query(F.data == "mmanual")
async def cb_mmanual(cb: CallbackQuery, state: FSMContext):
    await state.set_state(S.mmanual)
    await render(cb, "Пришлите точный ID модели, например openai/gpt-4o-mini")


@dp.message(StateFilter(S.mmanual), F.text)
async def got_mmanual(m: Message, state: FSMContext):
    await upd(m.from_user.id, model=m.text.strip()[:200])
    await state.clear()
    await m.answer(f"✅ Модель: {m.text.strip()[:200]}")


# ── память, навыки, промпт
@dp.callback_query(F.data == "ma")
async def cb_ma(cb: CallbackQuery, state: FSMContext):
    await state.set_state(S.mem)
    await render(cb, "Пришлите факт о себе одним сообщением (например: «Я учусь в 10 классе, люблю астрономию»).")


@dp.message(StateFilter(S.mem), F.text)
async def got_mem(m: Message, state: FSMContext):
    await run("INSERT INTO memory(user_id,text) VALUES(?,?)", (m.from_user.id, m.text.strip()[:500]))
    await state.clear()
    await render(m, *await screen_mem(m.from_user.id))


@dp.callback_query(F.data.startswith("md:"))
async def cb_md(cb: CallbackQuery):
    await run("DELETE FROM memory WHERE id=? AND user_id=?", (int(cb.data[3:]), cb.from_user.id))
    await render(cb, *await screen_mem(cb.from_user.id))


@dp.callback_query(F.data == "ska")
async def cb_ska(cb: CallbackQuery, state: FSMContext):
    await state.set_state(S.sk_name)
    await render(cb, "Название навыка (коротко, например «Проверка эссе»):")


@dp.message(StateFilter(S.sk_name), F.text)
async def got_sk_name(m: Message, state: FSMContext):
    await state.update_data(name=m.text.strip()[:60])
    await state.set_state(S.sk_text)
    await m.answer("Теперь пришлите инструкцию навыка текстом или файлом .md/.txt.")


@dp.message(StateFilter(S.sk_text), F.text | F.document)
async def got_sk_text(m: Message, state: FSMContext):
    text = m.text or ""
    if m.document:
        if (m.document.file_size or 0) > 100_000:
            return await m.answer("Файл слишком большой (максимум 100 КБ).")
        buf = io.BytesIO()
        await m.bot.download(m.document, destination=buf)
        try:
            text = buf.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            return await m.answer("Нужен текстовый файл в UTF-8.")
    name = (await state.get_data())["name"]
    await run("INSERT INTO skills(user_id,name,text) VALUES(?,?,?)", (m.from_user.id, name, text.strip()[:8000]))
    await state.clear()
    await render(m, *await screen_skills(m.from_user.id))


@dp.callback_query(F.data.startswith("skd:"))
async def cb_skd(cb: CallbackQuery):
    await run("DELETE FROM skills WHERE id=? AND user_id=?", (int(cb.data[4:]), cb.from_user.id))
    await render(cb, *await screen_skills(cb.from_user.id))


@dp.message(StateFilter(S.sysprompt), F.text)
async def got_sys(m: Message, state: FSMContext):
    await upd(m.from_user.id, sys=m.text.strip()[:4000])
    await state.clear()
    await render(m, *await screen_settings(m.from_user.id))


# ───────────────────────── Сессии ─────────────────────────
async def new_session(uid, title="Новый чат"):
    now = time.time()
    cur = await db.execute("INSERT INTO sessions(user_id,title,created,updated) VALUES(?,?,?,?)",
                           (uid, title[:60], now, now))
    await db.commit()
    sid = cur.lastrowid
    await run("UPDATE history SET session_id=? WHERE user_id=? AND session_id IS NULL", (sid, uid))  # старая история
    await upd(uid, session_id=sid)
    return sid


async def own_session(uid, sid):
    rows = await fetch("SELECT * FROM sessions WHERE id=? AND user_id=?", (sid, uid))
    return rows[0] if rows else None


async def active_session(uid):
    u = await get_user(uid)
    if u["session_id"] and await own_session(uid, u["session_id"]):
        return u["session_id"]
    return await new_session(uid)


async def msg_count(sid):
    return (await fetch("SELECT COUNT(*) c FROM history WHERE session_id=?", (sid,)))[0]["c"]


async def delete_session(uid, sid):
    await run("DELETE FROM history WHERE session_id=?", (sid,))
    await run("DELETE FROM sessions WHERE id=? AND user_id=?", (sid, uid))
    u = await get_user(uid)
    if u["session_id"] == sid:
        rows = await fetch("SELECT id FROM sessions WHERE user_id=? ORDER BY updated DESC LIMIT 1", (uid,))
        if rows:
            await upd(uid, session_id=rows[0]["id"])
        else:
            await new_session(uid)


async def screen_sessions(uid):
    cur = await active_session(uid)
    rows = await fetch("SELECT id, title FROM sessions WHERE user_id=? ORDER BY updated DESC LIMIT 12", (uid,))
    buttons = []
    for r in rows:
        n = await msg_count(r["id"])
        buttons.append((("▶️ " if r["id"] == cur else "") + f"{r['title'][:30]} ({n // 2})", f"ss:{r['id']}"))
    buttons.append(("➕ Новая сессия", "ssnew"))
    text = ("🗂 Сессии\n\nУ каждой сессии своя история: бот помнит всё, что вы писали в ней. "
            "Цифра в скобках — число обменов сообщениями. ▶️ — активная сессия.")
    return text, mk(buttons, 1)


async def screen_session(uid, sid):
    r = await own_session(uid, sid)
    if not r:
        return await screen_sessions(uid)
    cur = await active_session(uid)
    n = await msg_count(sid)
    created = time.strftime("%d.%m.%Y %H:%M", time.localtime(r["created"]))
    text = (f"🗂 {r['title']}\nСообщений: {n}\nСоздана: {created}\n"
            + ("▶️ Это активная сессия" if sid == cur else "Не активна"))
    rows = []
    if sid != cur:
        rows.append(("▶️ Переключиться", f"ssuse:{sid}"))
    rows += [("✏️ Переименовать", f"ssren:{sid}"), ("📤 Экспорт в файл", f"ssexp:{sid}"),
             ("🧹 Очистить историю", f"sscl:{sid}"), ("🗑 Удалить сессию", f"ssdel:{sid}")]
    return text, mk(rows, 1, back="m:sess")


async def export_session(target, uid, sid):
    if not await own_session(uid, sid):
        return
    rows = await fetch("SELECT role, content FROM history WHERE session_id=? ORDER BY id", (sid,))
    text = "\n\n".join(("Вы: " if r["role"] == "user" else "NOVA: ") + r["content"] for r in rows) or "(пусто)"
    await target.answer_document(BufferedInputFile(text.encode("utf-8"), filename=f"session_{sid}.txt"))


@dp.callback_query(F.data == "m:new")
async def cb_new(cb: CallbackQuery):
    await new_session(cb.from_user.id)
    await cb.answer("🆕 Новая сессия создана")
    await render(cb, *await screen_main(cb.from_user.id))


@dp.callback_query(F.data == "m:sess")
async def cb_sess(cb: CallbackQuery):
    await render(cb, *await screen_sessions(cb.from_user.id))


@dp.callback_query(F.data == "ssnew")
async def cb_ssnew(cb: CallbackQuery):
    await new_session(cb.from_user.id)
    await cb.answer("🆕 Новая сессия создана")
    await render(cb, *await screen_sessions(cb.from_user.id))


@dp.callback_query(F.data.startswith("ss:"))
async def cb_ss(cb: CallbackQuery):
    await render(cb, *await screen_session(cb.from_user.id, int(cb.data[3:])))


@dp.callback_query(F.data.startswith("ssuse:"))
async def cb_ssuse(cb: CallbackQuery):
    uid, sid = cb.from_user.id, int(cb.data[6:])
    if await own_session(uid, sid):
        await upd(uid, session_id=sid)
        await cb.answer("Сессия переключена")
    await render(cb, *await screen_session(uid, sid))


@dp.callback_query(F.data.startswith("ssren:"))
async def cb_ssren(cb: CallbackQuery, state: FSMContext):
    sid = int(cb.data[6:])
    if not await own_session(cb.from_user.id, sid):
        return await cb.answer()
    await state.set_state(S.rename)
    await state.update_data(sid=sid)
    await render(cb, "Пришлите новое название сессии. /cancel — отмена.")


@dp.message(StateFilter(S.rename), F.text)
async def got_rename(m: Message, state: FSMContext):
    uid, sid = m.from_user.id, (await state.get_data())["sid"]
    if await own_session(uid, sid):
        await run("UPDATE sessions SET title=? WHERE id=?", (m.text.strip()[:60], sid))
    await state.clear()
    await render(m, *await screen_session(uid, sid))


@dp.callback_query(F.data.startswith("ssexp:"))
async def cb_ssexp(cb: CallbackQuery):
    await export_session(cb.message, cb.from_user.id, int(cb.data[6:]))
    await cb.answer()


@dp.callback_query(F.data.startswith("sscl:"))
async def cb_sscl(cb: CallbackQuery):
    uid, sid = cb.from_user.id, int(cb.data[5:])
    if await own_session(uid, sid):
        await run("DELETE FROM history WHERE session_id=?", (sid,))
        await cb.answer("История очищена")
    await render(cb, *await screen_session(uid, sid))


@dp.callback_query(F.data.startswith("ssdel:"))
async def cb_ssdel(cb: CallbackQuery):
    uid, sid = cb.from_user.id, int(cb.data[6:])
    if await own_session(uid, sid):
        await delete_session(uid, sid)
        await cb.answer("Сессия удалена")
    await render(cb, *await screen_sessions(uid))


# ── команды сессий
@dp.message(Command("sessions"))
async def cmd_sessions(m: Message, state: FSMContext):
    await state.clear()
    await render(m, *await screen_sessions(m.from_user.id))


@dp.message(Command("session"))
async def cmd_session(m: Message, state: FSMContext):
    await state.clear()
    uid = m.from_user.id
    await render(m, *await screen_session(uid, await active_session(uid)))


@dp.message(Command("new", "newsession"))
async def cmd_new(m: Message, command: CommandObject, state: FSMContext):
    await state.clear()
    await new_session(m.from_user.id, (command.args or "Новый чат").strip())
    await m.answer("🆕 Новая сессия создана. Прошлые сессии сохранены: /sessions")


@dp.message(Command("rename"))
async def cmd_rename(m: Message, command: CommandObject, state: FSMContext):
    await state.clear()
    uid = m.from_user.id
    sid = await active_session(uid)
    if command.args:
        await run("UPDATE sessions SET title=? WHERE id=?", (command.args.strip()[:60], sid))
        return await m.answer("✏️ Сессия переименована.")
    await state.set_state(S.rename)
    await state.update_data(sid=sid)
    await m.answer("Пришлите новое название сессии. /cancel — отмена.")


@dp.message(Command("clear"))
async def cmd_clear(m: Message, state: FSMContext):
    await state.clear()
    await run("DELETE FROM history WHERE session_id=?", (await active_session(m.from_user.id),))
    await m.answer("🧹 История текущей сессии очищена.")


@dp.message(Command("export"))
async def cmd_export(m: Message, state: FSMContext):
    await state.clear()
    uid = m.from_user.id
    await export_session(m, uid, await active_session(uid))


@dp.message(Command("history"))
async def cmd_history(m: Message, state: FSMContext):
    await state.clear()
    sid = await active_session(m.from_user.id)
    rows = await fetch("SELECT role, content FROM history WHERE session_id=? ORDER BY id DESC LIMIT 6", (sid,))
    if not rows:
        return await m.answer("В этой сессии пока пусто.")
    lines = [("👤 " if r["role"] == "user" else "🤖 ") + r["content"][:200].replace("\n", " ") for r in reversed(rows)]
    await m.answer("Последние сообщения сессии:\n\n" + "\n\n".join(lines))


# ── прочие команды
@dp.message(Command("settings"))
async def cmd_settings(m: Message, state: FSMContext):
    await state.clear()
    await render(m, *await screen_settings(m.from_user.id))


@dp.message(Command("temp"))
async def cmd_temp(m: Message, command: CommandObject, state: FSMContext):
    await state.clear()
    try:
        t = float((command.args or "").replace(",", "."))
        assert 0 <= t <= 2
    except Exception:
        return await m.answer("Использование: /temp 0.7 (число от 0 до 2)")
    await upd(m.from_user.id, temperature=t)
    await m.answer(f"Temperature: {t}")


@dp.message(Command("system"))
async def cmd_system(m: Message, command: CommandObject, state: FSMContext):
    await state.clear()
    uid = m.from_user.id
    if not command.args:
        cur = (await get_user(uid))["sys"]
        return await m.answer(f"Текущий промпт: {cur or 'не задан'}\n\n/system <текст> — задать, /system - — сбросить")
    await upd(uid, sys="" if command.args.strip() == "-" else command.args.strip()[:4000])
    await m.answer("✅ Системный промпт обновлён.")


async def safe_resolve(uid, pid):
    try:
        return await resolve(uid, pid)
    except ValueError:
        return None


VISION_TXT = {True: "👁 да, видит изображения", False: "нет, только текст", None: "неизвестно — попробуйте прислать картинку"}


@dp.message(Command("vision"))
async def cmd_vision(m: Message, state: FSMContext):
    await state.clear()
    uid = m.from_user.id
    u = await get_user(uid)
    prov = await safe_resolve(uid, u["provider"])
    v = await vision_status(prov, u["model"]) if prov and u["model"] else None
    await m.answer(f"Модель: {u['model'] or '—'}\nПонимает изображения: {VISION_TXT[v]}\n\n"
                   "Vision-модели помечены 👁 в /models (кнопка «👁 Vision» оставляет только их). "
                   "Стикеры тоже передаются как картинки; для анимированных берётся превью.")


@dp.message(Command("status"))
async def cmd_status(m: Message, state: FSMContext):
    await state.clear()
    uid = m.from_user.id
    u = await get_user(uid)
    sid = await active_session(uid)
    title = (await own_session(uid, sid))["title"]
    prov = await safe_resolve(uid, u["provider"])
    v = await vision_status(prov, u["model"]) if prov and u["model"] else None
    mem = (await fetch("SELECT COUNT(*) c FROM memory WHERE user_id=?", (uid,)))[0]["c"]
    sk = (await fetch("SELECT COUNT(*) c FROM skills WHERE user_id=?", (uid,)))[0]["c"]
    has_key = bool(prov and prov["key"])
    await m.answer(
        f"📊 Состояние\n\nПровайдер: {await resolve_name(uid, u['provider'])} (ключ: {'есть' if has_key else 'нет'})\n"
        f"Модель: {u['model'] or '—'}\nVision: {VISION_TXT[v]}\nРежим: {MODES[u['mode']][0]}\n"
        f"Temperature: {u['temperature']}\nСессия: {title} ({await msg_count(sid)} сообщ.)\n"
        f"Фактов в памяти: {mem}, навыков: {sk}")


COMMANDS = [
    ("menu", "Главное меню"), ("models", "Выбрать модель"), ("providers", "Провайдеры и ключи"),
    ("sessions", "Сессии"), ("newsession", "Новая сессия"), ("session", "Текущая сессия"),
    ("rename", "Переименовать сессию"), ("history", "Последние сообщения"), ("export", "Экспорт сессии в файл"),
    ("clear", "Очистить текущую сессию"), ("mode", "Режим ассистента"), ("memory", "Память о вас"),
    ("remember", "Запомнить факт"), ("skills", "Навыки"), ("settings", "Настройки"),
    ("temp", "Temperature"), ("system", "Системный промпт"), ("vision", "Поддержка изображений"),
    ("status", "Текущее состояние"), ("help", "Справка"), ("cancel", "Отменить ввод"),
]


# ───────────────────────── Чат ─────────────────────────
locks: dict[int, asyncio.Lock] = {}


def to_md(t: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"*\1*", t, flags=re.S)
    return re.sub(r"^#{1,6}\s*(.+)$", r"*\1*", t, flags=re.M)


def split_text(t: str, n: int = 4000):
    out = []
    while len(t) > n:
        cut = t.rfind("\n", 0, n)
        cut = cut if cut > n // 2 else n
        out.append(t[:cut])
        t = t[cut:].lstrip("\n")
    return out + [t]


async def put(target, text, edit):
    for pm in ("Markdown", None):
        try:
            body = to_md(text) if pm else text
            if edit:
                return await target.edit_text(body, parse_mode=pm)
            return await target.answer(body, parse_mode=pm)
        except TelegramBadRequest as e:
            if "not modified" in str(e):
                return None
            if pm is None:
                raise


async def build_system(m: Message, u, sid, vision):
    uid = m.from_user.id
    title = (await own_session(uid, sid))["title"]
    s = MODES[u["mode"]][1]
    s += f"\nПользователя зовут {m.from_user.first_name}."
    s += (f"\n\nТы ведёшь непрерывный диалог в сессии «{title}». Предыдущие сообщения этой сессии переданы тебе выше: "
          "опирайся на них и не говори, что не помнишь прошлые сообщения.")
    if vision is not False:
        s += " Если к сообщению приложено изображение или стикер, ты можешь его видеть и описывать."
    if u["sys"]:
        s += "\n\nДополнительные инструкции пользователя:\n" + u["sys"]
    mem = await fetch("SELECT text FROM memory WHERE user_id=? ORDER BY id", (uid,))
    if mem:
        s += "\n\nФакты о пользователе:\n" + "\n".join(f"- {r['text']}" for r in mem)
    sk = await fetch("SELECT name, text FROM skills WHERE user_id=? ORDER BY id", (uid,))
    if sk:
        block = "\n\n".join(f"### {r['name']}\n{r['text']}" for r in sk)[:8000]
        s += "\n\nНавыки пользователя (применяй, когда запрос подходит):\n" + block
    return s


def prep_image(data: bytes) -> str:
    """Любую картинку (jpeg/png/webp-стикер…) → JPEG ≤1568px в base64."""
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    im.thumbnail((1568, 1568))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode()


async def download_img(bot, fobj):
    buf = io.BytesIO()
    await bot.download(fobj, destination=buf)
    try:
        return await asyncio.to_thread(prep_image, buf.getvalue())
    except Exception:
        raise ValueError("Не удалось обработать изображение.")


def image_content(prompt, img):
    if not img:
        return prompt
    return [{"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]


async def build_user(m: Message, vision):
    """→ (content для модели, текст для истории, base64-картинка или None)"""
    text = m.text or m.caption or ""
    fobj = None
    if m.sticker:
        st = m.sticker
        label = "[Стикер" + (f" {st.emoji}" if st.emoji else "") + (f", набор {st.set_name}" if st.set_name else "") + "]"
        fobj = st.thumbnail if (st.is_animated or st.is_video) else st
        img = None
        if vision is not False and fobj:
            try:
                img = await download_img(m.bot, fobj)
            except ValueError:
                img = None
        if img:
            prompt = f"{label}\nПользователь прислал стикер (картинка приложена). Посмотри на него и отреагируй по смыслу."
        else:
            prompt = f"{label}\nПользователь прислал стикер, но картинка недоступна — ориентируйся на эмодзи."
        return image_content(prompt, img), label, img
    if m.document:
        d = m.document
        if (d.mime_type or "").startswith("image/"):
            fobj = d
        else:
            if (d.file_size or 0) > 1_000_000:
                raise ValueError("Файл больше 1 МБ.")
            buf = io.BytesIO()
            await m.bot.download(d, destination=buf)
            try:
                body = buf.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("Поддерживаются только текстовые файлы в UTF-8 (txt, md, py, json, csv…) и изображения.")
            full = f"Файл «{d.file_name}»:\n```\n{body[:30000]}\n```\n\n{text}"
            return full, full[:12000], None
    if m.photo:
        fobj = m.photo[-1]
    if fobj:
        if vision is False:
            raise ValueError("Выбранная модель не умеет смотреть изображения. Выберите модель с пометкой 👁 "
                             "в /models (проверить: /vision).")
        if (getattr(fobj, "file_size", 0) or 0) > 15_000_000:
            raise ValueError("Изображение слишком большое (максимум 15 МБ).")
        img = await download_img(m.bot, fobj)
        return image_content(text or "Опиши изображение.", img), f"[изображение] {text}".strip(), img
    return text, text, None


async def load_history(sid, vision):
    rows = await fetch("SELECT role, content, img FROM history WHERE session_id=? ORDER BY id DESC LIMIT ?",
                       (sid, HISTORY_MSGS))
    out, total = [], 0
    for r in rows:
        total += len(r["content"])
        if total > HISTORY_CHARS and out:
            break
        c = r["content"]
        if r["img"] and r["role"] == "user" and vision is not False:
            c = image_content(c, r["img"])
        out.append({"role": r["role"], "content": c})
    out.reverse()
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


@dp.message(StateFilter(None), (F.text & ~F.text.startswith("/")) | F.photo | F.document | F.sticker)
async def chat(m: Message):
    uid = m.from_user.id
    lock = locks.setdefault(uid, asyncio.Lock())
    if lock.locked():
        return await m.answer("⏳ Я ещё отвечаю на предыдущее сообщение.")
    async with lock:
        u = await get_user(uid)
        try:
            prov = await resolve(uid, u["provider"])
            if not prov:
                return await m.answer("Провайдер не найден. Откройте /providers.")
            if not prov["key"] and not prov["custom"]:
                return await m.answer(f"Для {prov['name']} нужен API-ключ.",
                                      reply_markup=mk([("🔑 Вставить ключ", f"k:{prov['id']}")], back=None))
            if not u["model"]:
                return await m.answer("Сначала выберите модель.",
                                      reply_markup=mk([("🤖 Выбрать модель", "m:model")], back=None))
            vision = await vision_status(prov, u["model"])
            user_content, user_text, img = await build_user(m, vision)
        except ValueError as e:
            return await m.answer(f"⚠️ {e}")

        sid = await active_session(uid)
        messages = ([{"role": "system", "content": await build_system(m, u, sid, vision)}]
                    + await load_history(sid, vision) + [{"role": "user", "content": user_content}])

        sent = await m.answer("⏳")
        acc, shown, last, err = "", "", 0.0, None
        try:
            async for piece in stream_chat(prov["base"], prov["key"], u["model"], messages, u["temperature"]):
                acc += piece
                if time.monotonic() - last >= 1.5:
                    view = acc[-3900:] + " ▌"
                    if view != shown:
                        try:
                            await sent.edit_text(view)
                        except TelegramRetryAfter as e:
                            await asyncio.sleep(e.retry_after)
                        except TelegramBadRequest:
                            pass
                        shown = view
                    last = time.monotonic()
        except ProviderError as e:
            err = f"⚠️ Ошибка провайдера {prov['name']}:\n{e}"
            if img and e.status in (400, 404, 413, 415, 422):
                err += "\n\nВозможно, эта модель не принимает изображения. Выберите модель с пометкой 👁 в /models."
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            err = f"⚠️ Сетевая ошибка: {type(e).__name__}"

        if not acc.strip():
            return await sent.edit_text(err or "⚠️ Пустой ответ от модели. Попробуйте другую модель.")
        chunks = split_text(acc)
        await put(sent, chunks[0], True)
        for c in chunks[1:]:
            await put(m, c, False)
        if err:
            await m.answer(err)

        # сохраняем в историю сессии (картинку держим только у последнего сообщения с ней)
        if img:
            await run("UPDATE history SET img=NULL WHERE session_id=?", (sid,))
        await run("INSERT INTO history(user_id,session_id,role,content,img) VALUES(?,?,?,?,?)",
                  (uid, sid, "user", user_text[:12000], img))
        await run("INSERT INTO history(user_id,session_id,role,content) VALUES(?,?,?,?)",
                  (uid, sid, "assistant", acc[:12000]))
        title = (await own_session(uid, sid))["title"]
        if title == "Новый чат" and user_text.strip():
            title = re.sub(r"\s+", " ", user_text).strip()[:40]
        await run("UPDATE sessions SET title=?, updated=? WHERE id=?", (title, time.time(), sid))
        await run("DELETE FROM history WHERE session_id=? AND id NOT IN "
                  "(SELECT id FROM history WHERE session_id=? ORDER BY id DESC LIMIT ?)", (sid, sid, SESSION_CAP))


async def start_web():
    """HTTP-сервер для Render Web Service: /health нужен и платформе, и «будильнику»."""
    port = os.getenv("PORT")
    if not port:
        return None

    async def health(_):
        return web.Response(text="NOVA UNIVERSE bot: ok")

    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(port)).start()
    logging.info("health-сервер на порту %s", port)
    return runner


async def self_ping():
    """Будильник №1: бот сам стучится на свой публичный адрес, пока не спит."""
    url = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not url:
        return
    while True:
        await asyncio.sleep(240)
        try:
            async with session.get(url + "/health", timeout=aiohttp.ClientTimeout(total=30)) as r:
                await r.read()
        except Exception as e:
            logging.warning("self-ping: %s", type(e).__name__)


async def main():
    global session
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600, sock_read=180))
    backup = Backup(DB_PATH, fernet, session)
    if backup.enabled and await backup.restore():
        logging.info("БД восстановлена из резервной копии")
    await init_db()
    runner = await start_web()
    tasks = [asyncio.create_task(self_ping())]
    if backup.enabled:
        tasks.append(asyncio.create_task(backup.loop()))
    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
    try:
        await bot.set_my_commands([BotCommand(command=c, description=d) for c, d in COMMANDS])
    except Exception:
        logging.exception("set_my_commands failed")
    try:
        await dp.start_polling(bot)
    finally:
        for t in tasks:
            t.cancel()
        if backup.enabled:  # финальный снимок при остановке (Render шлёт SIGTERM)
            try:
                await asyncio.wait_for(backup.backup(force=True), 25)
            except Exception:
                logging.exception("final backup failed")
        if runner:
            await runner.cleanup()
        await session.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
