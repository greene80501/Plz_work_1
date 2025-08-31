import os, sys, shutil, json, time, csv, hashlib, mimetypes, argparse, re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import Counter

# --- File Extraction (No changes needed here) ---
from pypdf import PdfReader
from docx import Document
from PIL import Image, ExifTags
import chardet

try:
    import openpyxl
except Exception:
    openpyxl = None
try:
    import easyocr
except Exception:
    easyocr = None

# ---- NEW: Import the OpenAI library for DeepSeek ----
from openai import OpenAI

# --- Categories for classification ---
CATEGORIES = [
    "Personal", "School", "Work", "Finance", "Research", "Code", "Datasets",
    "Photos", "Screenshots", "Design", "Audio", "Video",
    "Installers", "Archives", "Other",
]

STOPWORDS = set("""
a an and are as at be by for from has have if in into is it its of on or that the their this to was were will with without your you we our
pdf doc docx ppt pptx xls xlsx csv txt page file report note photo image picture screenshot scan camera iphone samsung android
""".split())

# --- (All file extraction and helper functions like sha256, sniff_text, etc. remain unchanged) ---
def sha256(path: Path, max_bytes: int = 300 * 1024 * 1024) -> str | None:
    try:
        if path.stat().st_size > max_bytes: return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None
def sniff_text(path: Path, max_bytes=400_000) -> str:
    raw = path.read_bytes()[:max_bytes]
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    try: return raw.decode(enc, errors="ignore")
    except Exception: return raw.decode("utf-8", errors="ignore")
def extract_pdf(path: Path, max_pages=5) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages[:max_pages])
    except Exception:
        return ""
def extract_docx(path: Path, max_paras=200) -> str:
    try:
        d = Document(str(path))
        return "\n".join(p.text for p in d.paragraphs[:max_paras])
    except Exception:
        return ""
def extract_csv_peek(path: Path, lines=120) -> str:
    try:
        out = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                out.append(line.rstrip("\n"))
                if i + 1 >= lines: break
        return "\n".join(out)
    except Exception:
        return ""
def extract_xlsx_peek(path: Path, cells=220) -> str:
    if openpyxl is None: return ""
    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        ws = wb.active
        out, c = [], 0
        for r in ws.iter_rows(min_row=1, max_row=60, max_col=16, values_only=True):
            row = ",".join("" if v is None else str(v) for v in r)
            out.append(row); c += len(r)
            if c >= cells: break
        return "\n".join(out)
    except Exception:
        return ""
def extract_image_meta(path: Path) -> str:
    try:
        im = Image.open(str(path))
        info = {"format": im.format, "mode": im.mode, "size": im.size}
        exif_text = []
        exif = getattr(im, "getexif", lambda: None)()
        if exif:
            for k, v in exif.items():
                tag = ExifTags.TAGS.get(k, str(k))
                if isinstance(v, bytes): continue
                exif_text.append(f"{tag}: {v}")
        return json.dumps(info) + "\n" + "\n".join(exif_text[:40])
    except Exception:
        return ""
def extract_image_ocr_easyocr(reader, path: Path) -> str:
    if reader is None: return ""
    try:
        results = reader.readtext(str(path), detail=1)
        lines = [r[1] for r in results if isinstance(r, (list, tuple)) and len(r) >= 2]
        return "\n".join(lines)
    except Exception:
        return ""
def read_snippet(path: Path, mime: str, reader=None, max_bytes=400_000) -> Tuple[str, Dict[str, Any]]:
    ext = path.suffix.lower()
    meta = {"file_name": path.name, "mime": mime, "size": path.stat().st_size}
    text = ""
    if mime.startswith("text/") or ext in {".py",".js",".ts",".java",".cpp",".c",".cs",".go",".rs",".html",".css",".json",".md",".yml",".yaml"}:
        text = sniff_text(path, max_bytes=max_bytes)
    elif ext == ".pdf":
        text = extract_pdf(path)
    elif ext == ".docx":
        text = extract_docx(path)
    elif ext in {".csv", ".tsv"}:
        text = extract_csv_peek(path)
    elif ext in {".xlsx"}:
        text = extract_xlsx_peek(path)
    elif mime.startswith("image/"):
        meta_s = extract_image_meta(path)
        ocr = extract_image_ocr_easyocr(reader, path)
        text = (meta_s + ("\n\nOCR:\n" + ocr if ocr else "")).strip()
    return text[:max_bytes], meta
def simple_subcategory_snippet(text: str, fallback: str = "") -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) <= 24]
    common = [w for w, _ in Counter(words).most_common(6)]
    out = " ".join(common[:3]).strip()
    return out or fallback
def choose_target(base: Path, label: str, subcategory: str, auto_ok: bool) -> Path:
    if not auto_ok:
        return base / "Unsorted_Review"
    if label == "Personal":
        return base / "Personal"
    safe_cat = "".join(c for c in label if c.isalnum() or c in "-_ ").strip() or "Other"
    safe_sub = "".join(c for c in subcategory if c.isalnum() or c in "-_ ").strip()
    return (base / safe_cat / safe_sub) if safe_sub else (base / safe_cat)
def unique_destination(dst: Path) -> Path:
    if not dst.exists(): return dst
    stem, suf = dst.stem, dst.suffix
    for i in range(1, 9999):
        cand = dst.with_name(f"{stem}-{i}{suf}")
        if not cand.exists(): return cand
    return dst.with_name(f"{stem}-dup{int(time.time())}{suf}")

# ---- NEW: LLM CLASSIFICATION FUNCTION for DeepSeek ----
def classify_with_deepseek(client: OpenAI, file_name: str, content: str) -> Optional[str]:
    """
    Classifies content using the DeepSeek API and returns a single category name.
    """
    system_prompt = f"""
    You are an expert file organizer. Your task is to classify a file into ONE of the following categories:
    {', '.join(CATEGORIES)}

    Analyze the file's name and content, then respond with ONLY a JSON object containing the chosen category.
    Your response MUST be in the format: {{"category": "ChosenCategory"}}
    """
    
    user_prompt = f"Filename: {file_name}\n\nFile Content Preview:\n{content[:8000]}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            # Forcing JSON output is a good practice if the model supports it
            response_format={"type": "json_object"},
        )
        
        response_text = response.choices[0].message.content
        data = json.loads(response_text)
        category = data.get("category")

        if category in CATEGORIES:
            return category
        else:
            print(f"[WARN] LLM returned an invalid category: {category}")
            return "Other"
            
    except Exception as e:
        print(f"[ERROR] DeepSeek API call failed: {e}")
        return None

def main():
    ap = argparse.ArgumentParser(description="LLM-powered local file sorter using DeepSeek")
    ap.add_argument("source", help="Folder to sort")
    ap.add_argument("--dest", default=None, help="Destination base (default: <source>/Sorted_AI)")
    ap.add_argument("--move", action="store_true", help="Move files (default: dry-run)")
    ap.add_argument("--copy", action="store_true", help="Copy files (safe mode)")
    ap.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    ap.add_argument("--minconf", type=float, default=0.90, help="Confidence threshold (for logging). LLM is generally confident.")
    ap.add_argument("--ocr_langs", default="en", help="Comma-separated OCR languages (e.g., en,ko,de)")
    ap.add_argument("--ocr_gpu", action="store_true", help="Use GPU for EasyOCR if available")
    args = ap.parse_args()

    # --- API Key Check ---
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("Error: DEEPSEEK_API_KEY environment variable not set. Please get a key from platform.deepseek.com.")
    
    # --- Initialize the LLM Client for DeepSeek ---
    print("Initializing DeepSeek client...")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        sys.exit(f"Source not found: {source}")

    dest_base = Path(args.dest).expanduser().resolve() if args.dest else (source / "Sorted_AI")
    for d in (dest_base, dest_base / "Unsorted_Review", dest_base / "_logs"):
        d.mkdir(parents=True, exist_ok=True)

    def prelabel(path: Path, mime: str) -> str | None:
        ext = path.suffix.lower()
        name = path.name.lower()
        if ext in {".exe", ".msi", ".pkg", ".dmg"}: return "Installers"
        if ext in {".zip",".rar",".7z",".tar",".gz",".bz2",".xz"}: return "Archives"
        if mime.startswith("image/") and any(s in name for s in ["screenshot", "screen shot", "snip"]): return "Screenshots"
        if ext in {".mp3", ".wav", ".m4a", ".flac"}: return "Audio"
        if ext in {".mp4", ".mov", ".avi", ".mkv"}: return "Video"
        return None

    if easyocr is None:
        print("[WARN] easyocr not installed; OCR will be skipped.")
        reader = None
    else:
        langs = [s.strip() for s in args.ocr_langs.split(",") if s.strip()]
        print(f"Loading EasyOCR ({'+'.join(langs)}), gpu={args.ocr_gpu} …")
        reader = easyocr.Reader(langs, gpu=args.ocr_gpu)

    log_csv = dest_base / "_logs" / "moves.csv"
    new_log = not log_csv.exists()
    with log_csv.open("a", newline="", encoding="utf-8") as logf:
        writer = csv.writer(logf)
        if new_log:
            writer.writerow(["ts","action","from","to","label","subcategory","confidence","hash","model"])

        processed = 0
        for entry in source.iterdir():
            if entry.is_dir(): continue
            if args.limit and processed >= args.limit: break
            try:
                mime, _ = mimetypes.guess_type(entry.name)
                mime = mime or "application/octet-stream"
                
                early = prelabel(entry, mime)
                text, meta = read_snippet(entry, mime, reader=reader)

                if early is None:
                    label = classify_with_deepseek(client, entry.name, text)
                    conf = 0.95 if label and label != "Other" else 0.10
                    if label is None:
                        label = "Other"
                else:
                    label, conf = early, 0.99

                subcat = ""
                if label not in ["Personal", "Installers", "Archives"]:
                    subcat = simple_subcategory_snippet(text)

                auto_ok = conf >= args.minconf
                target_dir = choose_target(dest_base, label, subcat, auto_ok)
                target_dir.mkdir(parents=True, exist_ok=True)
                dst = unique_destination(target_dir / entry.name)

                action = "DRYRUN"
                if args.copy:
                    shutil.copy2(str(entry), str(dst)); action = "COPY"
                elif args.move:
                    shutil.move(str(entry), str(dst)); action = "MOVE"

                file_hash = sha256(dst if action != "DRYRUN" else entry) or ""
                writer.writerow([int(time.time()), action, str(entry), str(dst), label, subcat, f"{conf:.3f}", file_hash, "deepseek-chat"])
                print(f"[{action}] {entry.name} -> {target_dir.name} ({label}/{subcat or ''}, conf={conf:.2f})")
                processed += 1
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nInterrupted."); break
            except Exception as e:
                writer.writerow([int(time.time()), "ERROR", str(entry), "", "", "", "", "", f"{type(e).__name__}:{e}"])
                print(f"[ERROR] {entry.name}: {e}")

if __name__ == "__main__":
    main()