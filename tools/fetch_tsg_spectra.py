"""Download ONLY the spectra files (.tsg / .bip + .cal / .ini) for one hole from the NCI HyLogger zip,
skipping the core photos and the core-image raster (_cras.bip), using HTTP range requests (no full download needed).

Usage (from the repo folder, venv active):
    python tools/fetch_tsg_spectra.py https://thredds.nci.org.au/thredds/fileServer/rs07/WA/BWD072.zip
Files are written to INTERIM_HyLogger_Dataset_Package_GDA2020_DB/TSG/<HOLE>/  (git-ignored).
Add --list to only print the zip contents and sizes.
"""
import sys, io, os, zipfile, urllib.request
from pathlib import Path

class HttpFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP Range requests."""
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                print("Warning: server did not announce range support; trying anyway.")
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data
    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)

url = sys.argv[1]
hole = Path(url).stem
remote = HttpFile(url)
print(f"{hole}.zip on the server: {remote.size / 1e9:.2f} GB")
z = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=4 * 1024 * 1024))
items = z.infolist()
for i in items:
    print(f"  {i.file_size / 1e6:9.1f} MB  {i.filename}") if not i.filename.lower().endswith(".jpg") else None
print(f"  (+ {sum(i.filename.lower().endswith('.jpg') for i in items)} core photos, "
      f"{sum(i.file_size for i in items if i.filename.lower().endswith('.jpg')) / 1e9:.1f} GB, skipped)")
if "--list" in sys.argv:
    sys.exit()

out = Path(__file__).resolve().parents[1] / "INTERIM_HyLogger_Dataset_Package_GDA2020_DB" / "TSG" / hole
out.mkdir(parents=True, exist_ok=True)
for i in items:
    name = i.filename.replace("\\", "/").split("/")[-1]
    # spectra (.tsg/.bip) + their small calibration/settings files; skip core imagery (_cras, ~1 GB)
    if name.lower().endswith((".tsg", ".bip", ".cal", ".ini")) and "_cras" not in name.lower():
        dest = out / name
        if dest.exists() and dest.stat().st_size == i.file_size:
            print("already there:", name); continue
        print(f"downloading {name} ({i.file_size / 1e6:.0f} MB) ...", flush=True)
        with z.open(i) as src, open(dest, "wb") as dst:
            while chunk := src.read(8 * 1024 * 1024):
                dst.write(chunk)
print("Done ->", out)
