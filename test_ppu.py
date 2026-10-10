"""PPU regression tests on synthetic data (no ROM needed):  python test_ppu.py

Covers background colours/attributes/scrolling/nametable wrap/mid-frame split
and sprites (priority, flips, 8x16, clipping, 8-per-line limit, sprite 0 hit).
Tests poke p.vram directly, which bypasses the tile-row cache's invalidation,
so frame() marks the cache dirty first.
"""
import sys

from ppu import PPU

fails = 0


class FakeCart:
    def __init__(self):
        self.chr = bytearray(8192)
        self.chr_is_ram = True
        self.mirroring = "vertical"


def frame(p):
    p._bg_dirty = True
    p.run_scanline(261)
    for line in range(240):
        p.run_scanline(line)


def ok(name, cond):
    global fails
    print(f"{name:14}", "PASS" if cond else "FAIL")
    if not cond:
        fails += 1


def scroll(p, x, y=0):
    p.write_reg(5, x)
    p.write_reg(5, y)


# ---------------------------------------------------------------- background
def bg_tests():
    c = FakeCart()
    p = PPU(c)
    for r in range(8):
        c.chr[16 + r] = 0xF0                 # tile 1: left half colour 1
        c.chr[32 + r] = 0xFF                 # tile 2: solid colour 3
        c.chr[40 + r] = 0xFF
    p.palette[0], p.palette[1], p.palette[3], p.palette[5] = 0x22, 0x16, 0x30, 0x2A
    rgb = p.rgb

    def at(y, x, i):
        return tuple(p.frame_rgb()[y, x]) == tuple(rgb[i])

    p.vram[0] = 1
    p.vram[2] = 1
    p.vram[0x3C0] = 0b0100                   # top-right quad uses palette 1
    p.vram[32] = 2
    p.mask = 0x1E
    scroll(p, 0)
    frame(p)
    ok("bg c1", at(0, 0, 0x16) and at(0, 3, 0x16))
    ok("bg backdrop", at(0, 4, 0x22))
    ok("bg row1", at(8, 0, 0x30) and at(15, 7, 0x30) and at(16, 0, 0x22))
    ok("bg attr", at(0, 16, 0x2A))
    scroll(p, 3)
    frame(p)
    ok("bg fine x", at(0, 0, 0x16) and at(0, 1, 0x22))
    scroll(p, 16)
    frame(p)
    ok("bg coarse x", at(0, 0, 0x2A))
    p.vram[0x400] = 2
    scroll(p, 248)
    frame(p)
    ok("bg nt wrap", at(0, 8, 0x30))
    p.vram[0x400] = 0
    p.vram[5 * 32], p.vram[5 * 32 + 1] = 1, 2
    scroll(p, 0)
    p._bg_dirty = True
    p.run_scanline(261)
    for line in range(240):
        if line == 32:
            scroll(p, 8)                     # status-bar style split
        p.run_scanline(line)
    ok("bg split top", at(0, 0, 0x16))
    ok("bg split bot", at(40, 0, 0x30))
    scroll(p, 0, 8)
    frame(p)
    ok("bg scroll y", at(0, 0, 0x30))


# ---------------------------------------------------------------- sprites
def mk():
    c = FakeCart()
    p = PPU(c)
    for r in range(8):
        c.chr[16 * 3 + r] = 0xFF                       # tile 3 solid
        c.chr[16 * 4 + r] = 0xF0                       # tile 4 left half
        c.chr[16 * 5 + r] = 0xFF if r < 4 else 0x00    # tile 5 top half
    p.palette[0], p.palette[1] = 0x22, 0x30
    p.palette[0x11], p.palette[0x19] = 0x16, 0x2A
    p.oam[:] = bytes([0xFF] * 256)
    p.mask = 0x1E
    scroll(p, 0)
    return p


def spr(p, i, y, tile, at, x):
    p.oam[i * 4:i * 4 + 4] = bytes([y, tile, at, x])


def sprite_tests():
    S, BG, W = 0x16, 0x22, 0x30

    def at(p, y, x, i):
        return tuple(p.frame_rgb()[y, x]) == tuple(p.rgb[i])

    p = mk(); spr(p, 5, 9, 3, 0, 20); frame(p)
    ok("spr basic", at(p, 10, 20, S) and at(p, 17, 27, S) and at(p, 9, 20, BG)
       and at(p, 18, 20, BG) and at(p, 10, 28, BG) and at(p, 10, 19, BG))

    p = mk(); p.vram[2 * 32 + 5] = 3; spr(p, 6, 15, 3, 0x20, 36); frame(p)
    ok("spr behind", at(p, 16, 36, S) and at(p, 16, 40, W))
    p = mk(); p.vram[2 * 32 + 5] = 3; spr(p, 6, 15, 3, 0x00, 36); frame(p)
    ok("spr front", at(p, 16, 40, S))

    p = mk(); spr(p, 1, 0, 4, 0, 100); spr(p, 2, 20, 4, 0x40, 100); frame(p)
    ok("spr flip H", at(p, 1, 100, S) and at(p, 1, 104, BG) and at(p, 21, 107, S) and at(p, 21, 100, BG))
    p = mk(); spr(p, 1, 0, 5, 0, 100); spr(p, 2, 20, 5, 0x80, 100); frame(p)
    ok("spr flip V", at(p, 1, 100, S) and at(p, 8, 100, BG) and at(p, 21, 100, BG) and at(p, 28, 100, S))

    p = mk(); spr(p, 10, 50, 3, 0, 60); spr(p, 11, 50, 3, 2, 60); frame(p)
    ok("spr priority", at(p, 51, 60, S))

    p = mk()
    for k in range(9):
        spr(p, k + 8, 69, 3, 0, k * 10)
    frame(p)
    ok("8 per line", at(p, 70, 70, S) and at(p, 70, 80, BG) and p.overflow)

    p = mk(); p.mask = 0x18; spr(p, 3, 9, 3, 0, 4); frame(p)
    ok("left clip", at(p, 10, 4, BG) and at(p, 10, 8, S))

    p = mk(); p.vram[2 * 32 + 5] = 3; spr(p, 0, 15, 3, 0, 40)
    p._bg_dirty = True
    p.run_scanline(261)
    first = None
    for line in range(30):
        p.run_scanline(line)
        if p.sprite0 and first is None:
            first = line
    ok("s0 hit line", first == 16)
    p = mk(); spr(p, 0, 15, 3, 0, 40); frame(p)
    ok("s0 no bg", not p.sprite0)
    p = mk(); p.vram[2 * 32 + 31] = 3; spr(p, 0, 15, 3, 0, 255); frame(p)
    ok("s0 x=255", not p.sprite0)
    p = mk(); p.vram[2 * 32 + 31] = 3; spr(p, 0, 15, 3, 0, 248); frame(p)
    ok("s0 x<255", p.sprite0)
    p.run_scanline(261)
    ok("s0 clears", not p.sprite0)

    p = mk(); p.ctrl = 0x20; spr(p, 2, 29, 4, 0, 50); frame(p)
    ok("8x16", at(p, 30, 50, S) and at(p, 30, 54, BG) and at(p, 38, 50, S)
       and at(p, 38, 57, S) and at(p, 42, 50, BG))


if __name__ == "__main__":
    bg_tests()
    sprite_tests()
    print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)