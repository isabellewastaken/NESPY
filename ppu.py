"""PPU: registers, VRAM, scroll latches, vblank/NMI, and scanline rendering
of background and sprites.

run_scanline(n) is called at the START of scanline n, before the CPU runs
that line's cycles. Visible lines are drawn here using the scroll state
(v/t/x) as it stands at that moment, which is what makes SMB's status-bar
split work: the game writes scroll registers after the sprite 0 hit, and
the next line picks them up.

Shortcuts (fine for SMB): OAM is snapshotted once per frame at line 0, and
the sprite 0 hit flag is raised at the start of the first line where sprite
0 overlaps an opaque background pixel, not at the exact dot.
"""
import os

import numpy as np

# Default 64-colour NES palette. Drop a standard 192-byte "nes.pal" file next
# to this script to use a different one.
_PAL_HEX = """
7C7C7C 0000FC 0000BC 4428BC 940084 A80020 A81000 881400 503000 007800 006800 005800 004058 000000 000000 000000
BCBCBC 0078F8 0058F8 6844FC D800CC E40058 F83800 E45C10 AC7C00 00B800 00A800 00A844 008888 000000 000000 000000
F8F8F8 3CBCFC 6888FC 9878F8 F878F8 F85898 F87858 FCA044 F8B800 B8F818 58D854 58F898 00E8D8 787878 000000 000000
FCFCFC A4E4FC B8B8F8 D8B8F8 F8B8F8 F8A4C0 F0D0B0 FCE0A8 F8D878 D8F878 B8F8B8 B8F8D8 00FCFC F8D8F8 000000 000000
"""


def load_palette():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nes.pal")
    if os.path.exists(path):
        data = open(path, "rb").read()
        if len(data) >= 192:
            return np.frombuffer(data[:192], np.uint8).reshape(64, 3).copy()
    return np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)]
                     for h in _PAL_HEX.split()], dtype=np.uint8)


# LUT[lo_byte, hi_byte] -> the 8 two-bit pixel values of that pattern row,
# leftmost pixel first. Replaces per-pixel bit twiddling.
_BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
LUT = ((_BITS[None, :, :] << 1) | _BITS[:, None, :]).astype(np.uint8)


class PPU:
    def __init__(self, cart):
        self.chr = cart.chr
        self.chr_is_ram = cart.chr_is_ram
        if cart.mirroring == "vertical":
            self.nt_map = (0, 1, 0, 1)    # $2000=$2800, $2400=$2C00 (SMB)
        elif cart.mirroring == "horizontal":
            self.nt_map = (0, 0, 1, 1)
        else:
            raise NotImplementedError("Four-screen mirroring not supported")

        self.vram = bytearray(0x800)      # two nametables
        self.palette = bytearray(32)
        self.oam = bytearray(256)

        self.ctrl = 0                     # $2000
        self.mask = 0                     # $2001
        self.oam_addr = 0                 # $2003
        self.vblank = False
        self.sprite0 = False
        self.overflow = False
        self.nmi_pending = False

        # Loopy scroll registers
        self.v = 0       # current VRAM address (15 bits)
        self.t = 0       # temporary VRAM address (15 bits)
        self.x = 0       # fine X scroll (3 bits)
        self.w = 0       # write toggle

        self.buffer = 0  # $2007 read buffer

        self.rgb = load_palette()
        self.pal_np = np.frombuffer(self.palette, dtype=np.uint8)  # live view
        self.index_buf = np.zeros((240, 256), dtype=np.uint8)   # palette index per pixel
        self.chr_np = np.frombuffer(self.chr, dtype=np.uint8)   # live view of CHR
        self._bg_key = -1                 # tile-row cache, see _bg_line
        self._bg_dirty = True
        self._tile_base = None
        self._palshift = None
        self.line_sprites = [[] for _ in range(240)]   # sprite indices per line
        self._zeros = np.zeros(256, dtype=np.uint8)

    # ------------------------------------------------------------ VRAM bus
    @staticmethod
    def _pal_index(a):
        i = a & 0x1F
        if i & 0x13 == 0x10:              # $3F10/14/18/1C mirror $3F00/04/08/0C
            i &= 0x0F
        return i

    def _read(self, a):
        a &= 0x3FFF
        if a < 0x2000:
            return self.chr[a]
        if a < 0x3F00:
            return self.vram[self.nt_map[(a >> 10) & 3] * 0x400 + (a & 0x3FF)]
        return self.palette[self._pal_index(a)]

    def _write(self, a, val):
        a &= 0x3FFF
        if a < 0x2000:
            if self.chr_is_ram:
                self.chr[a] = val
        elif a < 0x3F00:
            self.vram[self.nt_map[(a >> 10) & 3] * 0x400 + (a & 0x3FF)] = val
            self._bg_dirty = True         # invalidate the tile-row cache
        else:
            self.palette[self._pal_index(a)] = val & 0x3F

    # ------------------------------------------------------------ registers
    def read_reg(self, r):
        if r == 2:                        # PPUSTATUS
            s = (self.vblank << 7) | (self.sprite0 << 6) | (self.overflow << 5)
            self.vblank = False
            self.w = 0
            return s
        if r == 4:                        # OAMDATA
            return self.oam[self.oam_addr]
        if r == 7:                        # PPUDATA
            addr = self.v & 0x3FFF
            if addr >= 0x3F00:            # palette reads are immediate
                out = self._read(addr)
                self.buffer = self._read(addr - 0x1000)
            else:
                out = self.buffer
                self.buffer = self._read(addr)
            self.v = (self.v + (32 if self.ctrl & 4 else 1)) & 0x7FFF
            return out
        return 0                          # write-only registers

    def write_reg(self, r, val):
        if r == 0:                        # PPUCTRL
            was_nmi = self.ctrl & 0x80
            self.ctrl = val
            self.t = (self.t & 0x73FF) | ((val & 3) << 10)
            if not was_nmi and (val & 0x80) and self.vblank:
                self.nmi_pending = True   # enabling NMI during vblank fires it
        elif r == 1:                      # PPUMASK
            self.mask = val
        elif r == 3:                      # OAMADDR
            self.oam_addr = val
        elif r == 4:                      # OAMDATA
            self.oam[self.oam_addr] = val
            self.oam_addr = (self.oam_addr + 1) & 0xFF
        elif r == 5:                      # PPUSCROLL
            if self.w == 0:
                self.t = (self.t & 0x7FE0) | (val >> 3)
                self.x = val & 7
                self.w = 1
            else:
                self.t = (self.t & 0x0C1F) | ((val & 7) << 12) | ((val >> 3) << 5)
                self.w = 0
        elif r == 6:                      # PPUADDR
            if self.w == 0:
                self.t = (self.t & 0x00FF) | ((val & 0x3F) << 8)
                self.w = 1
            else:
                self.t = (self.t & 0x7F00) | val
                self.v = self.t
                self.w = 0
        elif r == 7:                      # PPUDATA
            self._write(self.v, val)
            self.v = (self.v + (32 if self.ctrl & 4 else 1)) & 0x7FFF

    # ------------------------------------------------------------ background
    def _inc_y(self):
        v = self.v
        if (v & 0x7000) != 0x7000:
            v += 0x1000
        else:
            v &= ~0x7000
            y = (v & 0x03E0) >> 5
            if y == 29:
                y = 0
                v ^= 0x0800          # wrap into the other vertical nametable
            elif y == 31:
                y = 0
            else:
                y += 1
            v = (v & ~0x03E0) | (y << 5)
        self.v = v

    def _bg_line(self):
        """Returns (pix, idx): 2-bit background pixel values and palette
        indices for the 256 pixels of the current line.

        The 33 tile addresses and attribute palettes only change when the
        scroll enters a new tile row/column, nametable memory is written, or
        the pattern table select changes. They are cached, so each of the 8
        lines in a tile row only re-reads its CHR bytes."""
        v = self.v
        key = (v & 0x0FFF) | ((self.ctrl & 0x10) << 8)   # excludes fine Y
        if key != self._bg_key or self._bg_dirty:
            self._bg_key = key
            self._bg_dirty = False
            coarse_x = v & 0x1F
            coarse_y = (v >> 5) & 0x1F
            nt_sel = (v >> 10) & 3
            pt = 0x1000 if self.ctrl & 0x10 else 0
            vram, nt_map = self.vram, self.nt_map
            row = coarse_y << 5
            arow = 0x3C0 + ((coarse_y >> 2) << 3)
            ashift = (coarse_y & 2) << 1
            bases = [0] * 33
            pals = [0] * 33
            for i in range(33):                  # 32 tiles + 1 for fine-X scroll
                cx = coarse_x + i
                off = nt_map[nt_sel ^ ((cx >> 5) & 1)] << 10   # crossing x=32 flips nametable
                cx &= 0x1F
                bases[i] = pt + (vram[off + row + cx] << 4)
                pals[i] = ((vram[off + arow + (cx >> 2)] >> (ashift | (cx & 2))) & 3) << 2
            self._tile_base = np.array(bases, dtype=np.intp)
            self._palshift = np.repeat(np.array(pals, dtype=np.uint8), 8)

        base = self._tile_base + ((v >> 12) & 7)
        chr_ = self.chr_np
        fx = self.x
        pix = LUT[chr_[base], chr_[base + 8]].reshape(-1)[fx:fx + 256]
        if not self.mask & 0x02:                 # hide background in leftmost 8 pixels
            pix[:8] = 0
        idx = self.pal_np[np.where(pix == 0, 0, self._palshift[fx:fx + 256] | pix)]
        return pix, idx

    # ------------------------------------------------------------ sprites
    def _eval_sprites(self):
        """Snapshot OAM into per-scanline sprite lists (done once per frame)."""
        h = 16 if self.ctrl & 0x20 else 8
        lines = self.line_sprites
        for lst in lines:
            lst.clear()
        oam = self.oam
        for i in range(64):
            y0 = oam[i << 2] + 1                 # OAM Y is one less than the top row
            if y0 >= 240:
                continue
            for ln in range(y0, min(y0 + h, 240)):
                lines[ln].append(i)

    def _sprite_line(self, line, bgpix, idx):
        """Composite sprites over `idx`; also detects the sprite 0 hit."""
        lst = self.line_sprites[line]
        if not lst:
            return idx
        if len(lst) > 8:
            self.overflow = True
            lst = lst[:8]                        # hardware draws only 8 per line

        tall = self.ctrl & 0x20
        h = 16 if tall else 8
        sp_table = 0x1000 if self.ctrl & 0x08 else 0
        oam, chr_, mask = self.oam, self.chr, self.mask
        spr = np.zeros(256, dtype=np.uint8)      # bits0-1 pixel, 2-3 palette, 4 behind-bg

        for i in reversed(lst):                  # lowest index last => ends up on top
            b = i << 2
            x, at = oam[b + 3], oam[b + 2]
            r = line - (oam[b] + 1)
            if at & 0x80:
                r = h - 1 - r                    # vertical flip
            tile = oam[b + 1]
            if tall:
                a = (0x1000 if tile & 1 else 0) + ((tile & 0xFE) << 4) + (r & 7) + ((r & 8) << 1)
            else:
                a = sp_table + (tile << 4) + r
            row = LUT[chr_[a], chr_[a + 8]]
            if at & 0x40:
                row = row[::-1]                  # horizontal flip
            n = 256 - x if x > 248 else 8        # clip at right edge
            seg = row[:n]
            nz = seg != 0

            if i == 0 and not self.sprite0:      # sprite 0 hit
                xs = np.arange(x, x + n)
                valid = xs < 255                 # never hits at x = 255
                if mask & 6 != 6:
                    valid &= xs >= 8             # left-8 clipping disables hits there
                if (nz & (bgpix[x:x + n] != 0) & valid).any():
                    self.sprite0 = True

            code = ((at & 3) << 2) | (0x10 if at & 0x20 else 0)
            view = spr[x:x + n]
            view[nz] = seg[nz] | code

        if not mask & 0x04:
            spr[:8] = 0                          # hide sprites in leftmost 8 pixels
        sp = spr & 3
        show = (sp != 0) & ~(((spr & 0x10) != 0) & (bgpix != 0))
        return np.where(show, self.pal_np[0x10 + (spr & 0x0F)], idx)

    # ------------------------------------------------------------ timing
    def run_scanline(self, line):
        mask = self.mask
        rendering = mask & 0x18
        if line < 240:
            if line == 0:
                self._eval_sprites()
            if rendering:
                self.v = (self.v & 0x7BE0) | (self.t & 0x041F)   # copy horizontal bits
            if mask & 0x08:
                pix, idx = self._bg_line()
            else:
                pix = self._zeros
                idx = np.full(256, self.palette[0], dtype=np.uint8)
            if mask & 0x10:
                idx = self._sprite_line(line, pix, idx)
            self.index_buf[line] = idx
            if rendering:
                self._inc_y()
        elif line == 241:
            self.vblank = True
            if self.ctrl & 0x80:
                self.nmi_pending = True
        elif line == 261:
            self.vblank = False
            self.sprite0 = False
            self.overflow = False
            if rendering:
                self.v = (self.v & 0x041F) | (self.t & 0x7BE0)    # copy vertical bits

    # ------------------------------------------------------------ output
    def frame_rgb(self):
        """Convert the frame's palette indices to a (240, 256, 3) RGB array."""
        return self.rgb[self.index_buf & 0x3F]

    @property
    def framebuffer(self):
        return self.frame_rgb()