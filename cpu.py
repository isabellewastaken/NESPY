"""6502 core for the NES (2A03): official opcodes only, no decimal mode.

step() executes one instruction and returns the cycles it took.
Flags are stored as separate 0/1 values (fc, fz, ...) because that is
much faster in Python than masking a P byte on every instruction.
"""


class CPU:
    def __init__(self, bus):
        self.bus = bus
        bus.cpu = self
        self.a = self.x = self.y = 0
        self.sp = 0xFD
        self.pc = 0
        self.fc = self.fz = self.fv = self.fn = self.fd = 0
        self.fi = 1
        self.cycles = 0
        self.crossed = 0   # page-cross flag set by addressing modes
        self.extra = 0     # extra cycles from taken branches
        self.idle = False  # set when a JMP jumps to itself (see main.run_frame)

    # ---------------------------------------------------------------- setup
    def reset(self):
        r = self.bus.read
        self.pc = r(0xFFFC) | (r(0xFFFD) << 8)
        self.sp = 0xFD
        self.fi = 1
        self.cycles = 7

    def get_p(self, brk=0):
        return (self.fc | (self.fz << 1) | (self.fi << 2) | (self.fd << 3)
                | brk | 0x20 | (self.fv << 6) | (self.fn << 7))

    def set_p(self, v):
        self.fc = v & 1
        self.fz = (v >> 1) & 1
        self.fi = (v >> 2) & 1
        self.fd = (v >> 3) & 1
        self.fv = (v >> 6) & 1
        self.fn = (v >> 7) & 1

    # ---------------------------------------------------------------- helpers
    def push(self, v):
        self.bus.write(0x100 | self.sp, v)
        self.sp = (self.sp - 1) & 0xFF

    def pull(self):
        self.sp = (self.sp + 1) & 0xFF
        return self.bus.read(0x100 | self.sp)

    def _nz(self, v):
        self.fz = 1 if v == 0 else 0
        self.fn = v >> 7

    def _branch(self, cond, target):
        if cond:
            self.extra = 1 + (1 if (self.pc ^ target) & 0xFF00 else 0)
            self.pc = target

    def _adc(self, m):
        a = self.a
        r = a + m + self.fc
        self.fv = ((~(a ^ m) & (a ^ r)) >> 7) & 1
        self.fc = 1 if r > 0xFF else 0
        self.a = r & 0xFF
        self._nz(self.a)

    def _cmp(self, reg, m):
        self.fc = 1 if reg >= m else 0
        self._nz((reg - m) & 0xFF)

    # ---------------------------------------------------------------- interrupts
    def nmi(self):
        self.push(self.pc >> 8)
        self.push(self.pc & 0xFF)
        self.push(self.get_p(0))          # B flag clear for hardware interrupts
        self.fi = 1
        r = self.bus.read
        self.pc = r(0xFFFA) | (r(0xFFFB) << 8)
        self.cycles += 7

    # ---------------------------------------------------------------- addressing modes
    # Each returns an effective address (or None for implied/accumulator).
    def m_imp(self):
        return None

    def m_imm(self):
        a = self.pc
        self.pc = (a + 1) & 0xFFFF
        return a

    def m_zp(self):
        a = self.bus.read(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        return a

    def m_zpx(self):
        return (self.m_zp() + self.x) & 0xFF

    def m_zpy(self):
        return (self.m_zp() + self.y) & 0xFF

    def m_abs(self):
        r = self.bus.read
        pc = self.pc
        a = r(pc) | (r((pc + 1) & 0xFFFF) << 8)
        self.pc = (pc + 2) & 0xFFFF
        return a

    def m_abx(self):
        base = self.m_abs()
        a = (base + self.x) & 0xFFFF
        self.crossed = 1 if (base ^ a) & 0xFF00 else 0
        return a

    def m_aby(self):
        base = self.m_abs()
        a = (base + self.y) & 0xFFFF
        self.crossed = 1 if (base ^ a) & 0xFF00 else 0
        return a

    def m_ind(self):   # JMP ($xxxx), including the page-wrap bug
        r = self.bus.read
        ptr = self.m_abs()
        lo = r(ptr)
        hi = r((ptr & 0xFF00) | ((ptr + 1) & 0xFF))
        return lo | (hi << 8)

    def m_inx(self):
        r = self.bus.read
        p = (self.m_zp() + self.x) & 0xFF
        return r(p) | (r((p + 1) & 0xFF) << 8)

    def m_iny(self):
        r = self.bus.read
        p = self.m_zp()
        base = r(p) | (r((p + 1) & 0xFF) << 8)
        a = (base + self.y) & 0xFFFF
        self.crossed = 1 if (base ^ a) & 0xFF00 else 0
        return a

    def m_rel(self):   # returns the branch target
        off = self.bus.read(self.pc)
        self.pc = (self.pc + 1) & 0xFFFF
        if off & 0x80:
            off -= 256
        return (self.pc + off) & 0xFFFF

    # ---------------------------------------------------------------- loads / stores
    def op_lda(self, ad): self.a = self.bus.read(ad); self._nz(self.a)
    def op_ldx(self, ad): self.x = self.bus.read(ad); self._nz(self.x)
    def op_ldy(self, ad): self.y = self.bus.read(ad); self._nz(self.y)
    def op_sta(self, ad): self.bus.write(ad, self.a)
    def op_stx(self, ad): self.bus.write(ad, self.x)
    def op_sty(self, ad): self.bus.write(ad, self.y)

    # ---------------------------------------------------------------- transfers
    def op_tax(self, ad): self.x = self.a; self._nz(self.x)
    def op_tay(self, ad): self.y = self.a; self._nz(self.y)
    def op_txa(self, ad): self.a = self.x; self._nz(self.a)
    def op_tya(self, ad): self.a = self.y; self._nz(self.a)
    def op_tsx(self, ad): self.x = self.sp; self._nz(self.x)
    def op_txs(self, ad): self.sp = self.x

    # ---------------------------------------------------------------- stack
    def op_pha(self, ad): self.push(self.a)
    def op_php(self, ad): self.push(self.get_p(0x10))
    def op_pla(self, ad): self.a = self.pull(); self._nz(self.a)
    def op_plp(self, ad): self.set_p(self.pull())

    # ---------------------------------------------------------------- logic / arithmetic
    def op_and(self, ad): self.a &= self.bus.read(ad); self._nz(self.a)
    def op_ora(self, ad): self.a |= self.bus.read(ad); self._nz(self.a)
    def op_eor(self, ad): self.a ^= self.bus.read(ad); self._nz(self.a)

    def op_bit(self, ad):
        m = self.bus.read(ad)
        self.fz = 1 if (self.a & m) == 0 else 0
        self.fv = (m >> 6) & 1
        self.fn = m >> 7

    def op_adc(self, ad): self._adc(self.bus.read(ad))
    def op_sbc(self, ad): self._adc(self.bus.read(ad) ^ 0xFF)
    def op_cmp(self, ad): self._cmp(self.a, self.bus.read(ad))
    def op_cpx(self, ad): self._cmp(self.x, self.bus.read(ad))
    def op_cpy(self, ad): self._cmp(self.y, self.bus.read(ad))

    # ---------------------------------------------------------------- inc / dec
    def op_inc(self, ad):
        v = (self.bus.read(ad) + 1) & 0xFF
        self.bus.write(ad, v); self._nz(v)

    def op_dec(self, ad):
        v = (self.bus.read(ad) - 1) & 0xFF
        self.bus.write(ad, v); self._nz(v)

    def op_inx(self, ad): self.x = (self.x + 1) & 0xFF; self._nz(self.x)
    def op_iny(self, ad): self.y = (self.y + 1) & 0xFF; self._nz(self.y)
    def op_dex(self, ad): self.x = (self.x - 1) & 0xFF; self._nz(self.x)
    def op_dey(self, ad): self.y = (self.y - 1) & 0xFF; self._nz(self.y)

    # ---------------------------------------------------------------- shifts (ad None = accumulator)
    def op_asl(self, ad):
        v = self.a if ad is None else self.bus.read(ad)
        self.fc = v >> 7
        r = (v << 1) & 0xFF
        if ad is None: self.a = r
        else: self.bus.write(ad, r)
        self._nz(r)

    def op_lsr(self, ad):
        v = self.a if ad is None else self.bus.read(ad)
        self.fc = v & 1
        r = v >> 1
        if ad is None: self.a = r
        else: self.bus.write(ad, r)
        self._nz(r)

    def op_rol(self, ad):
        v = self.a if ad is None else self.bus.read(ad)
        r = ((v << 1) | self.fc) & 0xFF
        self.fc = v >> 7
        if ad is None: self.a = r
        else: self.bus.write(ad, r)
        self._nz(r)

    def op_ror(self, ad):
        v = self.a if ad is None else self.bus.read(ad)
        r = (v >> 1) | (self.fc << 7)
        self.fc = v & 1
        if ad is None: self.a = r
        else: self.bus.write(ad, r)
        self._nz(r)

    # ---------------------------------------------------------------- jumps / calls
    def op_jmp(self, ad):
        if ad == ((self.pc - 3) & 0xFFFF):
            self.idle = True        # JMP to itself: only an interrupt can break out
        self.pc = ad

    def op_jsr(self, ad):
        ret = (self.pc - 1) & 0xFFFF
        self.push(ret >> 8)
        self.push(ret & 0xFF)
        self.pc = ad

    def op_rts(self, ad):
        lo = self.pull()
        hi = self.pull()
        self.pc = (((hi << 8) | lo) + 1) & 0xFFFF

    def op_rti(self, ad):
        self.set_p(self.pull())
        lo = self.pull()
        hi = self.pull()
        self.pc = (hi << 8) | lo

    def op_brk(self, ad):
        pc = (self.pc + 1) & 0xFFFF     # BRK skips a padding byte
        self.push(pc >> 8)
        self.push(pc & 0xFF)
        self.push(self.get_p(0x10))
        self.fi = 1
        r = self.bus.read
        self.pc = r(0xFFFE) | (r(0xFFFF) << 8)

    # ---------------------------------------------------------------- branches (ad = target)
    def op_bcc(self, ad): self._branch(not self.fc, ad)
    def op_bcs(self, ad): self._branch(self.fc, ad)
    def op_beq(self, ad): self._branch(self.fz, ad)
    def op_bne(self, ad): self._branch(not self.fz, ad)
    def op_bmi(self, ad): self._branch(self.fn, ad)
    def op_bpl(self, ad): self._branch(not self.fn, ad)
    def op_bvs(self, ad): self._branch(self.fv, ad)
    def op_bvc(self, ad): self._branch(not self.fv, ad)

    # ---------------------------------------------------------------- flags / nop
    def op_clc(self, ad): self.fc = 0
    def op_sec(self, ad): self.fc = 1
    def op_cli(self, ad): self.fi = 0
    def op_sei(self, ad): self.fi = 1
    def op_cld(self, ad): self.fd = 0
    def op_sed(self, ad): self.fd = 1   # stored but BCD is not implemented (2A03)
    def op_clv(self, ad): self.fv = 0
    def op_nop(self, ad): pass

    # ---------------------------------------------------------------- execute
    def step(self):
        pc = self.pc
        op = self.bus.read(pc)
        self.pc = (pc + 1) & 0xFFFF
        entry = TABLE[op]
        if entry is None:
            raise NotImplementedError(f"Opcode {op:02X} at ${pc:04X}")
        fn, mode, cyc, penalty = entry
        self.crossed = 0
        self.extra = 0
        addr = mode(self)
        fn(self, addr)
        if penalty:
            cyc += self.crossed
        cyc += self.extra
        self.cycles += cyc
        return cyc


# name  opcode:mode:cycles  ('p' suffix = +1 cycle on page cross)
_SPEC = """
ADC 69:imm:2 65:zp:3 75:zpx:4 6D:abs:4 7D:abx:4p 79:aby:4p 61:inx:6 71:iny:5p
AND 29:imm:2 25:zp:3 35:zpx:4 2D:abs:4 3D:abx:4p 39:aby:4p 21:inx:6 31:iny:5p
ASL 0A:imp:2 06:zp:5 16:zpx:6 0E:abs:6 1E:abx:7
BCC 90:rel:2
BCS B0:rel:2
BEQ F0:rel:2
BIT 24:zp:3 2C:abs:4
BMI 30:rel:2
BNE D0:rel:2
BPL 10:rel:2
BRK 00:imp:7
BVC 50:rel:2
BVS 70:rel:2
CLC 18:imp:2
CLD D8:imp:2
CLI 58:imp:2
CLV B8:imp:2
CMP C9:imm:2 C5:zp:3 D5:zpx:4 CD:abs:4 DD:abx:4p D9:aby:4p C1:inx:6 D1:iny:5p
CPX E0:imm:2 E4:zp:3 EC:abs:4
CPY C0:imm:2 C4:zp:3 CC:abs:4
DEC C6:zp:5 D6:zpx:6 CE:abs:6 DE:abx:7
DEX CA:imp:2
DEY 88:imp:2
EOR 49:imm:2 45:zp:3 55:zpx:4 4D:abs:4 5D:abx:4p 59:aby:4p 41:inx:6 51:iny:5p
INC E6:zp:5 F6:zpx:6 EE:abs:6 FE:abx:7
INX E8:imp:2
INY C8:imp:2
JMP 4C:abs:3 6C:ind:5
JSR 20:abs:6
LDA A9:imm:2 A5:zp:3 B5:zpx:4 AD:abs:4 BD:abx:4p B9:aby:4p A1:inx:6 B1:iny:5p
LDX A2:imm:2 A6:zp:3 B6:zpy:4 AE:abs:4 BE:aby:4p
LDY A0:imm:2 A4:zp:3 B4:zpx:4 AC:abs:4 BC:abx:4p
LSR 4A:imp:2 46:zp:5 56:zpx:6 4E:abs:6 5E:abx:7
NOP EA:imp:2
ORA 09:imm:2 05:zp:3 15:zpx:4 0D:abs:4 1D:abx:4p 19:aby:4p 01:inx:6 11:iny:5p
PHA 48:imp:3
PHP 08:imp:3
PLA 68:imp:4
PLP 28:imp:4
ROL 2A:imp:2 26:zp:5 36:zpx:6 2E:abs:6 3E:abx:7
ROR 6A:imp:2 66:zp:5 76:zpx:6 6E:abs:6 7E:abx:7
RTI 40:imp:6
RTS 60:imp:6
SBC E9:imm:2 E5:zp:3 F5:zpx:4 ED:abs:4 FD:abx:4p F9:aby:4p E1:inx:6 F1:iny:5p
SEC 38:imp:2
SED F8:imp:2
SEI 78:imp:2
STA 85:zp:3 95:zpx:4 8D:abs:4 9D:abx:5 99:aby:5 81:inx:6 91:iny:6
STX 86:zp:3 96:zpy:4 8E:abs:4
STY 84:zp:3 94:zpx:4 8C:abs:4
TAX AA:imp:2
TAY A8:imp:2
TSX BA:imp:2
TXA 8A:imp:2
TXS 9A:imp:2
TYA 98:imp:2
"""

TABLE = [None] * 256
for _line in _SPEC.strip().splitlines():
    _name, *_entries = _line.split()
    _fn = getattr(CPU, "op_" + _name.lower())
    for _e in _entries:
        _opc, _mode, _cyc = _e.split(":")
        TABLE[int(_opc, 16)] = (
            _fn,
            getattr(CPU, "m_" + _mode),
            int(_cyc.rstrip("p")),
            _cyc.endswith("p"),
        )