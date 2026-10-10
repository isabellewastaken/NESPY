"""CPU memory map. PPU, pad and cpu are attached after construction;
everything tolerates them being None so the CPU can run nestest alone."""


class Bus:
    def __init__(self, cart):
        self.ram = bytearray(0x800)
        self.prg = cart.prg
        self.prg_mask = len(cart.prg) - 1   # 16KB mirrors, 32KB doesn't
        self.ppu = None
        self.pad = None
        self.cpu = None                     # needed for DMA stall cycles

    def read(self, addr):
        if addr >= 0x8000:
            return self.prg[addr & self.prg_mask]
        if addr < 0x2000:
            return self.ram[addr & 0x7FF]
        if addr < 0x4000:
            return self.ppu.read_reg(addr & 7) if self.ppu else 0
        if addr == 0x4016:
            return self.pad.read() if self.pad else 0
        return 0   # sound stubs, open bus, unused

    def write(self, addr, val):
        if addr < 0x2000:
            self.ram[addr & 0x7FF] = val
        elif addr < 0x4000:
            if self.ppu:
                self.ppu.write_reg(addr & 7, val)
        elif addr == 0x4014:
            self._oam_dma(val)
        elif addr == 0x4016:
            if self.pad:
                self.pad.write(val)
        # $4000-$4013, $4015, $4017: sound stubs, ignore writes
        # $8000+: ROM, ignore writes (NROM has no registers)

    def _oam_dma(self, page):
        base = page << 8
        if self.ppu:
            self.ppu.oam[:] = bytes(self.read(base + i) for i in range(256))
        if self.cpu:
            # 513 cycles, +1 if started on an odd CPU cycle
            self.cpu.cycles += 513 + (self.cpu.cycles & 1)