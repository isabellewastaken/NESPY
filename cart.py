"""iNES cartridge loader. Mapper 0 (NROM) only, which is all SMB needs."""


class Cart:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != b"NES\x1a":
            raise ValueError("Not an iNES file")

        prg_banks = data[4]          # 16KB units
        chr_banks = data[5]          # 8KB units (0 = CHR RAM)
        flags6, flags7 = data[6], data[7]

        self.mapper = (flags6 >> 4) | (flags7 & 0xF0)
        if self.mapper != 0:
            raise NotImplementedError(f"Mapper {self.mapper} not supported")

        if flags6 & 0x08:
            self.mirroring = "four"
        else:
            self.mirroring = "vertical" if flags6 & 1 else "horizontal"

        offset = 16 + (512 if flags6 & 0x04 else 0)   # skip trainer if present
        prg_size = prg_banks * 16384
        self.prg = data[offset:offset + prg_size]
        offset += prg_size

        if chr_banks:
            self.chr = bytearray(data[offset:offset + chr_banks * 8192])
            self.chr_is_ram = False
        else:
            self.chr = bytearray(8192)   # CHR RAM
            self.chr_is_ram = True