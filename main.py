"""Headless main loop (no window yet).

Usage:  python main.py smb.nes [frames]

Runs the given number of frames, then prints where the CPU is and dumps
nametable 0 as text, using SMB's tile layout (0-9 digits, $0A-$23 letters,
$24 blank). If the title screen text shows up, the CPU, bus, PPU registers,
vblank and NMI are all working.
"""
import sys

from cart import Cart
from bus import Bus
from cpu import CPU
from ppu import PPU
from pad import Pad

CYCLES_PER_LINE = 341 / 3      # 113.667 CPU cycles per scanline


def build(rom_path):
    cart = Cart(rom_path)
    bus = Bus(cart)
    cpu = CPU(bus)
    ppu = PPU(cart)
    pad = Pad()
    bus.ppu = ppu
    bus.pad = pad
    cpu.reset()
    return cpu, ppu, pad


def run_frame(cpu, ppu, target):
    """Run one 262-scanline frame. `target` is the running cycle goal, so any
    overshoot from the last instruction carries into the next line."""
    for line in range(262):
        ppu.run_scanline(line)
        if ppu.nmi_pending:
            ppu.nmi_pending = False
            cpu.nmi()
        target += CYCLES_PER_LINE
        while cpu.cycles < target:
            cpu.step()
            if cpu.idle:
                # JMP-to-self (SMB's main loop while waiting for NMI). Nothing can
                # change before the next scanline, so burn the remaining cycles
                # in one go: 3 per JMP, rounded up, same total as spinning.
                cpu.idle = False
                left = target - cpu.cycles
                if left > 0:
                    cpu.cycles += 3 * int(-(-left // 3))
    return target


def tile_char(t):
    if t < 10:
        return str(t)
    if t < 0x24:
        return chr(ord("A") + t - 10)
    return " " if t == 0x24 else "."


def dump_nametable(ppu, nt=0):
    base = ppu.nt_map[nt] * 0x400
    for row in range(30):
        line = "".join(tile_char(ppu.vram[base + row * 32 + col]) for col in range(32))
        if line.strip():
            print(f"{row:2d} {line}")


def main():
    rom = sys.argv[1] if len(sys.argv) > 1 else "smb.nes"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 120

    cpu, ppu, pad = build(rom)
    target = cpu.cycles
    for _ in range(frames):
        target = run_frame(cpu, ppu, target)

    print(f"After {frames} frames: PC=${cpu.pc:04X}  ctrl=${ppu.ctrl:02X}  mask=${ppu.mask:02X}")
    dump_nametable(ppu)


if __name__ == "__main__":
    main()