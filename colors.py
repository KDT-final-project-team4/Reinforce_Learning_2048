# colors.py
def getTileBg(v):
    m = {
        0: 0xeee4da,
        2: 0xeee4da,
        4: 0xede0c8,
        8: 0xf2b179,
        16: 0xf59563,
        32: 0xf67c5f,
        64: 0xf65e3b,
        128: 0xedcf72,
        256: 0xedcc61,
        512: 0xedc850,
        1024: 0xedc53f,
        2048: 0xedc22e,
        4096: 0x3c3a32,
        8192: 0x3c3a32,
    }
    hex_val = m.get(v, 0x3c3a32)
    # Pygame에서 사용하기 위해 Hex를 RGB 튜플로 변환
    return ((hex_val >> 16) & 255, (hex_val >> 8) & 255, hex_val & 255)

def getTileFg(v):
    # Hex 문자열에 해당하는 RGB 튜플 반환
    if v >= 8:
        return (249, 246, 242)  # "#f9f6f2"
    else:
        return (119, 110, 101)  # "#776e65"