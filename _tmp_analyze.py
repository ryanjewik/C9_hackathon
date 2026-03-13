data = [
    (55,  273, 428, 155, 283, 448, 165),
    (67,  275, 434, 159, 277, 432, 155),
    (68,  287, None, None, 220, 442, 222),
    (150, 245, 402, 157, 178, 357, 179),
    (178, 317, 442, 125, 296, 421, 125),
    (183, 234, 425, 191, 237, 423, 186),
    (191, 278, 434, 156, 211, 388, 177),
    (212, 314, 433, 119, 295, 413, 118),
    # comparison: good crops
    (154, 367, 412, 45,  370, 410, 40),
    (46,  416, 434, 18,  386, 424, 38),  # threshold fallback
    (205, 394, 410, 16,  403, 441, 38),  # threshold fallback
]
hdr = f"{'crop':>5} {'ktr':>5} {'vtl':>5} {'gap':>5} {'cropL':>6} {'cropR':>6} {'cropW':>6} {'L_over':>7} {'R_over':>7}"
print(hdr)
print("-" * len(hdr))
for c, ktr, vtl, gap, cl, cr, cw in data:
    lo = ktr - cl if cl is not None else None
    ro = (cr - vtl) if vtl is not None and cr is not None else None
    print(f"{c:>5} {ktr:>5} {str(vtl):>5} {str(gap):>5} {cl:>6} {cr:>6} {cw:>6} {str(lo):>7} {str(ro):>7}")
