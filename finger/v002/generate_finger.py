#!/usr/bin/env python3
"""Build a closed index-finger mesh and write a Unity OBJ.

The finger runs base-to-tip along +Y, curls toward the palm (+Z), and is
wider across X than it is deep. One skin shell plus a separate fingernail
shell. Units are meters so Unity's importer can leave the scale at 1.
"""

from __future__ import annotations

import math
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Assets" / "Models"
PREVIEW = ROOT / "_preview.png"

LENGTH = 0.095
CURL = math.radians(8.0)
CURL_EXPONENT = 1.55
AROUND = 28
RINGS = 72
# Free-edge cap. v001 shrank the shaft on a quarter-circle from s=0.90 and
# fanned that cone to one pole, so the silhouette ended in a spear.
TIP_CAP_RINGS = 6
TIP_CAP_FORWARD = 0.0026
TIP_CAP_PALMAR_BIAS = 0.0009
TIP_CAP_DORSAL_PINCH = 0.18

SKIN = 0
NAIL = 1


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    span = edge1 - edge0
    if span == 0.0:
        return 0.0 if value < edge0 else 1.0
    t = clamp((value - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bump(s: float, center: float, width: float, amount: float) -> float:
    if width <= 0.0 or amount == 0.0:
        return 0.0
    u = (s - center) / width
    return amount * math.exp(-u * u)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a):
    return math.sqrt(dot(a, a))


def normalize(a):
    mag = length(a)
    if mag < 1e-12:
        return (0.0, 1.0, 0.0)
    return (a[0] / mag, a[1] / mag, a[2] / mag)


def curl_angle(s: float) -> float:
    return CURL * (clamp(s, 0.0, 1.0) ** CURL_EXPONENT)


def sample_centerline(samples: int):
    """Integrate a Y-up centerline that bends toward the palm (+Z)."""
    centers = []
    tangents = []
    palmars = []
    position = (0.0, 0.0, 0.0)
    steps = samples - 1
    for index in range(samples):
        s = index / steps
        phi = curl_angle(s)
        tangent = (0.0, math.cos(phi), math.sin(phi))
        palmar = (0.0, -math.sin(phi), math.cos(phi))
        centers.append(position)
        tangents.append(tangent)
        palmars.append(palmar)
        if index < steps:
            s_next = (index + 1) / steps
            phi_next = curl_angle(s_next)
            step = LENGTH / steps
            mid = (
                0.0,
                math.cos((phi + phi_next) * 0.5),
                math.sin((phi + phi_next) * 0.5),
            )
            position = add(position, mul(mid, step))
    return centers, tangents, palmars


def half_extents(s: float):
    """Lateral half-width and dorsal/palmar half-depth before creases."""
    wx = lerp(0.0096, 0.0036, s)
    wz = lerp(0.0082, 0.0032, s)
    wx += bump(s, 0.07, 0.050, 0.00145)
    wz += bump(s, 0.07, 0.050, 0.00110)
    wx += bump(s, 0.48, 0.032, 0.00070)
    wz += bump(s, 0.48, 0.032, 0.00085)
    wx += bump(s, 0.745, 0.024, 0.00040)
    wz += bump(s, 0.745, 0.024, 0.00048)
    # The free edge is a separate cap. Shrinking the shaft here made a spear.
    # Full-ring creases at the two interphalangeal joints. Palmar-only
    # dents did not move the mean radius enough to read as two valleys.
    scale = 1.0
    scale -= 0.30 * math.exp(-((s - 0.44) / 0.028) ** 2)
    scale -= 0.28 * math.exp(-((s - 0.70) / 0.024) ** 2)
    wx *= scale
    wz *= scale
    return wx, wz


def skin_point(s: float, theta: float, center, palmar):
    """Skin point. theta 0 is +X; positive sine is palmar (+ local Z)."""
    wx, wz = half_extents(s)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    lateral_amount = cosine * wx
    if sine >= 0.0:
        palmar_amount = sine * wz * (1.0 + 0.24 * sine)
        palmar_amount += bump(s, 0.90, 0.055, 0.00115) * sine
        palmar_amount -= bump(s, 0.44, 0.014, 0.00135) * sine
        palmar_amount -= bump(s, 0.705, 0.012, 0.00090) * sine
    else:
        dorsal = -sine
        palmar_amount = sine * wz * (1.0 - 0.10 * dorsal)
        extra = bump(s, 0.08, 0.045, 0.00155) * dorsal
        extra += bump(s, 0.49, 0.028, 0.00095) * dorsal
        extra += bump(s, 0.75, 0.020, 0.00050) * dorsal
        extra += bump(s, 0.795, 0.012, 0.00038) * dorsal
        if 0.80 <= s <= 0.96:
            window = smoothstep(0.80, 0.84, s) * (1.0 - smoothstep(0.93, 0.96, s))
            extra -= 0.00055 * window * dorsal
        palmar_amount -= extra
    point = add(center, add((lateral_amount, 0.0, 0.0), mul(palmar, palmar_amount)))
    outward = normalize(sub(point, center))
    return point, outward


def add_tip_cap(mesh, last_ids, tip_center, tip_tangent, tip_palmar):
    """Round the free edge. The profile meets the last shaft ring with a flat tangent."""
    stride = AROUND + 1
    rings = [last_ids]
    for step in range(1, TIP_CAP_RINGS + 1):
        theta = (math.pi / 2.0) * step / (TIP_CAP_RINGS + 1)
        shrink = math.cos(theta)
        forward = TIP_CAP_FORWARD * math.sin(theta)
        center = add(tip_center, mul(tip_tangent, forward))
        center = add(center, mul(tip_palmar, TIP_CAP_PALMAR_BIAS * math.sin(theta)))
        column_ids = []
        for column in range(stride):
            angle = 2.0 * math.pi * column / AROUND
            point, _outward = skin_point(1.0, angle, tip_center, tip_palmar)
            radial = sub(point, tip_center)
            radial_len = length(radial) or 1.0
            dorsal = max(0.0, -dot(radial, tip_palmar) / radial_len)
            local_shrink = shrink * (1.0 - TIP_CAP_DORSAL_PINCH * dorsal * math.sin(theta))
            position = add(center, mul(radial, local_shrink))
            column_ids.append(
                mesh.add_vert(position, (column / AROUND, 1.0), sub(position, tip_center), SKIN)
            )
        rings.append(column_ids)
    pole = add(tip_center, mul(tip_tangent, TIP_CAP_FORWARD))
    pole = add(pole, mul(tip_palmar, TIP_CAP_PALMAR_BIAS))
    pole_index = mesh.add_vert(pole, (0.5, 1.0), sub(pole, tip_center), SKIN)
    for proximal, distal in zip(rings, rings[1:]):
        for column in range(AROUND):
            a = proximal[column]
            b = proximal[column + 1]
            c = distal[column]
            d = distal[column + 1]
            mesh.add_tri(a, c, d, SKIN)
            mesh.add_tri(a, d, b, SKIN)
    for column in range(AROUND):
        mesh.add_tri(pole_index, rings[-1][column], rings[-1][column + 1], SKIN)
    return pole_index


class Mesh:
    def __init__(self):
        self.verts = []
        self.uvs = []
        self.outward = []
        self.groups = []
        self.faces = []

    def add_vert(self, point, uv, outward, group):
        self.verts.append(point)
        self.uvs.append(uv)
        self.outward.append(normalize(outward))
        self.groups.append(group)
        return len(self.verts) - 1

    def add_tri(self, a, b, c, group):
        self.faces.append((a, b, c, group))


def build_mesh():
    mesh = Mesh()
    centers, tangents, palmars = sample_centerline(RINGS)
    stride = AROUND + 1
    for ring in range(RINGS):
        s = ring / (RINGS - 1)
        center = centers[ring]
        palmar = palmars[ring]
        for column in range(stride):
            theta = 2.0 * math.pi * column / AROUND
            point, outward = skin_point(s, theta, center, palmar)
            mesh.add_vert(point, (column / AROUND, s), outward, SKIN)

    base_center = centers[0]
    base_tangent = tangents[0]
    base_palmar = palmars[0]
    dome_rings = 4
    dome_radius = 0.003
    dome_ids = []
    for step in range(1, dome_rings + 1):
        theta = (math.pi / 2.0) * step / (dome_rings + 1)
        shrink = math.cos(theta)
        center = add(base_center, mul(base_tangent, -dome_radius * math.sin(theta)))
        column_ids = []
        for column in range(stride):
            angle = 2.0 * math.pi * column / AROUND
            point, _outward = skin_point(0.0, angle, base_center, base_palmar)
            position = add(center, mul(sub(point, base_center), shrink))
            column_ids.append(
                mesh.add_vert(position, (column / AROUND, 0.0), sub(position, base_center), SKIN)
            )
        dome_ids.append(column_ids)

    base_index = mesh.add_vert(
        add(base_center, mul(base_tangent, -dome_radius)),
        (0.5, 0.0),
        mul(base_tangent, -1.0),
        SKIN,
    )
    tip_center = centers[-1]
    tip_tangent = tangents[-1]
    tip_palmar = palmars[-1]

    def add_tube(proximal, distal):
        for column in range(AROUND):
            a = proximal[column]
            b = proximal[column + 1]
            c = distal[column]
            d = distal[column + 1]
            mesh.add_tri(a, c, d, SKIN)
            mesh.add_tri(a, d, b, SKIN)

    for ring in range(RINGS - 1):
        add_tube(
            [ring * stride + column for column in range(stride)],
            [(ring + 1) * stride + column for column in range(stride)],
        )
    add_tube(dome_ids[0], [column for column in range(stride)])
    for step in range(1, dome_rings):
        add_tube(dome_ids[step], dome_ids[step - 1])

    for column in range(AROUND):
        mesh.add_tri(base_index, dome_ids[-1][column + 1], dome_ids[-1][column], SKIN)
    last_ring = [(RINGS - 1) * stride + column for column in range(stride)]
    pole_index = add_tip_cap(mesh, last_ring, tip_center, tip_tangent, tip_palmar)
    mesh.caps = (base_index, pole_index)

    add_nail(mesh, centers, palmars)
    orient_outward(mesh)
    normals = smooth_normals(mesh)
    return mesh, normals


def add_nail(mesh, centers, palmars):
    across = 16
    along = 12
    grid = []
    for j in range(along):
        row = []
        b = -1.0 + 2.0 * j / (along - 1)
        along_t = (b + 1.0) * 0.5
        corner = math.sqrt(max(0.0, 1.0 - abs(b) ** 6))
        width_scale = (0.70 + 0.30 * along_t) * (0.40 + 0.60 * corner)
        s = lerp(0.808, 0.942, along_t)
        for i in range(across):
            a = -1.0 + 2.0 * i / (across - 1)
            a_w = a * width_scale
            s_point = s - 0.007 * (a_w * a_w) * along_t
            theta = -math.pi / 2.0 + a_w * 0.60
            ring_p = s_point * (RINGS - 1)
            k0 = int(math.floor(ring_p))
            k1 = min(max(k0 + 1, 0), RINGS - 1)
            k0 = min(max(k0, 0), RINGS - 1)
            blend_p = clamp(ring_p - k0, 0.0, 1.0)
            center_p = add(mul(centers[k0], 1.0 - blend_p), mul(centers[k1], blend_p))
            palmar_p = normalize(add(mul(palmars[k0], 1.0 - blend_p), mul(palmars[k1], blend_p)))
            point, outward = skin_point(clamp(s_point, 0.0, 1.0), theta, center_p, palmar_p)
            camber = 0.00028 * (1.0 - a_w * a_w) * (1.0 - b * b)
            lift_bottom = 0.00012 + 0.00005 * along_t
            lift_top = lift_bottom + 0.00042 + camber
            row.append((point, outward, lift_bottom, lift_top, a_w, along_t))
        grid.append(row)

    def emit(sign):
        ids = []
        for j in range(along):
            row_ids = []
            for i in range(across):
                point, outward, lift_bottom, lift_top, a_w, along_t = grid[j][i]
                lift = lift_top if sign > 0.0 else lift_bottom
                position = add(point, mul(outward, lift))
                uv = ((a_w + 1.0) * 0.5, along_t)
                row_ids.append(mesh.add_vert(position, uv, mul(outward, sign), NAIL))
            ids.append(row_ids)
        return ids

    top = emit(1.0)
    bottom = emit(-1.0)
    centroid = (0.0, 0.0, 0.0)
    count = 0
    for row in top:
        for index in row:
            centroid = add(centroid, mesh.verts[index])
            count += 1
    centroid = mul(centroid, 1.0 / count)

    def add_quad(i00, i10, i11, i01, desired):
        normal = cross(sub(mesh.verts[i10], mesh.verts[i00]), sub(mesh.verts[i11], mesh.verts[i00]))
        if dot(normal, desired) < 0.0:
            mesh.add_tri(i00, i01, i11, NAIL)
            mesh.add_tri(i00, i11, i10, NAIL)
        else:
            mesh.add_tri(i00, i10, i11, NAIL)
            mesh.add_tri(i00, i11, i01, NAIL)

    for j in range(along - 1):
        for i in range(across - 1):
            corners = (top[j][i], top[j][i + 1], top[j + 1][i + 1], top[j + 1][i])
            desired = (0.0, 0.0, 0.0)
            for index in corners:
                desired = add(desired, mesh.outward[index])
            add_quad(*corners, desired)
            corners = (bottom[j][i], bottom[j][i + 1], bottom[j + 1][i + 1], bottom[j + 1][i])
            desired = (0.0, 0.0, 0.0)
            for index in corners:
                desired = add(desired, mesh.outward[index])
            add_quad(*corners, desired)

    def add_side(loop_top, loop_bottom):
        for index in range(len(loop_top) - 1):
            corners = (
                loop_top[index],
                loop_top[index + 1],
                loop_bottom[index + 1],
                loop_bottom[index],
            )
            center = (0.0, 0.0, 0.0)
            for corner in corners:
                center = add(center, mesh.verts[corner])
            add_quad(*corners, sub(mul(center, 0.25), centroid))

    add_side(top[0], bottom[0])
    add_side(list(reversed(top[-1])), list(reversed(bottom[-1])))
    add_side(
        [top[j][0] for j in range(along - 1, -1, -1)],
        [bottom[j][0] for j in range(along - 1, -1, -1)],
    )
    add_side([top[j][-1] for j in range(along)], [bottom[j][-1] for j in range(along)])


def orient_outward(mesh):
    """Flip only a cap whose fan points into the finger. The tube is already outward."""
    def flip_cap(cap):
        flipped = []
        for a, b, c, group in mesh.faces:
            if cap in (a, b, c):
                flipped.append((c, b, a, group))
            else:
                flipped.append((a, b, c, group))
        mesh.faces = flipped

    def cap_balance(cap):
        good = bad = 0
        for a, b, c, _group in mesh.faces:
            if cap not in (a, b, c):
                continue
            normal = cross(sub(mesh.verts[b], mesh.verts[a]), sub(mesh.verts[c], mesh.verts[a]))
            if dot(normal, mesh.outward[cap]) > 0.0:
                good += 1
            else:
                bad += 1
        return good, bad

    for cap in mesh.caps:
        good, bad = cap_balance(cap)
        if bad > good:
            flip_cap(cap)

    skin_bad = 0
    for a, b, c, group in mesh.faces:
        if group != SKIN:
            continue
        normal = cross(sub(mesh.verts[b], mesh.verts[a]), sub(mesh.verts[c], mesh.verts[a]))
        reference = add(add(mesh.outward[a], mesh.outward[b]), mesh.outward[c])
        if length(normal) > 1e-14 and dot(normal, reference) <= 0.0:
            skin_bad += 1
    if skin_bad:
        raise SystemExit(f"inward skin faces: {skin_bad}")


def smooth_normals(mesh):
    accum = [(0.0, 0.0, 0.0) for _ in mesh.verts]
    for a, b, c, _group in mesh.faces:
        normal = cross(sub(mesh.verts[b], mesh.verts[a]), sub(mesh.verts[c], mesh.verts[a]))
        for index in (a, b, c):
            accum[index] = add(accum[index], normal)
    buckets = {}
    for index, point in enumerate(mesh.verts):
        key = (round(point[0], 6), round(point[1], 6), round(point[2], 6), mesh.groups[index])
        buckets.setdefault(key, []).append(index)
    normals = [(0.0, 1.0, 0.0) for _ in mesh.verts]
    for indices in buckets.values():
        total = (0.0, 0.0, 0.0)
        for index in indices:
            total = add(total, accum[index])
        total = normalize(total)
        for index in indices:
            normals[index] = total
    return normals


def validate(mesh, normals):
    if any(not all(math.isfinite(coord) for coord in point) for point in mesh.verts):
        raise SystemExit("non-finite vertex")
    def position_key(index):
        point = mesh.verts[index]
        return (round(point[0], 6), round(point[1], 6), round(point[2], 6))

    edges = {}
    for a, b, c, group in mesh.faces:
        for edge in ((a, b), (b, c), (c, a)):
            key = (group, tuple(sorted((position_key(edge[0]), position_key(edge[1])))))
            edges[key] = edges.get(key, 0) + 1
        area = length(cross(sub(mesh.verts[b], mesh.verts[a]), sub(mesh.verts[c], mesh.verts[a])))
        if area < 1e-12:
            raise SystemExit("degenerate triangle")
    boundary = [count for count in edges.values() if count != 2]
    if boundary:
        raise SystemExit(f"mesh is not closed: {len(boundary)} bad edges")
    volume = 0.0
    for a, b, c, group in mesh.faces:
        if group != SKIN:
            continue
        volume += dot(mesh.verts[a], cross(mesh.verts[b], mesh.verts[c]))
    volume /= 6.0
    if volume < 1.0e-6 or volume > 8.0e-5:
        raise SystemExit(f"unexpected skin volume {volume:.6e}")
    xs = [point[0] for point in mesh.verts]
    ys = [point[1] for point in mesh.verts]
    zs = [point[2] for point in mesh.verts]
    span = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    if span[1] < span[0] * 3.0 or span[1] < span[2] * 2.2:
        raise SystemExit(f"mesh is not finger-shaped: spans {span}")
    if span[1] < 0.08 or span[1] > 0.12:
        raise SystemExit(f"finger length out of range: {span[1]}")
    nail_points = [mesh.verts[index] for index, group in enumerate(mesh.groups) if group == NAIL]
    if len(nail_points) < 50:
        raise SystemExit("nail shell missing")
    nail_y = sum(point[1] for point in nail_points) / len(nail_points)
    if nail_y < span[1] * 0.65:
        raise SystemExit("nail is not on the distal phalanx")
    skin_inward = nail_inward = 0
    for normal, outward, group in zip(normals, mesh.outward, mesh.groups):
        agreement = dot(normal, outward)
        if group == SKIN and agreement < 0.25:
            skin_inward += 1
        elif group == NAIL and agreement < -0.2:
            nail_inward += 1
    if skin_inward or nail_inward:
        raise SystemExit(f"smoothed normals point inward: skin {skin_inward}, nail {nail_inward}")
    centers, _tangents, palmars = sample_centerline(RINGS)

    def radius_at(s, theta):
        ring = s * (RINGS - 1)
        lower = min(max(int(math.floor(ring)), 0), RINGS - 1)
        upper = min(lower + 1, RINGS - 1)
        blend = clamp(ring - lower, 0.0, 1.0)
        center = add(mul(centers[lower], 1.0 - blend), mul(centers[upper], blend))
        palmar = normalize(add(mul(palmars[lower], 1.0 - blend), mul(palmars[upper], blend)))
        point, _outward = skin_point(s, theta, center, palmar)
        return length(sub(point, center))

    dorsal = [radius_at(index / 80.0, -math.pi / 2.0) for index in range(81)]
    peaks = [
        index / 80.0
        for index in range(1, 80)
        if dorsal[index] > dorsal[index - 1] and dorsal[index] >= dorsal[index + 1]
    ]
    expected_peaks = (0.08, 0.49, 0.75)
    for expected in expected_peaks:
        if not any(abs(peak - expected) < 0.06 for peak in peaks):
            raise SystemExit(f"missing knuckle near {expected}: peaks {peaks}")
    palmar = [radius_at(index / 80.0, math.pi / 2.0) for index in range(81)]
    valleys = [
        index / 80.0
        for index in range(1, 80)
        if palmar[index] < palmar[index - 1] and palmar[index] <= palmar[index + 1]
    ]
    for expected in (0.44, 0.70):
        if not any(abs(valley - expected) < 0.05 for valley in valleys):
            raise SystemExit(f"missing palmar crease near {expected}: valleys {valleys}")
    return {
        "volume": volume,
        "span": span,
        "vertices": len(mesh.verts),
        "triangles": len(mesh.faces),
        "peaks": peaks,
        "valleys": valleys,
    }


def write_obj(mesh, normals):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = MODEL_DIR / "Finger.obj"
    mtl_path = MODEL_DIR / "Finger.mtl"
    mtl_path.write_text(
        "\n".join(
            [
                "# Skin and nail for the index-finger mesh.",
                "newmtl Skin",
                "Ka 0.20 0.14 0.11",
                "Kd 0.86 0.62 0.50",
                "Ks 0.10 0.08 0.07",
                "Ns 18",
                "illum 2",
                "",
                "newmtl Nail",
                "Ka 0.28 0.26 0.25",
                "Kd 0.94 0.88 0.86",
                "Ks 0.35 0.33 0.32",
                "Ns 90",
                "illum 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "# Index finger, meters, Unity Y-up. Palm is +Z, tip curls toward the palm.",
        "# Generated by Tools/generate_finger.py.",
        "mtllib Finger.mtl",
        "o IndexFinger",
    ]
    for point in mesh.verts:
        lines.append(f"v {point[0]:.7f} {point[1]:.7f} {point[2]:.7f}")
    for uv in mesh.uvs:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
    for normal in normals:
        lines.append(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
    current = None
    names = {SKIN: "Skin", NAIL: "Nail"}
    for a, b, c, group in mesh.faces:
        if group != current:
            lines.append(f"usemtl {names[group]}")
            current = group
        lines.append(f"f {a + 1}/{a + 1}/{a + 1} {b + 1}/{b + 1}/{b + 1} {c + 1}/{c + 1}/{c + 1}")
    text = "\n".join(lines) + "\n"
    obj_path.write_text(text, encoding="utf-8")
    resource_dir = ROOT / "Assets" / "Resources"
    resource_dir.mkdir(parents=True, exist_ok=True)
    # Same bytes the runtime TextAsset parses. Unity imports .txt as text and .obj as a model.
    (resource_dir / "Finger.txt").write_text(text, encoding="utf-8")
    return obj_path


def write_preview(mesh, normals, path: Path):
    width, height = 960, 420
    panel_w = width // 3
    pixels = [[(28, 30, 36) for _ in range(width)] for _ in range(height)]
    depth = [[1e9 for _ in range(width)] for _ in range(height)]
    light = normalize((-0.30, 0.82, -0.48))
    views = (
        ("dorsal", lambda p: (p[0], p[1], p[2])),
        ("side", lambda p: (p[2], p[1], -p[0])),
        ("palm", lambda p: (-p[0], p[1], -p[2])),
    )
    albedo = {SKIN: (214, 154, 124), NAIL: (236, 220, 214)}
    for panel, (name, project) in enumerate(views):
        projected = []
        for point, normal, group in zip(mesh.verts, normals, mesh.groups):
            x, y, z = project(point)
            nx, ny, nz = project(normal)
            projected.append((x, y, z, (nx, ny, nz), group))
        xs = [item[0] for item in projected]
        ys = [item[1] for item in projected]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        origin_x = panel * panel_w
        pad = 28

        def map_point(x, y):
            sx = origin_x + pad + (x - min_x) / span * (panel_w - 2 * pad)
            sy = height - pad - (y - min_y) / span * (height - 2 * pad)
            return sx, sy

        for a, b, c, group in mesh.faces:
            pa = projected[a]
            pb = projected[b]
            pc = projected[c]
            ax, ay = map_point(pa[0], pa[1])
            bx, by = map_point(pb[0], pb[1])
            cx, cy = map_point(pc[0], pc[1])
            min_px = max(int(min(ax, bx, cx)), origin_x)
            max_px = min(int(max(ax, bx, cx)) + 1, origin_x + panel_w - 1)
            min_py = max(int(min(ay, by, cy)), 0)
            max_py = min(int(max(ay, by, cy)) + 1, height - 1)
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-8:
                continue
            normal = normalize(add(add(pa[3], pb[3]), pc[3]))
            shade = 0.32 + 0.68 * abs(dot(normal, light))
            base = albedo[group]
            color = tuple(max(0, min(255, int(channel * shade))) for channel in base)
            for py in range(min_py, max_py + 1):
                for px in range(min_px, max_px + 1):
                    w0 = (bx - px) * (cy - py) - (by - py) * (cx - px)
                    w1 = (cx - px) * (ay - py) - (cy - py) * (ax - px)
                    w2 = (ax - px) * (by - py) - (ay - py) * (bx - px)
                    if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                        z = (pa[2] * w0 + pb[2] * w1 + pc[2] * w2) / area
                        if z < depth[py][px]:
                            depth[py][px] = z
                            pixels[py][px] = color
        label = name.encode("ascii")
        draw_label(pixels, origin_x + 12, 18, label)

    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for red, green, blue in row:
            raw.extend((red, green, blue))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


FONT = {
    "d": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "o": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "r": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "s": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "a": (0b01110, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111),
    "l": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "i": (0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110),
    "e": (0b01110, 0b10001, 0b10001, 0b11111, 0b10000, 0b10000, 0b01110),
    "p": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "m": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
}


def draw_label(pixels, x, y, text):
    cursor = x
    height = len(pixels)
    width = len(pixels[0])
    for byte in text:
        glyph = FONT.get(chr(byte), (0, 0, 0, 0, 0, 0, 0))
        for row, bits in enumerate(glyph):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    px = cursor + col
                    py = y + row
                    if 0 <= px < width and 0 <= py < height:
                        pixels[py][px] = (230, 230, 230)
        cursor += 6


def main():
    mesh, normals = build_mesh()
    stats = validate(mesh, normals)
    obj_path = write_obj(mesh, normals)
    if "--preview" in sys.argv:
        write_preview(mesh, normals, PREVIEW)
    print(f"obj={obj_path}")
    print(
        "vertices={vertices} triangles={triangles} volume={volume:.6e} span={span}".format(**stats)
    )


if __name__ == "__main__":
    main()
