import sqlite3
import struct
from contextlib import closing


def count_database_images(database):
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]


def read_components(sparse, folder):
    models = [
        path
        for path in sparse.iterdir()
        if path.name.isdigit() and (path / "images.bin").exists()
    ]
    components = []
    for path in sorted(models, key=lambda path: int(path.name)):
        images = read_image_names(path / "images.bin")
        with (path / "points3D.bin").open("rb") as file:
            (points,) = struct.unpack("<Q", file.read(8))
        components.append(
            {
                "index": int(path.name),
                "path": path.relative_to(folder).as_posix(),
                "registered_images": len(images),
                "points3D": points,
                "images": sorted(images),
            }
        )
    return components


def read_image_names(path):
    data = path.read_bytes()
    (count,) = struct.unpack_from("<Q", data, 0)
    offset = 8
    names = []
    for _ in range(count):
        offset += 64  # image_id, qvec, tvec, camera_id
        end = data.index(b"\0", offset)
        names.append(data[offset:end].decode())
        (points2d,) = struct.unpack_from("<Q", data, end + 1)
        offset = end + 9 + 24 * points2d  # x, y, point3D_id per point
    return names


def count_ply_vertices(path):
    with path.open("rb") as file:
        for line in file:
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.strip() == b"end_header":
                break
    return 0
