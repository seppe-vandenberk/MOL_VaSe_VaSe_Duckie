"""Find the next offset for which every Duckiematrix engine port is available."""

import argparse
import errno
import socket

BASE_PORTS = tuple(range(7501, 7506)) + tuple(range(17510, 17518))
MAX_OFFSET = 65535 - max(BASE_PORTS)


def find_free_offset(start=10, stop=MAX_OFFSET):
    if not 0 <= start <= stop <= MAX_OFFSET:
        raise ValueError(f"Offset search must fit between 0 and {MAX_OFFSET}")
    for offset in range(start, stop + 1):
        sockets = []
        try:
            for base in BASE_PORTS:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sockets.append(sock)
                # Wildcard binding detects listeners on any IPv4 interface.
                sock.bind(("0.0.0.0", base + offset))
            return offset
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError(f"No complete simulator port set is free for offsets {start} through {stop}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=10)
    args = parser.parse_args()
    try:
        print(find_free_offset(args.start))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"{error}\n")
