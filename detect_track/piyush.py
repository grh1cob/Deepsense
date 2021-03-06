import numpy as np
import socket


class Connect:
    def __init__(self):
        TCP_IP = '127.0.0.1'
        TCP_PORT = 5002
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print("waiting for connection")
        self.conn, address = s.accept()  # hangs until other end connects
        print('Connection address:', address)

    def put(self, val):
        val = np.atleast_2d(val)
        output = "%s" % "put"
        self.conn.send(output.encode('utf-8'))
        self.conn.send(("%10s" % str(val.shape).replace("(", "[").replace(")", "]")).encode('utf-8'))
        n_floats = val.size
        n_sent = 0
        val_str = val.astype('>f8').tostring(order='F')
        while n_sent < n_floats:
            n_tosend = min(128, n_floats - n_sent)
            self.conn.send(val_str[8 * n_sent:8 * (n_sent + n_tosend)])
            n_sent += n_tosend
            print("%i/%i floats sent" % (n_sent, n_floats))

    def get(self, name):
        self.conn.send(("%10s" % "radar_info").encode('utf-8'))
        # self.conn.send(("%10s"%name).encode('utf-8'))
        shape = tuple(map(int, self.conn.recv(30).split()))
        n_floats = np.prod(shape)
        val_flat = np.zeros(n_floats)
        n_read = 0
        while n_read < n_floats:
            n_toread = min(128, n_floats - n_read)
            val_flat[n_read:n_read + n_toread] = np.fromstring(self.conn.recv(n_toread * 8), '>f8')
            n_read += n_toread
            print("%i/%i floats read" % (n_read, n_floats))

        return np.ravel(val_flat.reshape(shape, order="F"))
