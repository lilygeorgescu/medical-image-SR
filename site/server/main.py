import time
from http.server import HTTPServer
from server import Server

HOST_NAME = 'localhost'
PORT_NUMBER = 8000
# TODO: verifica ca exista folderul si daca nu creaza-l


if __name__ == '__main__':
    httpd = HTTPServer((HOST_NAME, PORT_NUMBER), Server)
    print(time.asctime(), 'Server UP - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server DOWN - %s:%s' % (HOST_NAME, PORT_NUMBER))