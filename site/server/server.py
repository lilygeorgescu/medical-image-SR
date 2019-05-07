from http.server import BaseHTTPRequestHandler
import pdb


class Server(BaseHTTPRequestHandler):
  def do_HEAD(self):
    return
    
  def get_value(self, received, value):
  
    for item in received:
        if item[0] == value:
            return item[1]
    return None
    
  def do_POST(self):
    
    data = self.rfile.read(int(self.headers['Content-Length'])).decode("UTF-8").split('&')
    received = [x.split('=') for x in data]
    filename = self.get_value(received, 'file') + '.txt'
    image_name = self.get_value(received, 'img')
    method = self.get_value(received, 'method')
    
    file = open('answers/' + filename, 'w')
    file.write(image_name + '=' + method)
    file.close() 
    
    response = bytes("This is the response.", "utf-8") #create response

    self.send_response(200) #create header
    self.send_header("Content-Length", str(len(response)))
    self.end_headers()

    self.wfile.write(response) #send response
    return
    
  def do_GET(self):
    return
  def handle_http(self):
    return
  def respond(self):
    return