#!/usr/bin/python
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from os import curdir, sep
import cgi
from main import load_VGG_model, demo

PORT_NUMBER = 1234 # enter your port number here
HOST_NAME = 'abc.xyz.com' # enter your host name here


# This class will handles any incoming request from
# the browser
class myHandler(BaseHTTPRequestHandler):
    # Handler for the GET requests
    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"

        try:
            # Check the file extension required and
            # set the right mime type

            sendReply = False
            if self.path.endswith(".html"):
                mimetype = 'text/html'
                sendReply = True
            if self.path.endswith(".jpg"):
                mimetype = 'image/jpg'
                sendReply = True
            if self.path.endswith(".gif"):
                mimetype = 'image/gif'
                sendReply = True
            if self.path.endswith(".js"):
                mimetype = 'application/javascript'
                sendReply = True
            if self.path.endswith(".css"):
                mimetype = 'text/css'
                sendReply = True

            if sendReply == True:
                # Open the static file requested and send it
                f = open(curdir + sep + self.path)
                self.send_response(200)
                self.send_header('Content-type', mimetype)
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
            return

        except IOError:
            self.send_error(404, 'File Not Found: %s' % self.path)

            # Handler for the POST requests

    def do_POST(self):
        if self.path == "/send":
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                         'CONTENT_TYPE': self.headers['Content-Type'],
                         })

            image_link = form["image_link"].value

            result1, result2 = demo(image_link)

            print "Image Link: %s" % image_link
            self.send_response(200)
            self.end_headers()

            self.wfile.write('<html><header><title>Image Captioning</title></header>')
            self.wfile.write('<body><h1>Welcome to Generating Natural Description for images!</h1>')
            self.wfile.write('<h2>Final Project CS519 Deep Learning - Author: Khoi Nguyen</h2>')
            self.wfile.write('<p>You can search on Google Image or somewhere else for image link and paste link here</p>')
            self.wfile.write('<form method="POST" action="/send">')
            self.wfile.write('<label>Insert Image Link: </label>')
            self.wfile.write('<input type="text" name="image_link" value="'+image_link+'"/>')
            self.wfile.write('<input type="submit" value="Submit"/></form>')
            self.wfile.write('<h4>Result from model 1: '+result1+'</h4>')
            self.wfile.write('<h4>Result from model 2: '+result2+'</h4></form>')
            self.wfile.write('<img src="'+image_link+'" height="350">')
            self.wfile.write('</body></html>')

            return


try:
    # Create a web server and define the handler to manage the
    # incoming request
    server = HTTPServer((HOST_NAME, PORT_NUMBER), myHandler)
    print 'Started httpserver on port ', PORT_NUMBER
    load_VGG_model()
    print 'Loading Model Complete!, Start to Test now :)'

    # Wait forever for incoming htto requests
    server.serve_forever()

except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()
