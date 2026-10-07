from runner.service import app
import os

HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 5000))

if __name__ == '__main__':
     app.run(host=HOST, port = PORT_NUMBER)