import sys
import os
sys.path.append(f'{os.getcwd()}/cameratest/flask')
import cameratest.flask.app as app


if __name__ == '__main__':
    app.run()