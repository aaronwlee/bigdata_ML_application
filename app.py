from main import create_app, socketio
import os

app = create_app()

if __name__ == '__main__':
    if app.config['PORT']:
        port = app.config['PORT']
    else:
        port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port)

