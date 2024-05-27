from flask import Flask, request, jsonify

from Domain.VideoSynopsis.VideoSynopsis import video_synopsis

app = Flask(__name__)


@app.route('/video-path', methods=['POST'])
def video_path():
    data = request.get_json()
    filePath = data.get("filePath")
    path = fr"{filePath}"
    synopsis_output_name = video_synopsis(path)
    synopsis_output_path = fr"C:\Users\ashas\PycharmProjects\video-synopsis-ai\Output\{synopsis_output_name}"
    return jsonify({'synopsis_output_path': synopsis_output_path, 'message': 'Data received'}), 200


if __name__ == '__main__':
    app.run(port=5000)
