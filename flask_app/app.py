from utils import get_video_id
from video_feed import generate_video, pd
import json
from flask import Flask, render_template, request, Response

video_info_path = 'data/video_data.csv'

app = Flask(__name__)



# Demo page
@app.route('/', methods = ['POST', 'GET'])
def demo():
    if  request.method == 'POST':
        ytb_link = request.form.get('link')
        video_id = get_video_id(ytb_link)
        return render_template('demo.html', video_id = video_id)

    else:
        return render_template('demo.html')



# access youtube video info
@app.route('/video_info', methods=['POST'])
def get_video_info():
    output = request.get_json()
    with open('data/player_state.json', 'w') as outfile:
        json.dump(output, outfile)

    return ''


# serve webcam video
@app.route('/webcam')
def stream_video():
   
    return Response(generate_video(),
                mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
