# from utils import get_video_id
from video_feed import generate_video, json
from flask import Flask, render_template, request, Response

app = Flask(__name__)

STATE_PATH = '../data/player_state.json'

def get_video_id(ytb_link):
    """
    extract a youtube video id
    """
    idx_pattern = ytb_link.find('?v=')
    ytb_id = ytb_link[idx_pattern + 3 : idx_pattern + 3 + 11]
    return ytb_id


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
    with open(STATE_PATH, 'w') as outfile:
        json.dump(output, outfile)

    return ''


# serve webcam video
@app.route('/webcam')
def stream_video():
   
    return Response(generate_video(),
                mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
