from flask import Flask,render_template,redirect,request

import Caption_it_web


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def caption():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/" + f.filename # ./static/images.jpg
        f.save(path)
        print(path)
        caption = Caption_it_web.caption_this_image(path)
        print(caption)
        # result_dict = {
        #     'image':path,
        #     'caption':caption
        # }

    return render_template("index.html")



if __name__=='__main__':
    app.run(debug=True)