from flask import Flask,render_template,send_from_directory,request,url_for,redirect
from predict import SHARK_DETECTION
from datetime import datetime
import os

app = Flask(__name__, template_folder='template')
# ใส่ path ของ model ของเรา
pred = SHARK_DETECTION('best.pt')

@app.route('/public/<path:path>')
def send_report(path):
    return send_from_directory('public', path)

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/success', methods=['POST', 'GET'])
def successPOST():
    if request.method == 'POST':
        try:
            # โค้ดที่ใช้ในกรณี POST request
            f = request.files['file']
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            save_directory = 'D:/Senoir Proj/yolo webapp/public/'
            filename = os.path.join(save_directory, f"{date}.jpg")
            output = os.path.join(save_directory, f"{date}_output.jpg")

            f.save(filename)
            pred(filename, output)

            # ส่งรูปภาพที่ถูก upload โดยตรงเป็น response
            output_url = url_for('send_report', path=f"{date}_output.jpg")
            return render_template("success.html", image=output_url)
        except Exception as e:
            print(f"Error: {e}")
            return render_template("error.html", message="An error occurred.")
    else:
        return redirect("/", code=302)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80)
