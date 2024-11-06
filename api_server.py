# api_server.py
from flask import Flask, request, jsonify
import main_CAN_SLIM as MainPy  # main.pyをインポート

app = Flask(__name__)

@app.route('/run-future', methods=['POST'])
def run_future():
    # リクエストからパラメータを取得
    data = request.get_json()
    print(data)
    tickers = data.get('tickers')
    # flag = data.get('flag', False)

    # main.py内のfuture関数を呼び出し
    try:
        result = MainPy.main(tickers)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
