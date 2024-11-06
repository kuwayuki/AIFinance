# api_server.py
from flask import Flask, request, jsonify
import main_CAN_SLIM as MainPy  # main.pyをインポート

app = Flask(__name__)

@app.route('/main', methods=['POST'])
def run_future():
    # リクエストからパラメータを取得
    data = request.get_json()
    print(data)
    tickers = data.get('tickers')
    tickers = [ticker.strip().upper() for ticker in tickers.split(',')] 
    print(tickers)
    is_output_all_info = data.get('is_output_all_info', False)
    is_send_line = data.get('is_send_line', True)

    # return jsonify({"error": str(e)}), 500
    # main.py内のfuture関数を呼び出し
    try:
        result = MainPy.main(tickers, is_output_all_info = is_output_all_info, is_send_line = is_send_line)
        # return jsonify({"result": 'result'}), 200
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
