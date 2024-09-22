from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# 示例数据
data = {
    "bail_mill_name": None,
    "shot": None,
    "is_running": None,
    "data_time": None
}


# 处理API请求以更新数据
@app.route('/api/update_data', methods=['POST'])
def update_data():
    global data
    new_data = request.json
    data.update(new_data)
    return jsonify({"message": "Data updated successfully!"})


# 新增一个API路由，用于返回当前数据
@app.route('/api/get_data', methods=['GET'])
def get_data():
    return jsonify(data)  # 将当前的data变量返回为JSON格式


# 动态生成HTML页面
@app.route('/')
def index():
    html_content = f"""
    <html>
        <head>
            <title>Dynamic HTML</title>
        </head>
        <body>
            <h1>Ball Mill Information</h1>
            <p><strong>Name:</strong> {data['bail_mill_name']}</p>
            <p><strong>Shot:</strong> {data['shot']}</p>
            <p><strong>Is Running:</strong> {data['is_running']}</p>
            <p><strong>Date Time:</strong> {data['data_time']}</p>
        </body>
    </html>
    """
    return render_template_string(html_content)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
