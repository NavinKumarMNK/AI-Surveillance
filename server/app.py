from flask import Flask, request, jsonify

app = Flask(__name__)

# This dictionary will be used to store incoming MQTT messages.
mqtt_messages = []

@app.route('/mqtt', methods=['POST'])
def mqtt_handler():
    data = request.get_json()
    mqtt_messages.append(data)  # Store the MQTT message
    print(f"Received MQTT message: {data}")
    return jsonify({"message": "MQTT message received"})

@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify({"messages": mqtt_messages})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9002)

