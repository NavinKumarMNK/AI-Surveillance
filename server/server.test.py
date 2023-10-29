import paho.mqtt.client as mqtt
import json

# MQTT settings
mqtt_broker = "192.168.1.4"  # Set your MQTT broker address
mqtt_topic = "mqtt"  # Set the topic to send messages to
mqtt_port = 9002

# Create an MQTT client
client = mqtt.Client()
client.connect(mqtt_broker, mqtt_port, keepalive=60)

# Sample MQTT message data``
mqtt_message = {
    "message": "Hello from MQTT client!",
    "sender": "Client 1"
}

# Publish an MQTT message
client.publish(mqtt_topic, json.dumps(mqtt_message))
client.disconnect()
