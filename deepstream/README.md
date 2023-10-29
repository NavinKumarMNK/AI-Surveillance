# DeepStream

## Execute Graph

```
8555: host system rtsp server
8554: Deepstream RTSP output  
```

-> Host system
```bash
sudo ufw allow 8555
```

```bash
/opt/nvidia/graph-composer/execute_graph.sh ./composer/surveillance.yaml \
./composer/surveillance.parameters.yaml \
-d ./composer/target_x86_64.yaml -f
```

