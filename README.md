# SurveillanceAI

> A surveillance system that uses AI with DeepStream Backend

## DeepStream Development Environment

> Refer docker/deepstream-dev.Dockerfile

## Future work

- [ ] Deepstream Support
- [ ] Triton Server Support
- [ ] OpenVINO Model Server Support

### Build the container

```bash
docker build -t deepstream:dev -f deepstream-dev.Dockerfile ../
```

### Run the container

```bash
xhost +SI:localuser:root
```

```bash
docker run --network=bridge --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --privileged -v /var/run/docker.sock:/var/run/docker.sock -v ./workspace/SurveillanceAI:/workspace/SurveillanceAI -p 22:22 -p 8000:8000 -p 8554:8554 deepstream:dev
```
