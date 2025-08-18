# drone_project

# Docker to build ai deck firmware

```
docker run -it -v docker/ai_deck_bootloader:/module/ -v docker/ai_deck_examples:/examples --privileged --name "ai_deck" -P bitcraze/aideck /bin/bash
```
