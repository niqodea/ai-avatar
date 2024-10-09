# AI Avatar

Client-server application for AI avatar generation using diffusion models.

## Installation

We use Poetry to install dependencies, so make sure that you have Poetry installed and available in your PATH.

### Server

In a Python >= 3.11 environment:

```sh
make install-server
```

The machine running the server should have access to a GPU.

### Client

In a Python >= 3.11 environment:

```sh
make install-client
```

## Usage

### Server

To run the server:

```sh
make run-server
# You can specify a different port
make run-server -- --port=1234
# Set the MODEL_NAME env variable to use a custom base model
MODEL_NAME='Lykon/dreamshaper-8' make run-server
```

The server will start listening once all the models are loaded and ready.

### Client

To run the client:

```sh
make run-client -- \
    --output /path/to/avatar.png \
    --image /path/to/image.png \
    --prompt "A photo of an astronaut"
```

The client will request a generation to the server, wait for it to finish, and save the image to the output path.

To learn more about all possible arguments:

```sh
make run-client -- --help
```
