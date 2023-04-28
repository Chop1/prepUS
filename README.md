# prepUS

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`prepUS` is a Python package designed to preprocess ultrasound videos by removing layout and masking all pixels outside of the ultrasound field of view (FOV). The package is particularly useful for researchers and practitioners working with ultrasound data, as it helps to streamline the preprocessing workflow and improve the quality of the input data for further analysis.

## Features

- Removes layout from ultrasound videos
- Masks all pixels outside of the ultrasound FOV
- Supports multiple video formats (MP4, MOV, AVI)
- Easy-to-use command line interface (CLI)

## Installation

To install `prepUS`, simply run the following command:

```bash
pip install prepUS
```

## Usage

Once installed, you can use the prepUS CLI to preprocess your ultrasound videos. Here's an example:
```bash
removeLayout input_video.mp4 output_video.mp4 --thresh=0.05
```

In this example, input_video.mp4 is the input video file, output_video.mp4 is the output video file, and the thresh value is set to 0.05. Replace these values with your desired input and output file paths and threshold value.

## API

```python
from prepUS import removeLayout

result = removeLayout(input_file="input_video.mp4", output_file="output_video.mp4", thresh=0.05)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

