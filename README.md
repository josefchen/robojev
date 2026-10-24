# RoboJev

### Did the robot complete the task? RoboJev-27B looks at the robot trying to complete the task and returns a probability, in roughly a quarter of a second.

**[Launch page](https://huggingface.co/spaces/josefchen/RoboJev)** &nbsp;·&nbsp; **[Try your video](https://huggingface.co/spaces/josefchen/RoboJev-demo)** &nbsp;·&nbsp; **[Every result](https://huggingface.co/spaces/josefchen/RoboJev-data)** &nbsp;·&nbsp; **[Evaluation data](https://huggingface.co/datasets/josefchen/RoboJev-evals)** &nbsp;·&nbsp; **[Model card](https://huggingface.co/josefchen/RoboJev-27B)**

<picture><source media="(max-width: 700px)" srcset="assets/cost-tall.webp"><img src="assets/cost-wide.webp" alt="Accuracy against cost: RoboJev-27B at 95.2% balanced accuracy and $0.192 per 1,000 answers, frontier models at 91.9% to 93.3% and $2.08 to $9.29" width="100%"></picture>

On the 268 held-out SO-101 runs, RoboJev-27B achieves 95.2% balanced accuracy at a cost of $0.19 per 1,000 answers. GPT-6 Astra, GPT-6.1 Sol, Claude Opus 5.5, and Claude Sonnet 5.5 achieve 91.9%-93.3% accuracy at a cost of $2.08-$9.29 per 1,000 answers. The accuracy gap compared to GPT-6 Astra is within noise and its cost is \~48x higher.

The language models all received the same question from four frames of the main camera, while RoboJev-27B read ten frames plus the other cameras. Costs are API list prices from our runs, and one H100 at $2.69 an hour for RoboJev-27B.

All model responses are in the [data explorer](https://huggingface.co/spaces/josefchen/RoboJev-data) and the [model card](https://huggingface.co/josefchen/RoboJev-27B) contains the full tables and explains the methodology.

RoboJev-27B: Model card, benchmarks, weights available on Hugging Face. The code for the Python client, dataset grader, video overlay, and LeRobot plugin for RoboJev-27B can be found in this repo.

## Install

To install RoboJev and the RoboJev command-line tools, install the RoboJev client with pip:

```bash
pip install robojev
pip install pyarrow          # LeRobot v3.0 datasets
pip install opencv-python    # live camera
```

To play with videos you will need to install ffmpeg. To work with LeRobot Datasets v3.0 you will need pyarrow and to use a live camera you will need OpenCV.

## Use it

Serve the model using vLLM and then call it from the small client in this repo.

```bash
vllm serve josefchen/RoboJev-27B --served-model-name robojev --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 16}' --logprobs-mode processed_logprobs --trust-request-chat-template
```

```python
from robojev import RoboJev

judge = RoboJev("http://localhost:8000")
judge.done("Put the ring on the peg", "attempt.mp4")          # cuts ten frames from the video
judge.done("Put the ring on the peg", frames)                 # or pass ten frames, start to end
judge.done("Put the ring on the peg", "attempt.mp4",
           views={"wrist": "wrist_last.jpg"})                 # with other cameras' last frames
```

The `done` method returns P(completed). The model requires 1x 80 GB GPU (bf16).

Making the same call from a shell works, and displays the probability of having successfully completed the task.

```bash
robojev "Put the ring on the peg" attempt.mp4
```

If you want to draw the probability on a video or live camera input:

```bash
robojev-overlay "Put the ring on the peg" attempt.mp4 out.mp4
robojev-overlay "Put the ring on the peg" --camera 0       # requires opencv-python
```

## Grade a LeRobot dataset

The command below will grade all episodes in a LeRobot dataset on the Hub. Note that the server must be running.

```bash
robojev-grade owner/eval_dataset --server http://localhost:8000 --out results.csv
```

For each episode, the command prints the episode index, task, P(done), and verdict (Done or Not yet). The summary includes the number graded, the success rate at the threshold (default = 0.5), and how many episodes are worth a look (with P(done) between 0.2 and 0.8). Results can also be saved as CSV with the `--out` option.

It will pick a 3rd person camera and parse ten frames/episode via HTTP byte ranges. Use `--camera` to specify a camera, `--episodes` to specify episodes, and `--header` for additional server headers. No token is required for public datasets, but set `HF_TOKEN` when using private ones. LeRobot v2.0 and v2.1 require requests, pillow and ffmpeg, and v3.0 also requires `pyarrow` to read its Parquet episode metadata.

## LeRobot

The `lerobot_plugin` directory provides a robojev reward model for LeRobot to use on recorded episodes as well as a reward step for real-robot training that supersedes the per-task reward classifier. See its README for more info.

The plugin can also be installed with pip directly from the repository.

```bash
pip install "git+https://github.com/josefchen/robojev#subdirectory=lerobot_plugin"
```

## How it works

<picture><source media="(max-width: 700px)" srcset="assets/how-tall.webp"><img src="assets/how-wide.webp" alt="Ten frames of each attempt and the other cameras go into RoboJev-27B; the attempt that lands the ring gets P(success) 1.00, the one where it slips off gets 0.02" width="100%"></picture>

Task is sent as free text, along with the video. Client samples 10 frames from the main camera uniformly between the first and last frame. You can optionally add the last frame from each additional camera. No text output from the model. Instead, the model outputs a probability, p(true), at the answer token. This means each answer requires only a single forward pass.

The model doesn't actually generate anything. We just stop the prompt at `[decision]:`, and the client reads the log-probs of the tokens `false` and `true` at that position, adds the biases from `decision.json`, divides by the temperature and applies a softmax. `decision.json` also contains the number of frames the model reads.

## Limitations

- The weakest performance of RoboJev-27B is in wrist cameras and on long horizon tasks like stirring or folding.
- RoboJev-27B can only see what you show it. If the video cuts off before the result, its answer is no good.
- When using just the final frame as input, RoboJev-27B performs worse on new robots, achieving an AUROC of 0.776 on the unseen labs.
- In robots outside its training set, RoboJev-27B isn't any more accurate than JEV-27B-VL when given the same 10 frames.
- Community training labels are from 2 models, and have been verified against 200 human labels; some are incorrect.
- RoboJev-27B is not a safety measure.

## Tests

```bash
pip install -e ".[test]"
pytest
```

These tests use a mock vLLM server instead of an actual GPU-based model server. This allows them to validate the functionality without requiring a GPU.

## License

Like the weights, the code in this repository is licensed under Apache-2.0. For licenses of the data used for training, see the model card.
