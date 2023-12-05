# Instructions for Inference Your Own Data

Follow these steps to perform inference on your own data:

## 1. Step1: Prepare Your JSON file

We support inference on your own data by simply modifying the JSON file. Both single-reference and multiple-reference styles are supported.

For a single-reference style, use the following JSON format:

```json
[
    {
        "prompt": "Your prompt1",
        "style_path": "data/style_1.png"
    },
    {
        ...
    }
]

```


For a multiple-reference style, use the following JSON format:
```json
[
    {
        "prompt": "Your prompt1",
        "style_path": [
            "data/style_1.png",
            "data/style_2.png",
            "data/style_3.png"
        ]
    },
    {
        ...
    }
]
```

Note that the style path should be a relative path. We recommend organizing your test data according to the following structure:

```
.
├── your_eval_data
│   ├── data
│   │   ├── style_1.png
│   │   ├── style_2.png
│   │   └── style_3.png
│   └── eval_data.json
├── ...
```

## 2. Step2: modify the eval script

Update the `prompt_dir` and `filename` in the `scripts/run_infer_video.sh` script to point to your own test data. For example:
```bash
...
prompt_dir="/path/to/your_eval_data"
filename="eval_data.json"
...
```

After completing these steps, you can run the modified evaluation script to perform inference on your own data using the specified styles.