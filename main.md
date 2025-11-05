# RoadBuddy - Understanding the Road through Dashcam AI

## DESCRIPTION
Traffic is a critical issue in today's society. The challenge **"RoadBuddy – Understanding the Road through Dashcam AI"** aims to build a driving assistant capable of understanding video content from dashcams to quickly answer questions about traffic signs, signals, and driving instructions. This helps enhance safety, ensure legal compliance, and reduce driver distraction. The solution is also useful for post-accident analysis, evidence retrieval, logistics optimization, and enriching map/infrastructure data from common camera sources.

From a research perspective, the challenge creates a Vietnamese benchmark tailored to the traffic context in Vietnam.

**Input**: 
- A traffic video recorded by a car dashcam mounted on a car, lasting from 5 to 15 seconds in various scenarios such as urban/highway traffic, day/night, rain/sun. It may include traffic signs, signals, lane arrows, road markings, vehicles, etc.
- A user question.

 
**Output**: Corresponding answer to the question.

(Knowledge used to answer questions must comply with current Vietnamese road traffic laws.)

## EVALUATION
**Evaluation**:
- Each test case is run through the system to get results.
- Maximum runtime per test case: 30 seconds.

**Scoring**:
- Each submitted answer is scored as correct (1 point) if it matches the ground truth, or incorrect (0 points) otherwise.
- Answers that exceed the 30-second time limit receive 0 points.
- System score is calculated based on the total number of correct answers divided by the total number of test cases.

**Formula**:
Accuracy = Number of correct answers / Total number of test cases

**Tie-breaker**:
- When teams have equal accuracy scores, final ranking is determined by average inference time (lower is better).

## DATA
• Training data: ~600 videos, ~1000 samples including: questions, videos, answers, support frames 
• Public test: ~300 videos, ~500 samples including: questions, videos
• Private test: ~300 videos, ~500 samples including: questions, videos 

 
Training dataset including a train.json file and a folder of traffic videos. Each item in the annotations of training data include:

- video_path: the video which is used for the question 
- question: the question
- choices: choices of the questions 
- answer: the correct answer
- support_frames: reference frame in videos at a specific time

For example: **data\train\train.json**

{
    "__count__": 1490,
    "data": [
        {
            "id": "train_0001",
            "question": "Nếu xe ô tô đang chạy ở làn ngoài cùng bên phải trong video này thì xe đó chỉ được phép rẽ phải?",
            "choices": [
                "A. Đúng",
                "B. Sai"
            ],
            "answer": "B. Sai",
            "support_frames": [
                4.427402
            ],
            "video_path": "train/videos/2b840c67_386_clip_002_0008_0018_Y.mp4",
            "_unused_": "cbb77f7bf70be7d60ac580753f67ee61@1"
        },
        {
            "id": "train_0002",
            "question": "Phần đường trong video cho phép các phương tiện đi theo hướng nào khi đến nút giao?",
            "choices": [
                "A. Đi thẳng",
                "B. Đi thẳng và rẽ phải",
                "C. Đi thẳng, rẽ trái và rẽ phải",
                "D. Rẽ trái và rẽ phải"
            ],
            "answer": "C. Đi thẳng, rẽ trái và rẽ phải",
            "support_frames": [
                5.344766
            ],
            "video_path": "train/videos/2b840c67_386_clip_002_0008_0018_Y.mp4",
            "_unused_": "cbb77f7bf70be7d60ac580753f67ee61@2"
        },
        {
            "id": "train_0003",
            "question": "Biển chỉ dẫn 3 hướng di chuyển chính tiếp theo, đúng hay sai?",
            "choices": [
                "A. Đúng",
                "B. Sai"
            ],
            "answer": "A. Đúng",
            "support_frames": [
                3.845463
            ],
            "video_path": "train/videos/fe716b14_386_clip_003_0018_0024_N.mp4",
            "_unused_": "cbb77f7bf70be7d60ac580753f67ee61@3"
        },
    ]
}

The provided public test and private test dataset have the same format, but answer and supported frames are not provided.

**Submission format**:

Each team builds a solution for answering each question in the public and private test then submit to the submission page of the challenge.

The format of submission file is a CSV format as follow:
id,answer
testa_001,a
testb_002,a

## RULE
- Model size at inference time ≤ 8B parameters. You can use several small models if needed.
- Inference time ≤ 30s/sample.
- The machine for running the Docker of the final solution is configured with 1 GPU (RTX 3090 or NVIDIA A30), CPU: 16 cores, RAM: 64GB, Intel(R) Xeon(R) Gold 6442Y
- No Internet access during inference.
- Open-source data/models allowed.
- Synthetic data generation allowed using services or other models (LLM, VLM, etc.).
- After the competition ends, participants commit not to store any training data for personal purposes.
