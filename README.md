# Distilling-Anytime-Trajectories-with-Calibrated-Confidence
Instead of distilling only the best final answer (common in best-of-N / slow-teacher → fast-student), distill the entire improvement trajectory so the student can stream: draft → verified answer → corrected answer → final, each with a well-calibrated confidence and a learnable stop/continue policy.
Step 0
pip install -r requirements.txt
pip install -r requirements.openai.txt #(If you are creating the dataset)
pip install -r requirements.train.txt #(If you start the training)

Step 1 Dataset
Please execute the following command one per Windows Powershell in order to speed-up the process.

python scripts\make_batch_anytime_requests.py --split train --model gpt-4o-mini --out data\batch_anytime_train.jsonl

python scripts\submit_batch.py --infile data\batch_anytime_train.jsonl
