import textcnn.Infer as Infer
import pandas as pd
import math
from tqdm import tqdm

# prediction part
batch_size = 256
Infer = Infer.Infer()


def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        yield texts


test_df = pd.read_csv("data/test.csv")
all_preds = []
for x in tqdm(batch_gen(test_df)):
    labels, s = Infer.infer(x)
    all_preds.extend(labels)
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": all_preds})
submit_df.to_csv("data/submission.csv", index=False)
print('success')

