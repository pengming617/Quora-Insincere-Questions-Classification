import textcnn.Infer as Infer
import csv

out = open('data/sample_submission.csv', 'w', newline='')
csv_write = csv.writer(out,dialect='excel')
csv_write.writerow(['qid', 'prediction'])
Infer = Infer.Infer()
csv_file = csv.reader(open('data/test.csv', 'r'))
i = 1
for line in csv_file:
    print(i)
    i = i + 1
    if len(line) == 2 and line[0] != 'qid':
        labels, s = Infer.infer([line[1]])
        csv_write.writerow([line[0], labels[0]])
    else:
        print(line[0])
print("success")