# coding: utf-8
import logging
import os
import csv
root_path = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), ".")))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode='w')
_logger = logging.getLogger("translate_standard")


def transfer_yituxlsx(filename):
    i = 1
    f = open(root_path+"/data/train.txt", 'w', encoding='utf-8')
    csv_file = csv.reader(open(filename, 'r'))
    # f.truncate()
    for line in csv_file:
        if len(line) == 3 and line[2] in ['0', '1']:
            newline = line[0]+"\t\t"+str(line[1]).replace("\t", "")+"\t\t"+line[2]
            i += 1
            f.writelines(newline+"\n")
        else:
            print("第"+str(i)+"行数据格式错误")
    print("总共有"+str(i)+"行")
    f.close()


if __name__ == '__main__':
    transfer_yituxlsx(root_path+"/data/train.csv")