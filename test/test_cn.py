import torch
import argparse
import pandas as pd
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader,SequentialSampler
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model_path', default='../model/cn/model.pth', type=str, help='Path of the trained model')
parser.add_argument('-i','--input', default='./input.xml', type=str, help='Path of the input xml file')
parser.add_argument('-o','--output', default='./output.xml', type=str, help='Path of the output xml file')
args = parser.parse_args()

pooling = 'mean-max-avg'
model = 'yechen/bert-large-chinese'

class BertClassifier(nn.Module):
    def __init__(self, ):
        """
        freeze_bert (bool): 设置是否进行微调，0就是不，1就是调
        """
        super(BertClassifier, self).__init__()
        # 输入维度(hidden size of Bert)默认1024，输出维度(2)
        D_in, D_out = 1024,2

        # Bert模型
        self.dense = nn.Linear(D_in * 2, D_in)
        self.bert = BertModel.from_pretrained(model)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.LayerNorm(D_in),
            nn.Linear(D_in,D_out),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        # mean-max-avg:
        sequence_output = out.hidden_states[-1]
        avg_pooled = sequence_output.mean(1)
        max_pooled = torch.max(sequence_output, dim=1)
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
        out = self.dense(pooled)

        # 全连接，计算，输出label
        logits = self.classifier(out)

        return logits

dir = args.model_path.rstrip("model.pth")
batch_size = 24
tokenizer = BertTokenizer.from_pretrained(dir)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


class Test():
    def __init__(self,model_path,input_file,output_file) -> None:
        self.model = torch.load(model_path)
        self.input_file = input_file
        self.output_file = output_file
        self.preprocess(self.input_file)
        self.data = pd.read_xml(self.input_file)
        self.column_label = self.data.columns[1]
        self.data_length = len(self.data[self.column_label])
        with open(self.input_file,'r') as f:
            while True:
                line = f.readline()
                if not line.startswith('<'):
                    continue
                self.root_label = line.strip('\n').strip('<>')
                break

        
    def test(self):
        data = (self.data.copy())[self.column_label].values
        encoded_data = [tokenizer.encode(sent, add_special_tokens=True, truncation=True, max_length=512) for sent in data]
        MAX_LEN = max([len(sent) for sent in encoded_data])

        input_ids = []
        attention_masks = []
        for sent in data:
            encoded_sent = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,
                truncation= True,
                max_length=MAX_LEN,
                padding='max_length',
                return_attention_mask=True 
            )
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))        

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        test_data = TensorDataset(input_ids, attention_masks)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        predicts = []
        for batch in test_dataloader:
            b_input_ids,b_attn_mask = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask)
            preds = (torch.argmax(logits, dim=1).flatten()).cpu().numpy()
            preds = (preds * 2 - 1).tolist()   # 0,1调整为-1，1
            predicts+=preds

        predicts = predicts[:self.data_length]

        return self.generate_output_file(predicts)

    def generate_output_file(self,predicts):
        assert len(predicts) == self.data_length

        root = ET.Element(self.root_label)
        tree = ET.ElementTree(root)
        for row in self.data.index:
            entry = ET.Element(self.column_label)
            entry.set('id', str(row+1))
            entry.set('polarity',str(predicts[row]))
            entry.text = str(self.data[self.column_label][row])
            root.append(entry)

        self.__indent(root)
        tree.write(self.output_file, encoding='utf-8', xml_declaration=False)        


    def preprocess(self,input_file):
        text = []
        with open(input_file,'r') as f:
            text = f.readlines()

        with open(input_file,'w') as f:
            for line in text:
                line = line.replace('&','')
                f.write(line)

    def __indent(self,elem, level=0):
        i = "\n" + level*"\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.__indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

# Do testing on test dataset and generates a xml-formatted output file
test = Test(model_path=args.model_path, input_file=args.input, output_file=args.output)
test.test()
