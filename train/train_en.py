import torch
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import transformers.optimization
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='bert-large-cased', type=str, help='bert model')
parser.add_argument('--batch_size', default=20, type=int, help='batch size')
parser.add_argument('--dropout', default=0.4, type=float, help='dropout probability')
parser.add_argument('--epochs', default=10, type=int, help='epochs')
parser.add_argument('--reinit_pooler', default=False, type=bool, help='reinit_pooler')
parser.add_argument('--weight_decay', default=True, type=bool, help='weight decay')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--pooling', default='mean-max-avg', type=str, help='pooling layer type')
parser.add_argument('--out_dir', default='./model', type=str, help='output direction of the model to be saved')
parser.add_argument('--output', default='./output.txt', type=str, help='the file that stdout be redirected to')
args = parser.parse_args()

# 重定向标准输出至输出文件
savedStdout = sys.stdout
f = open(args.output,'w+')
sys.stdout = f

# 读入数据并打上对应标签
data_worse_en = pd.read_xml('data/en_sample_data/sample.negative.xml')
data_worse_en['label'] = 0

data_better_en = pd.read_xml('data/en_sample_data/sample.positive.xml')
data_better_en['label'] = 1

# 连接每个数据集作为训练集
data = pd.concat([data_worse_en[:-1], data_better_en[:-1]], axis=0).reset_index(drop=True)

"""
将将整个训练数据随机分为两组：一组包含80%的数据当作训练集和一组包含20%的数据当作测试集。
"""
X = data.review.values  # comment
y = data.label.values  # label
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, shuffle=True)

# GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.',flush=True)
    print('Device name:', torch.cuda.get_device_name(0),flush=True)

else:
    print('No GPU available, using the CPU instead.',flush=True)
    device = torch.device("cpu")

# bert tokenizer
tokenizer = BertTokenizer.from_pretrained(args.model)

# 进行token,预处理
def preprocessing_for_bert(data):
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

    return input_ids, attention_masks

# Encode 我们的数据
encoded_comment = [tokenizer.encode(sent, add_special_tokens=True, truncation=True, max_length=512) for sent in data.review.values]


# tokenize
MAX_LEN = max([len(sent) for sent in encoded_comment])
print("Max length of the encoded data:",MAX_LEN,flush=True)

# 在train，validate上运行preprocessing_for_bert
train_inputs, train_masks = preprocessing_for_bert(X_train)
validate_inputs, validate_masks = preprocessing_for_bert(X_validate)

# 转化为tensor类型
train_labels = torch.tensor(y_train)
validate_labels = torch.tensor(y_validate)

batch_size = args.batch_size

# 给训练集创建 DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, drop_last=True)

# 给验证集创建 DataLoader
validate_data = TensorDataset(validate_inputs, validate_masks, validate_labels)
validate_sampler = SequentialSampler(validate_data)
validate_dataloader = DataLoader(validate_data, sampler=validate_sampler, batch_size=batch_size)


class BertClassifier(nn.Module):
    def __init__(self, ):
        super(BertClassifier, self).__init__()
        D_in,D_out = 1024,2

        self.bert = BertModel.from_pretrained(args.model)
        self.dense = nn.Linear(D_in * 2, D_in)
        self.classifier = nn.Sequential(
            nn.Dropout(p=args.dropout),
            nn.LayerNorm(D_in),
            nn.Linear(D_in,D_out),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        # pooling
        if args.pooling == 'cls':
            out = out.last_hidden_state[:, 0, :]  # [batch, 768]
        elif args.pooling == 'pooler':
            out = out.pooler_output  # [batch, 768]
        elif args.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        elif args.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]
        elif args.pooling == 'mean-max-avg':
            sequence_output = out.hidden_states[-1]
            avg_pooled = sequence_output.mean(1)
            max_pooled = torch.max(sequence_output, dim=1)
            pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
            out = self.dense(pooled)

        logits = self.classifier(out)

        return logits


def initialize_model(epochs=2):
    # 初始化我们的Bert分类器
    bert_classifier = BertClassifier()
    # 用GPU运算
    bert_classifier.to(device)
    # 训练的总步数
    total_steps = len(train_dataloader) * epochs
    # 创建优化器
    if args.weight_decay:
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=args.lr,  # 默认学习率
                          eps=1e-8,  # 默认精度
                          correct_bias=True,
                          weight_decay=0.01
                          )
    else:
        optimizer = AdamW(bert_classifier.parameters(),
                          lr=args.lr,  # 默认学习率
                          eps=1e-8,  # 默认精度
                          correct_bias=True,
                          )
    # 学习率预热
    scheduler = transformers.optimization.get_polynomial_decay_schedule_with_warmup(optimizer,
                                                                        num_warmup_steps=0,
                                                                        num_training_steps=total_steps,
                                                                        power=4)
    return bert_classifier, optimizer, scheduler


loss_fn = nn.CrossEntropyLoss()  # 交叉熵


def train(model, train_dataloader, validate_dataloader=None, epochs=2, evaluation=False):
    model.train()
    
    # reinit pooler-layer
    if args.reinit_pooler:
        encoder_temp = model.bert
        encoder_temp.pooler.dense.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
        encoder_temp.pooler.dense.bias.data.zero_()
        for p in encoder_temp.pooler.parameters():
            p.requires_grad = True

    # reinit encoder layers
    # if args.reinit_layers > 0:
    #     # assert config.reinit_pooler
    #     logger.info(f"reinit  layers count of {str(args.reinit_layers)}")

    #     encoder_temp = model.bert
    #     for layer in encoder_temp.encoder.layer[-args.reinit_layers:]:
    #         for module in layer.modules():
    #             if isinstance(module, (nn.Linear, nn.Embedding)):
    #                 module.weight.data.normal_(mean=0.0, std=encoder_temp.config.initializer_range)
    #             elif isinstance(module, nn.LayerNorm):
    #                 module.bias.data.zero_()
    #                 module.weight.data.fill_(1.0)
    #             if isinstance(module, nn.Linear) and module.bias is not None:
    #                 module.bias.data.zero_()

    best_accuracy = 0.
    path = ''
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        print(
            f"{'Epoch':^7} | {'每40个Batch':^9} | {'训练集 Loss':^12} | {'测试集 Loss':^10} | {'测试集准确率':^9} | {'时间':^9}",flush=True)
        print("-" * 80,flush=True)

        t0_epoch, t0_batch = time.time(), time.time()

        total_loss, batch_loss, batch_counts = 0, 0, 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)

            loss = loss_fn(logits, b_labels)

            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()
            # 归一化，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数和学习率
            optimizer.step()
            scheduler.step()

            # Print每40个batch的loss和time
            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                # Print训练结果
                print(
                    f"{epoch_i + 1:^7} | {step:^10} | {batch_loss / batch_counts:^14.6f} | {'-':^12} | {'-':^13} | {time_elapsed:^9.2f}",flush=True)

                # 重置batch参数
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # 计算平均loss
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 80,flush=True)

        # =======================================
        #               Evaluation
        # =======================================
        if evaluation:
            # 每个epoch之后在我们的验证集上评估一下性能
            validate_loss, validate_accuracy = evaluate(model, validate_dataloader)

            # 保存当前性能最好的模型
            if validate_accuracy > best_accuracy:
                best_accuracy = validate_accuracy
                output_dir = os.path.join(args.out_dir,args.model,str(args.pooling))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_name = 'AUC'+str(validate_accuracy)+'_lr_' + str(args.lr) + '_pooling_' + str(args.pooling) + '_batch_' + str(args.batch_size)+'.pth'
                output_path = os.path.join(output_dir, output_name)
                if path and path != output_path:
                    # 删掉旧的准确率较低的模型
                    os.system('rm {}'.format(path))
                    print("Model {} has been removed".format(path),flush=True)
                path = output_path
                torch.save(model, output_path)
                print("Model Saved as :"+output_path,flush=True)

            # Print 整个训练集的耗时
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^10} | {avg_train_loss:^14.6f} | {validate_loss:^12.6f} | {validate_accuracy:^12.2f}% | {time_elapsed:^9.2f}",flush=True)
            print("-" * 80,flush=True)

def evaluate(model, test_dataloader):
    model.eval()

    test_accuracy = []
    test_loss = []

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels.long())
            test_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten()  # 返回一行中最大值的序号
            # 正确率
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            test_accuracy.append(accuracy)

    # 计算整体的平均正确率和loss
    val_loss = np.mean(test_loss)
    val_accuracy = np.mean(test_accuracy)

    return val_loss, val_accuracy

# 输出超参数加以保存
print("\n\nArgs for the training:",flush=True)
for k,v in sorted(vars(args).items()):
    print(k,'=',v,flush=True)
print("\n",flush=True)

# 实例化bert-classifier，并进行训练
bert_classifier, optimizer, scheduler = initialize_model(epochs=args.epochs)
print("Start training and testing:\n",flush=True)
train(bert_classifier, train_dataloader, validate_dataloader, epochs=args.epochs, evaluation=True)

# 输出参数总量
net = BertClassifier()
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())),flush=True)

# 关闭重定向文件，并将标准输出重新置为原输出
f.close()
sys.stdout = savedStdout