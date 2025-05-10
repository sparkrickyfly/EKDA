import torch
import transformers
import json
import sys
sys.path.append('/root/autodl-tmp/lako/LaKo-main/data_process')
import pickle
import nltk.stem.porter as pt
from re import template
from tqdm import tqdm
import numpy as np
import os.path as osp
import numpy as np
from rank_bm25 import BM25Okapi
import pdb
import os
from src.data import Dataset as LakoDataset
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
from transformers import AutoModel,AutoTokenizer
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from typing import Tuple
from torch import nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib


class opt(object):
    gpu=0
    epochs=1000
    early_stop=30
    dataset='okvqa'
    stream=1
    use_fact='yes'
    fact_use_way = 'concat'
    attention_score_style ='mean'
    consider_context_attention ='no'
    use_last_half_layer_attention = 'no'
    train_data="/root/autodl-tmp/lako/LaKo-main/vqa2_train_t5_3_v5_frequent_bm25_split21.json"
    eval_data="/root/autodl-tmp/lako/LaKo-main/vqa2_test_t5_1_v5_frequent_bm25_split21.json"
    model_size='base'

pt_stemmer = pt.PorterStemmer()  # Porter stemmer for word stemming

stop_words = ["yes","no","which","this","we","what","the","can","are","likely","you","where","does",'a','he','she','is',"","an","it","some","that","there",'how','other','or',
'bu','ha','hi','wa','ga','st','am','cd','rv','hp','uk','lo','ft','dc','pm','la','th','vw','ly','ox','my','lg','dr','\"i','\'s','mm','rd','3d','ny','ma','aa','re','fo','dy','nd','a ','ii','ex',
'av','ge','dj','tp','gp','os','de','wi','un','ct','pf','ot','al','co','ye','hu','mt','sa','bp','aw','tx','ca','ne','mr','jp','cb','\'a','fe','af','ar','du','od','vy','fa','bi','ti','si','ac','pa','tw',
'nw','iv','lb','  ',' ','ep','op','te','\"e','\"a','hd','oj','rm','a\'','o\'','ba','f5','ce','yo','yo','#2','mn','og','pt','sb','ds','$1','em','sd','ho','di','pn','db','ae','4h','cv','el','rc','le','v8',
'kk','na','vh','bt','qr','om','kc','ou','ln','b5','pu','mo','\"1','ah','kg','ax','pl','li','sw','fc','jr','sk','lf','jt','7,','mu','aq','pj','ky','jc','ab','ol','1.','2.','ay','ms','4,','bc','bo','km','ty',
'll','hr','oz','fi','cm','yr','pb','su','k9','k2','sr','uv','lu','j\'','mg','jk','ri','md','â½','hs','ed','eg','fu','gb','e2','sm','jo','\'i','fm','xl','bb','5g','da','et','ro','a1','io','a2','s8','v1','vx',
'ta','ww','cy','4\'','h4','ie','ki','4e','#1','rt','eu','ag','eo','i3','o2','ea','x3','\'o','nn','u-','$2','sl','>>','ec','nj','za','ck','mc','ra','ek','$4','4o','po','kw','sq','mj','e\"','nu','xx','b6','ei',
'5%','1x','cn','\"w','m\'','i','n','t','s','o',',','m','"','&','b','w','e','c','l','y','p','-','x','d','r','v','g','k','f','#','h','u','j','/','q','!','@','(','z',':','','of','with']

# Load various files
def load_init(base_path='/root/autodl-tmp/lako/LaKo-main/data_process/data/okvqa/cache/'):
    # Path to vqa2.0 training set
    with open(base_path+"3/train.json", "r", encoding='utf8') as ffp:
        trains = json.load(ffp)
    # Path to vqa2.0 test set
    with open(base_path+"1/valid.json", "r", encoding='utf8') as ffp:
        tests= json.load(ffp)
    # Image caption text (Note: this needs to be updated)
    with open(base_path+"all_coco_dict_caption.json", "r", encoding='utf8') as ffp:
        img2caption= json.load(ffp)
    # Note: Newly constructed kg triples, check format from this path, should be the same
    with open(base_path+"v5_tripleindex_database_frequent.json", "r") as ffp:
        tripleindex_database = json.load(ffp)
    # Note: Newly constructed kg triples, check format from this path, should be the same (stem version)
    with open(base_path+"v5_triplestemindex_database_frequent.json", "r") as ffp:
        triplestemindex_database = json.load(ffp)
    # Template for converting relations to sentences
    with open(base_path+"relation2template-v2.json", "r", encoding='utf8') as ffp:
        relation2template = json.load(ffp)
    # Previously processed text information contained in images
    with open(base_path+"image2text.json", "r", encoding='utf8') as ffp:
        image2text = json.load(ffp)
    return trains, tests, img2caption, tripleindex_database, triplestemindex_database, relation2template, image2text


# 将stem kg通过relation-template转化成句子
def convert_stemkg2sentence(triplestemindex_database, relation2template):
    four_tuple = dict()
    for i, triple_stem in tqdm(enumerate(triplestemindex_database.values()),
                               total=len(triplestemindex_database.values())):
        if triple_stem[1] in relation2template.keys():
            relation = relation2template[triple_stem[1]]
        elif triple_stem[1][-2] == "#" and triple_stem[1][-1] == "f":
            relation = "is more " + triple_stem[1][:-2] + " than"
        elif triple_stem[1][-2] == "#" and triple_stem[1][-1] == "r":
            relation = "is less " + triple_stem[1][:-2] + " than"
        else:
            relation = triple_stem[1]

        triple_sentence = triplestemindex_database[str(i)][0] + " " + relation + " " + triplestemindex_database[str(i)][
            2]
        four_tuple[i] = [triple_stem[0], triple_stem[1], triple_stem[2], triple_sentence]
    with open("four_tuple_stem.json", "w") as ffp:
        json.dump(four_tuple, ffp)
    return four_tuple


# 提取top500
def top_500kg(trains, img2caption, image2text, four_tuple):
    okvqa_train_list = list()
    for train in tqdm(trains, total=len(trains)):
        okvqa_train_dict = dict()
        question = train["sent"]
        # 训练用的最好的答案
        targets = list(train['label'].keys())
        if targets == []:
            continue
        else:
            target = targets[0]
        # eval的时候命中任意一个即可
        answer = train['label']

        img_id = train['img_id']
        captions = img2caption[str(train['img_id'])]
        caption_sentence = ""

        # 此处是将，如果图像本身包含文本，将此文本先添加（不用改动）
        if image2text.__contains__(str(train['img_id'])):
            caption_sentence = caption_sentence + image2text[str(train['img_id'])] + " "

        # 注意，此处本来是因为有5个，所以使用了一个循环把他们联结在一起
        for i, caption in enumerate(captions):
            cap = caption["caption"]
            if cap[-1] != ".":
                cap = cap + "."
            if i != len(captions) - 1:
                caption_sentence = caption_sentence + cap + " "
            else:
                caption_sentence = caption_sentence + cap

        caption_sentence = caption_sentence.replace("..", ".").replace(". .", ".")
        sentence = question + " " + caption_sentence
        sentence = sentence.replace("?", "").replace(".", "").replace(",", "")
        # ——————————————————至此得到了每一个图像对应的具体sentence，包含了问题，caption和图像本身包含的文本
        # 注意，作stem操作，并去除停用词，此处需要修改成spicy
        word_list = list(set(list(map(pt_stemmer.stem, sentence.split(" ")))))
        word_list_nostop = list()
        for word in word_list:
            if word not in stop_words:
                word_list_nostop.append(word)

        word_list_nostop = set(word_list_nostop)

        fact = dict()
        fact_500_list = list()
        for i, triple_stem in enumerate(four_tuple.values()):
            triple_stem_list = set((triple_stem[0] + " " + triple_stem[2]).split(" "))
            if word_list_nostop & triple_stem_list:
                # if any(x in word_list_nostop for x in triple_stem_list):
                fact[triple_stem[3]] = i

        caption_word = list(set(caption_sentence.replace(".", "").replace(",", "").split(" ")))
        caption_del_sentence = ""
        for capp in caption_word:
            caption_del_sentence = caption_del_sentence + capp + " "
        new_sentence_forfact = question + " " + caption_del_sentence[:-1].replace("?", "").replace(".", "").replace(",",
                                                                                                                    "")

        tokenized_fact = [doc.split(" ") for doc in list(fact.keys())]
        bm25 = BM25Okapi(tokenized_fact)

        if len(fact) >= 500:
            fact_500 = bm25.get_top_n(new_sentence_forfact.split(" "), list(fact.keys()), n=500)
        else:
            fact_500 = bm25.get_top_n(new_sentence_forfact.split(" "), list(fact.keys()), n=len(fact))
        for f in fact_500:
            fact_dic = dict()
            fact_dic["sentence"] = f + "."
            fact_dic["id"] = fact[f]
            fact_500_list.append(fact_dic)

        okvqa_train_dict["question"] = question
        okvqa_train_dict["target"] = target
        okvqa_train_dict["answer"] = answer
        okvqa_train_dict["img_id"] = img_id
        # okvqa_train_dict["score"] = score
        okvqa_train_dict["caption"] = caption_sentence
        okvqa_train_dict["fact"] = fact_500_list
        okvqa_train_list.append(okvqa_train_dict)

    print(len(okvqa_train_list))
    # 注意，此处根据train的划分不一样，命名也要不一样，如split21，split1这种
    with open("vqa2_train_t5_3_v5_frequent_bm25_split21.json", "w") as ffp:
        json.dump(okvqa_train_list, ffp)

        # 其实test和train的代码是一样的，把它归到一个函数，其实和top-500kg实现的是一样的事情，应该就是trains和tests对应一下
        def top_500kg_test(tests, img2caption, image2text, four_tuple):
            okvqa_test_list = list()
            print(len(tests))

            for test in tqdm(tests, total=len(tests)):
                okvqa_test_dict = dict()
                question = test["sent"]
                # 训练用的最好的答案
                targets = list(test['label'].keys())
                if targets == []:
                    target = 'NNNNNNN'
                    answer = {'NNNNNNN': 0}
                else:
                    target = targets[0]
                    answer = test['label']
                # eval的时候命中任意一个即可

                img_id = test['img_id']
                captions = img2caption[str(test['img_id'])]
                caption_sentence = ""

                if image2text.__contains__(str(test['img_id'])):
                    caption_sentence = caption_sentence + image2text[str(test['img_id'])] + " "

                for i, caption in enumerate(captions):
                    cap = caption["caption"]
                    if cap[-1] != ".":
                        cap = cap + "."
                    if i != len(captions) - 1:
                        caption_sentence = caption_sentence + cap + " "
                    else:
                        caption_sentence = caption_sentence + cap

                caption_sentence = caption_sentence.replace("..", ".").replace(". .", ".")

                sentence = question + " " + caption_sentence

                sentence = sentence.replace("?", "").replace(".", "").replace(",", "")
                # 作stem操作，并去除停用词
                word_list = list(set(list(map(pt_stemmer.stem, sentence.split(" ")))))
                word_list_nostop = list()
                for word in word_list:
                    if word not in stop_words:
                        word_list_nostop.append(word)
                word_list_nostop = set(word_list_nostop)

                fact = dict()
                fact_500_list = list()
                for i, triple_stem in enumerate(four_tuple.values()):
                    triple_stem_list = set((triple_stem[0] + " " + triple_stem[2]).split(" "))
                    if word_list_nostop & triple_stem_list:
                        # if any(x in word_list_nostop for x in triple_stem_list):
                        fact[triple_stem[3]] = i

                caption_word = list(set(caption_sentence.replace(".", "").replace(",", "").split(" ")))
                caption_del_sentence = ""
                for capp in caption_word:
                    caption_del_sentence = caption_del_sentence + capp + " "
                new_sentence_forfact = question + " " + caption_del_sentence[:-1].replace("?", "").replace(".",
                                                                                                           "").replace(
                    ",", "")

                tokenized_fact = [doc.split(" ") for doc in list(fact.keys())]
                bm25 = BM25Okapi(tokenized_fact)

                if len(fact) >= 500:
                    fact_500 = bm25.get_top_n(new_sentence_forfact.split(" "), list(fact.keys()), n=500)
                else:
                    fact_500 = bm25.get_top_n(new_sentence_forfact.split(" "), list(fact.keys()), n=len(fact))
                for f in fact_500:
                    fact_dic = dict()
                    fact_dic["sentence"] = f + "."
                    fact_dic["id"] = fact[f]
                    fact_500_list.append(fact_dic)

                okvqa_test_dict["question"] = question
                okvqa_test_dict["target"] = target
                okvqa_test_dict["answer"] = answer
                okvqa_test_dict["img_id"] = img_id
                # okvqa_train_dict["score"] = score
                okvqa_test_dict["caption"] = caption_sentence
                okvqa_test_dict["fact"] = fact_500_list
                okvqa_test_list.append(okvqa_test_dict)

            print(len(okvqa_test_list))
            with open("vqa2_test_t5_1_v5_frequent_bm25_split21.json", "w") as ffp:
                json.dump(okvqa_test_list, ffp)

   trains, tests, img2caption, tripleindex_database, triplestemindex_database, relation2template, image2text = load_init()
   four_tuple = convert_stemkg2sentence(triplestemindex_database, relation2template)
   # train数据集
   top_500kg(trains, img2caption, image2text, four_tuple)
   # test数据集
   top_500kg_test(tests, img2caption, image2text, four_tuple)
import joblib
base_path = '/root/autodl-tmp/lako/LaKo-main/'
eval_data_path = '/root/autodl-tmp/lako/LaKo-main/vqa2_test_t5_1_v5_frequent_bm25_split21.json'#base_path +opt.train_data
train_data_path = '/root/autodl-tmp/lako/LaKo-main/vqa2_train_t5_3_v5_frequent_bm25_split21.json'#base_path +opt.eval_data
with open(train_data_path, 'r') as fin:
    train_examples = json.load(fin)
    
with open(eval_data_path, 'r') as fin:
    eval_examples = json.load(fin)
texts = train_examples+eval_examples
texts = [item[ 'caption'] for item in tqdm(texts)]
joblib.dump(texts,'texts')
texts[0:3]
topk=5
texts = train_examples+eval_examples
texts = [' '.join(pd.DataFrame(item['fact']).sentence[0:topk].tolist()) for item in texts]
joblib.dump(texts,'facts')
texts[0:3]
from sklearn.decomposition import PCA
import joblib
from transformers import pipeline
pca = PCA(256)

feature_extractor_llama = pipeline("feature-extraction",framework="pt",model='decapoda-research/llama-7b-hf')
feature_extractor_flan = pipeline("feature-extraction",framework="pt",model='google/flan-t5-small')
feature_extractor_dolly = pipeline("feature-extraction",framework="pt",model='databricks/dolly-v2-12b')
feature_extractor_falcon = pipeline("feature-extraction",framework="pt",model='tiiuae/falcon-7b')
feature_extractor_llama = pipeline("feature-extraction",framework="pt",model='decapoda-research/llama-7b-hf')
feature_extractor_flan = pipeline("feature-extraction",framework="pt",model='google/flan-t5-small')
feature_extractor_dolly = pipeline("feature-extraction",framework="pt",model='databricks/dolly-v2-12b')
feature_extractor_falcon = pipeline("feature-extraction",framework="pt",model='tiiuae/falcon-7b')
embeddings = {}

features = feature_extractor_llama(texts)
features =[np.array(f).mean(1).squeeze() for f in features]
features = np.array(features)
features = pca.fit_transform(features) # 白化和降维
embeddings['llama'] = features

features = feature_extractor_flan(texts)
features =[np.array(f).mean(1).squeeze() for f in features]
features = np.array(features)
features = pca.fit_transform(features) # 白化和降维
embeddings['flan'] = features

features = feature_extractor_dolly(texts)
features =[np.array(f).mean(1).squeeze() for f in features]
features = np.array(features)
features = pca.fit_transform(features) # 白化和降维
embeddings['dolly'] = features


features = feature_extractor_falcon(texts)
features =[np.array(f).mean(1).squeeze() for f in features]
features = np.array(features)
features = pca.fit_transform(features) # 白化和降维
embeddings['falcon'] = features


joblib.dump(embeddings,'teacher_knowledge')
student_model = AutoModel.from_pretrained('/root/autodl-tmp/lako/LaKo-main/t5_models/t5-base')
student_tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/lako/LaKo-main/t5_models/t5-base')
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
#%%
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.15):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
import torch
import torch.nn.functional as F

    
    
    
class Distillator4Features(pl.LightningModule):

    def __init__(self, temperature: float = 1.0, student_model=None) -> None:
        super(Distillator4Features, self).__init__()
        self.student = student_model
        self.temperature = temperature
        self.alighment= nn.Linear(768,256)
        self.gnn_projection = nn.Linear(768,256)
        self.text_projection = nn.Linear(768,256)
        self.gnn = GCN(768,768)

    def configure_optimizers(self):
        # Define your optimizer here (e.g., Adam)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @property
    def temperature(self) -> float:
        return self._temperature if self.training else 1

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value < 1:
            raise ValueError(f"Temperature must be above 1, it cannot be {value}")
        self._temperature = value

    def ensemble_distillation_loss(self,student_alignment_features,teacher_features):
        loss =0.0
        for k in teacher_features:
            loss = loss +F.mse_loss(student_alignment_features,teacher_features[k])
        loss = loss / len(teacher_features)
        return loss

    def contrastive_distillation_loss(self,student_alignment_features,teacher_features):
        loss =0.0
        projections_student = F.normalize(student_alignment_features, p=2, dim=1)
        sim_student = torch.matmul(projections_student,projections_student.T)/self.temperature
        for k in teacher_features:
            projections_teacher = F.normalize(teacher_features[k], p=2, dim=1)
            sim_teacher = torch.matmul(projections_teacher,projections_teacher.T)
            loss = loss +F.cross_entropy(sim_student,sim_teacher)
            
        loss = loss / len(teacher_features)
        return loss
    
    def GClip(self,graph_features,sentence_features):
        gnn_embedding = self.gnn_projection(graph_features)
        text_embeddings = self.text_projection(sentence_features)

        logits = (text_embeddings @ gnn_embedding.T) / self.temperature
        graph_similarity = gnn_embedding @ gnn_embedding.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (graph_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        gnn_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (gnn_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
        
    def training_step(self, batch, batch_idx):
        texts = batch.pop('text')
        facts = batch.pop('facts')
        
        facts['decoder_input_ids'] = facts['input_ids']

        outputs =  self.student.forward(**facts,output_attentions=False)
        
        node_features= outputs.last_hidden_state.mean(1)
        normed_node_featurs = F.normalize(node_features,p=2)
        adj_matrix = torch.matmul(normed_node_featurs,normed_node_featurs.T)

        graph_features = self.gnn(node_features,adj_matrix)
        sentence_features = node_features
        graph_clip_loss = self.GClip(graph_features,sentence_features)
        
        
        student_features = self.student.forward(input_ids=texts['input_ids'],attention_mask=texts['attention_mask'],decoder_input_ids =texts['input_ids']).last_hidden_state.mean(1)
        student_alignment_features = self.alighment(student_features)
        
        ensemble_distillation_loss = self.ensemble_distillation_loss(student_alignment_features,batch)
        
        contrastive_distillation_loss = self.contrastive_distillation_loss(student_alignment_features,batch)
        total_loss = (ensemble_distillation_loss+contrastive_distillation_loss+graph_clip_loss)/3.0
        self.log('train_loss', total_loss)  # Log the loss for tracking in TensorBoard or other logging tools
        return total_loss
teacher_knowledge = joblib.load('teacher_knowledge')
texts = joblib.load('texts')
facts = joblib.load('facts')
from torch.nn import functional as F

class TextDataset(Dataset):
    def __init__(self, texts,teacher_knowledge,facts):
        self.texts = texts
        self.facts = facts
        self.teacher_knowledge = teacher_knowledge

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        knowledge = {k:self.teacher_knowledge.get(k)[idx] for k in self.teacher_knowledge.keys()}
        
        fact = self.facts[idx]
        
        knowledge['text'] = text
        knowledge['fact'] = fact
        return knowledge
    
    
datasets = TextDataset(texts,teacher_knowledge,facts)
print(next(iter(datasets)))


def collate_fn(batch):
    # print(batch)
    dicts = pd.DataFrame(batch).to_dict(orient='list')

    for k in dicts:
        if k!='text' and k!='fact':
            dicts[k]= torch.from_numpy(np.vstack(dicts[k])).to(torch.float32)
        elif k=='text':
            dicts[k] = student_tokenizer(dicts[k],padding =True,truncation =True,return_tensors='pt')
        else:
            dicts[k] = student_tokenizer(dicts[k],padding =True,truncation =True,return_tensors='pt')

    return dicts

dataloaders = DataLoader(datasets,batch_size=16,shuffle=True,num_workers=4,drop_last=False,collate_fn =collate_fn )#collate_fn 
batch = next(iter(dataloaders))
print(batch)
batch.keys()
model = Distillator4Features(temperature=1,student_model=student_model)

trainer = pl.Trainer(accelerator='gpu',devices=1,max_epochs=20,num_sanity_val_steps=0,reload_dataloaders_every_n_epochs=1,default_root_dir='/root/autodl-tmp/lako/LaKo-main/checkpoints')
trainer.fit(model,dataloaders)
model.student.save_pretrained('/root/autodl-tmp/lako/LaKo-main/t5_models_knowledge_distillation')
from sklearn.metrics import classification_report
from torchmetrics.aggregation import MeanMetric
from transformers import T5ForConditionalGeneration,T5Tokenizer

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

import torchmetrics

from torch.optim import AdamW
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torch import optim

from transformers import AutoModel

from torchmetrics.classification import Accuracy,Recall,Precision,F1Score

from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor,RichProgressBar,EarlyStopping

import torch
import numpy as np
import pandas as pd
from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
)
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelWithLMHead, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar

torch.cuda.empty_cache()
# pl.seed_everything(42)

class PyTorchDataModule(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):

        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[
            labels == 0
        ] = -100  # to make sure we have correct labels for T5 text generation

        return dict(
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )
class LightningDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        num_workers: int = 2,
    ):

        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = PyTorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
class LightningModel(pl.LightningModule):

    def __init__(
        self,
        tokenizer,
        model,
        outputdir: str = "outputs",
        save_only_last_epoch: bool = False,
    ):

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.save_only_last_epoch = save_only_last_epoch
        self.validation_step_outputs=[]
        self.training_step_outputs=[]

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        self.training_step_outputs.append(loss.detach())
        
        return loss

    def validation_step(self, batch, batch_size):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True
        )
        self.validation_step_outputs.append(loss.detach())
        return loss

    def test_step(self, batch, batch_size):
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=2e-5)

    def on_train_epoch_end(self): ## 这个写法不错，适用于t5 的保存
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x for x in self.training_step_outputs])).item(),
            4,
        )
        path = f"{self.outputdir}/simplet5-epoch-{self.current_epoch}-train-loss-{str(self.average_training_loss)}-val-loss-{str(self.average_validation_loss)}"
        if self.save_only_last_epoch:
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.tokenizer.save_pretrained(path)
                self.model.save_pretrained(path)
        else:
            self.tokenizer.save_pretrained(path)
            self.model.save_pretrained(path)
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        _loss = [x.cpu() for x in self.validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )
        self.validation_step_outputs =[]


        
        
        
        
class SimpleT5(object):

    def __init__(self) -> None:
        pass

    def from_pretrained(self, model_type="t5", model_name="t5-base") -> None:

        if model_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type =="codet5":
            self.tokenizer = RobertaTokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        max_epochs: int = 5,
        use_gpu: bool = True,
        outputdir: str = "outputs",
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        precision=32,
        logger="default",
        dataloader_num_workers: int = 2,
        save_only_last_epoch: bool = False,
    ):

        self.data_module = LightningDataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        self.T5Model = LightningModel(
            tokenizer=self.tokenizer,
            model=self.model,
            outputdir=outputdir,
            save_only_last_epoch=save_only_last_epoch,
        )

        # add callbacks
        callbacks = [RichProgressBar(leave=True)]

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=early_stopping_patience_epochs,
                verbose=True,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        # add gpu support
        gpus = 'gpu' if use_gpu else 'cpu'

        # add logger
        loggers = True if logger == "default" else logger

        # prepare trainer
        trainer = pl.Trainer(
            logger=loggers,
            callbacks=callbacks,
            max_epochs=max_epochs,
            accelerator='gpu',
            precision=precision,
            log_every_n_steps=1,fast_dev_run=False,devices=1,check_val_every_n_epoch=1,val_check_interval=1.0,num_sanity_val_steps=0
        )


        # fit trainer
        trainer.fit(self.T5Model, self.data_module)

    def load_model(
        self, model_type: str = "t5", model_dir: str = "outputs", use_gpu: bool = False
    ):

        if model_type == "t5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "mt5":
            self.model = MT5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "byt5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type =="codet5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = RobertaTokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        



    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,binary=False
    ):
        # 这里应该按照长度对输入的句子进行batch，然后groupby length 比较好

        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]

        return preds
import json
from transformers import AutoTokenizer,T5ForConditionalGeneration
import torch
import transformers

import json
import sys
sys.path.append('/root/autodl-tmp/lako/LaKo-main/data_process')
import pickle
import nltk.stem.porter as pt
from re import template
from tqdm import tqdm
import numpy as np
import os.path as osp
import numpy as np
from rank_bm25 import BM25Okapi
import pdb
import os
from src.data import Dataset as LakoDataset
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
from transformers import AutoModel,AutoTokenizer
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from typing import Tuple
from torch import nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
import src
from src import model as srcmodel
from src import data as srcdata
from torch.utils.data import RandomSampler

def get_target(example):
    if 'target' in example:
        target = example['target']
        return target + ' </s>'
    elif 'answers' in example:
        return random.choice(example['answers']) + ' </s>'
    else:
        return None

#%%
class opt(object):
    gpu=0
    epochs=1000
    early_stop=30
    dataset='okvqa'
    stream=1
    use_fact='yes'
    fact_use_way = 'concat'
    attention_score_style ='mean'
    consider_context_attention ='no'
    use_last_half_layer_attention = 'no' 
    train_data="vqa2_train_t5_3_v5_frequent_bm25_split21.json"
    eval_data="vqa2_test_t5_1_v5_frequent_bm25_split21.json"
    model_size='base'
    stream=2;
    use_fact="yes"
    n_context=10
    text_maxlength=130
    # mean / max / 21mean /
    attention_score_style="21mean"
    use_last_half_layer_attention="no"
    attention_part="full_attention_of"
    iter_name=""
    answer_maxlength = 20
    batch_size = 16
# from https://github.com/hackerchenzhuo/LaKo/blob/main/run_okvqa_train.sh

with_fact = ""
from_scratch = "_from_scratch"
if opt.use_fact == "yes":
    fact_para = f"_stream_{opt.stream}_content_{opt.n_context}_"
else:
    fact_para = ""
train_data_name = opt.train_data.split('/')[-1]
iter_name = ""
opt.device = opt.gpu

#%%

train_data_path = opt.train_data
eval_data_path = opt.eval_data
with open(train_data_path, 'r') as fin:
    train_examples = json.load(fin)
with open(eval_data_path, 'r') as fin:
    eval_examples = json.load(fin)
    
train_context = ['context:'+item.get('caption')+' '+'question:'+item.get('question') for item in tqdm(train_examples)]
train_answer = [item.get('answer') for item in tqdm(train_examples)]
train_target = [get_target(item) for item in tqdm(train_examples)]
train_data = pd.DataFrame()
train_data['source'] = train_context
train_data['target'] = train_target
train_data['answer'] = train_answer


test_data = pd.DataFrame()
test_context = ['context:'+item.get('caption')+' '+'question:'+item.get('question') for item in tqdm(eval_examples)]
test_answer = [item.get('answer') for item in tqdm(eval_examples)]
test_target = [get_target(item) for item in tqdm(eval_examples)]
test_data['source'] = test_context
test_data['target'] = test_target
test_data['answer'] = test_answer

train_data.to_csv('train_data.csv',index=False)
test_data.to_csv('test_data.csv',index=False)
import pandas as pd
model = SimpleT5()
model.load_model('t5','/root/autodl-tmp/lako/LaKo-main/t5_models_knowledge_distillation', use_gpu=True)

X_train = pd.read_csv('train_data.csv')
X_test = pd.read_csv('test_data.csv')
X_train.pop('answer')
X_test.pop('answer')
X_train.columns =  ['source_text','target_text']
X_test.columns =  ['source_text','target_text']
#%%


model.fit(train_df=X_train, 
            eval_df=X_test,
            source_max_token_len = 600, 
            target_max_token_len = 43,
            batch_size = 16,
            max_epochs = 30,
            use_gpu = True,
            outputdir = "/root/autodl-tmp/lako/LaKo-main/final_models",
            early_stopping_patience_epochs = 0,
            precision = 32,dataloader_num_workers=4,save_only_last_epoch =False
            )
model.load_model('t5','/root/autodl-tmp/lako/LaKo-main/best_model_2023_09_17', use_gpu=True)
predictions = []
for text in tqdm(X_test.source_text.tolist()):
    pred = model.predict(text)
    predictions.append(pred)
from evaluate_retrieved_facts import compute_acc

X_test['target'] = X_test['target_text'].agg(lambda x:x.split(' ')[0])
X_test['predictions'] = [p[0] for p in predictions]
print('acc:',compute_acc(X_test))

