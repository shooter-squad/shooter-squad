import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
#要实现patches Embeddings: CLS Token, Position Embedding
#要实现Transformer： Attention Residuals，MLP，TransformerEncoder 解码器
#要实现 Head
#要实现ViT
#Data 
"""
img = Image.open('./cat.jpg')

fig = plt.figure()
plt.imshow(img)
"""
#image预处理
# resize to imagenet size 
#transform = Compose([Resize((84, 84)), ToTensor()])
#x = transform(img)
#x = x.unsqueeze(0) # add batch dim
#print(x.shape)
y = torch.rand(64,4,84,84)
print("deep")
#网络默认的传入的图片size是torch.Size([1, 3, 224, 224])

#接下来实现第一个: PatchEmbedding
class PatchEmbedding(nn.Module):# N=（H*W）/P*P
    def __init__(self, in_channels: int = 4, patch_size: int = 4, emb_size: int = 64, img_size: int = 84):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    
#PatchEmbedding()(x).shape
#此时图片变成torch.Size([1, X, 128])
#embedding的size是超参数，我们可以自己改，包括patch_size,in_channels这些

#把embedding层做完之后，就输入到transformer层
#transformer:Attention->ResidualAdd->MLP

#MultiheadAttention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 64, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
#output_size = batch head values_len * embedding_size

#Residuals 
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

#MLP
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
#Combine the above, Get TransformerEncoderBlock
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 64,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
#最后输出还是[1,197,768],要看一下这个1和197怎么得到的
#test:
#patches_embedded = PatchEmbedding()(x)
#TransformerEncoderBlock()(patches_embedded).shape
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
                
#Encoder层就是叠几个这样的TransformerBlock模块，这些都可以我们自己调节

#patches_embedded = PatchEmbedding()(y)
#print(TransformerEncoderBlock()(patches_embedded).shape)







class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        print(input_dims[0])#???
        self.input_vector = input_dims[0]
        self.embedding_layer = PatchEmbedding() #定义embedding层 
        self.transformer = TransformerEncoderBlock() #定义transformer层 
        #()
        #patches_embedding_vector = PatchEmbedding()(input_dims[0])
        #output_vector = TransformerEncoderBlock()(patches_embedding_vector)
        #print(output_vector.shape) #[1,142,128]
        #[1,442,128]
        #self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        #self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        #fc_input_dims = self.calculate_conv_output_dims(input_dims)
        #print(fc_input_dims)
        #input_dims = torch.flatten(output_vector)
        self.fc1 = nn.Linear(28288, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        # print(state.shape)
        #conv1 = F.relu(self.conv1(state))
        # print(conv1.shape)
        #conv2 = F.relu(self.conv2(conv1))
        #conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        #conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        #print("输入的state的shape:")
        #print(state.shape)
        embedding_vector = self.embedding_layer(state)
        output_vector = self.transformer(embedding_vector)
        #print("transformer出来之后的shape")
        #print(output_vector.shape)
        transformer_state = output_vector.view(output_vector.size()[0],-1)
        #print("reshape之后的transformer_state出来之后的shape:")
        #print(transformer_state.shape)
        #input_dims = torch.flatten(output_vector)        
        flat1 = F.relu(self.fc1(transformer_state))
        actions = self.fc2(flat1) # // NOTE: actions: [32, 6]

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
